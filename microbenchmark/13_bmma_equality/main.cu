// 13: tensor-core equality test vs CUDA-core compare, all-pairs.
// A 64-bit key k is encoded as the 128-bit vector [k | ~k]. With the 1-bit tensor-core op (AND + popc),
//   popc([a|~a] & [b|~b]) = popc(a&b) + popc(~a&~b) = 64 - hamming(a, b),
// which is exactly 64 iff a == b. One mma.m8n8k128 produces 64 pair results (8 queries x 8 candidates).
// Baseline: thread-per-query loop comparing 64-bit keys against broadcast candidates.
// Needs sm_80+ (AND + popc form; the XOR form is deprecated on Hopper).
// Fragment layout (PTX ISA, m8n8k128): lane l holds A word [row l/4][bits (l%4)*32 ..], B word
// [col l/4][bits (l%4)*32 ..], and C/D values for row l/4, cols (l%4)*2 + {0,1}.

#include "bench.cuh"

constexpr uint32_t kSplits = 32; // candidate chunks per query tile, to fill the GPU at small sizes

__global__ void make_keys(u64* keys, uint32_t n, uint32_t domain, u64 seed)
{
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        keys[i] = (hash64(i ^ (seed << 32)) % domain) * 0x9E3779B97F4A7C15ULL;
}

// enc[4*i + {0,1,2,3}] = {lo, hi, ~lo, ~hi}
__global__ void encode(const u64* keys, uint32_t* enc, uint32_t n)
{
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const u64 k = keys[i];
        const uint32_t lo = static_cast<uint32_t>(k), hi = static_cast<uint32_t>(k >> 32);
        enc[4 * i + 0] = lo;
        enc[4 * i + 1] = hi;
        enc[4 * i + 2] = ~lo;
        enc[4 * i + 3] = ~hi;
    }
}

__device__ inline void bmma_and_popc(int& d0, int& d1, uint32_t a, uint32_t b)
{
    asm volatile("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc {%0, %1}, {%2}, {%3}, {%4, %5};\n"
                 : "=r"(d0), "=r"(d1)
                 : "r"(a), "r"(b), "r"(0), "r"(0));
}

// block = 4 warps, each warp owns one tile of 8 queries; blockIdx.y selects a candidate chunk.
__global__ void tensor_kernel(const uint32_t* qenc, const uint32_t* cenc, uint32_t M, uint32_t N, uint32_t chunk, u64* total)
{
    const uint32_t warp  = threadIdx.x / 32, lane = threadIdx.x % 32;
    const uint32_t qtile = blockIdx.x * 4 + warp;
    if (qtile * 8 >= M)
        return; // warp-uniform
    const uint32_t a = qenc[(size_t)(qtile * 8 + lane / 4) * 4 + lane % 4];

    const uint32_t c_begin = blockIdx.y * chunk, c_end = min(N, c_begin + chunk);
    uint32_t       cnt = 0;
#pragma unroll 8
    for (uint32_t c = c_begin; c < c_end; c += 8) {
        const uint32_t b = cenc[(size_t)(c + lane / 4) * 4 + lane % 4];
        int            d0, d1;
        bmma_and_popc(d0, d1, a, b);
        cnt += (d0 == 64) + (d1 == 64);
    }
    for (int off = 16; off > 0; off >>= 1)
        cnt += __shfl_xor_sync(0xffffffffu, cnt, off);
    if (lane == 0 && cnt)
        atomicAdd(total, (u64)cnt);
}

__global__ void cuda_core_kernel(const u64* q, const u64* c, uint32_t M, uint32_t N, uint32_t chunk, u64* total)
{
    const uint32_t qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= M)
        return;
    const u64      key     = q[qi];
    const uint32_t c_begin = blockIdx.y * chunk, c_end = min(N, c_begin + chunk);
    uint32_t       cnt = 0;
#pragma unroll 8
    for (uint32_t j = c_begin; j < c_end; ++j)
        cnt += (__ldg(&c[j]) == key);
    if (cnt)
        atomicAdd(total, (u64)cnt);
}

int main()
{
    print_device();
    constexpr uint32_t kMaxN = 65536;
    DeviceBuffer<u64>      qkeys(kMaxN), ckeys(kMaxN);
    DeviceBuffer<uint32_t> qenc(kMaxN * 4), cenc(kMaxN * 4);
    DeviceBuffer<u64>      total(1);

    std::printf("m_equals_n,variant,ms,gpairs_per_s,matches,ok\n");
    for (uint32_t n : {1024u, 4096u, 16384u, 65536u}) {
        make_keys<<<256, 256>>>(qkeys.get(), n, n, 1);
        make_keys<<<256, 256>>>(ckeys.get(), n, n, 2);
        encode<<<256, 256>>>(qkeys.get(), qenc.get(), n);
        encode<<<256, 256>>>(ckeys.get(), cenc.get(), n);
        CUDA_CHECK(cudaGetLastError());

        const uint32_t chunk = n / kSplits;
        const double   pairs = (double)n * n;

        const float cc_ms = time_ms([&] {
            CUDA_CHECK(cudaMemsetAsync(total.get(), 0, sizeof(u64)));
            cuda_core_kernel<<<dim3(n / 128, kSplits), 128>>>(qkeys.get(), ckeys.get(), n, n, chunk, total.get());
        });
        const u64 cc_matches = read_scalar(total.get());
        std::printf("%u,cuda_core,%.4f,%.3f,%llu,1\n", n, cc_ms, pairs / cc_ms / 1e6, cc_matches);

        const float tc_ms = time_ms([&] {
            CUDA_CHECK(cudaMemsetAsync(total.get(), 0, sizeof(u64)));
            tensor_kernel<<<dim3(n / 32, kSplits), 128>>>(qenc.get(), cenc.get(), n, n, chunk, total.get());
        });
        const u64 tc_matches = read_scalar(total.get());
        std::printf("%u,tensor_b1,%.4f,%.3f,%llu,%d\n", n, tc_ms, pairs / tc_ms / 1e6, tc_matches,
                    tc_matches == cc_matches);
    }
    return 0;
}
