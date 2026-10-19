// 04: cp.async staging ring depth and entry size.
// Each block streams chunks of `entry` bytes from global memory and does `passes` passes of ALU work over each
// chunk. depth 0 reads global memory directly; depth S >= 1 stages through an S-slot shared-memory ring
// (depth 1 = copy, wait, compute: no overlap; depth >= 2 prefetches S-1 chunks ahead).

#include "bench.cuh"

constexpr size_t kTotalBytes = 256u << 20;

__device__ inline uint32_t mix_chunk(const uint32_t* buf, uint32_t words, int passes, uint32_t acc)
{
    for (int p = 0; p < passes; ++p)
        for (uint32_t i = threadIdx.x; i < words; i += blockDim.x)
            acc = acc * 31u + (buf[i] ^ static_cast<uint32_t>(p));
    return acc;
}

__device__ inline uint32_t chunks_for_block(uint32_t total_chunks)
{
    return total_chunks > blockIdx.x ? (total_chunks - blockIdx.x + gridDim.x - 1) / gridDim.x : 0;
}

__global__ void direct_kernel(const uint32_t* __restrict__ data, uint32_t chunk_words, uint32_t total_chunks, int passes,
                              uint32_t* out)
{
    const uint32_t n   = chunks_for_block(total_chunks);
    uint32_t       acc = 0;
    for (uint32_t c = 0; c < n; ++c)
        acc = mix_chunk(data + (size_t)(blockIdx.x + (size_t)c * gridDim.x) * chunk_words, chunk_words, passes, acc);
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template <int S>
__global__ void ring_kernel(const uint32_t* __restrict__ data, uint32_t chunk_words, uint32_t total_chunks, int passes,
                            uint32_t* out)
{
    extern __shared__ __align__(16) uint32_t ring[];
    const uint32_t n = chunks_for_block(total_chunks);

    auto issue = [&](uint32_t c) {
        if (c < n) {
            uint32_t*       dst = ring + (size_t)(c % S) * chunk_words;
            const uint32_t* src = data + (size_t)(blockIdx.x + (size_t)c * gridDim.x) * chunk_words;
            for (uint32_t i = threadIdx.x; i < chunk_words / 4; i += blockDim.x)
                cp_async16(dst + 4 * i, src + 4 * i);
        }
        cp_commit(); // always commit so group counting stays uniform
    };

    for (int s = 0; s < S - 1; ++s)
        issue(s);

    uint32_t acc = 0;
    for (uint32_t c = 0; c < n; ++c) {
        issue(c + S - 1); // refills the slot consumed in the previous iteration
        cp_wait<S - 1>(); // chunk c has landed
        __syncthreads();
        acc = mix_chunk(ring + (size_t)(c % S) * chunk_words, chunk_words, passes, acc);
        __syncthreads();
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template <int S> void set_smem_limit()
{
    CUDA_CHECK(cudaFuncSetAttribute(ring_kernel<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
}

int main()
{
    print_device();
    DeviceBuffer<uint32_t> data(kTotalBytes / 4);
    fill_hash_u32(data.get(), data.size(), 7u);
    const int blocks = sm_count() * 2;
    DeviceBuffer<uint32_t> out((size_t)blocks * 256);

    set_smem_limit<1>();
    set_smem_limit<2>();
    set_smem_limit<3>();
    set_smem_limit<4>();

    std::printf("entry_bytes,depth,passes,ms,gb_per_s,ok\n");
    for (uint32_t entry : {1024u, 2048u, 4096u, 8192u}) {
        const uint32_t words  = entry / 4;
        const uint32_t chunks = static_cast<uint32_t>(kTotalBytes / entry);
        for (int passes : {2, 16}) {
            u64 ref = 0;
            for (int depth = 0; depth <= 4; ++depth) {
                const size_t smem = (size_t)depth * entry;
                auto         launch = [&] {
                    switch (depth) {
                    case 0:
                        direct_kernel<<<blocks, 256>>>(data.get(), words, chunks, passes, out.get());
                        break;
                    case 1:
                        ring_kernel<1><<<blocks, 256, smem>>>(data.get(), words, chunks, passes, out.get());
                        break;
                    case 2:
                        ring_kernel<2><<<blocks, 256, smem>>>(data.get(), words, chunks, passes, out.get());
                        break;
                    case 3:
                        ring_kernel<3><<<blocks, 256, smem>>>(data.get(), words, chunks, passes, out.get());
                        break;
                    default:
                        ring_kernel<4><<<blocks, 256, smem>>>(data.get(), words, chunks, passes, out.get());
                        break;
                    }
                };
                const float ms  = time_ms(launch);
                const u64   sum = device_sum_u32(out.get(), out.size());
                if (depth == 0)
                    ref = sum;
                std::printf("%u,%d,%d,%.4f,%.2f,%d\n", entry, depth, passes, ms, kTotalBytes / ms / 1e6, sum == ref);
            }
        }
    }
    return 0;
}
