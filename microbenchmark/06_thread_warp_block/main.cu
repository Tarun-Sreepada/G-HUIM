// 06: which unit should own a transaction: a thread, a warp or a block?
// Task per transaction: sum utils of the items passing a filter ((key & 3) != 0), like a "kept items" sum.
//   thread: one thread per transaction on the warp-wide tiled layout (S = 32), lanes read coalesced
//   warp  : one warp per transaction on the contiguous layout, lanes stride the items, shuffle reduce
//   block : 128 threads per transaction on the contiguous layout, warp reduce + shared memory
// Total items are constant (2^25), so only the transaction length changes.

#include "bench.cuh"

constexpr size_t kTotalItems = 1u << 25;
constexpr int    kBlock      = 128;

__host__ __device__ inline uint32_t key_value(uint32_t tx, uint32_t k) { return hash32(tx * 0x9E3779B1u + k * 0x85EBCA6Bu) & 7u; }
__host__ __device__ inline uint32_t util_value(uint32_t tx, uint32_t k) { return (hash32(tx ^ (k * 0xC2B2AE35u)) & 15u) + 1u; }

// S = 1 -> contiguous, S = 32 -> warp-wide tiles.
__global__ void init_kernel(uint32_t* keys, uint32_t* utils, uint32_t n_tx, uint32_t len, uint32_t S)
{
    const size_t n = (size_t)n_tx * len;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint32_t tx = static_cast<uint32_t>(i / len), k = static_cast<uint32_t>(i % len);
        const size_t   slot = (size_t)(tx / S) * S * len + (size_t)k * S + (tx % S);
        keys[slot]  = key_value(tx, k);
        utils[slot] = util_value(tx, k);
    }
}

__global__ void thread_kernel(const uint32_t* keys, const uint32_t* utils, u64* out, uint32_t n_tx, uint32_t len)
{
    const uint32_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx >= n_tx)
        return;
    const size_t base = (size_t)(tx / 32) * 32 * len + (tx % 32);
    u64          sum  = 0;
    for (uint32_t k = 0; k < len; ++k) {
        const size_t slot = base + (size_t)k * 32;
        if (keys[slot] & 3u)
            sum += utils[slot];
    }
    out[tx] = sum;
}

__device__ inline u64 warp_sum(u64 v)
{
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

__global__ void warp_kernel(const uint32_t* keys, const uint32_t* utils, u64* out, uint32_t n_tx, uint32_t len)
{
    const uint32_t tx   = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const uint32_t lane = threadIdx.x % 32;
    if (tx >= n_tx)
        return;
    const size_t base = (size_t)tx * len;
    u64          sum  = 0;
    for (uint32_t k = lane; k < len; k += 32)
        if (keys[base + k] & 3u)
            sum += utils[base + k];
    sum = warp_sum(sum);
    if (lane == 0)
        out[tx] = sum;
}

__global__ void block_kernel(const uint32_t* keys, const uint32_t* utils, u64* out, uint32_t len)
{
    __shared__ u64 partial[kBlock / 32];
    const uint32_t tx   = blockIdx.x;
    const size_t   base = (size_t)tx * len;
    u64            sum  = 0;
    for (uint32_t k = threadIdx.x; k < len; k += blockDim.x)
        if (keys[base + k] & 3u)
            sum += utils[base + k];
    sum = warp_sum(sum);
    if (threadIdx.x % 32 == 0)
        partial[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        u64 total = 0;
        for (int w = 0; w < kBlock / 32; ++w)
            total += partial[w];
        out[tx] = total;
    }
}

int main()
{
    print_device();
    DeviceBuffer<uint32_t> keys(kTotalItems), utils(kTotalItems);
    DeviceBuffer<u64>      out(kTotalItems / 4);

    std::printf("len,mapping,ms,gitems_per_s,ok\n");
    for (uint32_t len : {4u, 8u, 16u, 32u, 64u, 128u, 256u, 512u, 1024u, 4096u}) {
        const uint32_t n_tx = static_cast<uint32_t>(kTotalItems / len);

        init_kernel<<<1024, 256>>>(keys.get(), utils.get(), n_tx, len, 32);
        CUDA_CHECK(cudaGetLastError());
        const float t_ms  = time_ms([&] { thread_kernel<<<(n_tx + 255) / 256, 256>>>(keys.get(), utils.get(), out.get(), n_tx, len); });
        const u64   ref   = device_sum_u64(out.get(), n_tx);
        std::printf("%u,thread,%.4f,%.3f,1\n", len, t_ms, kTotalItems / t_ms / 1e6);

        init_kernel<<<1024, 256>>>(keys.get(), utils.get(), n_tx, len, 1);
        CUDA_CHECK(cudaGetLastError());
        const float w_ms = time_ms([&] { warp_kernel<<<(n_tx * 32 + kBlock - 1) / kBlock, kBlock>>>(keys.get(), utils.get(), out.get(), n_tx, len); });
        const u64   w_sum = device_sum_u64(out.get(), n_tx);
        std::printf("%u,warp,%.4f,%.3f,%d\n", len, w_ms, kTotalItems / w_ms / 1e6, w_sum == ref);

        const float b_ms = time_ms([&] { block_kernel<<<n_tx, kBlock>>>(keys.get(), utils.get(), out.get(), len); });
        const u64   b_sum = device_sum_u64(out.get(), n_tx);
        std::printf("%u,block,%.4f,%.3f,%d\n", len, b_ms, kTotalItems / b_ms / 1e6, b_sum == ref);
    }
    return 0;
}
