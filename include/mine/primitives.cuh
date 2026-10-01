#pragma once
// Warp/block/device-wide scan building blocks (used instead of thrust).

#include <cstdint>
#include <cuda_runtime.h>

namespace ghuim {

constexpr unsigned kFullMask = 0xffffffffu;

template <typename T> __device__ __forceinline__ T warp_inclusive_scan(T value)
{
    const unsigned lane = threadIdx.x & 31;
#pragma unroll
    for (unsigned offset = 1; offset < 32; offset <<= 1) {
        const T other = __shfl_up_sync(kFullMask, value, offset);
        if (lane >= offset)
            value += other;
    }
    return value;
}

template <typename T> __device__ __forceinline__ T warp_sum(T value)
{
#pragma unroll
    for (unsigned offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(kFullMask, value, offset);
    return value;
}

// Inclusive scan across the block (blockDim.x a multiple of 32). `shared` holds 32 values.
// Every thread of the block must call it.
template <typename T> __device__ T block_inclusive_scan(T value, T* shared)
{
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x / 32, warps = blockDim.x / 32;
    value = warp_inclusive_scan(value);
    if (lane == 31)
        shared[warp] = value;
    __syncthreads();
    if (warp == 0) {
        T total = lane < warps ? shared[lane] : T{0};
        shared[lane] = warp_inclusive_scan(total);
    }
    __syncthreads();
    if (warp > 0)
        value += shared[warp - 1];
    __syncthreads();
    return value;
}

// In-place exclusive scan of data[0..n) on the device; *total receives the sum.
// `block_sums` needs ceil(n / 1024) elements.
void exclusive_scan(uint32_t* data, uint32_t n, uint32_t* block_sums, uint32_t* total, cudaStream_t stream);

} // namespace ghuim
