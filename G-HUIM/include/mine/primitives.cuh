#pragma once
// Warp/block/device-wide scan building blocks (custom kernels; no Thrust or CUB).

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
        T total      = lane < warps ? shared[lane] : T{0};
        shared[lane] = warp_inclusive_scan(total);
    }
    __syncthreads();
    if (warp > 0)
        value += shared[warp - 1];
    __syncthreads();
    return value;
}

// In-place exclusive scan of data[0..n) on the device; *total receives the sum.
// `block_sums` needs scan_block_count(n) elements. Instantiated for uint32_t and uint64_t.
template <typename T>
void exclusive_scan(T* data, uint32_t n, T* block_sums, T* total, cudaStream_t stream);

constexpr uint32_t kScanBlock = 1024;

constexpr uint32_t scan_block_count(uint64_t n)
{
    return static_cast<uint32_t>((n + kScanBlock - 1) / kScanBlock);
}

} // namespace ghuim
