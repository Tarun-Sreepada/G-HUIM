#include "cuda/buffer.cuh"
#include "mine/primitives.cuh"

namespace ghuim {
namespace {

template <typename T> __global__ void scan_blocks(T* data, uint32_t n, T* block_sums)
{
    __shared__ T   shared[32];
    const uint32_t i     = blockIdx.x * kScanBlock + threadIdx.x;
    const T        value = i < n ? data[i] : T{0};
    const T        scan  = block_inclusive_scan(value, shared);
    if (i < n)
        data[i] = scan - value;
    if (threadIdx.x == kScanBlock - 1)
        block_sums[blockIdx.x] = scan;
}

// Exclusive scan of the per-block sums by a single block, carrying between chunks.
template <typename T> __global__ void scan_sums(T* sums, uint32_t n, T* total)
{
    __shared__ T shared[32];
    __shared__ T carry;
    if (threadIdx.x == 0)
        carry = 0;
    __syncthreads();
    for (uint32_t base = 0; base < n; base += kScanBlock) {
        const uint32_t i     = base + threadIdx.x;
        const T        value = i < n ? sums[i] : T{0};
        const T        scan  = block_inclusive_scan(value, shared);
        if (i < n)
            sums[i] = carry + scan - value;
        __syncthreads();
        if (threadIdx.x == kScanBlock - 1)
            carry += scan;
        __syncthreads();
    }
    if (threadIdx.x == 0)
        *total = carry;
}

template <typename T> __global__ void add_block_offsets(T* data, uint32_t n, const T* block_sums)
{
    const uint32_t i = blockIdx.x * kScanBlock + threadIdx.x;
    if (i < n)
        data[i] += block_sums[blockIdx.x];
}

} // namespace

template <typename T>
void exclusive_scan(T* data, uint32_t n, T* block_sums, T* total, cudaStream_t stream)
{
    if (n == 0) {
        check_cuda(cudaMemsetAsync(total, 0, sizeof(T), stream), "scan");
        return;
    }
    const uint32_t blocks = scan_block_count(n);
    scan_blocks<<<blocks, kScanBlock, 0, stream>>>(data, n, block_sums);
    scan_sums<<<1, kScanBlock, 0, stream>>>(block_sums, blocks, total);
    add_block_offsets<<<blocks, kScanBlock, 0, stream>>>(data, n, block_sums);
    check_launch("launch scan");
}

template void exclusive_scan<uint32_t>(uint32_t*, uint32_t, uint32_t*, uint32_t*, cudaStream_t);
template void exclusive_scan<uint64_t>(uint64_t*, uint32_t, uint64_t*, uint64_t*, cudaStream_t);

} // namespace ghuim
