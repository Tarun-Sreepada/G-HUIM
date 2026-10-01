#include "cuda/buffer.cuh"
#include "mine/primitives.cuh"

namespace ghuim {
namespace {

constexpr uint32_t kScanBlock = 1024;

__global__ void scan_blocks(uint32_t* data, uint32_t n, uint32_t* block_sums)
{
    __shared__ uint32_t shared[32];
    const uint32_t      i     = blockIdx.x * kScanBlock + threadIdx.x;
    const uint32_t      value = i < n ? data[i] : 0;
    const uint32_t      scan  = block_inclusive_scan(value, shared);
    if (i < n)
        data[i] = scan - value;
    if (threadIdx.x == kScanBlock - 1)
        block_sums[blockIdx.x] = scan;
}

// Exclusive scan of the per-block sums by a single block, carrying between chunks.
__global__ void scan_sums(uint32_t* sums, uint32_t n, uint32_t* total)
{
    __shared__ uint32_t shared[32];
    __shared__ uint32_t carry;
    if (threadIdx.x == 0)
        carry = 0;
    __syncthreads();
    for (uint32_t base = 0; base < n; base += kScanBlock) {
        const uint32_t i     = base + threadIdx.x;
        const uint32_t value = i < n ? sums[i] : 0;
        const uint32_t scan  = block_inclusive_scan(value, shared);
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

__global__ void add_block_offsets(uint32_t* data, uint32_t n, const uint32_t* block_sums)
{
    const uint32_t i = blockIdx.x * kScanBlock + threadIdx.x;
    if (i < n)
        data[i] += block_sums[blockIdx.x];
}

} // namespace

void exclusive_scan(uint32_t* data, uint32_t n, uint32_t* block_sums, uint32_t* total, cudaStream_t stream)
{
    const uint32_t blocks = (n + kScanBlock - 1) / kScanBlock;
    if (n == 0) {
        check_cuda(cudaMemsetAsync(total, 0, sizeof(uint32_t), stream), "scan");
        return;
    }
    scan_blocks<<<blocks, kScanBlock, 0, stream>>>(data, n, block_sums);
    scan_sums<<<1, kScanBlock, 0, stream>>>(block_sums, blocks, total);
    add_block_offsets<<<blocks, kScanBlock, 0, stream>>>(data, n, block_sums);
    check_cuda(cudaGetLastError(), "launch scan");
}

} // namespace ghuim
