// Static kernels: every task is processed by one block of a fixed size.

#include "cuda/buffer.cuh"
#include "mine/dfs/baseline/process.cuh"
#include "mine/dfs/kernels.cuh"

#include <algorithm>

namespace ghuim::dfs {

using namespace baseline;

namespace {

__global__ void block_tasks(BatchView b, const uint32_t* order, uint32_t count)
{
    for (uint32_t i = blockIdx.x; i < count; i += gridDim.x)
        process_task(BlockGroup{}, b, order[i]);
}

uint32_t grid_for(uint64_t units, uint32_t per_block)
{
    return static_cast<uint32_t>(std::min<uint64_t>((units + per_block - 1) / per_block, 1u << 20));
}

} // namespace

void launch_block_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, uint32_t block_size,
                        cudaStream_t stream)
{
    if (count == 0)
        return;
    block_tasks<<<grid_for(count, 1), block_size, 0, stream>>>(batch, order, count);
    check_cuda(cudaGetLastError(), "launch block tasks");
}

} // namespace ghuim::dfs
