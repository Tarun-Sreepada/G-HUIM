// Static-parallelism kernels: every task is processed by one fixed-size group (a block, a warp or a
// single thread).

#include "cuda/buffer.cuh"
#include "mine/baseline/process.cuh"
#include "mine/kernels.cuh"

#include <algorithm>

namespace ghuim {

using namespace baseline;

namespace {

__global__ void block_tasks(BatchView b, const uint32_t* order, uint32_t count)
{
    for (uint32_t i = blockIdx.x; i < count; i += gridDim.x)
        process_task(BlockGroup{}, b, order[i]);
}

__global__ void warp_tasks(BatchView b, const uint32_t* order, uint32_t count)
{
    const uint32_t warp  = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const uint32_t warps = gridDim.x * blockDim.x / 32;
    for (uint32_t i = warp; i < count; i += warps)
        process_task(WarpGroup{}, b, order[i]);
}

__global__ void thread_tasks(BatchView b, const uint32_t* order, uint32_t count)
{
    const uint32_t thread  = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t threads = gridDim.x * blockDim.x;
    for (uint32_t i = thread; i < count; i += threads)
        process_task(ThreadGroup{}, b, order[i]);
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

void launch_warp_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, cudaStream_t stream)
{
    if (count == 0)
        return;
    constexpr uint32_t block = 256;
    warp_tasks<<<grid_for(uint64_t{count} * 32, block), block, 0, stream>>>(batch, order, count);
    check_cuda(cudaGetLastError(), "launch warp tasks");
}

void launch_thread_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, cudaStream_t stream)
{
    if (count == 0)
        return;
    constexpr uint32_t block = 128;
    thread_tasks<<<grid_for(count, block), block, 0, stream>>>(batch, order, count);
    check_cuda(cudaGetLastError(), "launch thread tasks");
}

} // namespace ghuim
