// Kernels for tasks one thread group processes alone: a lane group, a warp, or a block.

#include "cuda/buffer.cuh"
#include "mine/bfs/device/process.cuh"
#include "mine/bfs/kernels.cuh"

#include <algorithm>

namespace ghuim::bfs {
namespace {

constexpr uint32_t kMaxGrid = 65535;

// Block-sized tasks: one block per task; its warps split the transactions.
__global__ void block_tasks(BatchView b, const uint32_t* order, uint32_t count)
{
    for (uint32_t i = blockIdx.x; i < count; i += gridDim.x) {
        process_task(BlockGroup{}, b, order[i], threadIdx.x / 32, blockDim.x / 32);
        __syncthreads(); // the next task reuses the block
    }
}

// Warp-sized tasks: every warp of the block runs its own task.
__global__ void warp_tasks(BatchView b, const uint32_t* order, uint32_t count)
{
    const uint32_t warp = threadIdx.x / 32, warps = blockDim.x / 32;
    for (uint32_t i = blockIdx.x * warps + warp; i < count; i += gridDim.x * warps) {
        process_task(WarpGroup{}, b, order[i], 0, 1);
        __syncwarp();
    }
}

// Tiny tasks (at most a warp's worth of transactions): a group of `group` lanes per task.
__global__ void subwarp_tasks(BatchView b, const uint32_t* order, uint32_t count, uint32_t group)
{
    const SubGroup g(group);
    const uint32_t first  = (blockIdx.x * blockDim.x + threadIdx.x) / group;
    const uint32_t groups = gridDim.x * blockDim.x / group;
    for (uint32_t i = first; i < count; i += groups) {
        process_task(g, b, order[i], 0, 1);
        g.sync();
    }
}

} // namespace

void launch_warp_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, const AutoTune& tune,
                       cudaStream_t stream)
{
    const uint32_t warps = tune.packed_block_warps;
    warp_tasks<<<std::min(ceil_div(count, warps), kMaxGrid), warps * 32, 0, stream>>>(batch, order, count);
    check_launch("launch warp-task kernel");
}

void launch_block_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, uint32_t bin,
                        cudaStream_t stream)
{
    block_tasks<<<std::min(count, kMaxGrid), 32u << bin, 0, stream>>>(batch, order, count);
    check_launch("launch block-task kernel");
}

void launch_subwarp_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, uint32_t group,
                          const AutoTune& tune, cudaStream_t stream)
{
    const uint32_t threads = tune.subwarp_block_warps * 32;
    const uint64_t blocks  = (uint64_t{count} * group + threads - 1) / threads;
    subwarp_tasks<<<static_cast<uint32_t>(std::min<uint64_t>(blocks, 1u << 20)), threads, 0, stream>>>(batch, order,
                                                                                                       count, group);
    check_launch("launch sub-warp kernel");
}

const void* block_kernel()
{
    return reinterpret_cast<const void*>(block_tasks);
}

const void* warp_kernel()
{
    return reinterpret_cast<const void*>(warp_tasks);
}

const void* subwarp_kernel()
{
    return reinterpret_cast<const void*>(subwarp_tasks);
}

} // namespace ghuim::bfs
