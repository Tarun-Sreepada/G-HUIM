// adaptive-optimized kernels for tasks that fit in a block or less.
//
// A task is processed by a group of lanes sharing a warp (tiny tasks), a warp, or a block, depending
// on its share of the batch's work (see place_task). Every lane handles one transaction at a time;
// the lanes of a group loop in lockstep over the longest work among them so they stay converged. All
// per-task state lives in global scratch.

#include "cuda/buffer.cuh"
#include "mine/dfs/kernels.cuh"
#include "mine/dfs/optimized/kernels.hpp"
#include "mine/dfs/optimized/process.cuh"

#include <algorithm>

namespace ghuim::inline GHUIM_WIDTHS::dfs {
namespace opt {
namespace {

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

// Tiny tasks (at most a warp's worth of transactions): a group of `group` lanes per task, one lane
// per transaction, so a warp runs 32 / group tasks. Tasks in a launch have similar sizes.
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

const void* block_kernel() { return reinterpret_cast<const void*>(block_tasks); }
const void* warp_kernel() { return reinterpret_cast<const void*>(warp_tasks); }
const void* subwarp_kernel() { return reinterpret_cast<const void*>(subwarp_tasks); }

} // namespace opt

using namespace opt;

void launch_optimized_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, uint32_t bin,
                            const AutoTune& tune, cudaStream_t stream)
{
    if (count == 0)
        return;
    if (bin == 0) {
        const uint32_t warps = tune.packed_block_warps;
        warp_tasks<<<std::min<uint32_t>(ceil_div(count, warps), 65535), warps * 32, 0, stream>>>(batch, order, count);
    }
    else
        block_tasks<<<std::min<uint32_t>(count, 65535), 32u << bin, 0, stream>>>(batch, order, count);
    check_cuda(cudaGetLastError(), "launch optimized tasks");
}

void launch_subwarp_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, uint32_t group,
                          const AutoTune& tune, cudaStream_t stream)
{
    if (count == 0)
        return;
    const uint32_t threads = tune.subwarp_block_warps * 32;
    const uint64_t blocks  = (uint64_t{count} * group + threads - 1) / threads;
    subwarp_tasks<<<static_cast<uint32_t>(std::min<uint64_t>(blocks, 1u << 20)), threads, 0, stream>>>(batch, order,
                                                                                                       count, group);
    check_cuda(cudaGetLastError(), "launch subwarp tasks");
}

} // namespace ghuim::inline GHUIM_WIDTHS::dfs
