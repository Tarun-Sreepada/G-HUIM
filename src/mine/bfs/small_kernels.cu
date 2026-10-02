// Kernels for tasks one thread group processes alone: a lane group, a warp, or a block.

#include "cuda/buffer.cuh"
#include "mine/bfs/device/process.cuh"
#include "mine/bfs/kernels.cuh"

#include <algorithm>

namespace ghuim::bfs {
namespace {

constexpr uint32_t kMaxGrid = 65535;

// Block-sized tasks: one block per task; its warps split the transactions.
template <typename Mode> __global__ void block_tasks(BatchView b, const uint32_t* order, uint32_t count)
{
    for (uint32_t i = blockIdx.x; i < count; i += gridDim.x) {
        process_task<Mode>(BlockGroup{}, b, order[i], threadIdx.x / 32, blockDim.x / 32, true);
        __syncthreads(); // the next task reuses the block
    }
}

// Warp-sized tasks: every warp of the block runs its own task.
template <typename Mode> __global__ void warp_tasks(BatchView b, const uint32_t* order, uint32_t count)
{
    const uint32_t warp = threadIdx.x / 32, warps = blockDim.x / 32;
    for (uint32_t i = blockIdx.x * warps + warp; i < count; i += gridDim.x * warps) {
        process_task<Mode>(WarpGroup{}, b, order[i], 0, 1);
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
        process_task<LaneTx>(g, b, order[i], 0, 1);
        g.sync();
    }
}

} // namespace

void launch_warp_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, const AutoTune& tune,
                       cudaStream_t stream)
{
    const uint32_t warps = tune.packed_block_warps;
    const uint32_t blocks = std::min(ceil_div(count, warps), kMaxGrid);
    if (tune.warp_mode)
        warp_tasks<WarpTx><<<blocks, warps * 32, 0, stream>>>(batch, order, count);
    else
        warp_tasks<LaneTx><<<blocks, warps * 32, 0, stream>>>(batch, order, count);
    check_launch("launch warp-task kernel");
}

void launch_block_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, uint32_t bin,
                        const AutoTune& tune, cudaStream_t stream)
{
    if (tune.staged) {
        // The block keeps the bin's compute warps and adds the IO warps (at most 32 warps in all).
        const uint32_t warps = std::min((1u << bin) + tune.io_warps, 32u);
        block_tasks<StagedTx><<<std::min(count, kMaxGrid), warps * 32, kStageBarrierBytes + tune.ring_bytes + tune.table_bytes,
                                stream>>>(batch, order, count);
    }
    else if (tune.warp_mode)
        block_tasks<WarpTx><<<std::min(count, kMaxGrid), 32u << bin, 0, stream>>>(batch, order, count);
    else
        block_tasks<LaneTx><<<std::min(count, kMaxGrid), 32u << bin, tune.table_bytes, stream>>>(batch, order, count);
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

const void* block_kernel(bool warp_mode)
{
    return warp_mode ? reinterpret_cast<const void*>(block_tasks<WarpTx>)
                     : reinterpret_cast<const void*>(block_tasks<LaneTx>);
}

const void* block_kernel_staged()
{
    return reinterpret_cast<const void*>(block_tasks<StagedTx>);
}

void prepare_staged_block_kernel(uint32_t dynamic_bytes)
{
    check_cuda(cudaFuncSetAttribute(block_tasks<StagedTx>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(dynamic_bytes)),
               "raise staged block kernel shared memory");
}

const void* warp_kernel(bool warp_mode)
{
    return warp_mode ? reinterpret_cast<const void*>(warp_tasks<WarpTx>)
                     : reinterpret_cast<const void*>(warp_tasks<LaneTx>);
}

const void* subwarp_kernel()
{
    return reinterpret_cast<const void*>(subwarp_tasks);
}

} // namespace ghuim::bfs
