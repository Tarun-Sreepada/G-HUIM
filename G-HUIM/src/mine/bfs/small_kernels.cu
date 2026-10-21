// Kernels for tasks one thread group processes alone: a lane group, a warp, or a block.

#include "cuda/buffer.cuh"
#include "mine/bfs/device/process.cuh"
#include "mine/bfs/kernels.cuh"

#include <algorithm>

namespace ghuim::inline GHUIM_WIDTHS::bfs {
namespace {

constexpr uint32_t kMaxGrid = 65535;

// Block-sized tasks: one block per task; its warps split the transactions.
template <typename Mode> __device__ __forceinline__ void block_tasks_body(BatchView b, const uint32_t* order, uint32_t count)
{
    for (uint32_t i = blockIdx.x; i < count; i += gridDim.x) {
        process_task<Mode>(BlockGroup{}, b, order[i], threadIdx.x / 32, blockDim.x / 32);
        __syncthreads(); // the next task reuses the block
    }
}

template <typename Mode> __global__ void block_tasks(BatchView b, const uint32_t* order, uint32_t count)
{
    block_tasks_body<Mode>(b, order, count);
}

// The lane kernel is capped at 48 registers: past that, 256-thread blocks stop filling the SM, which
// shrinks the largest full-occupancy block and moves tasks into the sliced (huge) kernels.
template <> __global__ void __maxnreg__(48) block_tasks<LaneTx>(BatchView b, const uint32_t* order, uint32_t count)
{
    block_tasks_body<LaneTx>(b, order, count);
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
template <typename Mode, bool LocalTail = false>
__global__ void subwarp_tasks(BatchView b, const uint32_t* order, uint32_t count, uint32_t group)
{
    const SubGroup g(group);
    const uint32_t first  = (blockIdx.x * blockDim.x + threadIdx.x) / group;
    const uint32_t groups = gridDim.x * blockDim.x / group;
    for (uint32_t i = first; i < count; i += groups) {
        process_task<Mode, LocalTail>(g, b, order[i], 0, 1);
        g.sync();
    }
}

} // namespace

void launch_warp_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, const AutoTune& tune, cudaStream_t stream)
{
    const uint32_t warps  = tune.packed_block_warps;
    const uint32_t blocks = std::min(ceil_div(count, warps), kMaxGrid);
    with_mode(tune, [&](auto mode) {
        warp_tasks<decltype(mode)><<<blocks, warps * 32, 0, stream>>>(batch, order, count);
    });
    check_launch("launch warp-task kernel");
}

void launch_block_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, uint32_t bin, const AutoTune& tune,
                        cudaStream_t stream)
{
    if (tune.staged) {
        // The block keeps the bin's compute warps and adds the IO warps (at most 32 warps in all).
        const uint32_t warps   = std::min((1u << bin) + tune.io_warps, 32u);
        const uint32_t dynamic = kStageBarrierBytes + tune.ring_bytes;
        block_tasks<StagedTx><<<std::min(count, kMaxGrid), warps * 32, dynamic, stream>>>(batch, order, count);
    }
    else
        with_mode(tune, [&](auto mode) {
            block_tasks<decltype(mode)><<<std::min(count, kMaxGrid), 32u << bin, 0, stream>>>(batch, order, count);
        });
    check_launch("launch block-task kernel");
}

void launch_subwarp_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, uint32_t group, bool local_tail,
                          const AutoTune& tune, cudaStream_t stream)
{
    const uint32_t threads = tune.subwarp_block_warps * 32;
    const uint64_t blocks  = (uint64_t{count} * group + threads - 1) / threads;
    const uint32_t grid = static_cast<uint32_t>(std::min<uint64_t>(blocks, 1u << 20));
    if (tune.tiled && local_tail)
        subwarp_tasks<TileTx, true><<<grid, threads, 0, stream>>>(batch, order, count, group);
    else if (tune.tiled)
        subwarp_tasks<TileTx><<<grid, threads, 0, stream>>>(batch, order, count, group);
    else if (local_tail)
        subwarp_tasks<LaneTx, true><<<grid, threads, 0, stream>>>(batch, order, count, group);
    else
        subwarp_tasks<LaneTx><<<grid, threads, 0, stream>>>(batch, order, count, group);
    check_launch("launch sub-warp kernel");
}

const void* block_kernel(bool tiled)
{
    return tiled ? reinterpret_cast<const void*>(block_tasks<TileTx>) : reinterpret_cast<const void*>(block_tasks<LaneTx>);
}

const void* block_kernel_staged()
{
    return reinterpret_cast<const void*>(block_tasks<StagedTx>);
}

void prepare_staged_block_kernel(uint32_t dynamic_bytes)
{
    check_cuda(cudaFuncSetAttribute(block_kernel_staged(), cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(dynamic_bytes)),
               "raise staged block kernel shared memory");
}

const void* warp_kernel(bool tiled)
{
    return tiled ? reinterpret_cast<const void*>(warp_tasks<TileTx>) : reinterpret_cast<const void*>(warp_tasks<LaneTx>);
}

const void* subwarp_kernel(bool tiled, bool local_tail)
{
    if (local_tail)
        return tiled ? reinterpret_cast<const void*>(subwarp_tasks<TileTx, true>)
                     : reinterpret_cast<const void*>(subwarp_tasks<LaneTx, true>);
    return tiled ? reinterpret_cast<const void*>(subwarp_tasks<TileTx>)
                 : reinterpret_cast<const void*>(subwarp_tasks<LaneTx>);
}

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
