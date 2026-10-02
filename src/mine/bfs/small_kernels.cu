// Kernels for tasks one thread group processes alone: a lane group, a warp, or a block.

#include "cuda/buffer.cuh"
#include "mine/bfs/device/process.cuh"
#include "mine/bfs/kernels.cuh"

#include <algorithm>

namespace ghuim::bfs {
namespace {

constexpr uint32_t kMaxGrid = 65535;

// Block-sized tasks: one block per task; its warps split the transactions.
template <typename Mode> __device__ __forceinline__ void block_tasks_body(BatchView b, const uint32_t* order, uint32_t count)
{
    for (uint32_t i = blockIdx.x; i < count; i += gridDim.x) {
        process_task<Mode>(BlockGroup{}, b, order[i], threadIdx.x / 32, blockDim.x / 32, true);
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

void launch_warp_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, uint32_t shape, const AutoTune& tune,
                       cudaStream_t stream)
{
    const ShapeTune& st     = tune.shape[shape];
    const uint32_t   warps  = st.packed_block_warps;
    const uint32_t   blocks = std::min(ceil_div(count, warps), kMaxGrid);
    with_mode(st.group, [&](auto mode) {
        warp_tasks<decltype(mode)><<<blocks, warps * 32, 0, stream>>>(batch, order, count);
    });
    check_launch("launch warp-task kernel");
}

void launch_block_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, uint32_t bin, uint32_t shape,
                        const AutoTune& tune, cudaStream_t stream)
{
    if (tune.staged) {
        // The block keeps the bin's compute warps and adds the IO warps (at most 32 warps in all).
        const uint32_t warps   = std::min((1u << bin) + tune.io_warps, 32u);
        const uint32_t dynamic = kStageBarrierBytes + tune.ring_bytes + tune.table_bytes;
        if (tune.warp_mode)
            block_tasks<StagedWarpTx><<<std::min(count, kMaxGrid), warps * 32, dynamic, stream>>>(batch, order, count);
        else
            block_tasks<StagedTx><<<std::min(count, kMaxGrid), warps * 32, dynamic, stream>>>(batch, order, count);
    }
    else {
        const uint32_t group = tune.shape[shape].group;
        with_mode(group, [&](auto mode) {
            block_tasks<decltype(mode)><<<std::min(count, kMaxGrid), 32u << bin, group == kLaneMode ? tune.table_bytes : 0,
                                          stream>>>(batch, order, count);
        });
    }
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

const void* block_kernel(uint32_t group)
{
    const void* kernel = nullptr;
    with_mode(group, [&](auto mode) { kernel = reinterpret_cast<const void*>(block_tasks<decltype(mode)>); });
    return kernel;
}

const void* block_kernel_staged(bool warp_mode)
{
    return warp_mode ? reinterpret_cast<const void*>(block_tasks<StagedWarpTx>)
                     : reinterpret_cast<const void*>(block_tasks<StagedTx>);
}

void prepare_staged_block_kernel(uint32_t dynamic_bytes, bool warp_mode)
{
    check_cuda(cudaFuncSetAttribute(block_kernel_staged(warp_mode), cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(dynamic_bytes)),
               "raise staged block kernel shared memory");
}

const void* warp_kernel(uint32_t group)
{
    const void* kernel = nullptr;
    with_mode(group, [&](auto mode) { kernel = reinterpret_cast<const void*>(warp_tasks<decltype(mode)>); });
    return kernel;
}

const void* subwarp_kernel()
{
    return reinterpret_cast<const void*>(subwarp_tasks);
}

} // namespace ghuim::bfs
