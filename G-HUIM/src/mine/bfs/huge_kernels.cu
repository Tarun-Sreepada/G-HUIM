// Kernels for huge tasks: gridDim.y blocks share each task (blockIdx.x), each taking a slice of the
// transactions. Passes that need the whole task finished are separate launches.

#include "cuda/buffer.cuh"
#include "mine/bfs/device/staging.cuh"
#include "mine/bfs/device/tile_passes.cuh"
#include "mine/bfs/kernels.cuh"

#include <algorithm>

namespace ghuim::inline GHUIM_WIDTHS::bfs {
namespace {

constexpr uint32_t kMaxGrid = 65535;
// Refinement rounds a huge task may run (each costs three launches); -1 means this many.
constexpr int32_t kMaxHugeRefine = 8;

__device__ __forceinline__ uint32_t slices_of(const BatchView& b, const AutoTune& tune, const TaskView& v)
{
    return place_task(v.task, tune, b.control->batch_work).slices;
}

// The transactions of a slice: warp w of block y starts at (y * warps + w) units, `unit` transactions
// per warp step (32: a lane per transaction).
template <typename Mode>
__device__ __forceinline__ void slice_range(const Projection& p, uint32_t slices, uint32_t& from, uint32_t& step)
{
    const uint32_t warps = blockDim.x / 32;
    const uint32_t unit  = Mode::unit(p, Lanes(WarpGroup{}));
    from                 = (blockIdx.y * warps + threadIdx.x / 32) * unit;
    step                 = slices * warps * unit;
}

__global__ void huge_init(BatchView b, AutoTune tune, const uint32_t* order)
{
    const TaskView v      = open_task(b, order[blockIdx.x]);
    const uint32_t slices = slices_of(b, tune, v);
    if (blockIdx.y >= slices)
        return;
    clear_task(v, blockIdx.y * blockDim.x + threadIdx.x, slices * blockDim.x);
}

template <typename Mode> __global__ void huge_local(BatchView b, AutoTune tune, const uint32_t* order)
{
    const TaskView v      = open_task(b, order[blockIdx.x]);
    const uint32_t slices = slices_of(b, tune, v);
    if (blockIdx.y >= slices)
        return;
    const Lanes lanes(WarpGroup{});
    uint32_t    from, step;
    slice_range<Mode>(v.p, slices, from, step);
    const auto ctx = Mode::make_ctx(b, v.p, blockDim.x / 32, slices, blockIdx.y);
    Mode::scan_local(v.p, lanes, from, step, *v.state, ctx);
}

__global__ void huge_rename(BatchView b, AutoTune, const uint32_t* order)
{
    const TaskView v           = open_task(b, order[blockIdx.x]);
    const uint32_t child_items = rename_items(BlockGroup{}, v.p, v.task.item + 1, v.header.n_items, b.minimum_utility);
    if (threadIdx.x == 0) {
        v.state->child_items = child_items;
        if (child_items == 0)
            finish_task(b, v);
    }
}

// Refinement rounds (see refine_items): clear, scan and rename, one launch each, repeated by the host;
// a task that has converged returns at once.
__global__ void huge_refine_clear(BatchView b, AutoTune, const uint32_t* order)
{
    const TaskView v     = open_task(b, order[blockIdx.x]);
    const uint32_t first = v.task.item + 1, n = v.header.n_items;
    const uint32_t kept  = v.state->child_items;
    if (v.state->refined)
        return;
    if (kept == 0 || kept >= n - first) {
        __syncthreads();
        if (threadIdx.x == 0)
            v.state->refined = 1;
        return;
    }
    clear_kept_local(v.p, first, n, threadIdx.x, blockDim.x);
}

__global__ void huge_refine_scan(BatchView b, AutoTune tune, const uint32_t* order)
{
    const TaskView v      = open_task(b, order[blockIdx.x]);
    const uint32_t slices = slices_of(b, tune, v);
    if (blockIdx.y >= slices || v.state->refined)
        return;
    const uint32_t warps = blockDim.x / 32;
    scan_refine(v.p, Lanes(WarpGroup{}), (blockIdx.y * warps + threadIdx.x / 32) * 32, slices * warps * 32);
}

__global__ void huge_refine_rename(BatchView b, AutoTune, const uint32_t* order)
{
    const TaskView v = open_task(b, order[blockIdx.x]);
    if (v.state->refined)
        return;
    const uint32_t kept = v.state->child_items;
    const uint32_t next = refine_rename(BlockGroup{}, v.p, v.task.item + 1, v.header.n_items, b.minimum_utility);
    if (threadIdx.x == 0) {
        v.state->child_items = next;
        if (next == kept || next == 0)
            v.state->refined = 1;
        if (next == 0)
            finish_task(b, v);
    }
}

template <typename Mode> __global__ void huge_subtree(BatchView b, AutoTune tune, const uint32_t* order)
{
    const TaskView v      = open_task(b, order[blockIdx.x]);
    const uint32_t slices = slices_of(b, tune, v);
    if (blockIdx.y >= slices || v.state->child_items == 0)
        return;
    const Lanes lanes(WarpGroup{});
    uint32_t    from, step;
    slice_range<Mode>(v.p, slices, from, step);
    const auto ctx = Mode::make_ctx(b, v.p, blockDim.x / 32, slices, blockIdx.y);
    Mode::scan_subtree(v.p, lanes, from, step, *v.state, ctx);
}

// Tile method: the class starts of the child's counting sort (one block per task).
__global__ void huge_tile_bases(BatchView b, AutoTune, const uint32_t* order)
{
    const TaskView v = open_task(b, order[blockIdx.x]);
    if (v.state->child_items == 0)
        return;
    tile_class_bases(BlockGroup{}, v.p, *v.state);
}

// Tile method: every representative takes a slot of its length class.
template <typename Mode> __global__ void huge_tile_assign(BatchView b, AutoTune tune, const uint32_t* order)
{
    const TaskView v      = open_task(b, order[blockIdx.x]);
    const uint32_t slices = slices_of(b, tune, v);
    if (blockIdx.y >= slices || v.state->child_items == 0)
        return;
    Projection p = v.p;
    p.child_stride = tile_stride(v.state->child_trans);
    const Lanes lanes(WarpGroup{});
    uint32_t    from, step;
    slice_range<Mode>(p, slices, from, step);
    const auto ctx = Mode::make_ctx(b, p, blockDim.x / 32, slices, blockIdx.y);
    Mode::assign_slots(p, lanes, from, step, ctx);
}

__global__ void huge_allocate(BatchView b, AutoTune, const uint32_t* order)
{
    const TaskView v = open_task(b, order[blockIdx.x]);
    if (v.state->child_items == 0)
        return;
    if (b.tiled) {
        Projection p   = v.p;
        p.child_stride = tile_stride(v.state->child_trans);
        tile_offsets(BlockGroup{}, p, *v.state);
        __syncthreads();
    }
    allocate_child(BlockGroup{}, b, v, v.state->child_items);
}

template <typename Mode> __global__ void huge_write(BatchView b, AutoTune tune, const uint32_t* order)
{
    const TaskView v      = open_task(b, order[blockIdx.x]);
    const uint32_t slices = slices_of(b, tune, v);
    if (blockIdx.y >= slices || v.state->child_items == 0 || v.state->child_primaries == 0)
        return;
    const ChildView c = child_view(b, *v.state, v.state->child_items);
    copy_child_items(v, c, blockIdx.y * blockDim.x + threadIdx.x, slices * blockDim.x);
    Projection base = v.p;
    if constexpr (TiledMode<Mode>)
        base.child_stride = tile_stride(v.state->child_trans);
    const Lanes lanes(WarpGroup{});
    uint32_t    from, step;
    slice_range<Mode>(base, slices, from, step);
    const auto ctx = Mode::make_ctx(b, base, blockDim.x / 32, slices, blockIdx.y);
    Mode::write_representatives(base, lanes, from, step, *v.state, c, ctx);
}

template <typename Mode> __global__ void huge_merge(BatchView b, AutoTune tune, const uint32_t* order)
{
    const TaskView v      = open_task(b, order[blockIdx.x]);
    const uint32_t slices = slices_of(b, tune, v);
    if (blockIdx.y >= slices || v.state->child_items == 0 || v.state->child_primaries == 0)
        return;
    const ChildView c = child_view(b, *v.state, v.state->child_items);
    Projection      base = v.p;
    if constexpr (TiledMode<Mode>)
        base.child_stride = tile_stride(v.state->child_trans);
    const Lanes     lanes(WarpGroup{});
    uint32_t        from, step;
    slice_range<Mode>(base, slices, from, step);
    const auto ctx = Mode::make_ctx(b, base, blockDim.x / 32, slices, blockIdx.y);
    Mode::merge_duplicates(base, lanes, from, step, c, ctx);
}

__global__ void huge_emit(BatchView b, AutoTune, const uint32_t* order)
{
    const TaskView v = open_task(b, order[blockIdx.x]);
    if (v.state->child_items == 0 || v.state->child_primaries == 0)
        return;
    const ChildView c = child_view(b, *v.state, v.state->child_items);
    build_child_index(BlockGroup{}, v, v.state->child_items, c);
    emit_children(BlockGroup{}, b, v, v.state->child_items, c);
}

// init is launched by the caller; passes local..merge run in this order, emit follows.
template <typename Mode>
void launch_huge_passes(const BatchView& batch, const AutoTune& tune, const uint32_t* part, dim3 sliced, dim3 single,
                        uint32_t threads, uint32_t dynamic_bytes, cudaStream_t stream)
{
    huge_local<Mode><<<sliced, threads, dynamic_bytes, stream>>>(batch, tune, part);
    huge_rename<<<single, threads, 0, stream>>>(batch, tune, part);
    const int32_t rounds = batch.refine_rounds < 0 ? kMaxHugeRefine : std::min(batch.refine_rounds, kMaxHugeRefine);
    for (int32_t r = 0; r < rounds; ++r) {
        huge_refine_clear<<<single, threads, 0, stream>>>(batch, tune, part);
        huge_refine_scan<<<sliced, threads, 0, stream>>>(batch, tune, part);
        huge_refine_rename<<<single, threads, 0, stream>>>(batch, tune, part);
    }
    huge_subtree<Mode><<<sliced, threads, dynamic_bytes, stream>>>(batch, tune, part);
    if constexpr (TiledMode<Mode>) {
        huge_tile_bases<<<single, threads, 0, stream>>>(batch, tune, part);
        huge_tile_assign<Mode><<<sliced, threads, dynamic_bytes, stream>>>(batch, tune, part);
    }
    huge_allocate<<<single, threads, 0, stream>>>(batch, tune, part);
    huge_write<Mode><<<sliced, threads, dynamic_bytes, stream>>>(batch, tune, part);
    huge_merge<Mode><<<sliced, threads, dynamic_bytes, stream>>>(batch, tune, part);
}

} // namespace

void launch_huge_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, uint32_t max_slices,
                       const AutoTune& tune, cudaStream_t stream)
{
    const uint32_t threads = tune.max_block_warps * 32;
    for (uint32_t done = 0; done < count; done += kMaxGrid) {
        const uint32_t  n    = std::min(count - done, kMaxGrid);
        const uint32_t* part = order + done;
        const dim3      sliced(n, max_slices), single(n, 1);
        huge_init<<<sliced, threads, 0, stream>>>(batch, tune, part);
        if (tune.staged)
            launch_huge_passes<StagedTx>(batch, tune, part, sliced, single, threads, kStageBarrierBytes + tune.ring_bytes,
                                         stream);
        else if (tune.tiled)
            launch_huge_passes<TileTx>(batch, tune, part, sliced, single, threads, 0, stream);
        else
            launch_huge_passes<LaneTx>(batch, tune, part, sliced, single, threads, 0, stream);
        huge_emit<<<single, threads, 0, stream>>>(batch, tune, part);
        check_launch("launch huge-task kernels");
    }
}

template <typename Mode> void huge_handles(const void* (&kernels)[kHugeKernels])
{
    const void* handles[kHugeKernels] = {
            reinterpret_cast<const void*>(huge_init),
            reinterpret_cast<const void*>(huge_local<Mode>),
            reinterpret_cast<const void*>(huge_rename),
            reinterpret_cast<const void*>(huge_subtree<Mode>),
            reinterpret_cast<const void*>(huge_allocate),
            reinterpret_cast<const void*>(huge_write<Mode>),
            reinterpret_cast<const void*>(huge_merge<Mode>),
            reinterpret_cast<const void*>(huge_emit)};
    std::copy(std::begin(handles), std::end(handles), kernels);
}

void huge_kernels(const void* (&kernels)[kHugeKernels], bool tiled)
{
    if (tiled)
        huge_handles<TileTx>(kernels);
    else
        huge_handles<LaneTx>(kernels);
}

void tile_huge_kernels(const void* (&kernels)[kTileHugeKernels])
{
    kernels[0] = reinterpret_cast<const void*>(huge_tile_bases);
    kernels[1] = reinterpret_cast<const void*>(huge_tile_assign<TileTx>);
}

void huge_kernels_staged(const void* (&kernels)[kHugeKernels])
{
    huge_handles<StagedTx>(kernels);
}

void prepare_staged_kernels(uint32_t dynamic_bytes)
{
    prepare_staged_block_kernel(dynamic_bytes);
    const void* kernels[kHugeKernels];
    huge_kernels_staged(kernels);
    for (uint32_t k : {1u, 3u, 5u, 6u}) // the local, subtree, write and merge passes
        check_cuda(cudaFuncSetAttribute(kernels[k], cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(dynamic_bytes)),
                   "raise staged huge kernel shared memory");
}

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
