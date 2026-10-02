// Kernels for huge tasks: gridDim.y blocks share each task (blockIdx.x), each taking a slice of the
// transactions. Passes that need the whole task finished are separate launches.

#include "cuda/buffer.cuh"
#include "mine/bfs/device/staging.cuh"
#include "mine/bfs/kernels.cuh"

#include <algorithm>

namespace ghuim::bfs {
namespace {

constexpr uint32_t kMaxGrid = 65535;

__device__ __forceinline__ uint32_t slices_of(const BatchView& b, const AutoTune& tune, const TaskView& v)
{
    return place_task(v.task, tune, b.control->batch_work).slices;
}

// The transactions of a slice: warp w of block y starts at (y * warps + w) units, `unit` transactions
// per warp step (32 for lane per transaction, 32 / group size for lane groups).
template <typename Mode>
__device__ __forceinline__ void slice_range(const Projection& p, uint32_t slices, uint32_t& from, uint32_t& step)
{
    const uint32_t warps = blockDim.x / 32;
    const uint32_t unit  = Mode::unit(p, Lanes(WarpGroup{}));
    from                 = (blockIdx.y * warps + threadIdx.x / 32) * unit;
    step                 = slices * warps * unit;
}

// Shared-memory copies of the per-item tables for one slice block: local[n] and subtree[n] (64-bit)
// followed by rename[n]. Used when the task's items fit in the block's table region; the block then
// accumulates privately and flushes with one atomic per item.
__device__ __forceinline__ bool use_tables(const BatchView& b, uint32_t n_items)
{
    return b.table_bytes != 0 && uint64_t{n_items} * 20 <= b.table_bytes;
}

__device__ __forceinline__ unsigned long long* table_region(const BatchView& b)
{
    return reinterpret_cast<unsigned long long*>(stage_smem + b.table_offset);
}

// Copies rename[first, n) into the shared table and returns a projection that reads it.
__device__ __forceinline__ Projection with_shared_rename(const BatchView& b, const Projection& p, uint32_t first, uint32_t n)
{
    Projection q = p;
    auto*      t = table_region(b);
    q.subtree    = t + n;
    q.rename     = reinterpret_cast<uint32_t*>(t + 2 * n);
    for (uint32_t i = first + threadIdx.x; i < n; i += blockDim.x)
        q.rename[i] = p.rename[i];
    __syncthreads();
    return q;
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
    const auto     ctx   = Mode::make_ctx(b, v.p, blockDim.x / 32, slices, blockIdx.y);
    const uint32_t first = v.task.item + 1, n = v.header.n_items;
    if (!use_tables(b, n)) {
        Mode::scan_local(v.p, lanes, from, step, *v.state, ctx);
        return;
    }
    Projection p   = v.p;
    auto*      loc = table_region(b);
    p.local        = loc;
    for (uint32_t i = first + threadIdx.x; i < n; i += blockDim.x)
        loc[i] = 0;
    __syncthreads();
    Mode::scan_local(p, lanes, from, step, *v.state, ctx);
    __syncthreads();
    for (uint32_t i = first + threadIdx.x; i < n; i += blockDim.x)
        if (loc[i] != 0)
            add(v.p.local[i], loc[i]);
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

template <typename Mode> __global__ void huge_subtree(BatchView b, AutoTune tune, const uint32_t* order)
{
    const TaskView v      = open_task(b, order[blockIdx.x]);
    const uint32_t slices = slices_of(b, tune, v);
    if (blockIdx.y >= slices || v.state->child_items == 0)
        return;
    const Lanes lanes(WarpGroup{});
    uint32_t    from, step;
    slice_range<Mode>(v.p, slices, from, step);
    const auto     ctx         = Mode::make_ctx(b, v.p, blockDim.x / 32, slices, blockIdx.y);
    const uint32_t first       = v.task.item + 1, n = v.header.n_items;
    const uint32_t child_items = v.state->child_items;
    if (!use_tables(b, n)) {
        Mode::scan_subtree(v.p, lanes, from, step, *v.state, ctx);
        return;
    }
    Projection p = with_shared_rename(b, v.p, first, n);
    for (uint32_t z = threadIdx.x; z < child_items; z += blockDim.x)
        p.subtree[z] = 0;
    __syncthreads();
    Mode::scan_subtree(p, lanes, from, step, *v.state, ctx);
    __syncthreads();
    for (uint32_t z = threadIdx.x; z < child_items; z += blockDim.x)
        if (p.subtree[z] != 0)
            add(v.p.subtree[z], p.subtree[z]);
}

__global__ void huge_allocate(BatchView b, AutoTune, const uint32_t* order)
{
    const TaskView v = open_task(b, order[blockIdx.x]);
    if (v.state->child_items == 0)
        return;
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
    const Lanes lanes(WarpGroup{});
    uint32_t    from, step;
    slice_range<Mode>(v.p, slices, from, step);
    const auto ctx = Mode::make_ctx(b, v.p, blockDim.x / 32, slices, blockIdx.y);
    if (!use_tables(b, v.header.n_items)) {
        Mode::write_representatives(v.p, lanes, from, step, *v.state, c, ctx);
        return;
    }
    const Projection p = with_shared_rename(b, v.p, v.task.item + 1, v.header.n_items);
    Mode::write_representatives(p, lanes, from, step, *v.state, c, ctx);
}

template <typename Mode> __global__ void huge_merge(BatchView b, AutoTune tune, const uint32_t* order)
{
    const TaskView v      = open_task(b, order[blockIdx.x]);
    const uint32_t slices = slices_of(b, tune, v);
    if (blockIdx.y >= slices || v.state->child_items == 0 || v.state->child_primaries == 0)
        return;
    const ChildView c = child_view(b, *v.state, v.state->child_items);
    const Lanes     lanes(WarpGroup{});
    uint32_t        from, step;
    slice_range<Mode>(v.p, slices, from, step);
    const auto ctx = Mode::make_ctx(b, v.p, blockDim.x / 32, slices, blockIdx.y);
    if (!use_tables(b, v.header.n_items)) {
        Mode::merge_duplicates(v.p, lanes, from, step, c, ctx);
        return;
    }
    const Projection p = with_shared_rename(b, v.p, v.task.item + 1, v.header.n_items);
    Mode::merge_duplicates(p, lanes, from, step, c, ctx);
}

__global__ void huge_emit(BatchView b, AutoTune, const uint32_t* order)
{
    const TaskView v = open_task(b, order[blockIdx.x]);
    if (v.state->child_items == 0 || v.state->child_primaries == 0)
        return;
    const ChildView c = child_view(b, *v.state, v.state->child_items);
    emit_children(BlockGroup{}, b, v, v.state->child_items, c);
}

// init is launched by the caller; passes local..merge run in this order, emit follows.
template <typename Mode>
void launch_huge_passes(const BatchView& batch, const AutoTune& tune, const uint32_t* part, dim3 sliced, dim3 single,
                        uint32_t threads, uint32_t dynamic_bytes, cudaStream_t stream)
{
    huge_local<Mode><<<sliced, threads, dynamic_bytes, stream>>>(batch, tune, part);
    huge_rename<<<single, threads, 0, stream>>>(batch, tune, part);
    huge_subtree<Mode><<<sliced, threads, dynamic_bytes, stream>>>(batch, tune, part);
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
            launch_huge_passes<StagedTx>(batch, tune, part, sliced, single, threads,
                                         kStageBarrierBytes + tune.ring_bytes + tune.table_bytes, stream);
        else if (tune.warp_mode)
            launch_huge_passes<WarpTx>(batch, tune, part, sliced, single, threads, 0, stream);
        else
            launch_huge_passes<LaneTx>(batch, tune, part, sliced, single, threads, tune.table_bytes, stream);
        huge_emit<<<single, threads, 0, stream>>>(batch, tune, part);
        check_launch("launch huge-task kernels");
    }
}

void huge_kernels(const void* (&kernels)[kHugeKernels], bool warp_mode)
{
    const void* handles[kHugeKernels] = {
            reinterpret_cast<const void*>(huge_init),
            warp_mode ? reinterpret_cast<const void*>(huge_local<WarpTx>) : reinterpret_cast<const void*>(huge_local<LaneTx>),
            reinterpret_cast<const void*>(huge_rename),
            warp_mode ? reinterpret_cast<const void*>(huge_subtree<WarpTx>) : reinterpret_cast<const void*>(huge_subtree<LaneTx>),
            reinterpret_cast<const void*>(huge_allocate),
            warp_mode ? reinterpret_cast<const void*>(huge_write<WarpTx>) : reinterpret_cast<const void*>(huge_write<LaneTx>),
            warp_mode ? reinterpret_cast<const void*>(huge_merge<WarpTx>) : reinterpret_cast<const void*>(huge_merge<LaneTx>),
            reinterpret_cast<const void*>(huge_emit)};
    std::copy(std::begin(handles), std::end(handles), kernels);
}

void huge_kernels_staged(const void* (&kernels)[kHugeKernels])
{
    const void* handles[kHugeKernels] = {
            reinterpret_cast<const void*>(huge_init),
            reinterpret_cast<const void*>(huge_local<StagedTx>),
            reinterpret_cast<const void*>(huge_rename),
            reinterpret_cast<const void*>(huge_subtree<StagedTx>),
            reinterpret_cast<const void*>(huge_allocate),
            reinterpret_cast<const void*>(huge_write<StagedTx>),
            reinterpret_cast<const void*>(huge_merge<StagedTx>),
            reinterpret_cast<const void*>(huge_emit)};
    std::copy(std::begin(handles), std::end(handles), kernels);
}

void prepare_staged_kernels(uint32_t dynamic_bytes)
{
    prepare_staged_block_kernel(dynamic_bytes);
    const void* kernels[] = {reinterpret_cast<const void*>(huge_local<StagedTx>),
                             reinterpret_cast<const void*>(huge_subtree<StagedTx>),
                             reinterpret_cast<const void*>(huge_write<StagedTx>),
                             reinterpret_cast<const void*>(huge_merge<StagedTx>)};
    for (const void* kernel : kernels)
        check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(dynamic_bytes)),
                   "raise staged huge kernel shared memory");
}

} // namespace ghuim::bfs
