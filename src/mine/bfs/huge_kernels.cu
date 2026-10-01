// Kernels for huge tasks: gridDim.y blocks share each task (blockIdx.x), each taking a slice of the
// transactions. Passes that need the whole task finished are separate launches.

#include "cuda/buffer.cuh"
#include "mine/bfs/device/child.cuh"
#include "mine/bfs/kernels.cuh"

#include <algorithm>

namespace ghuim::bfs {
namespace {

constexpr uint32_t kMaxGrid = 65535;

__device__ __forceinline__ uint32_t slices_of(const BatchView& b, const AutoTune& tune, const TaskView& v)
{
    return place_task(v.task, tune, b.control->batch_work).slices;
}

__device__ __forceinline__ void slice_range(uint32_t slices, uint32_t& from, uint32_t& step)
{
    const uint32_t warps = blockDim.x / 32;
    from                 = (blockIdx.y * warps + threadIdx.x / 32) * 32;
    step                 = slices * warps * 32;
}

__global__ void huge_init(BatchView b, AutoTune tune, const uint32_t* order)
{
    const TaskView v      = open_task(b, order[blockIdx.x]);
    const uint32_t slices = slices_of(b, tune, v);
    if (blockIdx.y >= slices)
        return;
    clear_task(v, blockIdx.y * blockDim.x + threadIdx.x, slices * blockDim.x);
}

__global__ void huge_local(BatchView b, AutoTune tune, const uint32_t* order)
{
    const TaskView v      = open_task(b, order[blockIdx.x]);
    const uint32_t slices = slices_of(b, tune, v);
    if (blockIdx.y >= slices)
        return;
    const Lanes lanes(WarpGroup{});
    uint32_t    from, step;
    slice_range(slices, from, step);
    scan_local(v.p, lanes, from, step, *v.state);
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

__global__ void huge_subtree(BatchView b, AutoTune tune, const uint32_t* order)
{
    const TaskView v      = open_task(b, order[blockIdx.x]);
    const uint32_t slices = slices_of(b, tune, v);
    if (blockIdx.y >= slices || v.state->child_items == 0)
        return;
    const Lanes lanes(WarpGroup{});
    uint32_t    from, step;
    slice_range(slices, from, step);
    scan_subtree(v.p, lanes, from, step, *v.state);
}

__global__ void huge_allocate(BatchView b, AutoTune, const uint32_t* order)
{
    const TaskView v = open_task(b, order[blockIdx.x]);
    if (v.state->child_items == 0)
        return;
    allocate_child(BlockGroup{}, b, v, v.state->child_items);
}

__global__ void huge_write(BatchView b, AutoTune tune, const uint32_t* order)
{
    const TaskView v      = open_task(b, order[blockIdx.x]);
    const uint32_t slices = slices_of(b, tune, v);
    if (blockIdx.y >= slices || v.state->child_items == 0 || v.state->child_primaries == 0)
        return;
    const ChildView c = child_view(b, *v.state, v.state->child_items);
    copy_child_items(v, c, blockIdx.y * blockDim.x + threadIdx.x, slices * blockDim.x);
    const Lanes lanes(WarpGroup{});
    uint32_t    from, step;
    slice_range(slices, from, step);
    write_representatives(v.p, lanes, from, step, *v.state, c);
}

__global__ void huge_merge(BatchView b, AutoTune tune, const uint32_t* order)
{
    const TaskView v      = open_task(b, order[blockIdx.x]);
    const uint32_t slices = slices_of(b, tune, v);
    if (blockIdx.y >= slices || v.state->child_items == 0 || v.state->child_primaries == 0)
        return;
    const ChildView c = child_view(b, *v.state, v.state->child_items);
    const Lanes     lanes(WarpGroup{});
    uint32_t        from, step;
    slice_range(slices, from, step);
    merge_duplicates(v.p, lanes, from, step, c);
}

__global__ void huge_emit(BatchView b, AutoTune, const uint32_t* order)
{
    const TaskView v = open_task(b, order[blockIdx.x]);
    if (v.state->child_items == 0 || v.state->child_primaries == 0)
        return;
    const ChildView c = child_view(b, *v.state, v.state->child_items);
    emit_children(BlockGroup{}, b, v, v.state->child_items, c);
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
        huge_local<<<sliced, threads, 0, stream>>>(batch, tune, part);
        huge_rename<<<single, threads, 0, stream>>>(batch, tune, part);
        huge_subtree<<<sliced, threads, 0, stream>>>(batch, tune, part);
        huge_allocate<<<single, threads, 0, stream>>>(batch, tune, part);
        huge_write<<<sliced, threads, 0, stream>>>(batch, tune, part);
        huge_merge<<<sliced, threads, 0, stream>>>(batch, tune, part);
        huge_emit<<<single, threads, 0, stream>>>(batch, tune, part);
        check_launch("launch huge-task kernels");
    }
}

void huge_kernels(const void* (&kernels)[kHugeKernels])
{
    const void* handles[kHugeKernels] = {
            reinterpret_cast<const void*>(huge_init),     reinterpret_cast<const void*>(huge_local),
            reinterpret_cast<const void*>(huge_rename),   reinterpret_cast<const void*>(huge_subtree),
            reinterpret_cast<const void*>(huge_allocate), reinterpret_cast<const void*>(huge_write),
            reinterpret_cast<const void*>(huge_merge),    reinterpret_cast<const void*>(huge_emit)};
    std::copy(std::begin(handles), std::end(handles), kernels);
}

} // namespace ghuim::bfs
