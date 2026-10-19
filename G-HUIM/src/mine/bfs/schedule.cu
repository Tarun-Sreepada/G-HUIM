// Scheduling kernels: scratch sizes per task, and binning of a chunk [begin, end) of the level.

#include "cuda/buffer.cuh"
#include "mine/bfs/device/task_view.cuh"
#include "mine/bfs/kernels.cuh"

#include <algorithm>

namespace ghuim::bfs {
namespace {

uint32_t grid(uint64_t n, uint32_t block = 256)
{
    const uint64_t blocks = (n + block - 1) / block;
    return static_cast<uint32_t>(blocks == 0 ? 1 : std::min<uint64_t>(blocks, 65535));
}

__global__ void scratch_sizes(const Task* tasks, uint32_t count, uint64_t* sizes, bool tiled)
{
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i <= count; i += gridDim.x * blockDim.x)
        sizes[i] = i < count ? scratch_bytes(tasks[i].n_items, tasks[i].n_trans, tiled) : 0;
}

// Prefix sums are monotonic. Find exactly the same maximal fitting chunks as the host's
// linear walk, reading only O(chunks * log(max_tasks)) offsets on the device.
__global__ void plan_chunks(const uint64_t* prefix, uint32_t begin, uint32_t count, uint32_t max_tasks,
                            uint64_t scratch_capacity, ChunkPlan* plan)
{
    plan->count = 0;
    plan->oversized = 0;
    plan->cuts[0] = begin;
    while (begin < count && plan->count < kChunkPlanPage) {
        const uint64_t base = prefix[begin];
        uint32_t low = begin, high = begin + min(count - begin, max_tasks);
        while (low < high) {
            const uint32_t mid = low + (high - low + 1) / 2;
            if (prefix[mid] - base <= scratch_capacity)
                low = mid;
            else
                high = mid - 1;
        }
        if (low == begin) {
            plan->oversized = 1;
            return;
        }
        plan->bases[plan->count] = base;
        plan->cuts[++plan->count] = low;
        begin = low;
    }
}

__global__ void begin_chunk(Control* control)
{
    for (uint32_t b = threadIdx.x; b < kMaxCells; b += blockDim.x)
        control->cell_count[b] = control->cell_start[b] = control->cell_cursor[b] = control->cell_max_slices[b] = 0;
    if (threadIdx.x == 0)
        control->batch_work = 0;
}

__global__ void measure_chunk(Control* control, const Task* tasks, uint32_t begin, uint32_t end)
{
    __shared__ unsigned long long block_work;
    if (threadIdx.x == 0)
        block_work = 0;
    __syncthreads();
    unsigned long long work = 0;
    for (uint32_t i = begin + blockIdx.x * blockDim.x + threadIdx.x; i < end; i += gridDim.x * blockDim.x)
        work += tasks[i].n_trans;
    add(block_work, work);
    __syncthreads();
    if (threadIdx.x == 0 && block_work != 0)
        add(control->batch_work, block_work);
}

__global__ void count_bins(Control* control, const Task* tasks, uint32_t begin, uint32_t end, AutoTune tune)
{
    for (uint32_t i = begin + blockIdx.x * blockDim.x + threadIdx.x; i < end; i += gridDim.x * blockDim.x) {
        const Placement place = place_task(tasks[i], tune, control->batch_work);
        add(control->cell_count[place.cell()], 1u);
        fetch_max(control->cell_max_slices[place.cell()], place.slices);
    }
}

// Exclusive prefix of the cell counts: where each cell's tasks start in `order`.
__global__ void start_cells(Control* control)
{
    uint32_t start = 0;
    for (uint32_t c = 0; c < kCells; ++c) {
        control->cell_start[c] = start;
        start += control->cell_count[c];
    }
}

__global__ void scatter_bins(Control* control, const Task* tasks, uint32_t begin, uint32_t end, uint32_t* order,
                             AutoTune tune)
{
    for (uint32_t i = begin + blockIdx.x * blockDim.x + threadIdx.x; i < end; i += gridDim.x * blockDim.x) {
        const uint32_t cell = place_task(tasks[i], tune, control->batch_work).cell();
        order[control->cell_start[cell] + fetch_add(control->cell_cursor[cell], 1u)] = i;
    }
}

} // namespace

void launch_plan_chunks(const uint64_t* prefix, uint32_t begin, uint32_t count, uint32_t max_tasks,
                        uint64_t scratch_capacity, ChunkPlan* plan, cudaStream_t stream)
{
    plan_chunks<<<1, 1, 0, stream>>>(prefix, begin, count, max_tasks, scratch_capacity, plan);
    check_launch("launch chunk planning");
}

void launch_scratch_sizes(const Task* tasks, uint32_t count, uint64_t* sizes, bool tiled, cudaStream_t stream)
{
    scratch_sizes<<<grid(count + 1), 256, 0, stream>>>(tasks, count, sizes, tiled);
    check_launch("launch scratch sizes");
}

void launch_bin_chunk(Control* control, const Task* tasks, uint32_t begin, uint32_t end, uint32_t* order,
                      const AutoTune& tune, cudaStream_t stream)
{
    const uint32_t blocks = grid(end - begin);
    begin_chunk<<<1, 32, 0, stream>>>(control);
    measure_chunk<<<blocks, 256, 0, stream>>>(control, tasks, begin, end);
    count_bins<<<blocks, 256, 0, stream>>>(control, tasks, begin, end, tune);
    start_cells<<<1, 1, 0, stream>>>(control);
    scatter_bins<<<blocks, 256, 0, stream>>>(control, tasks, begin, end, order, tune);
    check_launch("launch chunk binning");
}

} // namespace ghuim::bfs
