#include "cuda/buffer.cuh"
#include "mine/device/common.cuh"
#include "mine/primitives.cuh"
#include "mine/schedule.cuh"

namespace ghuim {
namespace {

constexpr uint32_t kSelectBlock = 1024;

__global__ void select_kernel(SelectArgs a)
{
    __shared__ unsigned long long scan[32];
    __shared__ unsigned long long used_arena, used_scratch, used_children;
    __shared__ uint32_t           taken;
    __shared__ bool               done;

    Control&       control    = *a.control;
    const uint32_t top        = control.stack_top;
    const uint64_t arena_top  = control.arena_top < a.arena_capacity ? control.arena_top : a.arena_capacity;
    const uint64_t arena_free = (a.arena_capacity - arena_top) / a.headroom_divisor;
    const uint64_t child_free = a.stack_capacity - top;
    if (threadIdx.x == 0) {
        used_arena = used_scratch = used_children = 0;
        taken                                     = 0;
        done                                      = top == 0;
    }
    __syncthreads();

    // Take tasks from the top while the running sums of their bounds fit. The sums only grow,
    // so the tasks that fit form a prefix of each chunk.
    for (uint32_t base = 0; !done; base += kSelectBlock) {
        const uint32_t i     = base + threadIdx.x;
        const bool     valid = i < top && i < a.max_batch;
        Task           task{};
        uint64_t       arena = 0, scratch = 0, children = 0;
        if (valid) {
            task     = a.stack[top - 1 - i];
            arena    = task.arena_bound() / a.arena_divisor;
            scratch  = scratch_bytes(task.n_items, task.n_trans);
            children = task.child_bound();
        }
        const unsigned long long arena_sum    = block_inclusive_scan<unsigned long long>(arena, scan);
        const unsigned long long scratch_sum  = block_inclusive_scan<unsigned long long>(scratch, scan);
        const unsigned long long children_sum = block_inclusive_scan<unsigned long long>(children, scan);
        const bool fits = valid && used_arena + arena_sum <= arena_free &&
                          used_scratch + scratch_sum <= a.scratch_capacity &&
                          used_children + children_sum <= child_free;
        const uint32_t count = __syncthreads_count(fits);
        if (fits) {
            a.batch[taken + threadIdx.x]          = task;
            a.scratch_offset[taken + threadIdx.x] = used_scratch + scratch_sum - scratch;
        }
        __syncthreads();
        if (count != 0 && threadIdx.x == count - 1) {
            used_arena += arena_sum;
            used_scratch += scratch_sum;
            used_children += children_sum;
            taken += count;
            done = count < kSelectBlock;
        }
        if (count == 0 && threadIdx.x == 0)
            done = true;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        // Bounds are pessimistic: when nothing fits, run the top task anyway; it is deferred
        // if its child database really does not fit.
        control.forced = 0;
        if (taken == 0 && top != 0) {
            a.batch[0]          = a.stack[top - 1];
            a.scratch_offset[0] = 0;
            taken               = 1;
            control.forced      = 1;
        }
        control.batch_count = taken;
        control.stack_top   = top - taken;
        control.deferred    = 0;
        for (uint32_t b = 0; b < kMaxBins; ++b)
            control.bin_count[b] = control.bin_cursor[b] = control.bin_max_slices[b] = 0;
        control.batch_work = 0;
    }
}

__device__ Placement placement_of(const Task& task, const BinRule& rule, const Control& control)
{
    if (rule.optimized)
        return place_task(task, rule.tune, control.batch_work);
    for (uint32_t b = 0; b + 1 < rule.bins; ++b)
        if (task.n_trans <= rule.max_trans[b])
            return {b, 1};
    return {rule.bins - 1, 1};
}

__global__ void measure_batch(Control* control, const Task* batch)
{
    __shared__ unsigned long long block_work;
    if (threadIdx.x == 0)
        block_work = 0;
    __syncthreads();
    const uint32_t     count = control->batch_count;
    unsigned long long work  = 0;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
        work += batch[i].n_trans;
    add(block_work, work);
    __syncthreads();
    if (threadIdx.x == 0 && block_work != 0)
        add(control->batch_work, block_work);
}

__global__ void count_bins(Control* control, const Task* batch, BinRule rule)
{
    const uint32_t count = control->batch_count;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
    {
        const Placement place = placement_of(batch[i], rule, *control);
        const uint32_t  bin   = place.bin;
        add(control->bin_count[bin], 1u);
        fetch_max(control->bin_max_slices[bin], place.slices);
    }
}

__global__ void scatter_bins(Control* control, const Task* batch, uint32_t* order, BinRule rule)
{
    const uint32_t count = control->batch_count;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        const uint32_t bin   = placement_of(batch[i], rule, *control).bin;
        uint32_t       start = 0;
        for (uint32_t b = 0; b < bin; ++b)
            start += control->bin_count[b];
        order[start + fetch_add(control->bin_cursor[bin], 1u)] = i;
    }
}

__global__ void mark_kernel(const Task* stack, uint32_t size, uint32_t* bitmap)
{
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
        const uint32_t unit = stack[i].db;
        fetch_or(bitmap[unit / 32], 1u << (unit % 32));
    }
}

__global__ void popcount_kernel(const uint32_t* bitmap, uint32_t* counts, uint32_t words)
{
    for (uint32_t w = blockIdx.x * blockDim.x + threadIdx.x; w < words; w += gridDim.x * blockDim.x)
        counts[w] = cuda::std::popcount(bitmap[w]);
}

__global__ void extract_kernel(const uint32_t* bitmap, const uint32_t* offsets, uint32_t words, uint32_t* live)
{
    for (uint32_t w = blockIdx.x * blockDim.x + threadIdx.x; w < words; w += gridDim.x * blockDim.x) {
        uint32_t bits = bitmap[w], at = offsets[w];
        while (bits) {
            const uint32_t bit = cuda::std::countr_zero(bits);
            live[at++]         = w * 32 + bit;
            bits &= bits - 1;
        }
    }
}

__global__ void sizes_kernel(const char* arena, const uint32_t* live, uint32_t count, uint32_t* units)
{
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        const auto* header = reinterpret_cast<const DbHeader*>(arena + uint64_t{live[i]} * kArenaUnit);
        units[i]           = static_cast<uint32_t>(header->bytes / kArenaUnit);
    }
}

__global__ void copy_kernel(char* dst, const uint64_t* dst_off, const char* src, const uint64_t* src_off,
                            const uint64_t* bytes, uint32_t count)
{
    for (uint32_t i = blockIdx.x; i < count; i += gridDim.x) {
        auto*          to    = reinterpret_cast<uint4*>(dst + dst_off[i]);
        const auto*    from  = reinterpret_cast<const uint4*>(src + src_off[i]);
        const uint64_t words = bytes[i] / sizeof(uint4);
        for (uint64_t w = threadIdx.x; w < words; w += blockDim.x)
            to[w] = from[w];
    }
}

__global__ void remap_kernel(Task* stack, uint32_t size, const uint32_t* live, const uint32_t* moved_to,
                             uint32_t count)
{
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
        uint32_t low = 0, high = count; // live[] is sorted; find the task's database
        while (low < high) {
            const uint32_t mid = (low + high) / 2;
            if (live[mid] < stack[i].db)
                low = mid + 1;
            else
                high = mid;
        }
        stack[i].db = moved_to[low];
    }
}

uint32_t grid(uint64_t n, uint32_t block = 256)
{
    const uint64_t blocks = (n + block - 1) / block;
    return static_cast<uint32_t>(blocks == 0 ? 1 : blocks < 65535 ? blocks : 65535);
}

} // namespace

void select_batch(const SelectArgs& args, cudaStream_t stream)
{
    select_kernel<<<1, kSelectBlock, 0, stream>>>(args);
    check_cuda(cudaGetLastError(), "launch batch selection");
}

void bin_batch(Control* control, const Task* batch, uint32_t* order, const BinRule& rule, uint32_t max_batch,
               cudaStream_t stream)
{
    measure_batch<<<grid(max_batch), 256, 0, stream>>>(control, batch);
    count_bins<<<grid(max_batch), 256, 0, stream>>>(control, batch, rule);
    scatter_bins<<<grid(max_batch), 256, 0, stream>>>(control, batch, order, rule);
    check_cuda(cudaGetLastError(), "launch binning");
}

void mark_live(const Control*, const Task* stack, uint32_t stack_size, uint32_t* bitmap, uint32_t words,
               cudaStream_t stream)
{
    check_cuda(cudaMemsetAsync(bitmap, 0, uint64_t{words} * sizeof(uint32_t), stream), "clear live bitmap");
    if (stack_size != 0)
        mark_kernel<<<grid(stack_size), 256, 0, stream>>>(stack, stack_size, bitmap);
    check_cuda(cudaGetLastError(), "launch mark");
}

void count_live(const uint32_t* bitmap, uint32_t* counts, uint32_t words, cudaStream_t stream)
{
    popcount_kernel<<<grid(words), 256, 0, stream>>>(bitmap, counts, words);
    check_cuda(cudaGetLastError(), "launch popcount");
}

void extract_live(const uint32_t* bitmap, const uint32_t* offsets, uint32_t words, uint32_t* live, cudaStream_t stream)
{
    extract_kernel<<<grid(words), 256, 0, stream>>>(bitmap, offsets, words, live);
    check_cuda(cudaGetLastError(), "launch extract");
}

void live_sizes(const char* arena, const uint32_t* live, uint32_t count, uint32_t* units, cudaStream_t stream)
{
    if (count != 0)
        sizes_kernel<<<grid(count), 256, 0, stream>>>(arena, live, count, units);
    check_cuda(cudaGetLastError(), "launch sizes");
}

void copy_databases(char* dst, const uint64_t* dst_off, const char* src, const uint64_t* src_off,
                    const uint64_t* bytes, uint32_t count, cudaStream_t stream)
{
    if (count != 0)
        copy_kernel<<<count < 65535 ? count : 65535, 256, 0, stream>>>(dst, dst_off, src, src_off, bytes, count);
    check_cuda(cudaGetLastError(), "launch database copy");
}

void remap_tasks(Task* stack, uint32_t stack_size, const uint32_t* live, const uint32_t* moved_to, uint32_t count,
                 cudaStream_t stream)
{
    if (stack_size != 0)
        remap_kernel<<<grid(stack_size), 256, 0, stream>>>(stack, stack_size, live, moved_to, count);
    check_cuda(cudaGetLastError(), "launch remap");
}

} // namespace ghuim
