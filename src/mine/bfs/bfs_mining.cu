// Level-synchronous (BFS) optimized miner.
//
// Each level is one batch: its tasks are binned by size, each bin runs on its own stream with a
// thread-group size chosen from the level's total work, and children are written into the arena
// region the previous level is not using. Per-task working memory comes from a scratch buffer that
// is reused every chunk, so the arenas only ever hold projected databases.

#include "mine/bfs/bfs_mining.cuh"
#include "mine/bfs/device.cuh"

#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <array>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bfs {
namespace {

void check_cuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

// Owns a device allocation of `count` elements.
template <typename T> class DeviceBuffer
{
  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) : count_(count)
    {
        if (count != 0)
            check_cuda(cudaMalloc(&data_, count * sizeof(T)), "allocate device memory");
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), count_(other.count_)
    {
        other.data_  = nullptr;
        other.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        std::swap(data_, other.data_);
        std::swap(count_, other.count_);
        return *this;
    }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T*     get() const { return data_; }
    size_t size() const { return count_; }

  private:
    T*     data_  = nullptr;
    size_t count_ = 0;
};

// Tasks per chunk. Thread groups are sized from a task's share of the chunk's work, so a chunk as
// large as a whole level would give nearly every task one warp and leave heavy tasks as stragglers.
constexpr uint32_t kMaxChunkTasks = 1u << 20;

uint32_t grid(uint64_t n, uint32_t block = 256)
{
    const uint64_t blocks = (n + block - 1) / block;
    return static_cast<uint32_t>(blocks == 0 ? 1 : blocks < 65535 ? blocks : 65535);
}

// ---------------------------------------------------------------------------------------------
// Root database: the built database in the offset layout, with local item names.
// ---------------------------------------------------------------------------------------------

__global__ void root_lengths(const Database* db, uint32_t* lengths, uint32_t* kept)
{
    const uint32_t n = db->num_transactions;
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t <= n; t += gridDim.x * blockDim.x) {
        const uint32_t length = t < n ? db->transactions[t].length : 0;
        lengths[t]            = length;
        kept[t]               = length != 0;
    }
}

__global__ void root_fill(const Database* db, const uint32_t* entry_offset, const uint32_t* tx_index, char* root,
                          DbLayout layout, uint32_t n_trans, uint32_t n_entries, uint32_t n_items, uint32_t min_id,
                          unsigned long long* total_utility)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *reinterpret_cast<DbHeader*>(root) = DbHeader{n_trans, n_entries, n_items, 0, layout.bytes};
    auto* globals = reinterpret_cast<uint32_t*>(root + layout.items);
    auto* txs     = reinterpret_cast<Tx*>(root + layout.tx);
    auto* entries = reinterpret_cast<Entry*>(root + layout.entries);
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_items; i += gridDim.x * blockDim.x)
        globals[i] = min_id + i;

    const uint32_t     n   = db->num_transactions;
    unsigned long long sum = 0;
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += gridDim.x * blockDim.x) {
        const Transaction& in = db->transactions[t];
        if (in.length == 0)
            continue;
        const uint32_t begin = entry_offset[t];
        txs[tx_index[t]]     = Tx{begin, in.length, 0};
        for (uint32_t j = 0; j < in.length; ++j) {
            entries[begin + j] = Entry{in.data[j].key - min_id, in.data[j].util};
            sum += in.data[j].util;
        }
    }
    if (sum != 0)
        add(*total_utility, sum);
}

// ---------------------------------------------------------------------------------------------
// Mining kernels
// ---------------------------------------------------------------------------------------------

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

// Huge tasks: gridDim.y blocks share each task (blockIdx.x), each taking a slice of the
// transactions. Passes that need the whole task finished are separate launches.
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

void launch_bin(const BatchView& b, const uint32_t* order, uint32_t count, uint32_t bin, uint32_t max_slices,
                const AutoTune& tune, cudaStream_t s)
{
    if (count == 0)
        return;
    if (bin == 0) {
        const uint32_t warps = tune.packed_block_warps;
        warp_tasks<<<std::min<uint32_t>(ceil_div(count, warps), 65535), warps * 32, 0, s>>>(b, order, count);
    }
    else if (bin < kOptHugeBin)
        block_tasks<<<std::min<uint32_t>(count, 65535), 32u << bin, 0, s>>>(b, order, count);
    else if (bin == kOptHugeBin) {
        const uint32_t threads = tune.max_block_warps * 32;
        for (uint32_t done = 0; done < count; done += 65535) {
            const uint32_t  n    = std::min<uint32_t>(count - done, 65535);
            const uint32_t* part = order + done;
            const dim3      sliced(n, max_slices), single(n, 1);
            huge_init<<<sliced, threads, 0, s>>>(b, tune, part);
            huge_local<<<sliced, threads, 0, s>>>(b, tune, part);
            huge_rename<<<single, threads, 0, s>>>(b, tune, part);
            huge_subtree<<<sliced, threads, 0, s>>>(b, tune, part);
            huge_allocate<<<single, threads, 0, s>>>(b, tune, part);
            huge_write<<<sliced, threads, 0, s>>>(b, tune, part);
            huge_merge<<<sliced, threads, 0, s>>>(b, tune, part);
            huge_emit<<<single, threads, 0, s>>>(b, tune, part);
        }
    }
    else {
        const uint32_t group   = 1u << (bin - kOptSubBin);
        const uint32_t threads = tune.subwarp_block_warps * 32;
        const uint64_t blocks  = (uint64_t{count} * group + threads - 1) / threads;
        subwarp_tasks<<<static_cast<uint32_t>(std::min<uint64_t>(blocks, 1u << 20)), threads, 0, s>>>(b, order, count,
                                                                                                     group);
    }
    check_cuda(cudaGetLastError(), "launch mining kernel");
}

// ---------------------------------------------------------------------------------------------
// Device profile: block sizes derived from kernel occupancy.
// ---------------------------------------------------------------------------------------------

AutoTune make_auto_tune()
{
    int device = 0;
    check_cuda(cudaGetDevice(&device), "get device");
    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device), "query device");
    const uint32_t sms = static_cast<uint32_t>(props.multiProcessorCount);

    AutoTune tune{};
    tune.resident_warps = sms * (props.maxThreadsPerMultiProcessor / 32);

    auto occupancy = [](const void* kernel, uint32_t threads) {
        int count = 0;
        check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&count, kernel, static_cast<int>(threads), 0),
                   "query occupancy");
        return static_cast<uint32_t>(count);
    };
    // Kernels launched with the same block size: a size is usable only if all of them fit.
    const void* block_kernels[] = {
            reinterpret_cast<const void*>(block_tasks),   reinterpret_cast<const void*>(huge_init),
            reinterpret_cast<const void*>(huge_local),    reinterpret_cast<const void*>(huge_rename),
            reinterpret_cast<const void*>(huge_subtree),  reinterpret_cast<const void*>(huge_allocate),
            reinterpret_cast<const void*>(huge_write),    reinterpret_cast<const void*>(huge_merge),
            reinterpret_cast<const void*>(huge_emit)};
    auto block_occupancy = [&](uint32_t threads) {
        uint32_t blocks = ~0u;
        for (const void* kernel : block_kernels)
            blocks = std::min(blocks, occupancy(kernel, threads));
        return blocks;
    };

    // The largest block size whose resident warps per SM equal the best any size achieves.
    uint32_t best = 0, best_size = 1;
    for (uint32_t size = 1; size < kOptBlockSizes; ++size) {
        const uint32_t resident = block_occupancy(32u << size) << size;
        if (resident != 0 && resident >= best) {
            best      = resident;
            best_size = size;
        }
    }
    tune.max_block_warps = 1u << best_size;
    tune.max_slices      = sms * std::max(1u, block_occupancy(32u << best_size));

    auto warp_block = [&](const void* kernel) {
        uint32_t best_warps = 0, block_warps = 1;
        for (uint32_t size = 0; size < kOptBlockSizes; ++size) {
            const uint32_t resident = occupancy(kernel, 32u << size) << size;
            if (resident != 0 && resident >= best_warps) {
                best_warps  = resident;
                block_warps = 1u << size;
            }
        }
        return block_warps;
    };
    tune.packed_block_warps  = warp_block(reinterpret_cast<const void*>(warp_tasks));
    tune.subwarp_block_warps = warp_block(reinterpret_cast<const void*>(subwarp_tasks));

    std::printf("Auto-tune: %s (%u SMs), %u resident warps, max block %u threads, packed-warp block %u threads, "
                "lane-group block %u threads, huge tasks up to %u blocks\n",
                props.name, sms, tune.resident_warps, tune.max_block_warps * 32, tune.packed_block_warps * 32,
                tune.subwarp_block_warps * 32, tune.max_slices);
    return tune;
}

// ---------------------------------------------------------------------------------------------
// Scheduling kernels: scratch sizes, and binning of a chunk [begin, end) of the level.
// ---------------------------------------------------------------------------------------------

__global__ void scratch_sizes(const Task* tasks, uint32_t count, uint64_t* sizes)
{
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i <= count; i += gridDim.x * blockDim.x)
        sizes[i] = i < count ? scratch_bytes(tasks[i].n_items, tasks[i].n_trans) : 0;
}

__global__ void begin_chunk(Control* control)
{
    for (uint32_t b = threadIdx.x; b < kMaxBins; b += blockDim.x)
        control->bin_count[b] = control->bin_cursor[b] = control->bin_max_slices[b] = 0;
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
        add(control->bin_count[place.bin], 1u);
        fetch_max(control->bin_max_slices[place.bin], place.slices);
    }
}

__global__ void scatter_bins(Control* control, const Task* tasks, uint32_t begin, uint32_t end, uint32_t* order,
                             AutoTune tune)
{
    for (uint32_t i = begin + blockIdx.x * blockDim.x + threadIdx.x; i < end; i += gridDim.x * blockDim.x) {
        const uint32_t bin   = place_task(tasks[i], tune, control->batch_work).bin;
        uint32_t       start = 0;
        for (uint32_t b = 0; b < bin; ++b)
            start += control->bin_count[b];
        order[start + fetch_add(control->bin_cursor[bin], 1u)] = i;
    }
}

// ---------------------------------------------------------------------------------------------
// Output: itemsets are rebuilt by following parent links back through the levels.
// ---------------------------------------------------------------------------------------------

void write_itemsets(const std::string& path, const std::vector<std::vector<Node>>& levels,
                    const Item_mapping* item_map, uint32_t max_item, uint64_t minimum_utility)
{
    uint32_t max_assigned_id = 0;
    for (uint32_t i = 0; i < max_item; ++i)
        max_assigned_id = std::max(max_assigned_id, item_map[i].id);
    std::vector<std::string> id_to_item(static_cast<size_t>(max_assigned_id) + 1);
    for (uint32_t i = 0; i < max_item; ++i)
        id_to_item[item_map[i].id] = std::to_string(item_map[i].item);

    std::FILE* out = std::fopen(path.c_str(), "w");
    if (out == nullptr) {
        std::cerr << "Error: Could not open output file: " << path << "\n";
        return;
    }
    std::vector<uint32_t> items;
    std::string           line;
    for (size_t level = 0; level < levels.size(); ++level) {
        for (const Node& node : levels[level]) {
            if (node.utility < minimum_utility)
                continue;
            items.clear();
            items.push_back(node.item);
            uint32_t parent = node.parent;
            for (size_t up = level; up-- > 0;) {
                const Node& p = levels[up][parent];
                items.push_back(p.item);
                parent = p.parent;
            }
            line.clear();
            for (size_t k = items.size(); k-- > 0;) {
                line += id_to_item[items[k]];
                line.push_back(' ');
            }
            line += "UTIL ";
            line += std::to_string(node.utility);
            line.push_back('\n');
            std::fwrite(line.data(), 1, line.size(), out);
        }
    }
    std::fclose(out);
}

} // namespace
} // namespace bfs

using namespace bfs;

void write_bfs_itemsets(const std::string& path, const BfsResult& result, const Item_mapping* item_map,
                        uint32_t max_item, uint64_t minimum_utility)
{
    write_itemsets(path, result.levels, item_map, max_item, minimum_utility);
}

BfsResult bfs_optimized_mining(const built_file& bf, const Item_mapping* item_map, uint32_t max_item,
                               const BfsOptions& options, Timer& timer)
{
    // Promising items carry consecutive ids (assigned in TWU order); pruned items carry id 0.
    uint32_t min_id = ~0u, max_id = 0, promising = 0;
    for (uint32_t i = 0; i < max_item; ++i)
        if (item_map[i].id != 0) {
            min_id = std::min(min_id, item_map[i].id);
            max_id = std::max(max_id, item_map[i].id);
            ++promising;
        }
    BfsResult result;
    if (promising == 0 || bf.primary.empty())
        return result;
    const uint32_t n_items = max_id - min_id + 1;
    if (n_items != promising)
        throw std::runtime_error("promising item ids are not consecutive");

    const AutoTune               tune = make_auto_tune();
    std::array<cudaStream_t, 4>  streams{};
    for (auto& stream : streams)
        check_cuda(cudaStreamCreate(&stream), "create stream");
    cudaStream_t main_stream = streams[0];

    // --- Root database ---
    const uint32_t         n_raw = static_cast<uint32_t>(bf.db->num_transactions);
    DeviceBuffer<uint32_t> lengths(n_raw + 1), kept(n_raw + 1), entry_offset(n_raw + 1), tx_index(n_raw + 1);
    size_t                 scan_temp_bytes = 0;
    check_cuda(cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_bytes, lengths.get(), entry_offset.get(), n_raw + 1),
               "size scan");
    {
        DeviceBuffer<char> temp(std::max<size_t>(scan_temp_bytes, 1));
        root_lengths<<<grid(n_raw + 1), 256, 0, main_stream>>>(bf.db, lengths.get(), kept.get());
        check_cuda(cub::DeviceScan::ExclusiveSum(temp.get(), scan_temp_bytes, lengths.get(), entry_offset.get(),
                                                 n_raw + 1, main_stream),
                   "scan lengths");
        check_cuda(cub::DeviceScan::ExclusiveSum(temp.get(), scan_temp_bytes, kept.get(), tx_index.get(), n_raw + 1,
                                                 main_stream),
                   "scan transactions");
    }
    uint32_t n_entries = 0, n_trans = 0;
    check_cuda(cudaMemcpyAsync(&n_entries, entry_offset.get() + n_raw, sizeof(uint32_t), cudaMemcpyDeviceToHost,
                               main_stream),
               "read entry count");
    check_cuda(cudaMemcpyAsync(&n_trans, tx_index.get() + n_raw, sizeof(uint32_t), cudaMemcpyDeviceToHost,
                               main_stream),
               "read transaction count");
    check_cuda(cudaStreamSynchronize(main_stream), "build root");

    const DbLayout                         root_layout = DbLayout::of(n_trans, n_entries, n_items);
    DeviceBuffer<char>                     root(root_layout.bytes);
    DeviceBuffer<unsigned long long>       total_utility(1);
    check_cuda(cudaMemsetAsync(total_utility.get(), 0, sizeof(unsigned long long), main_stream), "clear total");
    root_fill<<<grid(std::max(n_raw, n_items)), 256, 0, main_stream>>>(bf.db, entry_offset.get(), tx_index.get(),
                                                                       root.get(), root_layout, n_trans, n_entries,
                                                                       n_items, min_id, total_utility.get());
    unsigned long long total = 0;
    check_cuda(cudaMemcpyAsync(&total, total_utility.get(), sizeof(total), cudaMemcpyDeviceToHost, main_stream),
               "read total utility");
    check_cuda(cudaStreamSynchronize(main_stream), "fill root");
    if (total > 0xffffffffull)
        throw std::runtime_error("total database utility exceeds 32 bits");
    lengths = {}, kept = {}, entry_offset = {}, tx_index = {};

    // --- Memory budget: task arrays, scratch, and two arena regions ---
    const uint64_t capacity = options.task_capacity;
    size_t         free_bytes = 0, total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "query device memory");
    size_t prefix_temp_bytes = 0;
    check_cuda(cub::DeviceScan::ExclusiveSum(nullptr, prefix_temp_bytes, static_cast<uint64_t*>(nullptr),
                                             static_cast<uint64_t*>(nullptr), capacity + 1),
               "size prefix scan");
    const uint64_t per_task   = 2 * sizeof(Task) + sizeof(Node) + 2 * sizeof(uint64_t) + sizeof(uint32_t);
    const uint64_t task_bytes = capacity * per_task + prefix_temp_bytes;
    const uint64_t budget     = std::min<uint64_t>(options.memory_bytes, free_bytes - (512ull << 20));
    const uint64_t scratch_capacity = std::clamp<uint64_t>(budget / 8, 256ull << 20, 4ull << 30);
    if (budget < task_bytes + scratch_capacity + (64ull << 20))
        throw std::runtime_error("not enough device memory for the task arrays; lower --stack_capacity");
    const uint64_t region_bytes =
            std::min<uint64_t>((budget - task_bytes - scratch_capacity) / 2 / kArenaUnit * kArenaUnit,
                               0xffffffffull * kArenaUnit);
    std::printf("BFS memory: %llu MiB per arena region (x2), scratch %llu MiB, %llu tasks per level (%llu MiB)\n",
                (unsigned long long)(region_bytes >> 20), (unsigned long long)(scratch_capacity >> 20),
                (unsigned long long)capacity, (unsigned long long)(task_bytes >> 20));

    DeviceBuffer<Task>     level_a(capacity), level_b(capacity);
    DeviceBuffer<Node>     nodes(capacity);
    DeviceBuffer<uint64_t> sizes(capacity + 1), prefix(capacity + 1);
    DeviceBuffer<uint32_t> order(capacity);
    DeviceBuffer<char>     prefix_temp(std::max<size_t>(prefix_temp_bytes, 1));
    DeviceBuffer<char>     scratch(scratch_capacity);
    DeviceBuffer<char>     regions[2] = {DeviceBuffer<char>(region_bytes), DeviceBuffer<char>(region_bytes)};
    DeviceBuffer<Control>  control(1);

    // --- Level 1: one task per primary item of the root ---
    std::vector<Task> seeds;
    for (uint32_t id : bf.primary)
        seeds.push_back(Task{0, static_cast<uint32_t>(root_layout.bytes / kArenaUnit), n_trans, n_items, id - min_id,
                             kNone});
    if (seeds.size() > capacity)
        throw std::runtime_error("too many first-level tasks; raise --stack_capacity");
    check_cuda(cudaMemcpy(level_a.get(), seeds.data(), seeds.size() * sizeof(Task), cudaMemcpyHostToDevice),
               "upload seeds");

    Task*                          current = level_a.get();
    Task*                          next    = level_b.get();
    uint32_t                       count   = static_cast<uint32_t>(seeds.size());
    Control                        host{};
    std::vector<std::vector<Node>>& levels = result.levels;
    const bool                     collect = !options.output.empty() && options.output != "/dev/null";
    uint64_t                       peak    = 0;
    uint64_t                       peak_scratch = 0, peak_tasks = 0;
    std::vector<uint64_t>          host_prefix;

    for (uint32_t level = 1; count != 0; ++level) {
        const char* src = level == 1 ? root.get() : regions[(level - 1) & 1].get();
        char*       dst = regions[level & 1].get();

        host.arena_top  = 0;
        host.next_count = 0;
        host.error      = 0;
        check_cuda(cudaMemcpyAsync(control.get(), &host, sizeof(Control), cudaMemcpyHostToDevice, main_stream),
                   "reset control");

        // Scratch offsets over the whole level; split it into chunks that fit the scratch buffer.
        scratch_sizes<<<grid(count + 1), 256, 0, main_stream>>>(current, count, sizes.get());
        check_cuda(cub::DeviceScan::ExclusiveSum(prefix_temp.get(), prefix_temp_bytes, sizes.get(), prefix.get(),
                                                 count + 1, main_stream),
                   "scan scratch sizes");
        uint64_t level_scratch = 0;
        check_cuda(cudaMemcpyAsync(&level_scratch, prefix.get() + count, sizeof(uint64_t), cudaMemcpyDeviceToHost,
                                   main_stream),
                   "read scratch total");
        check_cuda(cudaStreamSynchronize(main_stream), "scratch sizes");

        peak_scratch = std::max<uint64_t>(peak_scratch, std::min<uint64_t>(level_scratch, scratch_capacity));
        peak_tasks   = std::max<uint64_t>(peak_tasks, count);
        std::vector<uint32_t> cuts{0};
        if (level_scratch > scratch_capacity || count > kMaxChunkTasks) {
            host_prefix.resize(count + 1);
            check_cuda(cudaMemcpy(host_prefix.data(), prefix.get(), (count + 1) * sizeof(uint64_t),
                                  cudaMemcpyDeviceToHost),
                       "read scratch offsets");
            uint32_t begin = 0;
            while (begin < count) {
                uint32_t end = begin;
                while (end < count && end - begin < kMaxChunkTasks &&
                       host_prefix[end + 1] - host_prefix[begin] <= scratch_capacity)
                    ++end;
                if (end == begin)
                    throw std::runtime_error("one task needs more scratch than available; raise --total_memory");
                cuts.push_back(end);
                begin = end;
            }
        }
        else
            cuts.push_back(count);

        for (size_t c = 0; c + 1 < cuts.size(); ++c) {
            const uint32_t begin = cuts[c], end = cuts[c + 1];
            const uint64_t base  = cuts.size() > 2 ? host_prefix[begin] : 0;

            begin_chunk<<<1, 32, 0, main_stream>>>(control.get());
            measure_chunk<<<grid(end - begin), 256, 0, main_stream>>>(control.get(), current, begin, end);
            count_bins<<<grid(end - begin), 256, 0, main_stream>>>(control.get(), current, begin, end, tune);
            scatter_bins<<<grid(end - begin), 256, 0, main_stream>>>(control.get(), current, begin, end, order.get(),
                                                                    tune);
            check_cuda(cudaMemcpyAsync(&host, control.get(), sizeof(Control), cudaMemcpyDeviceToHost, main_stream),
                       "read bins");
            check_cuda(cudaStreamSynchronize(main_stream), "bin chunk");

            const BatchView view{src,         dst,           region_bytes, control.get(), current,
                                 prefix.get(), base,         scratch.get(), next,         static_cast<uint32_t>(capacity),
                                 nodes.get(), options.minimum_utility};
            uint32_t        first = 0;
            for (uint32_t b = 0; b < kBinCount; ++b) {
                launch_bin(view, order.get() + first, host.bin_count[b], b, host.bin_max_slices[b], tune,
                           streams[b % streams.size()]);
                first += host.bin_count[b];
            }
            check_cuda(cudaDeviceSynchronize(), "mine level");
        }

        check_cuda(cudaMemcpy(&host, control.get(), sizeof(Control), cudaMemcpyDeviceToHost), "read control");
        if (host.error & kArenaFull)
            throw std::runtime_error("level " + std::to_string(level) +
                                     " does not fit in its arena region; raise --total_memory");
        if (host.error & kNextLevelFull)
            throw std::runtime_error("level " + std::to_string(level + 1) + " has more than " +
                                     std::to_string(capacity) + " tasks; raise --stack_capacity");
        peak = std::max<uint64_t>(peak, host.arena_top);

        if (collect) {
            levels.emplace_back(count);
            check_cuda(cudaMemcpy(levels.back().data(), nodes.get(), count * sizeof(Node), cudaMemcpyDeviceToHost),
                       "read itemsets");
        }

        timer.recordPoint("Level " + std::to_string(level), 0, host.arena_top);
        std::swap(current, next);
        count = host.next_count;
    }

    for (auto stream : streams)
        cudaStreamDestroy(stream);
    std::printf("BFS: peak arena region use %llu MiB, peak scratch %llu MiB, peak level %llu tasks (%llu MiB of task "
                "arrays)\n",
                (unsigned long long)(peak >> 20), (unsigned long long)(peak_scratch >> 20),
                (unsigned long long)peak_tasks, (unsigned long long)((peak_tasks * per_task) >> 20));

    result.patterns = host.patterns;
    return result;
}
