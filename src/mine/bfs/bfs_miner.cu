// Host driver of the level-synchronous (BFS) miner.
//
// Each level is mined in chunks: the chunk's tasks are binned by size, each bin runs on its own
// stream with a thread-group size chosen from the chunk's total work, and children are written into
// the arena region the previous level is not using. Per-task working memory comes from a scratch
// buffer that is reused every chunk, so the arenas only ever hold projected databases.

#include "cuda/nvtx.hpp"
#include "cuda/buffer.cuh"
#include "mine/bfs/bfs_miner.hpp"
#include "mine/bfs/kernels.cuh"
#include "mine/bfs/root_image.hpp"
#include "mine/host/transfer.hpp"
#include "mine/output.hpp"
#include "mine/primitives.cuh"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <memory>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

namespace ghuim {
namespace {

using namespace bfs;

// Tasks per chunk. Thread groups are sized from a task's share of the chunk's work, so a chunk as
// large as a whole level would give nearly every task one warp and leave heavy tasks as stragglers.
constexpr uint32_t kMaxChunkTasks = 1u << 20;
// Cap on tasks per level: each costs 76 bytes of task arrays.
constexpr uint64_t kMaxLevelTasks = 32u << 20;
constexpr uint64_t kReservedBytes = 512ull << 20; // runtime, kernels and slack kept free
constexpr uint32_t kStreams       = 4;

class Streams
{
  public:
    Streams()
    {
        for (auto& stream : streams_)
            check_cuda(cudaStreamCreate(&stream), "create stream");
    }
    ~Streams()
    {
        for (auto stream : streams_)
            cudaStreamDestroy(stream);
    }
    Streams(const Streams&)                = delete;
    Streams&     operator=(const Streams&) = delete;
    cudaStream_t operator[](size_t i) const
    {
        return streams_[i % kStreams];
    }

  private:
    std::array<cudaStream_t, kStreams> streams_{};
};

// Device profile of the current GPU.
AutoTune make_auto_tune(const MiningOptions& options)
{
    int device = 0;
    check_cuda(cudaGetDevice(&device), "get device");
    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device), "query device");
    const uint32_t sms = static_cast<uint32_t>(props.multiProcessorCount);

    AutoTune tune{};
    tune.warp_mode  = options.method == Method::Warp;
    tune.staged     = options.method == Method::Staged;
    tune.io_warps   = options.io_warps;
    tune.ring_bytes = options.ring_kib * 1024;
    tune.resident_warps = sms * (props.maxThreadsPerMultiProcessor / 32);
    measure_occupancy(tune, sms);
    // Static: every task runs in the block-per-task bin of --block-size threads (bin 0 is one warp).
    tune.fixed_bin = options.method == Method::Static ? static_cast<uint32_t>(std::countr_zero(options.block_size / 32))
                                                      : kNone;
    spdlog::info("Auto-tune ({}): {} ({} SMs), {} resident warps, max block {} threads, packed-warp block {} threads, "
                 "lane-group block {} threads, huge tasks up to {} blocks",
                 options.method == Method::Warp ? "lane group per transaction"
                                                    : options.method == Method::Staged ? "staged lanes" : "lane per transaction", props.name, sms,
                 tune.resident_warps, tune.max_block_warps * 32, tune.packed_block_warps * 32,
                 tune.subwarp_block_warps * 32, tune.max_slices);
    return tune;
}

// Device buffers sized from the free memory.
struct Budget
{
    uint64_t capacity;         // tasks per level
    uint64_t scratch_capacity; // bytes
    uint64_t region_bytes;     // each of the two arena regions
    uint64_t task_bytes;
    uint64_t per_task;
};

Budget plan_budget(const MiningOptions& options, uint64_t first_level_tasks)
{
    size_t free_bytes = 0, total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "query device memory");
    if (free_bytes <= kReservedBytes + (64ull << 20))
        throw std::runtime_error("not enough free device memory");
    const uint64_t usable = free_bytes - kReservedBytes;
    const uint64_t budget = options.memory_bytes == 0 ? usable : std::min<uint64_t>(options.memory_bytes, usable);

    Budget plan{};
    plan.per_task = 2 * sizeof(Task) + sizeof(Node) + sizeof(uint64_t) + sizeof(uint32_t);
    // The task arrays may take a quarter of the budget; a smaller level cap only costs a retry message.
    const uint64_t fit = budget / 4 / plan.per_task;
    plan.capacity      = std::min<uint64_t>({options.stack_capacity, kMaxLevelTasks, fit});
    if (plan.capacity < first_level_tasks)
        throw std::runtime_error("not enough device memory for the first level's tasks");
    if (plan.capacity < std::min<uint64_t>(options.stack_capacity, kMaxLevelTasks))
        spdlog::warn("Level capacity limited to {} tasks by device memory", plan.capacity);

    plan.task_bytes       = plan.capacity * plan.per_task + scan_block_count(plan.capacity + 1) * sizeof(uint64_t);
    plan.scratch_capacity = std::clamp<uint64_t>(budget / 8, 256ull << 20, 4ull << 30);
    if (budget < plan.task_bytes + plan.scratch_capacity + (64ull << 20))
        throw std::runtime_error("not enough device memory for the task arrays; lower --stack-capacity");
    plan.region_bytes =
            std::min<uint64_t>((budget - plan.task_bytes - plan.scratch_capacity) / 2 / kArenaUnit * kArenaUnit,
                               0xffffffffull * kArenaUnit);
    spdlog::info("BFS memory: {} MiB per arena region (x2), scratch {} MiB, {} tasks per level ({} MiB)",
                 plan.region_bytes >> 20, plan.scratch_capacity >> 20, plan.capacity, plan.task_bytes >> 20);
    return plan;
}

class LevelMiner
{
  public:
    LevelMiner(const Dataset& data, const MiningOptions& options)
        : data_(data), options_(options), tune_(make_auto_tune(options)), budget_(plan_budget(options, data.primary.size())),
          collect_(!options.output.empty())
    {
        allocate();
    }

    MiningResult run();

    // The itemsets of every level, when --output is set.
    const std::vector<std::vector<Node>>& levels() const
    {
        return levels_;
    }

  private:
    void     allocate();
    void     seed_first_level();
    void     reset_control();
    void     plan_chunks(uint32_t count, std::vector<uint32_t>& cuts);
    void     mine_chunk(uint32_t level, uint32_t begin, uint32_t end, uint64_t scratch_base);
    void     check_level(uint32_t level);
    uint32_t mine_level(uint32_t level);

    const Dataset&       data_;
    const MiningOptions& options_;
    const AutoTune       tune_;
    const Budget         budget_;
    const bool           collect_;
    Streams              streams_;

    DeviceBuffer<char>                root_;
    DeviceBuffer<Task>                level_a_, level_b_;
    DeviceBuffer<Node>                nodes_;
    DeviceBuffer<uint64_t>            prefix_, scan_sums_, scan_total_;
    DeviceBuffer<uint32_t>            order_;
    DeviceBuffer<char>                scratch_;
    std::array<DeviceBuffer<char>, 2> regions_;
    DeviceBuffer<Control>             control_;

    uint64_t                       root_bytes_ = 0;
    uint32_t                       root_trans_ = 0, root_items_ = 0;
    Task*                          current_ = nullptr;
    Task*                          next_    = nullptr;
    uint32_t                       count_   = 0;
    Control                        host_{};
    std::vector<uint64_t>          host_prefix_;
    std::vector<std::vector<Node>> levels_;
    uint64_t                       peak_arena_ = 0, peak_scratch_ = 0, peak_tasks_ = 0;
    MiningResult                   result_;
};

void LevelMiner::allocate()
{
    GHUIM_RANGE("bfs: build + upload root database");
    const std::vector<char> root = root_image(data_);
    root_bytes_                  = root.size();
    root_trans_                  = static_cast<uint32_t>(data_.transaction_count());
    root_items_                  = static_cast<uint32_t>(data_.original_item.size());
    if (root_bytes_ / kArenaUnit > 0xffffffffull)
        throw std::runtime_error("the input database is too large for 32-bit arena offsets");
    root_ = DeviceBuffer<char>(root_bytes_);
    upload(root_.get(), root.data(), root.size());

    const uint64_t capacity = budget_.capacity;
    level_a_                = DeviceBuffer<Task>(capacity);
    level_b_                = DeviceBuffer<Task>(capacity);
    nodes_                  = DeviceBuffer<Node>(capacity);
    prefix_                 = DeviceBuffer<uint64_t>(capacity + 1);
    scan_sums_              = DeviceBuffer<uint64_t>(scan_block_count(capacity + 1));
    scan_total_             = DeviceBuffer<uint64_t>(1);
    order_                  = DeviceBuffer<uint32_t>(capacity);
    scratch_                = DeviceBuffer<char>(budget_.scratch_capacity);
    regions_                = {DeviceBuffer<char>(budget_.region_bytes), DeviceBuffer<char>(budget_.region_bytes)};
    control_                = DeviceBuffer<Control>(1);
}

// Level 1: one task per primary item of the root.
void LevelMiner::seed_first_level()
{
    std::vector<Task> seeds;
    seeds.reserve(data_.primary.size());
    for (uint32_t item : data_.primary)
        seeds.push_back(
                Task{0, static_cast<uint32_t>(root_bytes_ / kArenaUnit), root_trans_, root_items_, item, kNone});
    upload(level_a_.get(), seeds.data(), seeds.size());
    current_ = level_a_.get();
    next_    = level_b_.get();
    count_   = static_cast<uint32_t>(seeds.size());
}

void LevelMiner::reset_control()
{
    GHUIM_RANGE("bfs: reset control");
    host_.arena_top  = 0;
    host_.next_count = 0;
    host_.error      = 0;
    check_cuda(cudaMemcpyAsync(control_.get(), &host_, sizeof(Control), cudaMemcpyHostToDevice, streams_[0]),
               "reset control");
}

// Scratch offsets over the whole level, split into chunks that fit the scratch buffer (`cuts` holds
// the chunk boundaries).
void LevelMiner::plan_chunks(uint32_t count, std::vector<uint32_t>& cuts)
{
    GHUIM_RANGE("bfs: plan chunks (scratch sizes + scan + sync)");
    launch_scratch_sizes(current_, count, prefix_.get(), streams_[0]);
    exclusive_scan(prefix_.get(), count + 1, scan_sums_.get(), scan_total_.get(), streams_[0]);
    uint64_t level_scratch = 0;
    check_cuda(cudaMemcpyAsync(&level_scratch, prefix_.get() + count, sizeof(uint64_t), cudaMemcpyDeviceToHost,
                               streams_[0]),
               "read scratch total");
    check_cuda(cudaStreamSynchronize(streams_[0]), "scratch sizes");

    peak_scratch_ = std::max<uint64_t>(peak_scratch_, std::min<uint64_t>(level_scratch, budget_.scratch_capacity));
    peak_tasks_   = std::max<uint64_t>(peak_tasks_, count);
    cuts          = {0};
    if (level_scratch <= budget_.scratch_capacity && count <= kMaxChunkTasks) {
        cuts.push_back(count);
        return;
    }
    host_prefix_.resize(count + 1);
    download(host_prefix_.data(), prefix_.get(), count + 1);
    for (uint32_t begin = 0; begin < count;) {
        uint32_t end = begin;
        while (end < count && end - begin < kMaxChunkTasks &&
               host_prefix_[end + 1] - host_prefix_[begin] <= budget_.scratch_capacity)
            ++end;
        if (end == begin)
            throw std::runtime_error("one task needs more scratch than available; raise --total-memory");
        cuts.push_back(end);
        begin = end;
    }
}

// Bins the chunk [begin, end) and runs every bin on its own stream.
void LevelMiner::mine_chunk(uint32_t level, uint32_t begin, uint32_t end, uint64_t scratch_base)
{
    const char* src = level == 1 ? root_.get() : regions_[(level - 1) & 1].get();
    char*       dst = regions_[level & 1].get();

    {
        GHUIM_RANGE("bfs: bin chunk + readback");
        launch_bin_chunk(control_.get(), current_, begin, end, order_.get(), tune_, streams_[0]);
        check_cuda(cudaMemcpyAsync(&host_, control_.get(), sizeof(Control), cudaMemcpyDeviceToHost, streams_[0]),
                   "read bins");
        check_cuda(cudaStreamSynchronize(streams_[0]), "bin chunk");
    }
    GHUIM_RANGE("bfs: launch bins + sync");

    const BatchView view{src,
                         dst,
                         budget_.region_bytes,
                         control_.get(),
                         current_,
                         prefix_.get(),
                         scratch_base,
                         scratch_.get(),
                         next_,
                         static_cast<uint32_t>(budget_.capacity),
                         nodes_.get(),
                         options_.minimum_utility,
                         tune_.staged ? tune_.io_warps : 0,
                         tune_.ring_bytes};
    uint32_t        first = 0;
    for (uint32_t bin = 0; bin < kBinCount; ++bin) {
        const uint32_t  count  = host_.bin_count[bin];
        const uint32_t* tasks  = order_.get() + first;
        cudaStream_t    stream = streams_[bin];
        first += count;
        if (count == 0)
            continue;
        if (bin == 0)
            launch_warp_tasks(view, tasks, count, tune_, stream);
        else if (bin < kOptHugeBin)
            launch_block_tasks(view, tasks, count, bin, tune_, stream);
        else if (bin == kOptHugeBin)
            launch_huge_tasks(view, tasks, count, host_.bin_max_slices[bin], tune_, stream);
        else
            launch_subwarp_tasks(view, tasks, count, 1u << (bin - kOptSubBin), tune_, stream);
    }
    synchronize("mine level");
}

void LevelMiner::check_level(uint32_t level)
{
    if (host_.error & kArenaFull)
        throw std::runtime_error("level " + std::to_string(level) +
                                 " does not fit in its arena region (raise --total-memory)");
    if (host_.error & kStageTimeout)
        throw std::runtime_error("a staging wait timed out (protocol error)");
    if (host_.error & kStageOverflow)
        throw std::runtime_error("a transaction is longer than its database's recorded longest");
    if (host_.error & kNextLevelFull)
        throw std::runtime_error("level " + std::to_string(level + 1) + " has more than " +
                                 std::to_string(budget_.capacity) + " tasks (raise --stack-capacity)");
}

// Mines the whole current level; returns the number of tasks of the next one.
uint32_t LevelMiner::mine_level(uint32_t level)
{
    reset_control();
    std::vector<uint32_t> cuts;
    plan_chunks(count_, cuts);
    for (size_t c = 0; c + 1 < cuts.size(); ++c) {
        const uint64_t base = cuts.size() > 2 ? host_prefix_[cuts[c]] : 0;
        mine_chunk(level, cuts[c], cuts[c + 1], base);
    }

    GHUIM_RANGE("bfs: level readback");
    download(&host_, control_.get(), 1);
    check_level(level);
    peak_arena_ = std::max<uint64_t>(peak_arena_, host_.arena_top);
    if (collect_) {
        levels_.emplace_back(count_);
        download(levels_.back().data(), nodes_.get(), count_);
    }
    return host_.next_count;
}

MiningResult LevelMiner::run()
{
    seed_first_level();
    uint64_t counted = 0;
    for (uint32_t level = 1; count_ != 0; ++level) {
        const auto     start = std::chrono::steady_clock::now();
        const uint32_t tasks = count_;
        const uint32_t next  = mine_level(level);
        const double   ms    = ms_since(start);

        result_.tasks += tasks;
        result_.kernel_ms += ms;
        result_.iterations.push_back({tasks, host_.patterns - counted, host_.arena_top, ms});
        spdlog::debug("Level {:>3}: {:>10} tasks, {:>10} itemsets, {} MiB of child databases, {:.1f} ms", level, tasks,
                      host_.patterns - counted, host_.arena_top >> 20, ms);
        counted = host_.patterns;
        std::swap(current_, next_);
        count_ = next;
    }
    spdlog::info("BFS: peak arena region use {} MiB, peak scratch {} MiB, peak level {} tasks ({} MiB of task arrays)",
                 peak_arena_ >> 20, peak_scratch_ >> 20, peak_tasks_, (peak_tasks_ * budget_.per_task) >> 20);
    if (host_.stage_consumer_cycles != 0)
        spdlog::info("Staging: compute warps waited {:.1f}% of their time for data, IO warps waited {:.1f} Mcycles for free slots",
                     100.0 * static_cast<double>(host_.stage_full_wait) / static_cast<double>(host_.stage_consumer_cycles),
                     static_cast<double>(host_.stage_empty_wait) / 1e6);
    result_.patterns   = host_.patterns;
    result_.peak_arena = peak_arena_;
    return result_;
}

// Rebuilds each itemset by following parent links back through the levels.
void write_itemsets(const std::vector<std::vector<Node>>& levels, uint64_t minimum_utility, ItemsetWriter& writer)
{
    std::vector<uint32_t> names;
    for (size_t level = 0; level < levels.size(); ++level) {
        for (const Node& node : levels[level]) {
            if (node.utility < minimum_utility)
                continue;
            names.assign(1, node.item);
            uint32_t parent = node.parent;
            for (size_t up = level; up-- > 0;) {
                const Node& ancestor = levels[up][parent];
                names.push_back(ancestor.item);
                parent = ancestor.parent;
            }
            std::reverse(names.begin(), names.end());
            writer.write(names.data(), names.size(), node.utility);
        }
    }
}

} // namespace

MiningResult mine_bfs(const Dataset& data, const MiningOptions& options)
{
    if (data.primary.empty())
        return {};
    MiningOptions effective = options;
    if (options.output == "/dev/null")
        effective.output.clear();
    LevelMiner   miner(data, effective);
    MiningResult result = miner.run();
    if (!effective.output.empty()) {
        GHUIM_RANGE("bfs: write itemsets");
        ItemsetWriter writer(effective.output, data.original_item, 0);
        write_itemsets(miner.levels(), effective.minimum_utility, writer);
    }
    return result;
}

} // namespace ghuim
