// Host driver of the depth-first miners (static and adaptive).
//
// Tasks live on device stacks. Each iteration the kernels consume every task of the current stack
// and push the children onto the next one. Projected databases are bump-allocated from an arena;
// when the space above the live databases runs low, allocation wraps to the free space below them
// (databases never reference older ones, so everything outside the live range is dead).

#include "cuda/buffer.cuh"
#include "mine/dfs/dfs_miner.hpp"
#include "mine/dfs/kernels.cuh"
#include "mine/host/transfer.hpp"
#include "mine/output.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ghuim {
namespace {

using namespace dfs;

// Kept free for the runtime and the kernels' per-thread local memory, which is reserved at launch.
constexpr uint64_t kReservedBytes = 4ull << 30;
constexpr uint32_t kDistribution  = 8;

// A task stack: the control struct in managed memory, the tasks in device memory.
class DeviceStack
{
  public:
    explicit DeviceStack(uint32_t capacity) : tasks_(capacity)
    {
        stack_->data     = tasks_.get();
        stack_->capacity = capacity;
    }
    Stack<Task>* get() const
    {
        return stack_.get();
    }
    uint32_t size() const
    {
        return stack_->top;
    }
    void clear()
    {
        stack_->top = 0;
    }
    Task* storage() const
    {
        return tasks_.get();
    }

  private:
    ManagedObject<Stack<Task>> stack_;
    DeviceBuffer<Task>         tasks_;
};

std::string describe(uint32_t error)
{
    std::string text;
    auto        add = [&](uint32_t bit, const char* message) {
        if (error & bit)
            text += std::string(text.empty() ? "" : "; ") + message;
    };
    add(kArenaFull, "projected-database memory exhausted (raise --total-memory)");
    add(kStackFull, "task stack overflow (raise --stack-capacity)");
    add(kPatternsFull, "itemset buffer overflow (raise --pattern-memory)");
    add(kLongTransaction, "a projected transaction has more than 256 items (use --method bfs-optimized)");
    add(kTableFull, "utility table overflow");
    add(kLimitExceeded, "task exceeds a fixed per-thread limit");
    return text;
}

// Block size for the next adaptive iteration: the average size class of the queued tasks.
uint32_t adaptive_block_threads(const uint32_t* distribution, uint32_t previous)
{
    uint64_t weighted = 0, tasks = 0;
    for (uint32_t bucket = 1; bucket < kDistribution; ++bucket) {
        weighted += uint64_t{distribution[bucket]} << (bucket + 3);
        tasks += distribution[bucket];
    }
    uint32_t threads = previous;
    if (tasks > 0)
        threads = static_cast<uint32_t>((weighted / tasks + 31) / 32 * 32);
    return std::clamp(threads, 32u, 1024u);
}

class Search
{
  public:
    Search(const Dataset& data, const MiningOptions& options, ItemsetWriter* writer)
        : data_(data), options_(options), writer_(writer), adaptive_(options.method == Method::Adaptive)
    {
    }

    MiningResult run();

  private:
    void     validate() const;
    void     allocate();
    void     upload_root_and_seeds();
    uint64_t root_bytes() const;
    void     run_static();
    void     run_adaptive();
    void     finish_iteration(uint64_t live_begin, uint64_t tasks, double ms);
    void     check_errors() const;
    void     reclaim_arena(uint64_t live_begin);
    void     drain_patterns();

    const Dataset&       data_;
    const MiningOptions& options_;
    ItemsetWriter*       writer_;
    const bool           adaptive_;

    uint32_t                                           capacity_ = 0;
    DeviceBuffer<char>                                 arena_;
    DeviceBuffer<uint32_t>                             patterns_;
    ManagedObject<Control>                             control_;
    ManagedObject<std::array<uint32_t, kDistribution>> distribution_;
    std::unique_ptr<DeviceStack>                       current_, next_, singles_, singles_next_;
    std::vector<uint32_t>                              staging_;
    uint64_t                                           root_end_ = 0;
    uint32_t                                           threads_  = 0;
    MiningResult                                       result_;
};

void Search::validate() const
{
    if (options_.minimum_utility > 0xffffffffull)
        throw std::runtime_error("the depth-first methods need --minutil below 2^32; use --method bfs-optimized");
    if (data_.original_item.size() + 2 >= 0xffffffffull)
        throw std::runtime_error("too many items");
}

uint64_t Search::root_bytes() const
{
    return sizeof(Database) + sizeof(Item) * data_.items.size() + sizeof(Transaction) * data_.transaction_count();
}

// Allocates the stacks, the itemset buffer and the arena within the free device memory.
void Search::allocate()
{
    size_t free_bytes = 0, total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "query device memory");
    const uint32_t stacks = adaptive_ ? 4 : 2;

    // Stacks may use up to 40% of the free memory; a smaller capacity only costs a retry message.
    const uint64_t fit = static_cast<uint64_t>(free_bytes * 0.4) / (stacks * sizeof(Task));
    capacity_          = static_cast<uint32_t>(std::min<uint64_t>({options_.stack_capacity, fit, 0x7fffffffull}));
    if (capacity_ < data_.primary.size())
        throw std::runtime_error("not enough device memory for the task stacks");
    if (capacity_ < options_.stack_capacity)
        spdlog::warn("Task stacks limited to {} tasks by free device memory", capacity_);

    current_ = std::make_unique<DeviceStack>(capacity_);
    next_    = std::make_unique<DeviceStack>(capacity_);
    if (adaptive_) {
        singles_      = std::make_unique<DeviceStack>(capacity_);
        singles_next_ = std::make_unique<DeviceStack>(capacity_);
    }
    if (!options_.output.empty() && options_.pattern_bytes / sizeof(uint32_t) > 0)
        patterns_ = DeviceBuffer<uint32_t>(options_.pattern_bytes / sizeof(uint32_t));

    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "query device memory");
    if (free_bytes < kReservedBytes + 2 * root_bytes())
        throw std::runtime_error("not enough free device memory for the projected databases");
    const uint64_t available = free_bytes - kReservedBytes;
    uint64_t       arena     = options_.memory_bytes == 0 ? available : options_.memory_bytes;
    if (arena > available)
        throw std::runtime_error("--total-memory exceeds the free device memory (" + std::to_string(available >> 20) +
                                 " MiB usable)");
    arena_ = DeviceBuffer<char>(arena);
    spdlog::info(
            "Device memory: {} MiB free; arena {} MiB, {} stack(s) of {} tasks ({} MiB each), itemset buffer {} MiB",
            free_bytes >> 20, arena >> 20, stacks, capacity_, (uint64_t{capacity_} * sizeof(Task)) >> 20,
            (patterns_.size() * sizeof(uint32_t)) >> 20);

    control_->arena       = arena_.get();
    control_->arena_limit = arena;
    if (patterns_.size() != 0) {
        control_->patterns         = patterns_.get();
        control_->pattern_capacity = patterns_.size();
    }
}

// The root database goes to the start of the arena; one task per first-level item points at it.
void Search::upload_root_and_seeds()
{
    const uint64_t bytes = root_bytes();
    if (bytes > arena_.size())
        throw std::runtime_error("the input database does not fit in the arena (raise --total-memory)");
    std::vector<char> image(bytes);
    auto*             db           = reinterpret_cast<Database*>(image.data());
    auto*             items        = reinterpret_cast<Item*>(db + 1);
    auto*             transactions = reinterpret_cast<Transaction*>(items + data_.items.size());

    Database*    device_db    = reinterpret_cast<Database*>(arena_.get());
    Item*        device_items = reinterpret_cast<Item*>(device_db + 1);
    Transaction* device_tx    = reinterpret_cast<Transaction*>(device_items + data_.items.size());

    for (size_t i = 0; i < data_.items.size(); ++i)
        items[i] = Item{data_.items[i] + 1, data_.utilities[i]}; // names are 1-based: 0 marks empty slots
    for (size_t t = 0; t < data_.transaction_count(); ++t)
        transactions[t] = Transaction{device_items + data_.tx_begin[t], data_.tx_begin[t + 1] - data_.tx_begin[t], 0};
    *db = Database{device_items, device_tx, static_cast<uint32_t>(data_.transaction_count()),
                   static_cast<uint32_t>(data_.items.size())};
    upload(arena_.get(), image.data(), bytes);
    root_end_              = (bytes + 7) & ~uint64_t{7};
    control_->arena_offset = root_end_;

    std::vector<Task> seeds;
    seeds.reserve(data_.primary.size());
    for (uint32_t name : data_.primary) {
        Task task;
        task.db           = device_db;
        task.unique_items = static_cast<uint32_t>(data_.original_item.size()) + 1;
        task.primary      = name + 1;
        seeds.push_back(task);
    }
    upload(current_->storage(), seeds.data(), seeds.size());
    current_->get()->top = static_cast<uint32_t>(seeds.size());
}

void Search::check_errors() const
{
    if (control_->error != 0)
        throw std::runtime_error(describe(control_->error));
}

// Wraps allocation to the free space below the live databases when less room is left above them
// than below. The live range is [live_begin, arena_offset).
void Search::reclaim_arena(uint64_t live_begin)
{
    const uint64_t live_end = control_->arena_offset;
    const uint64_t above    = arena_.size() - live_end;
    const uint64_t below    = live_begin - root_end_;
    if (above < below) {
        spdlog::debug("  arena wraps: {} MiB below the live range, {} MiB above", below >> 20, above >> 20);
        control_->arena_offset = root_end_;
        control_->arena_limit  = live_begin;
    }
    else
        control_->arena_limit = arena_.size();
    result_.peak_arena = std::max(result_.peak_arena, live_end - live_begin);
}

void Search::drain_patterns()
{
    if (writer_ == nullptr || control_->pattern_offset == 0)
        return;
    const uint64_t words = control_->pattern_offset;
    staging_.resize(words);
    download(staging_.data(), patterns_.get(), words);
    writer_->write_records(staging_.data(), words);
    control_->pattern_offset = 0;
}

void Search::finish_iteration(uint64_t live_begin, uint64_t tasks, double ms)
{
    const uint64_t live_bytes = control_->arena_offset - live_begin;
    reclaim_arena(live_begin);
    drain_patterns();
    const uint64_t found = control_->pattern_count - result_.patterns;
    result_.patterns     = control_->pattern_count;
    result_.tasks += tasks;
    result_.kernel_ms += ms;
    result_.iterations.push_back({tasks, found, live_bytes, ms});
    spdlog::debug("Iteration {:>3}: {:>10} tasks, {:>10} itemsets, {:.1f} ms", result_.iterations.size(), tasks, found,
                  ms);
}

void Search::run_static()
{
    while (current_->size() != 0) {
        const uint64_t live_begin = control_->arena_offset;
        const uint64_t tasks      = current_->size();
        const auto     start      = std::chrono::steady_clock::now();
        launch_block_tasks(
                {control_.get(), static_cast<uint32_t>(options_.minimum_utility), current_->get(), next_->get()},
                current_->size(), options_.block_size);
        synchronize("static mining kernel");
        check_errors();
        std::swap(current_, next_);
        next_->clear();
        finish_iteration(live_begin, tasks, ms_since(start));
    }
}

void Search::run_adaptive()
{
    auto& distribution = *distribution_;
    distribution[7]    = static_cast<uint32_t>(data_.primary.size()); // keeps the first launch non-empty
    threads_           = options_.block_size;
    const auto minimum = static_cast<uint32_t>(options_.minimum_utility);

    while (current_->size() != 0 || singles_->size() != 0) {
        const uint64_t live_begin = control_->arena_offset;
        const uint64_t tasks      = uint64_t{current_->size()} + singles_->size();
        const auto     start      = std::chrono::steady_clock::now();

        if (current_->size() != 0) {
            threads_ = adaptive_block_threads(distribution.data(), threads_);
            distribution.fill(0);
            launch_block_tasks(
                    {control_.get(), minimum, current_->get(), next_->get(), distribution.data(), singles_->get()},
                    current_->size(), threads_);
            synchronize("adaptive block kernel");
            check_errors();
            std::swap(current_, next_);
            next_->clear();
        }

        distribution[0] = singles_->size();
        if (singles_->size() != 0) {
            launch_single_tasks({control_.get(), minimum, singles_->get(), singles_next_->get()}, singles_->size());
            synchronize("adaptive single-thread kernel");
            check_errors();
        }
        std::swap(singles_, singles_next_);
        singles_next_->clear();

        finish_iteration(live_begin, tasks, ms_since(start));
    }
}

MiningResult Search::run()
{
    validate();
    if (data_.primary.empty())
        return result_;
    const auto start = std::chrono::steady_clock::now();
    allocate();
    upload_root_and_seeds();
    result_.setup_ms = ms_since(start);

    spdlog::info("Running {} depth-first mining", method_name(options_.method));
    if (adaptive_)
        run_adaptive();
    else
        run_static();
    result_.patterns = control_->pattern_count;
    return result_;
}

} // namespace

MiningResult mine_dfs(const Dataset& data, const MiningOptions& options)
{
    std::unique_ptr<ItemsetWriter> writer;
    if (!options.output.empty() && options.output != "/dev/null")
        writer = std::make_unique<ItemsetWriter>(options.output, data.original_item, 1);
    MiningOptions effective = options;
    if (writer == nullptr)
        effective.output.clear();
    return Search(data, effective, writer.get()).run();
}

} // namespace ghuim
