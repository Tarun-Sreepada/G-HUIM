// Host driver for the GPU EFIM search.
//
// All tasks live on a device stack. Each iteration a selection kernel pops a batch from the
// top (so the search is depth-first and few projected databases are live), a binning kernel
// groups the batch by size, and one kernel per bin processes it. Children and retried tasks
// are pushed straight back onto the stack; the host only reads a small Control block.
//
// Admission is optimistic: a child database is usually far smaller than its parent, so
// batches reserve less than the worst case. A task whose child does not fit is deferred
// (pushed back), and the next batch uses the worst-case bound, so the search always advances.
//
// Arena reclamation: databases are bump-allocated. When the arena fills up, the databases
// still referenced from the stack are found with a bitmap, slid down to the start of the
// arena in address order, and the stack is re-pointed (databases use internal offsets only).

#include "cuda/buffer.cuh"
#include "cuda/nvtx.hpp"
#include "mine/dfs/arena_gc.hpp"
#include "mine/dfs/bins.hpp"
#include "mine/host/transfer.hpp"
#include "mine/dfs/kernels.cuh"
#include "mine/dfs/dfs_miner.hpp"
#include "mine/primitives.cuh"
#include "mine/dfs/root_image.hpp"
#include "mine/dfs/schedule.cuh"

#include <algorithm>
#include <array>
#include <chrono>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

namespace ghuim::inline GHUIM_WIDTHS::dfs {
namespace {

constexpr uint32_t kMaxBatchTasks = 1u << 20;
constexpr uint32_t kStackCapacity = 1u << 25;
constexpr uint64_t kScratchBytes  = 1ull << 30;
constexpr uint64_t kReservedBytes = 384ull << 20; // runtime, kernels, and free-memory slack
constexpr uint32_t kOptimism      = 8; // optimistic batches assume children are 1/8 of the parent
constexpr uint32_t kHeadroom      = 2; // and leave half the free arena for later batches

class Miner
{
  public:
    Miner(const Dataset& data, const MiningOptions& options)
        : data_(data), options_(options), bins_(bins_for(options.method))
    {
        size_t free_bytes = 0, total_bytes = 0;
        check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "query device memory");
        const uint64_t per_batch = sizeof(Task) + sizeof(uint64_t) + sizeof(uint32_t);
        const uint64_t fixed     = kScratchBytes + uint64_t{kMaxBatchTasks} * per_batch +
                               uint64_t{kStackCapacity} * sizeof(Task);
        uint64_t arena = options.memory_bytes;
        if (arena == 0) {
            if (free_bytes < fixed + kReservedBytes)
                throw std::runtime_error("not enough free device memory");
            // The compaction bitmap and its counts take 8 bytes per 32 arena units (1/64).
            arena = (free_bytes - fixed - kReservedBytes) * 64 / 65;
        }
        arena  = std::min<uint64_t>(arena / kArenaUnit * kArenaUnit, uint64_t{0xffffffffu} * kArenaUnit);
        words_ = static_cast<uint32_t>((arena / kArenaUnit + 31) / 32);
        spdlog::info("Device memory: {} MiB free; arena {} MiB, scratch {} MiB, task stack {} MiB", free_bytes >> 20,
                     arena >> 20, kScratchBytes >> 20, (uint64_t{kStackCapacity} * sizeof(Task)) >> 20);

        arena_      = DeviceBuffer<char>(arena);
        scratch_    = DeviceBuffer<char>(kScratchBytes);
        stack_      = DeviceBuffer<Task>(kStackCapacity);
        batch_      = DeviceBuffer<Task>(kMaxBatchTasks);
        offsets_    = DeviceBuffer<uint64_t>(kMaxBatchTasks);
        order_      = DeviceBuffer<uint32_t>(kMaxBatchTasks);
        control_    = DeviceBuffer<Control>(1);
        bitmap_     = DeviceBuffer<uint32_t>(words_);
        counts_     = DeviceBuffer<uint32_t>(words_);
        block_sums_ = DeviceBuffer<uint32_t>(words_ / 32 + 2);
        scan_total_ = DeviceBuffer<uint32_t>(1);
        for (auto& stream : streams_)
            check_cuda(cudaStreamCreate(&stream), "create stream"); // orders with the plain cudaMemcpy calls
        for (auto& pair : bin_events_)
            for (auto& event : pair)
                check_cuda(cudaEventCreate(&event), "create event");
    }

    ~Miner()
    {
        for (auto stream : streams_)
            cudaStreamDestroy(stream);
        for (auto& pair : bin_events_)
            for (auto event : pair)
                cudaEventDestroy(event);
    }

    MiningResult run()
    {
        const auto              build_start = std::chrono::steady_clock::now();
        std::vector<char>       root;
        {
            GHUIM_RANGE("dfs: build root database");
            root = root_image(data_);
        }
        result_.setup_ms                    = ms_since(build_start);
        const auto              upload_start = std::chrono::steady_clock::now();
        if (root.size() > arena_.size())
            throw std::runtime_error("the input database does not fit in the arena; increase --memory");
        {
            GHUIM_RANGE("dfs: upload root database");
            upload(arena_.get(), root.data(), root.size());
        }

        std::vector<Task> tasks;
        const auto        n_trans = static_cast<uint32_t>(data_.transaction_count());
        const auto        n_items = static_cast<uint32_t>(data_.original_item.size());
        for (uint32_t item : data_.primary)
            tasks.push_back(Task{0, static_cast<uint32_t>(root.size() / kArenaUnit), n_trans, n_items, item, 1});
        upload(stack_.get(), tasks.data(), tasks.size());
        result_.setup_ms += ms_since(upload_start);
        control_host_           = Control{};
        control_host_.arena_top = root.size();
        control_host_.stack_top = static_cast<uint32_t>(tasks.size());
        upload(control_.get(), &control_host_, 1);
        gc_threshold_ = root.size() + (arena_.size() - root.size()) / 2;

        while (control_host_.stack_top != 0)
            run_batch();
        result_.patterns = control_host_.patterns;
        for (uint32_t d = 0; d < kMaxDepth; ++d) {
            result_.depth_tasks[d]    = control_host_.depth_tasks[d];
            result_.depth_patterns[d] = control_host_.depth_patterns[d];
        }
        spdlog::debug("Host-side scheduling: selection {:.0f} ms, compaction {:.0f} ms", select_ms_, gc_ms_);
        for (uint32_t b = 0; b < bins_.rule.bins; ++b)
            if (bin_tasks_[b] != 0)
                spdlog::debug("Bin {:>2}: {:>10} tasks, {:8.1f} ms (stream time, bins overlap)", b, bin_tasks_[b],
                              bin_ms_[b]);
        return result_;
    }

  private:
    void run_batch()
    {
        GHUIM_RANGE("dfs: batch");
        if (control_host_.arena_top > gc_threshold_ || control_host_.forced)
            collect_garbage();

        const uint64_t patterns_before = control_host_.patterns;
        const uint64_t arena_top_before = control_host_.arena_top;
        auto           lap      = std::chrono::steady_clock::now();
        cudaStream_t   stream   = streams_[0];
        const bool     cautious = control_host_.deferred != 0;
        const uint32_t optimism = cautious ? 1 : kOptimism, headroom = cautious ? 1 : kHeadroom;
        PhaseRanges phase;
        phase.next("dfs: select + bin batch");
        select_batch(SelectArgs{control_.get(), stack_.get(), batch_.get(), offsets_.get(), arena_.size(),
                                kScratchBytes, kStackCapacity, kMaxBatchTasks, optimism, headroom},
                     stream);
        bin_batch(control_.get(), batch_.get(), order_.get(), bins_.rule, kMaxBatchTasks, stream);
        check_cuda(cudaStreamSynchronize(stream), "select batch");
        download(&control_host_, control_.get(), 1);
        const double select_ms = ms_since(lap);
        select_ms_ += select_ms;
        const uint32_t count = control_host_.batch_count;

        lap = std::chrono::steady_clock::now();
        phase.next("dfs: launch bins + sync");
        const BatchView view{arena_.get(),   arena_.size(), control_.get(), batch_.get(),
                             offsets_.get(), scratch_.get(), stack_.get(),  kStackCapacity,
                             options_.minimum_utility, options_.refine_rounds};
        uint32_t        begin = 0;
        for (uint32_t b = 0; b < bins_.rule.bins; ++b) {
            const uint32_t  size  = control_host_.bin_count[b];
            const uint32_t* order = order_.get() + begin;
            cudaStream_t    s     = streams_[b % streams_.size()];
            if (size != 0)
                check_cuda(cudaEventRecord(bin_events_[b][0], s), "record event");
            switch (bins_.kind[b]) {
            case Bins::Block: launch_block_tasks(view, order, size, options_.block_size, s); break;
            case Bins::Optimized:
                launch_optimized_tasks(view, order, size, b, bins_.rule.tune, s);
                break;
            case Bins::Subwarp: launch_subwarp_tasks(view, order, size, 1u << (b - kOptSubBin), bins_.rule.tune, s); break;
            case Bins::Huge: launch_huge_tasks(view, order, size, control_host_.bin_max_slices[b], bins_.rule.tune, s); break;
            }
            if (size != 0)
                check_cuda(cudaEventRecord(bin_events_[b][1], s), "record event");
            begin += size;
        }
        check_cuda(cudaDeviceSynchronize(), "mine batch");
        for (uint32_t b = 0; b < bins_.rule.bins; ++b) {
            if (control_host_.bin_count[b] == 0)
                continue;
            float ms = 0;
            check_cuda(cudaEventElapsedTime(&ms, bin_events_[b][0], bin_events_[b][1]), "time bin");
            bin_ms_[b] += ms;
            bin_tasks_[b] += control_host_.bin_count[b];
        }
        const double kernel_ms = ms_since(lap);
        result_.kernel_ms += kernel_ms;

        phase.next("dfs: batch readback");
        download(&control_host_, control_.get(), 1);
        if (control_host_.error & kStackExhausted)
            throw std::runtime_error("task stack exhausted");
        const uint32_t deferred = control_host_.deferred;
        if (deferred == count && ++stalled_ == 2)
            throw std::runtime_error("projected databases do not fit in device memory; increase --memory");
        if (deferred != count)
            stalled_ = 0;
        if (control_host_.arena_top > arena_.size()) {
            // Failed allocations overshoot the bump pointer; pull it back.
            control_host_.arena_top = arena_.size();
            upload(&control_.get()->arena_top, &control_host_.arena_top, 1);
        }

        result_.iterations.push_back({count - deferred, control_host_.patterns - patterns_before,
                                      control_host_.arena_top > arena_top_before ? control_host_.arena_top - arena_top_before : 0,
                                      kernel_ms});
        result_.tasks += count - deferred;
        result_.deferrals += deferred;
        result_.peak_arena = std::max<uint64_t>(result_.peak_arena, control_host_.arena_top);

        if (spdlog::should_log(spdlog::level::debug)) {
            std::string bins;
            for (uint32_t b = 0; b < bins_.rule.bins; ++b)
                if (control_host_.bin_count[b] != 0)
                    bins += fmt::format(" {}:{}", b, control_host_.bin_count[b]);
            spdlog::debug("Batch {:>4}: {:>8} tasks [bins{}], deferred {}, +{} patterns, select {:.2f} ms, "
                          "kernels {:.2f} ms, arena {} MiB, stack {}",
                          result_.iterations.size(), count, bins, deferred, control_host_.patterns - patterns_before,
                          select_ms, kernel_ms, control_host_.arena_top >> 20, control_host_.stack_top);
        }
    }

    // Slides every database still referenced by the stack to the bottom of the arena.
    void collect_garbage()
    {
        GHUIM_RANGE("dfs: arena compaction");
        const auto       start = std::chrono::steady_clock::now();
        const Compaction done  = compact_arena({arena_, scratch_, stack_, control_, bitmap_, counts_, block_sums_,
                                                scan_total_, words_},
                                               control_host_.stack_top, streams_[0]);
        spdlog::debug("Arena compaction: {} live databases, {} MiB", done.live_databases, done.arena_top >> 20);
        ++result_.compactions;
        control_host_.arena_top = done.arena_top;
        control_host_.forced    = 0;
        gc_threshold_           = control_host_.arena_top + (arena_.size() - control_host_.arena_top) / 2;
        gc_ms_ += ms_since(start);
    }

    const Dataset&       data_;
    const MiningOptions& options_;
    Bins                 bins_;

    DeviceBuffer<char>          arena_, scratch_;
    DeviceBuffer<Task>          stack_, batch_;
    DeviceBuffer<uint64_t>      offsets_;
    DeviceBuffer<uint32_t>      order_, bitmap_, counts_, block_sums_, scan_total_;
    DeviceBuffer<Control>       control_;
    std::array<cudaStream_t, 4> streams_{};
    uint32_t                    words_ = 0;
    std::array<std::array<cudaEvent_t, 2>, kMaxBins> bin_events_{};
    std::array<double, kMaxBins>   bin_ms_{};
    std::array<uint64_t, kMaxBins> bin_tasks_{};

    Control      control_host_{};
    uint64_t     gc_threshold_ = 0;
    int          stalled_      = 0;
    double       select_ms_ = 0, gc_ms_ = 0;
    MiningResult result_;
};

} // namespace

} // namespace ghuim::inline GHUIM_WIDTHS::dfs

namespace ghuim::inline GHUIM_WIDTHS {

MiningResult mine_dfs(const Dataset& data, const MiningOptions& options)
{
    return dfs::Miner(data, options).run();
}

} // namespace ghuim::inline GHUIM_WIDTHS
