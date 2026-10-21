#include "mine/bfs/host/level_miner.hpp"

#include "mine/host/transfer.hpp"

#include <algorithm>
#include <chrono>
#include <spdlog/spdlog.h>
#include <string>

namespace ghuim::inline GHUIM_WIDTHS::bfs {

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
        if (options_.bin_times) {
            std::string line;
            for (uint32_t cell = 0; cell < kCells; ++cell)
                if (bin_tasks_[cell] != 0)
                    line += fmt::format(" [bin {}{}: {} tasks {:.2f} ms]", cell / kCellKinds, cell % kCellKinds == kTailKind ? " tail" : cell % kCellKinds == kVertKind ? " vertical" : "", bin_tasks_[cell],
                                        bin_ms_[cell]);
            spdlog::info("Level {:>3} bins:{}", level, line);
            bin_ms_.fill(0);
            bin_tasks_.fill(0);
        }
        std::swap(current_, next_);
        count_ = next;
    }
    spdlog::info("BFS: peak arena region use {} MiB, peak scratch {} MiB, peak level {} tasks ({} MiB of task arrays)",
                 peak_arena_ >> 20, peak_scratch_ >> 20, peak_tasks_, (peak_tasks_ * budget_.per_task) >> 20);
    if (host_.stage_consumer_cycles != 0)
        spdlog::info("Staging: compute warps waited {:.1f}% of their time for data, IO warps waited {:.1f} Mcycles for free slots",
                     100.0 * static_cast<double>(host_.stage_full_wait) / static_cast<double>(host_.stage_consumer_cycles),
                     static_cast<double>(host_.stage_empty_wait) / 1e6);
#ifdef GHUIM_WORK_STATS
    spdlog::info("Work: pass 1 scans {} transactions over all tasks, {} of them ({:.1f}%) contain the extension item",
                 host_.work_total, host_.work_present,
                 100.0 * static_cast<double>(host_.work_present) / static_cast<double>(std::max<uint64_t>(host_.work_total, 1)));
#endif
    result_.patterns   = host_.patterns;
    result_.peak_arena = peak_arena_;
    return result_;
}

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
