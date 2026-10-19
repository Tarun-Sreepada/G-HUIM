#include "mine/bfs/host/level_miner.hpp"

#include "cuda/nvtx.hpp"
#include "mine/host/transfer.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ghuim::bfs {

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
        const uint64_t base = chunk_bases_[c];
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

} // namespace ghuim::bfs
