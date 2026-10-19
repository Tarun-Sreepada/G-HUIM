#include "mine/bfs/host/level_miner.hpp"

#include "mine/host/transfer.hpp"

namespace ghuim::bfs {

// Level 1: one task per primary item of the root.
void LevelMiner::seed_first_level()
{
    std::vector<Task> seeds;
    seeds.reserve(data_.primary.size());
    for (uint32_t item : data_.primary)
        seeds.push_back(
                Task{0, static_cast<uint32_t>(root_bytes_ / kArenaUnit),
                     root_occ_counts_.empty() ? root_trans_ : root_occ_counts_[item], root_items_, item, kNone, root_max_len_,
                     root_entries_});
    upload(level_a_.get(), seeds.data(), seeds.size());
    current_ = level_a_.get();
    next_    = level_b_.get();
    count_   = static_cast<uint32_t>(seeds.size());
}

} // namespace ghuim::bfs
