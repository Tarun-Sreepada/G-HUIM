#pragma once
// The level-synchronous driver. mine_level() reads as: reset control, plan chunks, mine each chunk,
// read the level back. Each step lives in src/mine/bfs/host/level_miner/.

#include "app/config.hpp"
#include "cuda/buffer.cuh"
#include "data/dataset.hpp"
#include "mine/bfs/host/auto_tune.hpp"
#include "mine/bfs/host/budget.hpp"
#include "mine/bfs/host/streams.hpp"
#include "mine/bfs/kernels.cuh"
#include "mine/result.hpp"

#include <array>
#include <vector>

namespace ghuim::inline GHUIM_WIDTHS::bfs {

// Tasks per chunk. Thread groups are sized from a task's share of the chunk's work, so a chunk as
// large as a whole level would give nearly every task one warp and leave heavy tasks as stragglers.
constexpr uint32_t kMaxChunkTasks = 1u << 20;

class LevelMiner
{
  public:
    LevelMiner(const Dataset& data, const MiningOptions& options)
        : data_(data), options_(options), tune_(make_auto_tune(options, data)),
          budget_(plan_budget(options, tune_, data.primary.size())), collect_(!options.output.empty())
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
    DeviceBuffer<uint32_t>            vertical_;
    DeviceBuffer<Control>             control_;
    DeviceBuffer<ChunkPlan>           chunk_plan_;

    uint64_t                       root_bytes_ = 0;
    uint32_t                       root_trans_ = 0, root_items_ = 0, root_max_len_ = 0, root_entries_ = 0;
    std::vector<uint32_t>           root_occ_counts_;
    Task*                          current_ = nullptr;
    Task*                          next_    = nullptr;
    uint32_t                       count_   = 0;
    Control                        host_{};
    std::vector<uint64_t>          chunk_bases_;
    std::vector<std::vector<Node>> levels_;
    uint64_t                       peak_arena_ = 0, peak_scratch_ = 0, peak_tasks_ = 0;
    std::array<double, kMaxCells>  bin_ms_{};    // --bin-times: time and tasks of each cell in the current level
    std::array<uint64_t, kMaxCells> bin_tasks_{};
    MiningResult                   result_;
};

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
