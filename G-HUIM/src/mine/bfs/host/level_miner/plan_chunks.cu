#include "mine/bfs/host/level_miner.hpp"

#include "cuda/nvtx.hpp"
#include "mine/primitives.cuh"

#include <algorithm>
#include <stdexcept>

namespace ghuim::bfs {

// Scratch offsets over the whole level, split into chunks that fit the scratch buffer (`cuts` holds
// the chunk boundaries).
void LevelMiner::plan_chunks(uint32_t count, std::vector<uint32_t>& cuts)
{
    GHUIM_RANGE("bfs: plan chunks (scratch sizes + scan + sync)");
    launch_scratch_sizes(current_, count, prefix_.get(), tune_.tiled != 0, streams_[0]);
    exclusive_scan(prefix_.get(), count + 1, scan_sums_.get(), scan_total_.get(), streams_[0]);
    uint64_t level_scratch = 0;
    check_cuda(cudaMemcpyAsync(&level_scratch, prefix_.get() + count, sizeof(uint64_t), cudaMemcpyDeviceToHost,
                               streams_[0]),
               "read scratch total");
    check_cuda(cudaStreamSynchronize(streams_[0]), "scratch sizes");

    peak_scratch_ = std::max<uint64_t>(peak_scratch_, std::min<uint64_t>(level_scratch, budget_.scratch_capacity));
    peak_tasks_   = std::max<uint64_t>(peak_tasks_, count);
    cuts          = {0};
    chunk_bases_.clear();
    if (level_scratch <= budget_.scratch_capacity && count <= kMaxChunkTasks) {
        cuts.push_back(count);
        chunk_bases_.push_back(0);
        return;
    }
    ChunkPlan page{};
    while (cuts.back() < count) {
        launch_plan_chunks(prefix_.get(), cuts.back(), count, kMaxChunkTasks, budget_.scratch_capacity,
                           chunk_plan_.get(), streams_[0]);
        check_cuda(cudaMemcpyAsync(&page, chunk_plan_.get(), sizeof(page), cudaMemcpyDeviceToHost, streams_[0]),
                   "read chunk boundaries");
        check_cuda(cudaStreamSynchronize(streams_[0]), "chunk boundaries");
        if (page.oversized)
            throw std::runtime_error("one task needs more scratch than available; raise --total-memory");
        cuts.insert(cuts.end(), page.cuts + 1, page.cuts + page.count + 1);
        chunk_bases_.insert(chunk_bases_.end(), page.bases, page.bases + page.count);
    }
}

} // namespace ghuim::bfs
