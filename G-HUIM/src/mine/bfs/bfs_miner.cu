// Host driver of the level-synchronous (BFS) miner.
//
// Each level is mined in chunks: the chunk's tasks are binned by size, each bin runs on its own
// stream with a thread-group size chosen from the chunk's total work, and children are written into
// the arena region the previous level is not using. Per-task working memory comes from a scratch
// buffer that is reused every chunk, so the arenas only ever hold projected databases.

#include "cuda/nvtx.hpp"
#include "mine/bfs/bfs_miner.hpp"
#include "mine/bfs/host/itemsets.hpp"
#include "mine/bfs/host/level_miner.hpp"
#include "mine/host/transfer.hpp"

#include <chrono>
#include <spdlog/spdlog.h>

namespace ghuim {

MiningResult mine_bfs(const Dataset& data, const MiningOptions& options)
{
    using namespace bfs;
    if (data.primary.empty())
        return {};
    MiningOptions effective = options;
    if (options.output == "/dev/null")
        effective.output.clear();
    const auto   start = std::chrono::steady_clock::now();
    LevelMiner   miner(data, effective);
    spdlog::debug("  {:<28} {:9.1f} ms", "BFS setup (tune + buffers)", ms_since(start));
    MiningResult result = miner.run();
    if (!effective.output.empty()) {
        GHUIM_RANGE("bfs: write itemsets");
        ItemsetWriter writer(effective.output, data.original_item, 0);
        write_itemsets(miner.levels(), effective.minimum_utility, writer);
    }
    return result;
}

} // namespace ghuim
