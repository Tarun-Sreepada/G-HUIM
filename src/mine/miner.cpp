#include "mine/miner.hpp"

#include "mine/bfs/bfs_miner.hpp"
#include "mine/dfs/dfs_miner.hpp"

namespace ghuim {

MiningResult mine(const Dataset& data, const MiningOptions& options)
{
    return options.search == Search::Bfs ? mine_bfs(data, options) : mine_dfs(data, options);
}

} // namespace ghuim
