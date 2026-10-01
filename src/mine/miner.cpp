#include "mine/miner.hpp"

#include "mine/bfs/bfs_miner.hpp"

namespace ghuim {

MiningResult mine(const Dataset& data, const MiningOptions& options)
{
    return mine_bfs(data, options);
}

} // namespace ghuim
