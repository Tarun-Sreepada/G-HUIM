#include "mine/miner.hpp"

#include "mine/bfs/bfs_miner.hpp"
#include "mine/dfs/dfs_miner.hpp"

namespace ghuim {

MiningResult mine(const Dataset& data, const MiningOptions& options)
{
    switch (options.method) {
    case Method::BfsOptimized:
        return mine_bfs(data, options);
    case Method::Static:
    case Method::Adaptive:
        return mine_dfs(data, options);
    }
    return {};
}

} // namespace ghuim
