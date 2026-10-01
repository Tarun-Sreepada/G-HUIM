#pragma once

#include "app/config.hpp"
#include "data/dataset.hpp"
#include "mine/result.hpp"

namespace ghuim {

// Level-synchronous search: every task of a level runs before any task of the next. Level L's tasks
// are binned by size and run with a thread-group size chosen from the level's total work.
MiningResult mine_bfs(const Dataset& data, const MiningOptions& options);

} // namespace ghuim
