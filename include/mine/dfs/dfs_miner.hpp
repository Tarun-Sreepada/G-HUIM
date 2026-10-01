#pragma once

#include "app/config.hpp"
#include "data/dataset.hpp"
#include "mine/result.hpp"

namespace ghuim {

// Depth-first search driven by task stacks: one iteration processes every task of the current stack
// and pushes the children onto the next. method is Static or Adaptive.
MiningResult mine_dfs(const Dataset& data, const MiningOptions& options);

} // namespace ghuim
