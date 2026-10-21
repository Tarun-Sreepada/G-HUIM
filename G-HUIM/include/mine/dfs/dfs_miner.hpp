#pragma once

#include "app/config.hpp"
#include "data/dataset.hpp"
#include "mine/result.hpp"
#include "mine/widths.hpp"

namespace ghuim::inline GHUIM_WIDTHS {

// Depth-first search: tasks live on a device stack and are processed in batches popped from its top.
MiningResult mine_dfs(const Dataset& data, const MiningOptions& options);

} // namespace ghuim::inline GHUIM_WIDTHS
