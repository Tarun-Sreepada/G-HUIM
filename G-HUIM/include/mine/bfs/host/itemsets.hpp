#pragma once
// Writes the itemsets of a finished BFS run.

#include "mine/bfs/layout.cuh"
#include "mine/output.hpp"

#include <cstdint>
#include <vector>

namespace ghuim::inline GHUIM_WIDTHS::bfs {

void write_itemsets(const std::vector<std::vector<Node>>& levels, uint64_t minimum_utility, ItemsetWriter& writer);

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
