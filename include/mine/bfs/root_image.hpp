#pragma once

#include "data/dataset.hpp"

#include <vector>

namespace ghuim::bfs {

// The whole input as one projected database in arena format (item names are the identity).
std::vector<char> root_image(const Dataset& data, bool tiled = false, uint32_t index_min_trans = 0,
                             uint32_t index_sparsity = 0);

} // namespace ghuim::bfs
