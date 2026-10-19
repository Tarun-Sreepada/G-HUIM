#pragma once

#include "data/dataset.hpp"

#include <vector>

namespace ghuim::dfs {

// The whole input as one projected database in arena format (item names are the identity).
std::vector<char> root_image(const Dataset& data);

} // namespace ghuim::dfs
