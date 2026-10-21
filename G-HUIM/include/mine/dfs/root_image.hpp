#pragma once

#include "data/dataset.hpp"
#include "mine/widths.hpp"

#include <vector>

namespace ghuim::inline GHUIM_WIDTHS::dfs {

// The whole input as one projected database in arena format (item names are the identity).
std::vector<char> root_image(const Dataset& data);

} // namespace ghuim::inline GHUIM_WIDTHS::dfs
