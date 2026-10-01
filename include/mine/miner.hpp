#pragma once

#include "app/config.hpp"
#include "data/dataset.hpp"
#include "mine/result.hpp"

namespace ghuim {

// Runs the search selected by options.method on the GPU.
MiningResult mine(const Dataset& data, const MiningOptions& options);

} // namespace ghuim
