#pragma once
// Device profile of the current GPU, plus the options that shape the kernels.

#include "app/config.hpp"
#include "data/dataset.hpp"
#include "mine/bfs/placement.cuh"

namespace ghuim::bfs {

AutoTune make_auto_tune(const MiningOptions& options, const Dataset& data);

} // namespace ghuim::bfs
