#pragma once

#include "app/config.hpp"
#include "data/dataset.hpp"
#include "mine/result.hpp"

namespace ghuim {

// Creates the CUDA context of device 0 (about 0.3 s; the program runs it while the dataset loads).
// Driver initialization costs about 0.2-0.3 s per visible GPU and only device 0 is used, so when
// CUDA_VISIBLE_DEVICES is unset this sets it to "0" first. Set it yourself to run on another GPU.
void initialize_device();

// Runs the search selected by options.search with the work partitioning of options.method.
MiningResult mine(const Dataset& data, const MiningOptions& options);

} // namespace ghuim
