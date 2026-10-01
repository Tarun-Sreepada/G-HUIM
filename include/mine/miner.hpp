#pragma once

#include "app/config.hpp"
#include "data/dataset.hpp"

#include <array>
#include <cstdint>

namespace ghuim {

struct MiningResult
{
    uint64_t patterns    = 0; // high-utility itemsets found
    uint64_t batches     = 0;
    uint64_t tasks       = 0;
    uint64_t peak_arena  = 0;
    uint64_t compactions = 0;
    uint64_t deferrals   = 0;
    double   kernel_ms   = 0;
    double   root_build_ms  = 0; // laying out the input database for the GPU
    double   root_upload_ms = 0; // copying it and the first-level tasks to the device

    std::array<uint64_t, 64> depth_tasks{};    // tasks evaluated, by itemset length (last slot: 63+)
    std::array<uint64_t, 64> depth_patterns{}; // high-utility itemsets, by itemset length
};

MiningResult mine(const Dataset& data, const MiningOptions& options);

} // namespace ghuim
