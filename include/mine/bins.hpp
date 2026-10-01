#pragma once
// How a method maps task size classes to kernels.

#include "mine/miner.hpp"
#include "mine/schedule.cuh"

namespace ghuim {

struct Bins
{
    enum Kind
    {
        Thread,
        Warp,
        Block,
        Optimized,
        Subwarp,
        Huge
    };
    BinRule rule;
    Kind    kind[kMaxBins];
};

// Measures the device (adaptive-optimized only) and returns the bins of `method`.
Bins bins_for(Method method);

} // namespace ghuim
