#pragma once
// How a method maps task size classes to kernels.

#include "app/config.hpp"
#include "mine/dfs/schedule.cuh"

namespace ghuim::dfs {

struct Bins
{
    enum Kind
    {
        Block,
        Optimized,
        Subwarp,
        Huge
    };
    BinRule rule;
    Kind    kind[kMaxBins];
};

// Measures the device (adaptive only) and returns the bins of `method`.
Bins bins_for(Method method);

} // namespace ghuim::dfs
