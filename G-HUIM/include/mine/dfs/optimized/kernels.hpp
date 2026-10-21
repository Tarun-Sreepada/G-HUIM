#pragma once
// Handles to the adaptive-optimized kernels, for occupancy queries from another translation unit.

#include "mine/widths.hpp"

#include <cstdint>

namespace ghuim::inline GHUIM_WIDTHS::dfs::opt {

constexpr uint32_t kHugeKernels = 8;

const void* block_kernel();
const void* warp_kernel();
const void* subwarp_kernel();
// The phase kernels of a huge task, in launch order.
void huge_kernels(const void* (&kernels)[kHugeKernels]);

} // namespace ghuim::inline GHUIM_WIDTHS::dfs::opt
