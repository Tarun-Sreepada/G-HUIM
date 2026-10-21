#pragma once
// Vector kernels for the projection loop, for the instruction set chosen at startup (mine/simd_mode.hpp).
// Both give identical results.

#include "mine/level.hpp"

inline namespace REFIM_WIDTHS {

// Keeps the items x of a transaction suffix with mark[x] >= depth (and their utilities), packed
// into out_items/out_utils, and stores the sum of the kept utilities in *sum. Returns the kept
// count. The outputs need n + 16 slots (vector stores overrun).
using FilterFn = std::uint32_t (*)(const Item* items, const Util* utils, std::uint32_t n, const int* mark,
								   int depth, Item* out_items, Util* out_utils, Util* sum);

constexpr std::uint32_t kSimdSlack = 16;

// The kernel of the selected instruction set.
FilterFn filter_kernel();

}  // namespace REFIM_WIDTHS
