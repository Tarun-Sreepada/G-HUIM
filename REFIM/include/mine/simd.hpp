#pragma once
// Vector kernels for the projection loop, for the instruction set chosen at startup (mine/simd_mode.hpp).
// All give identical results.

#include "mine/level.hpp"

inline namespace REFIM_WIDTHS {

// Keeps the items x of a transaction suffix with mark[x] >= depth (and their utilities), packed
// into out_items/out_utils, and stores the sum of the kept utilities in *sum. Returns the kept
// count. The outputs need n + kSimdSlack slots (vector stores overrun). `keep` is the same test as a
// 256-byte table (0xFF = keep), filled by fill_keep_table() when needs_keep_table(); other kernels ignore it.
using FilterFn = std::uint32_t (*)(const Item* items, const Util* utils, std::uint32_t n, const int* mark,
								   int depth, const std::uint8_t* keep, Item* out_items, Util* out_utils, Util* sum);

constexpr std::uint32_t kSimdSlack = 64;

// The kernel of the selected instruction set.
FilterFn filter_kernel();
// Whether that kernel reads `keep` (avx512vbmi with 8-bit item ids).
bool needs_keep_table();

}  // namespace REFIM_WIDTHS
