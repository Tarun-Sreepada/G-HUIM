#pragma once
// Vector kernels for the projection loop, chosen once at startup (AVX-512 if the CPU has it,
// scalar otherwise). Both give identical results.

#include "mine/level.hpp"

#include <string_view>

// Keeps the items x of a transaction suffix with mark[x] >= depth (and their utilities), packed
// into out_items/out_utils, and stores the sum of the kept utilities in *sum. Returns the kept
// count. The outputs need n + 16 slots (vector stores overrun).
using FilterFn = std::uint32_t (*)(const Item* items, const Util* utils, std::uint32_t n, const int* mark,
								   int depth, Item* out_items, Util* out_utils, Utility* sum);

constexpr std::uint32_t kSimdSlack = 16;

// "auto", "avx512" or "scalar". Returns false if the requested set is not supported by this CPU.
bool select_simd(std::string_view mode);
FilterFn filter_kernel();
const char* simd_name();
