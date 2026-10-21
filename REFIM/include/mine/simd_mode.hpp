#pragma once
// The vector instruction set of the projection loop (mine/simd.hpp), chosen once at startup.

#include <string_view>

// "auto", "avx512vbmi", "avx512", "avx2" or "scalar". Returns false if the requested set is not supported by this CPU.
bool select_simd(std::string_view mode);
// "avx512vbmi", "avx512", "avx2" or "scalar". avx512vbmi runs the avx512 kernel unless item ids are 8-bit.
const char* simd_name();
// The sets this CPU can run, best first, e.g. "avx512vbmi, avx512, avx2, scalar".
const char* simd_supported();
