#pragma once
// The vector instruction set of the projection loop (mine/simd.hpp), chosen once at startup.

#include <string_view>

// "auto", "avx512" or "scalar". Returns false if the requested set is not supported by this CPU.
bool select_simd(std::string_view mode);
// "avx512" or "scalar".
const char* simd_name();
