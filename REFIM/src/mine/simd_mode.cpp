#include "mine/simd_mode.hpp"

namespace {

bool cpu_has_avx512() {
	__builtin_cpu_init();
	return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
		   __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq");
}

const char* g_name = "scalar";

}  // namespace

bool select_simd(std::string_view mode) {
	if (mode == "scalar" || mode == "auto") {  // auto, measured: vpgatherdd is slower than the scalar loop on Zen 4
		g_name = "scalar";
		return true;
	}
	if (mode == "avx512" && cpu_has_avx512()) {
		g_name = "avx512";
		return true;
	}
	return false;
}

const char* simd_name() { return g_name; }
