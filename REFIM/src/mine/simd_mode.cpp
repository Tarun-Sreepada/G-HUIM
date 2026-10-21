#include "mine/simd_mode.hpp"

namespace {

bool cpu_has_avx512() {
	__builtin_cpu_init();
	return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
		   __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq");
}

bool cpu_has_avx512vbmi() {
	return cpu_has_avx512() && __builtin_cpu_supports("avx512vbmi") && __builtin_cpu_supports("avx512vbmi2");
}

bool cpu_has_avx2() {
	__builtin_cpu_init();
	return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("popcnt");
}

const char* g_name = "scalar";

}  // namespace

bool select_simd(std::string_view mode) {
	// auto, measured on Zen 4 (EPYC 9224, 48 threads): the scalar loop, which filters and updates the bounds in one
	// pass, beats both gather kernels (avx2 within 2% on chess/connect, 5-8% slower on accidents/pumsb; avx512 slower still).
	if (mode == "scalar" || mode == "auto") {
		g_name = "scalar";
		return true;
	}
	if (mode == "avx512vbmi" && cpu_has_avx512vbmi()) {
		g_name = "avx512vbmi";
		return true;
	}
	if (mode == "avx512" && cpu_has_avx512()) {
		g_name = "avx512";
		return true;
	}
	if (mode == "avx2" && cpu_has_avx2()) {
		g_name = "avx2";
		return true;
	}
	return false;
}

const char* simd_name() { return g_name; }

const char* simd_supported() {
	return cpu_has_avx512vbmi() ? "avx512vbmi, avx512, avx2, scalar" : cpu_has_avx512() ? "avx512, avx2, scalar" : cpu_has_avx2() ? "avx2, scalar" : "scalar";
}
