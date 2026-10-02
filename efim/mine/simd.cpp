#include "mine/simd.hpp"

#include <immintrin.h>

namespace {

std::uint32_t filter_scalar(const Item* items, const Util* utils, std::uint32_t n, const int* mark, int depth,
							Item* out_items, Util* out_utils, Utility* sum) {
	std::uint32_t kept = 0;
	Utility s = 0;
	for (std::uint32_t i = 0; i < n; ++i) {
		if (mark[items[i]] < depth) continue;
		out_items[kept] = items[i];
		out_utils[kept] = utils[i];
		s += utils[i];
		++kept;
	}
	*sum = s;
	return kept;
}

// 32 items per step: widen the 16-bit items to 32-bit lanes, gather mark[], compare, and pack
// the survivors with vpcompress (to registers; compress-to-memory is microcoded on Zen 4).
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,popcnt"))) std::uint32_t
filter_avx512(const Item* items, const Util* utils, std::uint32_t n, const int* mark, int depth, Item* out_items,
			  Util* out_utils, Utility* sum) {
	const __m512i vdepth = _mm512_set1_epi32(depth);
	__m512i acc = _mm512_setzero_si512();
	std::uint32_t kept = 0;
	for (std::uint32_t i = 0; i < n; i += 16) {
		const std::uint32_t rem = n - i;
		const __mmask16 valid = rem >= 16 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>((1u << rem) - 1);
		const __m512i x = _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(valid, items + i));
		const __m512i m = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), valid, x, mark, 4);
		const __mmask16 k = _mm512_mask_cmpge_epi32_mask(valid, m, vdepth);
		const __m512i u = _mm512_maskz_compress_epi32(k, _mm512_maskz_loadu_epi32(valid, utils + i));
		acc = _mm512_add_epi64(acc, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(u)));
		acc = _mm512_add_epi64(acc, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(u, 1)));
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(out_items + kept), _mm512_cvtepi32_epi16(_mm512_maskz_compress_epi32(k, x)));
		_mm512_storeu_si512(out_utils + kept, u);
		kept += static_cast<std::uint32_t>(_mm_popcnt_u32(k));
	}
	*sum = _mm512_reduce_add_epi64(acc);
	return kept;
}

bool cpu_has_avx512() {
	__builtin_cpu_init();
	return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
		   __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq");
}

FilterFn g_filter = filter_scalar;
const char* g_name = "scalar";

}  // namespace

bool select_simd(std::string_view mode) {
	if (mode == "scalar") {
		g_filter = filter_scalar;
		g_name = "scalar";
		return true;
	}
	if (mode == "auto") {  // measured: vpgatherdd is slower than the scalar loop on Zen 4
		g_filter = filter_scalar;
		g_name = "scalar";
		return true;
	}
	if (mode == "avx512") {
		if (cpu_has_avx512()) {
			g_filter = filter_avx512;
			g_name = "avx512";
			return true;
		}
		return false;
	}
	return false;
}

FilterFn filter_kernel() { return g_filter; }
const char* simd_name() { return g_name; }
