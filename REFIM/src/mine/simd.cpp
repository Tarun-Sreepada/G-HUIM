#include "mine/simd.hpp"

#include "mine/simd_mode.hpp"

#include <immintrin.h>
#include <string_view>

inline namespace REFIM_WIDTHS {

namespace {

std::uint32_t filter_scalar(const Item* items, const Util* utils, std::uint32_t n, const int* mark, int depth,
							Item* out_items, Util* out_utils, Util* sum) {
	std::uint32_t kept = 0;
	Util s = 0;
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

// 16 lanes of 8-, 16- or 32-bit values, widened to 32 bits (masked: lanes past `valid` read nothing).
template <class T>
__attribute__((target("avx512f,avx512bw,avx512vl"))) __m512i load_lanes(__mmask16 valid, const T* p) {
	if constexpr (sizeof(T) == 1) return _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(valid, p));
	else if constexpr (sizeof(T) == 2) return _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(valid, p));
	else return _mm512_maskz_loadu_epi32(valid, p);
}

// Stores 16 lanes narrowed back to T.
template <class T>
__attribute__((target("avx512f,avx512bw,avx512vl"))) void store_lanes(T* p, __m512i v) {
	if constexpr (sizeof(T) == 1) _mm_storeu_si128(reinterpret_cast<__m128i*>(p), _mm512_cvtepi32_epi8(v));
	else if constexpr (sizeof(T) == 2) _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), _mm512_cvtepi32_epi16(v));
	else _mm512_storeu_si512(p, v);
}

// 16 items per step: gather mark[] for the items (widened to 32 bits), compare, and pack the survivors
// with vpcompress (to registers; compress-to-memory is microcoded on Zen 4). 64-bit utilities are packed
// in two halves of 8; narrower ones are widened to 32-bit lanes, whose sums wrap like Util and so give the
// same (fitting) total.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,popcnt"))) std::uint32_t
filter_avx512(const Item* items, const Util* utils, std::uint32_t n, const int* mark, int depth, Item* out_items,
			  Util* out_utils, Util* sum) {
	const __m512i vdepth = _mm512_set1_epi32(depth);
	__m512i acc = _mm512_setzero_si512();
	std::uint32_t kept = 0;
	for (std::uint32_t i = 0; i < n; i += 16) {
		const std::uint32_t rem = n - i;
		const __mmask16 valid = rem >= 16 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>((1u << rem) - 1);
		const __m512i x = load_lanes(valid, items + i);
		const __m512i m = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), valid, x, mark, 4);
		const __mmask16 k = _mm512_mask_cmpge_epi32_mask(valid, m, vdepth);
		store_lanes(out_items + kept, _mm512_maskz_compress_epi32(k, x));
		if constexpr (sizeof(Util) == 8) {
			const __mmask8 lo = static_cast<__mmask8>(k), hi = static_cast<__mmask8>(k >> 8);
			const __m512i ulo =
				_mm512_maskz_compress_epi64(lo, _mm512_maskz_loadu_epi64(static_cast<__mmask8>(valid), utils + i));
			const __m512i uhi =
				_mm512_maskz_compress_epi64(hi, _mm512_maskz_loadu_epi64(static_cast<__mmask8>(valid >> 8), utils + i + 8));
			acc = _mm512_add_epi64(acc, _mm512_add_epi64(ulo, uhi));
			const std::uint32_t nlo = static_cast<std::uint32_t>(_mm_popcnt_u32(lo));
			_mm512_storeu_si512(out_utils + kept, ulo);
			_mm512_storeu_si512(out_utils + kept + nlo, uhi);
		} else {
			const __m512i u = _mm512_maskz_compress_epi32(k, load_lanes(valid, utils + i));
			acc = _mm512_add_epi32(acc, u);
			store_lanes(out_utils + kept, u);
		}
		kept += static_cast<std::uint32_t>(_mm_popcnt_u32(k));
	}
	if constexpr (sizeof(Util) == 8) *sum = static_cast<Util>(_mm512_reduce_add_epi64(acc));
	else *sum = static_cast<Util>(_mm512_reduce_add_epi32(acc));
	return kept;
}

}  // namespace

FilterFn filter_kernel() { return std::string_view(simd_name()) == "avx512" ? filter_avx512 : filter_scalar; }

}  // namespace REFIM_WIDTHS
