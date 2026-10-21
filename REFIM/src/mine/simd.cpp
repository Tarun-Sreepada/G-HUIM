#include "mine/simd.hpp"

#include "mine/simd_mode.hpp"

#include <cstdint>
#include <immintrin.h>
#include <string_view>

inline namespace REFIM_WIDTHS {

namespace {

std::uint32_t filter_scalar(const Item* items, const Util* utils, std::uint32_t n, const int* mark, int depth,
							const std::uint8_t*, Item* out_items, Util* out_utils, Util* sum) {
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

// For each 8-bit lane mask, the 32-bit lane indices that pack the set lanes to the front (vpermd has no compress).
struct CompressTable {
	alignas(8) std::uint8_t lanes[256][8];
	constexpr CompressTable() : lanes{} {
		for (int m = 0; m < 256; ++m) {
			int k = 0;
			for (int i = 0; i < 8; ++i)
				if (m >> i & 1) lanes[m][k++] = static_cast<std::uint8_t>(i);
		}
	}
};
constexpr CompressTable kCompress;

template <class T>
__attribute__((target("avx2"))) __m256i load_lanes8(const T* p) {
	if constexpr (sizeof(T) == 1) return _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
	else if constexpr (sizeof(T) == 2) return _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
	else return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}

// Stores 8 lanes narrowed back to T (writes 8 * sizeof(T) bytes).
template <class T>
__attribute__((target("avx2"))) void store_lanes8(T* p, __m256i v) {
	if constexpr (sizeof(T) == 4) {
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
	} else {
		// Low byte (or two) of every 32-bit lane to the front of each 128-bit half, then join the halves.
		const __m256i pick = sizeof(T) == 1 ? _mm256_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
															   0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
											: _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1,
															   0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
		const __m256i packed = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(v, pick), _mm256_setr_epi32(0, 4, 1, 5, 2, 3, 6, 7));
		if constexpr (sizeof(T) == 1) _mm_storel_epi64(reinterpret_cast<__m128i*>(p), _mm256_castsi256_si128(packed));
		else _mm_storeu_si128(reinterpret_cast<__m128i*>(p), _mm256_castsi256_si128(packed));
	}
}

// 8 items per step: gather mark[], compare, and pack the survivors with vpermd and a lane table. 64-bit
// utilities are packed in two halves of 4. Narrower utilities are summed in 32-bit lanes, which wrap like Util.
// The tail (n % 8) is scalar so no load reads past the transaction.
__attribute__((target("avx2,popcnt"))) std::uint32_t filter_avx2(const Item* items, const Util* utils, std::uint32_t n,
																   const int* mark, int depth, const std::uint8_t*, Item* out_items,
																   Util* out_utils, Util* sum) {
	const __m256i below = _mm256_set1_epi32(depth - 1);
	__m256i acc = _mm256_setzero_si256();
	std::uint32_t kept = 0, i = 0;
	for (; i + 8 <= n; i += 8) {
		const __m256i x = load_lanes8(items + i);
		const __m256i m = _mm256_i32gather_epi32(mark, x, 4);
		const unsigned k = static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(m, below))));
		const __m256i perm = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(kCompress.lanes[k])));
		store_lanes8(out_items + kept, _mm256_permutevar8x32_epi32(x, perm));
		if constexpr (sizeof(Util) == 8) {
			// A 64-bit lane is two 32-bit lanes: lane j moves as 2j, 2j + 1.
			const auto pack4 = [](unsigned k4, __m256i u) {
				const __m256i idx = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(kCompress.lanes[k4])));
				const __m256i pairs = _mm256_add_epi32(_mm256_slli_epi32(_mm256_permutevar8x32_epi32(idx, _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3)), 1),
													   _mm256_setr_epi32(0, 1, 0, 1, 0, 1, 0, 1));
				const __m256i keep = _mm256_permutevar8x32_epi32(u, pairs);
				const __m256i live = _mm256_cmpgt_epi64(_mm256_set1_epi64x(_mm_popcnt_u32(k4)), _mm256_setr_epi64x(0, 1, 2, 3));
				return _mm256_and_si256(keep, live);
			};
			const unsigned lo = k & 0xF, hi = k >> 4;
			const __m256i ulo = pack4(lo, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(utils + i)));
			const __m256i uhi = pack4(hi, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(utils + i + 4)));
			acc = _mm256_add_epi64(acc, _mm256_add_epi64(ulo, uhi));
			const std::uint32_t nlo = static_cast<std::uint32_t>(_mm_popcnt_u32(lo));
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(out_utils + kept), ulo);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(out_utils + kept + nlo), uhi);
		} else {
			const __m256i live = _mm256_cmpgt_epi32(_mm256_set1_epi32(_mm_popcnt_u32(k)), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
			const __m256i u = _mm256_and_si256(_mm256_permutevar8x32_epi32(load_lanes8(utils + i), perm), live);
			acc = _mm256_add_epi32(acc, u);
			store_lanes8(out_utils + kept, u);
		}
		kept += static_cast<std::uint32_t>(_mm_popcnt_u32(k));
	}
	Util s = 0;
	alignas(32) std::uint64_t lanes64[4];
	alignas(32) std::uint32_t lanes32[8];
	if constexpr (sizeof(Util) == 8) {
		_mm256_store_si256(reinterpret_cast<__m256i*>(lanes64), acc);
		s = static_cast<Util>(lanes64[0] + lanes64[1] + lanes64[2] + lanes64[3]);
	} else {
		_mm256_store_si256(reinterpret_cast<__m256i*>(lanes32), acc);
		std::uint32_t t = 0;
		for (const std::uint32_t l : lanes32) t += l;
		s = static_cast<Util>(t);
	}
	for (; i < n; ++i) {
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
filter_avx512(const Item* items, const Util* utils, std::uint32_t n, const int* mark, int depth, const std::uint8_t*,
			  Item* out_items, Util* out_utils, Util* sum) {
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

// 8-bit item ids only: 64 items per step, looked up in the 256-byte keep table with two vpermi2b (one per
// 128-entry half, bit 7 picks the half) instead of gathering mark[], and packed with vpcompressb. Utilities
// are packed and summed in lanes of their own width (sums wrap exactly like Util).
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vbmi,avx512vbmi2,popcnt"))) std::uint32_t
filter_avx512vbmi(const Item* items, const Util* utils, std::uint32_t n, const int* mark, int depth, const std::uint8_t* keep,
				  Item* out_items, Util* out_utils, Util* sum) {
	if constexpr (sizeof(Item) != 1) {
		return filter_avx512(items, utils, n, mark, depth, keep, out_items, out_utils, sum);
	} else {
		constexpr std::uint32_t lanes = 64 / sizeof(Util);
		const __m512i t0 = _mm512_loadu_si512(keep), t1 = _mm512_loadu_si512(keep + 64);
		const __m512i t2 = _mm512_loadu_si512(keep + 128), t3 = _mm512_loadu_si512(keep + 192);
		__m512i acc = _mm512_setzero_si512();
		std::uint32_t kept = 0;
		for (std::uint32_t i = 0; i < n; i += 64) {
			const std::uint32_t rem = n - i;
			const __mmask64 valid = rem >= 64 ? ~__mmask64{0} : (__mmask64{1} << rem) - 1;
			const __m512i x = _mm512_maskz_loadu_epi8(valid, items + i);
			const __m512i lo = _mm512_permutex2var_epi8(t0, x, t1), hi = _mm512_permutex2var_epi8(t2, x, t3);
			const __m512i hit = _mm512_mask_blend_epi8(_mm512_movepi8_mask(x), lo, hi);
			const __mmask64 k = _mm512_test_epi8_mask(hit, hit) & valid;
			_mm512_storeu_si512(out_items + kept, _mm512_maskz_compress_epi8(k, x));
			for (std::uint32_t c = 0; c < sizeof(Util) && c * lanes < rem; ++c) {
				const __mmask64 part = static_cast<__mmask64>(-1) >> (64 - lanes);
				const __mmask64 kc = (k >> (c * lanes)) & part, vc = (valid >> (c * lanes)) & part;
				const Util* src = utils + i + c * lanes;
				__m512i u;
				if constexpr (sizeof(Util) == 2) {
					u = _mm512_maskz_compress_epi16(static_cast<__mmask32>(kc), _mm512_maskz_loadu_epi16(static_cast<__mmask32>(vc), src));
					acc = _mm512_add_epi16(acc, u);
				} else if constexpr (sizeof(Util) == 4) {
					u = _mm512_maskz_compress_epi32(static_cast<__mmask16>(kc), _mm512_maskz_loadu_epi32(static_cast<__mmask16>(vc), src));
					acc = _mm512_add_epi32(acc, u);
				} else {
					u = _mm512_maskz_compress_epi64(static_cast<__mmask8>(kc), _mm512_maskz_loadu_epi64(static_cast<__mmask8>(vc), src));
					acc = _mm512_add_epi64(acc, u);
				}
				_mm512_storeu_si512(out_utils + kept, u);
				kept += static_cast<std::uint32_t>(_mm_popcnt_u64(kc));
			}
		}
		if constexpr (sizeof(Util) == 2)
			*sum = static_cast<Util>(_mm512_reduce_add_epi32(_mm512_add_epi32(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(acc)),
																			  _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(acc, 1)))));
		else if constexpr (sizeof(Util) == 4)
			*sum = static_cast<Util>(_mm512_reduce_add_epi32(acc));
		else
			*sum = static_cast<Util>(_mm512_reduce_add_epi64(acc));
		return kept;
	}
}

}  // namespace

FilterFn filter_kernel() {
	const std::string_view name = simd_name();
	if (name == "avx512vbmi") return sizeof(Item) == 1 ? filter_avx512vbmi : filter_avx512;
	return name == "avx512" ? filter_avx512 : name == "avx2" ? filter_avx2 : filter_scalar;
}

bool needs_keep_table() { return sizeof(Item) == 1 && std::string_view(simd_name()) == "avx512vbmi"; }

}  // namespace REFIM_WIDTHS
