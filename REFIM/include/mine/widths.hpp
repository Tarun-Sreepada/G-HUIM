#pragma once
// Widths of item ids and utilities.
//
// The search (mine/ and the root build in io/root.hpp) is compiled once per (item bits, utility bits) pair
// listed in CMake's REFIM_WIDTHS, with REFIM_KEY_BITS and REFIM_UTIL_BITS set, into the inline namespace
// REFIM_WIDTHS (k<item>_u<utility>), so every pair links into one binary; Refim::mine() picks one at run time.
//
// Items keep ids 1..num_items, so num_items < 2^bits. Every utility the search stores or sums (an entry,
// a transaction's prefix utility, a local or subtree bound, an itemset's utility) is at most the largest
// transaction-weighted utility of the renamed transactions (Transactions::max_twu), so max_twu < 2^bits is
// enough and no sum can overflow.
//
// The search is also compiled per occurrence word width (CMake's REFIM_OCC_WIDTHS, REFIM_OCC_BITS), into
// k<item>_u<utility>_o<occurrence>. An occurrence packs a transaction index above a position (mine/level.hpp);
// an index needs bit_width(transactions - 1) bits and a position bit_width(longest transaction - 1), and no
// projected database has more transactions or a longer one than the input.

#include "mine/built_widths.hpp"  // REFIM_FOR_EACH_BUILT_WIDTH(X): X(item bits, utility bits, occurrence bits) per build

#include <algorithm>
#include <bit>
#include <cstdint>
#include <type_traits>

#define REFIM_WIDTH_NAME(item, util, occ) REFIM_WIDTH_NAME_(item, util, occ)
#define REFIM_WIDTH_NAME_(item, util, occ) k##item##_u##util##_o##occ

template <unsigned Bits>
using UnsignedBits = std::conditional_t<
	Bits == 8, std::uint8_t,
	std::conditional_t<Bits == 16, std::uint16_t, std::conditional_t<Bits == 32, std::uint32_t, std::uint64_t>>>;

// Position bits of a packed occurrence when no transaction has more than `max_len` items.
inline std::uint32_t occurrence_pos_bits(std::uint64_t max_len) {
	return static_cast<std::uint32_t>(std::bit_width(max_len > 1 ? max_len - 1 : 0));
}

// Smallest occurrence word (32 or 64 bits) holding indices 0..n_trans-1 above the position bits. The index
// keeps at least one bit, so a shift never spans the whole word.
inline int occurrence_bits_for(std::uint64_t n_trans, std::uint64_t max_len) {
	const std::uint32_t tid_bits = std::max<std::uint32_t>(1, static_cast<std::uint32_t>(std::bit_width(n_trans > 1 ? n_trans - 1 : 0)));
	return tid_bits + occurrence_pos_bits(max_len) <= 32 ? 32 : 64;
}

#if defined(REFIM_KEY_BITS) && defined(REFIM_UTIL_BITS) && defined(REFIM_OCC_BITS)
#define REFIM_WIDTHS REFIM_WIDTH_NAME(REFIM_KEY_BITS, REFIM_UTIL_BITS, REFIM_OCC_BITS)

inline namespace REFIM_WIDTHS {

using Item = UnsignedBits<REFIM_KEY_BITS>;   // renamed item id, 1..num_items (larger id = larger TWU)
using Util = UnsignedBits<REFIM_UTIL_BITS>;  // every utility stored or summed by the search
using OccWord = UnsignedBits<REFIM_OCC_BITS>;  // one packed occurrence

}  // namespace REFIM_WIDTHS
#endif
