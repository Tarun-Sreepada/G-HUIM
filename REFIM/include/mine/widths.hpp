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

#include "mine/built_widths.hpp"  // REFIM_FOR_EACH_BUILT_WIDTH(X): X(item bits, utility bits) per built pair

#include <cstdint>
#include <type_traits>

#define REFIM_WIDTH_NAME(item, util) REFIM_WIDTH_NAME_(item, util)
#define REFIM_WIDTH_NAME_(item, util) k##item##_u##util

template <unsigned Bits>
using UnsignedBits = std::conditional_t<
	Bits == 8, std::uint8_t,
	std::conditional_t<Bits == 16, std::uint16_t, std::conditional_t<Bits == 32, std::uint32_t, std::uint64_t>>>;

#if defined(REFIM_KEY_BITS) && defined(REFIM_UTIL_BITS)
#define REFIM_WIDTHS REFIM_WIDTH_NAME(REFIM_KEY_BITS, REFIM_UTIL_BITS)

inline namespace REFIM_WIDTHS {

using Item = UnsignedBits<REFIM_KEY_BITS>;   // renamed item id, 1..num_items (larger id = larger TWU)
using Util = UnsignedBits<REFIM_UTIL_BITS>;  // every utility stored or summed by the search

}  // namespace REFIM_WIDTHS
#endif
