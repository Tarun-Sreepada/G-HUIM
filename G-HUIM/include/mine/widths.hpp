#pragma once
// Widths of the entry keys and utilities on the device.
//
// The mining code (mine/bfs, mine/dfs) is compiled once per (key bits, utility bits) pair listed in
// CMake's GHUIM_WIDTHS, with GHUIM_KEY_BITS and GHUIM_UTIL_BITS set, into the inline namespace
// GHUIM_WIDTHS (k<key>_u<utility>), so every pair links into one binary; mine() picks one at run time.
//
// Keys are local item names 0..n_items-1, so n_items <= 2^bits. Every utility the miners store or sum
// (an entry, a transaction's prefix utility, a local or subtree bound, an itemset's utility) is at most
// the largest transaction-weighted utility of the pruned database (Dataset::max_twu), so max_twu < 2^bits
// is enough and no sum can overflow.

#include "mine/built_widths.hpp" // GHUIM_FOR_EACH_BUILT_WIDTH(X): X(key bits, utility bits) per built pair

#include <cstdint>
#include <type_traits>

#define GHUIM_WIDTH_NAME(key, util) GHUIM_WIDTH_NAME_(key, util)
#define GHUIM_WIDTH_NAME_(key, util) k##key##_u##util

namespace ghuim {

template <unsigned Bits>
using UnsignedBits = std::conditional_t<Bits == 8, uint8_t,
                                        std::conditional_t<Bits == 16, uint16_t, std::conditional_t<Bits == 32, uint32_t, uint64_t>>>;

} // namespace ghuim

#if defined(GHUIM_KEY_BITS) && defined(GHUIM_UTIL_BITS)
#define GHUIM_WIDTHS GHUIM_WIDTH_NAME(GHUIM_KEY_BITS, GHUIM_UTIL_BITS)

namespace ghuim::inline GHUIM_WIDTHS {

using Key  = UnsignedBits<GHUIM_KEY_BITS>;  // local item name of an entry
using Util = UnsignedBits<GHUIM_UTIL_BITS>; // every utility stored or summed on the device

} // namespace ghuim::inline GHUIM_WIDTHS
#endif
