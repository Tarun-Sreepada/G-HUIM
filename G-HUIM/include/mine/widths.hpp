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
//
// The BFS miner is also compiled per occurrence word width (CMake's GHUIM_OCC_WIDTHS, GHUIM_OCC_BITS), into
// k<key>_u<utility>_o<occurrence>. An occurrence packs a transaction slot above a position (mine/bfs/layout.cuh);
// a slot needs bit_width(transactions - 1) bits and a position bit_width(longest transaction - 1), and no
// projected database has more transactions or a longer one than the root.

#include "mine/built_widths.hpp" // GHUIM_FOR_EACH_BUILT_WIDTH(X) / GHUIM_FOR_EACH_BUILT_BFS_WIDTH(X): one X per miner built

#include <algorithm>
#include <bit>
#include <cstdint>
#include <type_traits>

#define GHUIM_WIDTH_NAME(key, util) GHUIM_WIDTH_NAME_(key, util)
#define GHUIM_WIDTH_NAME_(key, util) k##key##_u##util
#define GHUIM_BFS_WIDTH_NAME(key, util, occ) GHUIM_BFS_WIDTH_NAME_(key, util, occ)
#define GHUIM_BFS_WIDTH_NAME_(key, util, occ) k##key##_u##util##_o##occ

namespace ghuim {

template <unsigned Bits>
using UnsignedBits = std::conditional_t<Bits == 8, uint8_t,
                                        std::conditional_t<Bits == 16, uint16_t, std::conditional_t<Bits == 32, uint32_t, uint64_t>>>;

// Position bits of a packed occurrence when no transaction has more than `max_len` entries.
inline uint32_t occurrence_pos_bits(uint64_t max_len)
{
    return static_cast<uint32_t>(std::bit_width(max_len > 1 ? max_len - 1 : 0));
}

// Smallest occurrence word (32 or 64 bits) holding slots 0..n_trans-1 above the position bits. The slot keeps
// at least one bit, so a shift never spans the whole word.
inline uint32_t occurrence_bits_for(uint64_t n_trans, uint64_t max_len)
{
    const uint32_t slot_bits = std::max<uint32_t>(1, static_cast<uint32_t>(std::bit_width(n_trans > 1 ? n_trans - 1 : 0)));
    return slot_bits + occurrence_pos_bits(max_len) <= 32 ? 32 : 64;
}

} // namespace ghuim

#if defined(GHUIM_KEY_BITS) && defined(GHUIM_UTIL_BITS)
#ifdef GHUIM_OCC_BITS
#define GHUIM_WIDTHS GHUIM_BFS_WIDTH_NAME(GHUIM_KEY_BITS, GHUIM_UTIL_BITS, GHUIM_OCC_BITS)
#else
#define GHUIM_WIDTHS GHUIM_WIDTH_NAME(GHUIM_KEY_BITS, GHUIM_UTIL_BITS)
#endif

namespace ghuim::inline GHUIM_WIDTHS {

using Key  = UnsignedBits<GHUIM_KEY_BITS>;  // local item name of an entry
using Util = UnsignedBits<GHUIM_UTIL_BITS>; // every utility stored or summed on the device
#ifdef GHUIM_OCC_BITS
using OccWord = UnsignedBits<GHUIM_OCC_BITS>; // one packed occurrence (BFS)
#endif

} // namespace ghuim::inline GHUIM_WIDTHS
#endif
