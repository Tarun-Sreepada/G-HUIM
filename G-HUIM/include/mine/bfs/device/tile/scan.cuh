#pragma once
// Passes 1 and 2: local utilities and child sizes over tiles.

#include "mine/bfs/device/child.cuh"
#include "mine/bfs/device/modes.cuh"
#include "mine/bfs/placement.cuh"
#include "mine/bfs/device/tile/search.cuh"

namespace ghuim::inline GHUIM_WIDTHS::bfs {

// Lanes of the group that hold the same `key` (a class number, or `none`) take consecutive positions:
// returns this lane's position among them and, on the group's first such lane, adds the count to
// `counter` and returns the old value in `first`. Every lane of the group must call it.
__device__ inline uint32_t take_in_class(const Lanes& lanes, uint32_t key, uint32_t none, uint32_t* counter, uint32_t& first)
{
    const uint32_t same   = __match_any_sync(lanes.mask, key);
    const uint32_t lane   = threadIdx.x & 31;
    const uint32_t leader = __ffs(same) - 1;
    uint32_t       start  = 0;
    if (key != none && lane == leader)
        start = fetch_add(*counter, static_cast<uint32_t>(__popc(same)));
    first = __shfl_sync(lanes.mask, start, leader);
    return __popc(same & ((1u << lane) - 1));
}

// Pass 1: local utilities (as local_tile, over tiled entries).
__device__ inline void tile_local(const Projection& p, const Lanes& lanes, uint32_t i0, LocalAcc& acc)
{
    const uint32_t  i      = i0 + lanes.rank;
    const Tx        tx     = i < p.n_trans ? p.txs[tx_slot(p, i)] : Tx{0, 0, 0};
    const Key*      k_item = p.eitem + tx.begin;
    const Util*     k_util = p.eutil + tx.begin;
    const int       k      = p.occ ? (i < p.n_trans ? static_cast<int>(p.occ[i].pos) : -1)
                                   : find_item_tiled(k_item, tx.length, p.item, p.stride);
    if (i < p.n_trans)
        p.pos[i] = static_cast<uint32_t>(k + 1);
    const uint32_t  from   = static_cast<uint32_t>(k + 1);
    const uint32_t  rest   = k >= 0 ? tx.length - from : 0;
    const uint32_t  span   = lanes.max(rest);
    const Util      prefix = k >= 0 ? tx.utility + k_util[static_cast<uint32_t>(k) * p.stride] : 0;
    Util            after  = 0;
    for (uint32_t d = 0; d < span; ++d)
        if (d < rest)
            after += k_util[(from + d) * p.stride];
    for (uint32_t d = 0; d < span; ++d)
        if (d < rest)
            add(p.local[k_item[(from + d) * p.stride]], prefix + after);
    acc.utility += prefix;
}

// Pass 2: subtree utilities, merging of identical projected transactions, and the representatives of
// each length class.
__device__ inline void tile_subtree(const Projection& p, const Lanes& lanes, uint32_t i0, ChildSize& size)
{
    const uint32_t  i      = i0 + lanes.rank;
    const bool      valid  = i < p.n_trans;
    const uint32_t  pos    = valid ? p.pos[i] : 0;
    const Tx        tx     = pos != 0 ? p.txs[tx_slot(p, i)] : Tx{0, 0, 0};
    const Key*      k_item = p.eitem + tx.begin;
    const Util*     k_util = p.eutil + tx.begin;
    const int       k      = static_cast<int>(pos) - 1;
    const uint32_t  rest   = k >= 0 ? tx.length - pos : 0;
    const uint32_t  span   = lanes.max(rest);
    // One backward sweep, as in subtree_tile.
    const Util     prefix = k >= 0 ? tx.utility + k_util[static_cast<uint32_t>(k) * p.stride] : 0;
    Util           after  = 0;
    uint32_t       kept = 0, hash = 0;
    for (uint32_t d = span; d-- > 0;) {
        if (d >= rest)
            continue;
        const uint32_t z = p.rename[k_item[(pos + d) * p.stride]];
        if (z == kNone)
            continue;
        after += k_util[(pos + d) * p.stride];
        add(p.subtree[z], prefix + after);
        hash ^= mix(z * 0x9e3779b9u + kept++);
    }
    uint32_t bucket = kTileBuckets; // no class: not a representative
    if (valid) {
        if (kept == 0)
            p.rep[i] = kNone;
        else {
            size.max_len = max(size.max_len, kept);
            p.kept[i]    = kept;
            for (uint32_t h = hash % p.table_size;; h = h + 1 == p.table_size ? 0 : h + 1) {
                const uint32_t other = compare_swap(p.table[h], kNone, i);
                if (other == kNone) {
                    p.rep[i] = i;
                    ++size.trans;
                    size.entries += kept;
                    bucket = min(kept, kTileBuckets - 1);
                    break;
                }
                const Tx o = p.txs[tx_slot(p, other)];
                if (same_kept_items_tiled(p.eitem + o.begin, o.length, p.pos[other], k_item, tx.length, pos, p.rename, p.stride)) {
                    p.rep[i] = other;
                    break;
                }
            }
        }
    }
    uint32_t first;
    take_in_class(lanes, bucket, kTileBuckets, bucket < kTileBuckets ? p.hist + bucket : p.hist, first);
}

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
