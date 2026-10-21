#pragma once
// Pass 3: representatives write their child transaction; duplicates merge.

#include "mine/bfs/device/child.cuh"
#include "mine/bfs/device/modes.cuh"
#include "mine/bfs/placement.cuh"
#include "mine/bfs/device/tile/search.cuh"

namespace ghuim::inline GHUIM_WIDTHS::bfs {

// Pass 3a: representatives write their child transaction into their slot's tile.
__device__ inline void tile_write(const Projection& p, const Lanes& lanes, uint32_t i0, const ChildView& c)
{
    const uint32_t  i      = i0 + lanes.rank;
    const bool      write  = i < p.n_trans && p.rep[i] == i;
    const Tx        tx     = write ? p.txs[tx_slot(p, i)] : Tx{0, 0, 0};
    const Key*      k_item = p.eitem + tx.begin;
    const Util*     k_util = p.eutil + tx.begin;
    const int       k      = write ? static_cast<int>(p.pos[i]) - 1 : -1;
    const uint32_t  pos    = static_cast<uint32_t>(k + 1);
    const uint32_t  rest   = k >= 0 ? tx.length - pos : 0;
    const uint32_t  span   = lanes.max(rest);
    const uint32_t  kept   = write ? p.kept[i] : 0;
    Key*      out_item = nullptr;
    Util*     out_util = nullptr;
    if (write) {
        const uint32_t slot  = p.slot_of[i];
        const uint32_t begin = p.tile_off[slot / p.child_stride] + slot % p.child_stride;
        c.txs[slot]          = Tx{begin, kept, static_cast<Util>(tx.utility + k_util[static_cast<uint32_t>(k) * p.stride])};
        out_item             = c.eitem + begin;
        out_util             = c.eutil + begin;
    }
    for (uint32_t d = 0; d < span; ++d) {
        if (d >= rest)
            continue;
        const uint32_t z = p.rename[k_item[(pos + d) * p.stride]];
        if (z != kNone) {
            *out_item = static_cast<Key>(z);
            *out_util = k_util[(pos + d) * p.stride];
            out_item += p.child_stride;
            out_util += p.child_stride;
        }
    }
}

// Pass 3b: duplicates add their utilities into their representative's child transaction.
__device__ inline void tile_merge(const Projection& p, const Lanes& lanes, uint32_t i0, const ChildView& c)
{
    const uint32_t  i      = i0 + lanes.rank;
    const uint32_t  r      = i < p.n_trans ? p.rep[i] : kNone;
    const bool      merge  = r != kNone && r != i;
    const Tx        tx     = merge ? p.txs[tx_slot(p, i)] : Tx{0, 0, 0};
    const Key*      k_item = p.eitem + tx.begin;
    const Util*     k_util = p.eutil + tx.begin;
    const int       k      = merge ? static_cast<int>(p.pos[i]) - 1 : -1;
    const uint32_t  pos    = static_cast<uint32_t>(k + 1);
    const uint32_t  rest   = k >= 0 ? tx.length - pos : 0;
    const uint32_t  span   = lanes.max(rest);
    Util*           out    = nullptr;
    if (merge) {
        const uint32_t dst = p.slot_of[r];
        add(c.txs[dst].utility, tx.utility + k_util[static_cast<uint32_t>(k) * p.stride]);
        out = c.eutil + c.txs[dst].begin;
    }
    for (uint32_t d = 0; d < span; ++d) {
        if (d >= rest)
            continue;
        if (p.rename[k_item[(pos + d) * p.stride]] != kNone) {
            add(*out, k_util[(pos + d) * p.stride]);
            out += p.child_stride;
        }
    }
}

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
