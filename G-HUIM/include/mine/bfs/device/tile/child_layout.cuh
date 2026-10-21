#pragma once
// Length-class bases, slot assignment and tile offsets of the child.

#include "mine/bfs/device/child.cuh"
#include "mine/bfs/device/modes.cuh"
#include "mine/bfs/placement.cuh"
#include "mine/bfs/device/tile/scan.cuh"

namespace ghuim::inline GHUIM_WIDTHS::bfs {

// Class starts: the first child slot of every length class. One lane of the group works; the caller
// synchronizes the group afterwards.
template <typename Group> __device__ inline void tile_class_bases(const Group& g, const Projection& p, const TaskState& state)
{
    if (g.rank() != 0)
        return;
    const uint32_t classes = min(state.child_max_len, kTileBuckets - 1) + 1;
    uint32_t       run     = 0;
    for (uint32_t c = 0; c < classes; ++c) {
        p.base[c] = run;
        run += p.hist[c];
    }
}

// Every representative takes the next slot of its length class and raises its tile's length.
__device__ inline void tile_assign(const Projection& p, const Lanes& lanes, uint32_t i0)
{
    const uint32_t i    = i0 + lanes.rank;
    const bool     take = i < p.n_trans && p.rep[i] == i;
    const uint32_t kept = take ? p.kept[i] : 0;
    const uint32_t cls  = take ? min(kept, kTileBuckets - 1) : kTileBuckets;
    uint32_t       first;
    const uint32_t rank = take_in_class(lanes, cls, kTileBuckets, take ? p.cursor + cls : p.cursor, first);
    if (take) {
        const uint32_t slot = p.base[cls] + first + rank;
        p.slot_of[i]        = slot;
        fetch_max(p.tile_len[slot / p.child_stride], kept);
    }
}

// Offsets of the child's tiles from their lengths, and the child's padded entry count. The whole group
// calls it; the caller synchronizes the group afterwards.
template <typename Group> __device__ inline void tile_offsets(const Group& g, const Projection& p, TaskState& state)
{
    const uint32_t tiles = ceil_div(state.child_trans, p.child_stride);
    uint32_t       carry = 0, empty = 0;
    for (uint32_t t0 = 0; t0 < tiles; t0 += g.size()) {
        const uint32_t t     = t0 + g.rank();
        const uint32_t width = t < tiles ? p.tile_len[t] * p.child_stride : 0;
        const uint32_t scan  = g.scan(width);
        if (t < tiles)
            p.tile_off[t] = carry + scan - width;
        carry += g.last(scan);
        const uint32_t used = t < tiles ? min(p.child_stride, state.child_trans - t * p.child_stride) : 0;
        empty += g.last(g.scan(t < tiles ? p.tile_len[t] * (p.child_stride - used) : 0));
    }
    if (g.rank() == 0) {
        state.child_empty_slots  = empty;
        state.child_real_entries = state.child_entries;
        state.child_entries      = carry;
    }
}

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
