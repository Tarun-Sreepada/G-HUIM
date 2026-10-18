#pragma once
// Tile method: lane-per-transaction passes over databases in the tile layout (layout.cuh). Entry k of
// a transaction is at begin + k * kTile, so the lanes of a warp step (one tile of the parent) read
// neighbouring words whenever they read the same row.
//
// The child is built in tile order: pass 2 counts the representatives of each length class, a class
// scan gives every class its first slot, every representative takes a slot of its class, the longest
// transaction of each child tile sets the tile's length, and a scan over the tiles gives their offsets.
// Only then is the child allocated, at its exact size, and written.

#include "mine/bfs/device/child.cuh"
#include "mine/bfs/device/modes.cuh"
#include "mine/bfs/placement.cuh"

namespace ghuim::bfs {

// Position of `item` in a sorted tiled transaction (keys at col[k * kTile]), or -1.
__device__ inline int find_item_tiled(const uint32_t* col, uint32_t length, uint32_t item, uint32_t stride)
{
    int low = 0, high = static_cast<int>(length) - 1;
    while (low <= high) {
        const int      mid = (low + high) >> 1;
        const uint32_t key = col[static_cast<uint32_t>(mid) * stride];
        if (key == item)
            return mid;
        if (key < item)
            low = mid + 1;
        else
            high = mid - 1;
    }
    return -1;
}

// same_kept_items over tiled key arrays.
__device__ inline bool same_kept_items_tiled(const uint32_t* ea, uint32_t a_length, uint32_t i, const uint32_t* eb,
                                             uint32_t b_length, uint32_t j, const uint32_t* rename, uint32_t stride)
{
    for (;;) {
        while (i < a_length && rename[ea[i * stride]] == kNone)
            ++i;
        while (j < b_length && rename[eb[j * stride]] == kNone)
            ++j;
        if (i == a_length || j == b_length)
            return i == a_length && j == b_length;
        if (ea[i * stride] != eb[j * stride])
            return false;
        ++i;
        ++j;
    }
}

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
    const Tx        tx     = i < p.n_trans ? p.txs[i] : Tx{0, 0, 0};
    const uint32_t* k_item = p.eitem + tx.begin;
    const uint32_t* k_util = p.eutil + tx.begin;
    const int       k      = find_item_tiled(k_item, tx.length, p.item, p.stride);
    if (i < p.n_trans)
        p.pos[i] = static_cast<uint32_t>(k + 1);
    const uint32_t  from   = static_cast<uint32_t>(k + 1);
    const uint32_t  rest   = k >= 0 ? tx.length - from : 0;
    const uint32_t  span   = lanes.max(rest);
    const uint64_t  prefix = k >= 0 ? uint64_t{tx.utility} + k_util[static_cast<uint32_t>(k) * p.stride] : 0;
    uint64_t        after  = 0;
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
    const Tx        tx     = pos != 0 ? p.txs[i] : Tx{0, 0, 0};
    const uint32_t* k_item = p.eitem + tx.begin;
    const uint32_t* k_util = p.eutil + tx.begin;
    const int       k      = static_cast<int>(pos) - 1;
    const uint32_t  rest   = k >= 0 ? tx.length - pos : 0;
    const uint32_t  span   = lanes.max(rest);
    uint64_t kept_sum = 0;
    uint32_t kept     = 0;
    for (uint32_t d = 0; d < span; ++d)
        if (d < rest && p.rename[k_item[(pos + d) * p.stride]] != kNone) {
            ++kept;
            kept_sum += k_util[(pos + d) * p.stride];
        }
    const uint64_t remaining = (k >= 0 ? uint64_t{tx.utility} + k_util[static_cast<uint32_t>(k) * p.stride] : 0) + kept_sum;
    uint64_t       before    = 0;
    uint32_t       index = 0, hash = 0;
    for (uint32_t d = 0; d < span; ++d) {
        if (d >= rest)
            continue;
        const uint32_t item_d = k_item[(pos + d) * p.stride], util_d = k_util[(pos + d) * p.stride];
        const uint32_t z      = p.rename[item_d];
        if (z == kNone)
            continue;
        add(p.subtree[z], remaining - before);
        before += util_d;
        hash ^= mix(z * 0x9e3779b9u + index++);
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
                const Tx o = p.txs[other];
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
    uint32_t       carry = 0;
    for (uint32_t t0 = 0; t0 < tiles; t0 += g.size()) {
        const uint32_t t     = t0 + g.rank();
        const uint32_t width = t < tiles ? p.tile_len[t] * p.child_stride : 0;
        const uint32_t scan  = g.scan(width);
        if (t < tiles)
            p.tile_off[t] = carry + scan - width;
        carry += g.last(scan);
    }
    if (g.rank() == 0)
        state.child_entries = carry;
}

// Pass 3a: representatives write their child transaction into their slot's tile.
__device__ inline void tile_write(const Projection& p, const Lanes& lanes, uint32_t i0, const ChildView& c)
{
    const uint32_t  i      = i0 + lanes.rank;
    const bool      write  = i < p.n_trans && p.rep[i] == i;
    const Tx        tx     = write ? p.txs[i] : Tx{0, 0, 0};
    const uint32_t* k_item = p.eitem + tx.begin;
    const uint32_t* k_util = p.eutil + tx.begin;
    const int       k      = write ? static_cast<int>(p.pos[i]) - 1 : -1;
    const uint32_t  pos    = static_cast<uint32_t>(k + 1);
    const uint32_t  rest   = k >= 0 ? tx.length - pos : 0;
    const uint32_t  span   = lanes.max(rest);
    const uint32_t  kept   = write ? p.kept[i] : 0;
    uint32_t* out_item = nullptr;
    uint32_t* out_util = nullptr;
    if (write) {
        const uint32_t slot  = p.slot_of[i];
        const uint32_t begin = p.tile_off[slot / p.child_stride] + slot % p.child_stride;
        c.txs[slot]          = Tx{begin, kept, tx.utility + k_util[static_cast<uint32_t>(k) * p.stride]};
        out_item             = c.eitem + begin;
        out_util             = c.eutil + begin;
    }
    for (uint32_t d = 0; d < span; ++d) {
        if (d >= rest)
            continue;
        const uint32_t z = p.rename[k_item[(pos + d) * p.stride]];
        if (z != kNone) {
            *out_item = z;
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
    const Tx        tx     = merge ? p.txs[i] : Tx{0, 0, 0};
    const uint32_t* k_item = p.eitem + tx.begin;
    const uint32_t* k_util = p.eutil + tx.begin;
    const int       k      = merge ? static_cast<int>(p.pos[i]) - 1 : -1;
    const uint32_t  pos    = static_cast<uint32_t>(k + 1);
    const uint32_t  rest   = k >= 0 ? tx.length - pos : 0;
    const uint32_t  span   = lanes.max(rest);
    uint32_t*       out    = nullptr;
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

// Lane per transaction over tiled databases: a warp covers one tile of the parent per step.
struct TileTx
{
    using Ctx                  = NoCtx;
    static constexpr bool kTiled = true;
    static __device__ Ctx make_ctx(const BatchView&, const Projection&, uint32_t, uint32_t, uint32_t) { return {}; }
    static __device__ uint32_t unit(const Projection&, const Lanes& lanes) { return lanes.count; }
    static __device__ void scan_local(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step, TaskState& s,
                                      const Ctx&)
    {
        LocalAcc acc;
        for (uint32_t i0 = from; i0 < p.n_trans; i0 += step)
            tile_local(p, lanes, i0, acc);
        local_finish(lanes, acc, s);
    }
    static __device__ void scan_subtree(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step, TaskState& s,
                                        const Ctx&)
    {
        ChildSize size;
        for (uint32_t i0 = from; i0 < p.n_trans; i0 += step)
            tile_subtree(p, lanes, i0, size);
        subtree_finish(lanes, size, s);
    }
    static __device__ void assign_slots(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step, const Ctx&)
    {
        for (uint32_t i0 = from; i0 < p.n_trans; i0 += step)
            tile_assign(p, lanes, i0);
    }
    static __device__ void write_representatives(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step,
                                                 TaskState&, const ChildView& c, const Ctx&)
    {
        for (uint32_t i0 = from; i0 < p.n_trans; i0 += step)
            tile_write(p, lanes, i0, c);
    }
    static __device__ void merge_duplicates(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step,
                                            const ChildView& c, const Ctx&)
    {
        for (uint32_t i0 = from; i0 < p.n_trans; i0 += step)
            tile_merge(p, lanes, i0, c);
    }
};

template <typename Mode> concept TiledMode = requires { Mode::kTiled; };

// Calls `f` with the transaction mode a launch uses: TileTx for the tile method, else LaneTx.
template <typename F> void with_mode(const AutoTune& tune, F&& f)
{
    if (tune.tiled)
        f(TileTx{});
    else
        f(LaneTx{});
}

} // namespace ghuim::bfs
