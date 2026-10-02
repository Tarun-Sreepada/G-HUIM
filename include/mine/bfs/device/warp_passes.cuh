#pragma once
// Lane-group-per-transaction passes. A group of 4, 8, 16 or 32 lanes owns one transaction at a time
// and its lanes sit on consecutive entries, so the key and utility arrays are read in coalesced
// chunks of `size` entries. The group size comes from the task's longest transaction, so short
// transactions share a warp (32 / size groups per warp) instead of wasting lanes. Collectives use the
// group's own lane mask, and the transaction index is uniform within a group, so there are no
// lockstep loops; positions inside a transaction come from ballots and group scans.

#include "mine/bfs/device/child.cuh"
#include "mine/bfs/device/sources.cuh"

namespace ghuim::bfs {

// Lanes per transaction for a task whose longest transaction has `max_len` entries.
__host__ __device__ inline uint32_t tx_group_size(uint32_t max_len)
{
    return max_len <= 4 ? 4u : max_len <= 8 ? 8u : max_len <= 16 ? 16u : 32u;
}

// The lanes of a warp that work on one transaction.
struct TxGroup
{
    uint32_t size;   // lanes per transaction
    uint32_t rank;   // lane within the group
    uint32_t index;  // group within the warp
    uint32_t groups; // groups per warp
    unsigned mask;   // this group's lanes
    unsigned below;  // lanes of the warp below this one

    __device__ explicit TxGroup(uint32_t group_size)
    {
        const uint32_t lane = threadIdx.x & 31;
        size                = group_size;
        rank                = lane & (size - 1);
        index               = lane / size;
        groups              = 32 / size;
        mask                = size == 32 ? kFullMask : (((1u << size) - 1) << (lane & ~(size - 1)));
        below               = (1u << lane) - 1;
    }
    __device__ uint32_t ballot(bool flag) const { return __ballot_sync(mask, flag); }
    __device__ bool all(bool flag) const { return __all_sync(mask, flag); }
    // Group-wide 32-bit sum and xor. A transaction's utilities sum to at most the database total, which
    // load_dataset checks fits in 32 bits, so 32-bit sums are exact.
    __device__ uint32_t total(uint32_t value) const { return __reduce_add_sync(mask, value); }
    __device__ uint32_t xor_all(uint32_t value) const { return __reduce_xor_sync(mask, value); }
    __device__ uint32_t from_rank(uint32_t value, uint32_t src) const { return __shfl_sync(mask, value, src, size); }
    __device__ uint32_t inclusive_scan(uint32_t value) const
    {
        for (uint32_t offset = 1; offset < size; offset <<= 1) {
            const uint32_t other = __shfl_up_sync(mask, value, offset, size);
            if (rank >= offset)
                value += other;
        }
        return value;
    }
    // Ranks of the group's set lanes below this one in `ballot`.
    __device__ uint32_t before(uint32_t ballot_bits) const { return __popc(ballot_bits & below); }
};

// Number of keys below `item` in a sorted transaction. Chunks of `size` keys are read coalesced and
// the search stops at the first chunk holding a key >= item.
__device__ inline uint32_t group_lower_bound(const TxGroup& g, const uint32_t* keys, uint32_t length, uint32_t item)
{
    uint32_t below = 0;
    for (uint32_t base = 0; base < length; base += g.size) {
        const uint32_t idx = base + g.rank;
        const uint32_t n   = __popc(g.ballot(idx < length && keys[idx] < item));
        below += n;
        if (n < g.size)
            break;
    }
    return below;
}

// A transaction seen by one group: where the extension item sits and what follows it.
struct GroupTx
{
    const uint32_t* keys;
    const uint32_t* utils;
    uint32_t        length;
    uint32_t        pos;  // position behind the extension item (k + 1), 0 when absent
    uint32_t        k;    // position of the extension item (valid when found)
    uint32_t        rest; // entries behind it
    bool            found;
    uint64_t        prefix; // utility of the itemset in this transaction (0 when not found)
};

// The transaction `tx` once pass 1 has found the extension item at `pos - 1` (`pos` 0: absent).
// `slot` is the transaction's position in the source's step (a tile slot for staged sources).
template <typename Src>
__device__ __forceinline__ GroupTx make_group_tx(const Tx& tx, const Src& src, uint32_t slot, uint32_t pos)
{
    GroupTx t;
    t.keys   = src.keys(tx, slot);
    t.utils  = src.utils(tx, slot);
    t.length = tx.length;
    t.pos    = pos;
    t.found  = pos != 0;
    t.k      = pos - 1;
    t.rest   = t.found ? tx.length - pos : 0;
    t.prefix = t.found ? uint64_t{tx.utility} + t.utils[t.k] : 0;
    return t;
}

// Pass 1: searches for the extension item.
template <typename Src>
__device__ __forceinline__ GroupTx open_group_tx(const TxGroup& g, const Projection& p, const Tx& tx, const Src& src,
                                                 uint32_t slot)
{
    const uint32_t* keys = src.keys(tx, slot);
    const uint32_t  k    = group_lower_bound(g, keys, tx.length, p.item);
    return make_group_tx(tx, src, slot, k < tx.length && keys[k] == p.item ? k + 1 : 0);
}

// Pass 1: local utilities of later items, and the itemset's utility. Each pass is a per-transaction
// body (`slot` is the transaction's position in its source step) plus a finish; the drivers below run
// the bodies over global memory and the staged kernels run them over shared-memory tiles.
template <typename Src>
__device__ inline void group_local_tx(const Projection& p, const TxGroup& g, uint32_t i, uint32_t slot, const Src& src,
                                      uint64_t& utility)
{
    const GroupTx t = open_group_tx(g, p, src.tx(i, slot), src, slot);
    if (g.rank == 0)
        p.pos[i] = t.pos;
    if (!t.found)
        return;
    uint32_t after_part = 0;
    for (uint32_t base = 0; base < t.rest; base += g.size)
        if (base + g.rank < t.rest)
            after_part += t.utils[t.k + 1 + base + g.rank];
    const uint64_t after = g.total(after_part);
    for (uint32_t base = 0; base < t.rest; base += g.size)
        if (base + g.rank < t.rest)
            add(p.local[t.keys[t.k + 1 + base + g.rank]], t.prefix + after);
    utility += t.prefix;
}

__device__ inline void group_local_finish(const TxGroup& g, uint64_t utility, TaskState& state)
{
    if (g.rank == 0 && utility != 0)
        add(state.utility, utility);
}

__device__ inline void group_scan_local(const Projection& p, uint32_t group_size, uint32_t from, uint32_t step, TaskState& state)
{
    const TxGroup      g(group_size);
    const GlobalSource src{p};
    uint64_t           utility = 0;
    for (uint32_t i = from + g.index; i < p.n_trans; i += step)
        group_local_tx(p, g, i, 0, src, utility);
    group_local_finish(g, utility, state);
}

// Kept entries of `a` equal those of `b` (both sorted, so equal sets are equal sequences): the counts
// match and every kept item of `a` is a key of `b`.
__device__ inline bool group_same_kept_items(const TxGroup& g, const Projection& p, const GroupTx& a, uint32_t a_kept,
                                             uint32_t other_index)
{
    const Tx        other  = p.txs[other_index];
    const uint32_t* bkey   = p.eitem + other.begin;
    uint32_t        b_kept = 0;
    for (uint32_t base = p.pos[other_index]; base < other.length; base += g.size) {
        const uint32_t idx = base + g.rank;
        b_kept += __popc(g.ballot(idx < other.length && p.rename[bkey[idx]] != kNone));
    }
    if (b_kept != a_kept)
        return false;
    for (uint32_t base = 0; base < a.rest; base += g.size) {
        const uint32_t pos  = base + g.rank;
        const uint32_t item = pos < a.rest ? a.keys[a.k + 1 + pos] : 0;
        const bool     keep = pos < a.rest && p.rename[item] != kNone;
        if (!g.all(!keep || find_item(bkey, other.length, item) >= 0))
            return false;
    }
    return true;
}

// Pass 2: subtree utilities of kept items, merging of identical projected transactions, and the
// exact size of the child database.
template <typename Src>
__device__ inline void group_subtree_tx(const Projection& p, const TxGroup& g, uint32_t i, uint32_t slot, const Src& src,
                                        ChildSize& size)
{
    {
        const uint32_t pos = p.pos[i];
        if (pos == 0) { // absent: a staged tile holds no copy
            if (g.rank == 0)
                p.rep[i] = kNone;
            return;
        }
        const GroupTx t = make_group_tx(src.tx(i, slot), src, slot, pos);
        // Sweep 1: kept count and kept utility.
        uint32_t kept      = 0;
        uint32_t kept_part = 0;
        for (uint32_t base = 0; base < t.rest; base += g.size) {
            const uint32_t pos  = base + g.rank;
            const bool     keep = pos < t.rest && p.rename[t.keys[t.k + 1 + pos]] != kNone;
            kept += __popc(g.ballot(keep));
            kept_part += keep ? t.utils[t.k + 1 + pos] : 0;
        }
        if (kept == 0) {
            if (g.rank == 0)
                p.rep[i] = kNone;
            return;
        }
        const uint64_t kept_sum = g.total(kept_part);
        // Sweep 2: subtree(z) = prefix + utility of z and every kept item after it; hash the sequence.
        const uint64_t remaining = t.prefix + kept_sum;
        uint64_t       before    = 0;
        uint32_t       index     = 0, hash = 0;
        for (uint32_t base = 0; base < t.rest; base += g.size) {
            const uint32_t pos  = base + g.rank;
            const uint32_t z    = pos < t.rest ? p.rename[t.keys[t.k + 1 + pos]] : kNone;
            const bool     keep = z != kNone;
            const uint32_t bits = g.ballot(keep);
            const uint32_t util = keep ? t.utils[t.k + 1 + pos] : 0;
            const uint32_t scan = g.inclusive_scan(util);
            if (keep) {
                add(p.subtree[z], remaining - (before + scan - util));
                hash ^= mix(z * 0x9e3779b9u + index + g.before(bits));
            }
            before += g.from_rank(scan, g.size - 1);
            index += __popc(bits);
        }
        hash         = g.xor_all(hash);
        size.max_len = max(size.max_len, kept);
        if (g.rank == 0)
            p.kept[i] = kept;
        for (uint32_t h = hash % p.table_size;; h = h + 1 == p.table_size ? 0 : h + 1) {
            uint32_t other = g.rank == 0 ? compare_swap(p.table[h], kNone, i) : 0;
            other          = g.from_rank(other, 0);
            if (other == kNone) {
                if (g.rank == 0)
                    p.rep[i] = i;
                ++size.trans;
                size.entries += kept;
                break;
            }
            if (group_same_kept_items(g, p, t, kept, other)) {
                if (g.rank == 0)
                    p.rep[i] = other;
                break;
            }
        }
    }
}

__device__ inline void group_subtree_finish(const TxGroup& g, const ChildSize& size, TaskState& state)
{
    if (g.rank == 0 && size.trans != 0) {
        add(state.child_trans, size.trans);
        add(state.child_entries, size.entries);
        fetch_max(state.child_max_len, size.max_len);
    }
}

__device__ inline void group_scan_subtree(const Projection& p, uint32_t group_size, uint32_t from, uint32_t step, TaskState& state)
{
    const TxGroup      g(group_size);
    const GlobalSource src{p};
    ChildSize          size;
    for (uint32_t i = from + g.index; i < p.n_trans; i += step)
        group_subtree_tx(p, g, i, 0, src, size);
    group_subtree_finish(g, size, state);
}

// Pass 3a: representatives write their child transaction and entries (coalesced stores).
template <typename Src>
__device__ inline void group_write_tx(const Projection& p, const TxGroup& g, uint32_t i, uint32_t slot, const Src& src,
                                      TaskState& state, const ChildView& c)
{
    if (p.rep[i] != i)
        return;
    {
        const Tx       tx   = src.tx(i, slot);
        const GroupTx  t    = make_group_tx(tx, src, slot, p.pos[i]);
        const uint32_t kept = p.kept[i];
        uint32_t begin = 0;
        if (g.rank == 0) {
            const uint32_t out_slot = fetch_add(state.tx_cursor, 1u);
            begin                   = fetch_add(state.entry_cursor, kept);
            p.slot_of[i]            = out_slot;
            c.txs[out_slot]         = Tx{begin, kept, tx.utility + t.utils[t.k]};
        }
        uint32_t out = g.from_rank(begin, 0);
        for (uint32_t base = 0; base < t.rest; base += g.size) {
            const uint32_t pos  = base + g.rank;
            const uint32_t z    = pos < t.rest ? p.rename[t.keys[t.k + 1 + pos]] : kNone;
            const bool     keep = z != kNone;
            const uint32_t bits = g.ballot(keep);
            if (keep) {
                const uint32_t at = out + g.before(bits);
                c.eitem[at]       = z;
                c.eutil[at]       = t.utils[t.k + 1 + pos];
            }
            out += __popc(bits);
        }
    }
}

__device__ inline void group_write_representatives(const Projection& p, uint32_t group_size, uint32_t from, uint32_t step,
                                                   TaskState& state, const ChildView& c)
{
    const TxGroup      g(group_size);
    const GlobalSource src{p};
    for (uint32_t i = from + g.index; i < p.n_trans; i += step)
        group_write_tx(p, g, i, 0, src, state, c);
}

// Pass 3b: duplicates add their utilities into their representative's child transaction.
template <typename Src>
__device__ inline void group_merge_tx(const Projection& p, const TxGroup& g, uint32_t i, uint32_t slot, const Src& src,
                                      const ChildView& c)
{
    const uint32_t r = p.rep[i];
    if (r == kNone || r == i)
        return;
    {
        const Tx       tx  = src.tx(i, slot);
        const GroupTx  t   = make_group_tx(tx, src, slot, p.pos[i]);
        const uint32_t dst = p.slot_of[r];
        if (g.rank == 0)
            add(c.txs[dst].utility, tx.utility + t.utils[t.k]);
        uint32_t* out = c.eutil + c.txs[dst].begin;
        for (uint32_t base = 0; base < t.rest; base += g.size) {
            const uint32_t pos  = base + g.rank;
            const bool     keep = pos < t.rest && p.rename[t.keys[t.k + 1 + pos]] != kNone;
            const uint32_t bits = g.ballot(keep);
            if (keep)
                add(out[g.before(bits)], t.utils[t.k + 1 + pos]);
            out += __popc(bits);
        }
    }
}

__device__ inline void group_merge_duplicates(const Projection& p, uint32_t group_size, uint32_t from, uint32_t step, const ChildView& c)
{
    const TxGroup      g(group_size);
    const GlobalSource src{p};
    for (uint32_t i = from + g.index; i < p.n_trans; i += step)
        group_merge_tx(p, g, i, 0, src, c);
}

// ---------------------------------------------------------------------------------------------
// Transaction modes: how a task's passes map transactions onto lanes. process_task and the huge-task
// kernels are templates over a mode. `unit` is the number of consecutive transactions one warp covers
// per step, so a warp starts at warp_index * unit and the whole group of warps strides by their sum.
// ---------------------------------------------------------------------------------------------

// No per-task context: the plain modes need none.
struct NoCtx
{
};

// Lane per transaction: a warp covers 32 consecutive transactions per step.
struct LaneTx
{
    using Ctx = NoCtx;
    static __device__ Ctx make_ctx(const BatchView&, const Projection&, uint32_t, uint32_t, uint32_t) { return {}; }
    static __device__ uint32_t unit(const Projection&, const Lanes& lanes) { return lanes.count; }
    static __device__ void scan_local(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step, TaskState& s,
                                      const Ctx&)
    {
        bfs::scan_local(p, lanes, from, step, s);
    }
    static __device__ void scan_subtree(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step, TaskState& s,
                                        const Ctx&)
    {
        bfs::scan_subtree(p, lanes, from, step, s);
    }
    static __device__ void write_representatives(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step,
                                                 TaskState& s, const ChildView& c, const Ctx&)
    {
        bfs::write_representatives(p, lanes, from, step, s, c);
    }
    static __device__ void merge_duplicates(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step,
                                            const ChildView& c, const Ctx&)
    {
        bfs::merge_duplicates(p, lanes, from, step, c);
    }
};

// Lane group per transaction: a warp covers 32 / group-size transactions per step. `Size` is
// kGroupMode (group size from the task's longest transaction, at run time) or a fixed group size, which
// the compiler can unroll the loops and masks for.
template <uint32_t Size> struct WarpTxS
{
    using Ctx = NoCtx;
    static __device__ uint32_t size(const Projection& p) { return Size > kGroupMode ? Size : tx_group_size(p.max_len); }
    static __device__ Ctx make_ctx(const BatchView&, const Projection&, uint32_t, uint32_t, uint32_t) { return {}; }
    static __device__ uint32_t unit(const Projection& p, const Lanes&) { return 32 / size(p); }
    static __device__ void scan_local(const Projection& p, const Lanes&, uint32_t from, uint32_t step, TaskState& s, const Ctx&)
    {
        group_scan_local(p, size(p), from, step, s);
    }
    static __device__ void scan_subtree(const Projection& p, const Lanes&, uint32_t from, uint32_t step, TaskState& s,
                                        const Ctx&)
    {
        group_scan_subtree(p, size(p), from, step, s);
    }
    static __device__ void write_representatives(const Projection& p, const Lanes&, uint32_t from, uint32_t step,
                                                 TaskState& s, const ChildView& c, const Ctx&)
    {
        group_write_representatives(p, size(p), from, step, s, c);
    }
    static __device__ void merge_duplicates(const Projection& p, const Lanes&, uint32_t from, uint32_t step,
                                            const ChildView& c, const Ctx&)
    {
        group_merge_duplicates(p, size(p), from, step, c);
    }
};

using WarpTx = WarpTxS<kGroupMode>;

// Calls `f` with a default-constructed transaction mode for ShapeTune::group `group`.
template <typename F> void with_mode(uint32_t group, F&& f)
{
    switch (group) {
    case 4: f(WarpTxS<4>{}); break;
    case 8: f(WarpTxS<8>{}); break;
    case 16: f(WarpTxS<16>{}); break;
    case 32: f(WarpTxS<32>{}); break;
    case kGroupMode: f(WarpTx{}); break;
    default: f(LaneTx{}); break;
    }
}

} // namespace ghuim::bfs
