#pragma once
// Transaction passes of one task: local utilities, item renaming, subtree utilities and merging.

#include "mine/bfs/device/groups.cuh"
#include "mine/bfs/device/sources.cuh"
#include "mine/bfs/device/task_view.cuh"

namespace ghuim::bfs {

// ---------------------------------------------------------------------------------------------
// Transaction passes. A caller covers transactions i0 = from, from + step, ...; lane `rank` of the
// group handles transaction i0 + rank. After the search, a lane's work is the entries behind the
// extension item: `rest` of them, starting at e[k + 1].
// ---------------------------------------------------------------------------------------------

// Pass 1: local utilities of later items, and the itemset's utility. A pass is a `tile` body (one
// 32-transaction step: lane `rank` takes transaction i0 + rank) plus a `finish`; the drivers below
// run the bodies over global memory, and the staged kernels run them over shared-memory tiles.
struct LocalAcc
{
    uint64_t utility = 0;
};

template <typename Src>
__device__ inline void local_tile(const Projection& p, const Lanes& lanes, uint32_t i0, const Src& src, LocalAcc& acc)
{
    const uint32_t  i      = i0 + lanes.rank;
    const Tx        tx     = i < p.n_trans ? src.tx(i, lanes.rank) : Tx{0, 0, 0};
    const uint32_t* k_item = src.keys(tx, lanes.rank);
    const uint32_t* k_util = src.utils(tx, lanes.rank);
    const int       k      = find_item(k_item, tx.length, p.item);
    const uint32_t  rest   = k >= 0 ? tx.length - k - 1 : 0;
    const uint32_t  span   = lanes.max(rest);
    const uint64_t  prefix = k >= 0 ? uint64_t{tx.utility} + k_util[k] : 0;
    uint64_t        after  = 0;
    for (uint32_t d = 0; d < span; ++d)
        if (d < rest)
            after += k_util[k + 1 + d];
    for (uint32_t d = 0; d < span; ++d)
        if (d < rest)
            add(p.local[k_item[k + 1 + d]], prefix + after);
    acc.utility += prefix;
}

__device__ inline void local_finish(const Lanes& lanes, LocalAcc& acc, TaskState& state)
{
    const uint64_t utility = lanes.sum(acc.utility);
    if (lanes.leader() && utility != 0)
        add(state.utility, utility);
}

// Transactions a pass reads, for the staging IO warp (it copies only these).
__device__ inline bool local_wants(const Projection& p, uint32_t i)
{
    return i < p.n_trans;
}

__device__ inline void scan_local(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step,
                                  TaskState& state)
{
    const GlobalSource src{p};
    LocalAcc           acc;
    for (uint32_t i0 = from; i0 < p.n_trans; i0 += step)
        local_tile(p, lanes, i0, src, acc);
    local_finish(lanes, acc, state);
}

// Keeps the items after the extension item whose local utility meets the threshold and renames
// them 0, 1, ... (group scan); returns the kept count.
template <typename Group>
__device__ uint32_t rename_items(const Group& g, const Projection& p, uint32_t first, uint32_t n, uint64_t minutil)
{
    uint32_t kept_items = 0;
    for (uint32_t chunk = first; chunk < n; chunk += g.size()) {
        const uint32_t i    = chunk + g.rank();
        const bool     keep = i < n && p.local[i] >= minutil;
        const uint32_t scan = g.scan(keep);
        if (i < n)
            p.rename[i] = keep ? kept_items + scan - 1 : kNone;
        kept_items += g.count(keep);
    }
    return kept_items;
}

struct ChildSize
{
    uint32_t trans = 0, entries = 0, max_len = 0;
};

// Pass 2: subtree utilities of kept items, merging of identical projected transactions, and the
// exact size of the child database.
template <typename Src>
__device__ inline void subtree_tile(const Projection& p, const Lanes& lanes, uint32_t i0, const Src& src, ChildSize& size)
{
    const uint32_t  i      = i0 + lanes.rank;
    const bool      valid  = i < p.n_trans;
    const Tx        tx     = valid ? src.tx(i, lanes.rank) : Tx{0, 0, 0};
    const uint32_t* k_item = src.keys(tx, lanes.rank);
    const uint32_t* k_util = src.utils(tx, lanes.rank);
    const int       k      = find_item(k_item, tx.length, p.item);
    const uint32_t  rest   = k >= 0 ? tx.length - k - 1 : 0;
    const uint32_t  span   = lanes.max(rest);
    // Sweep 1: kept count and kept utility.
    uint64_t kept_sum = 0;
    uint32_t kept     = 0;
    for (uint32_t d = 0; d < span; ++d)
        if (d < rest && p.rename[k_item[k + 1 + d]] != kNone) {
            ++kept;
            kept_sum += k_util[k + 1 + d];
        }
    // Sweep 2: subtree(z) = prefix + utility of z and every kept item after it; hash the sequence.
    const uint64_t remaining = (k >= 0 ? uint64_t{tx.utility} + k_util[k] : 0) + kept_sum;
    uint64_t       before    = 0;
    uint32_t       index = 0, hash = 0;
    for (uint32_t d = 0; d < span; ++d) {
        if (d >= rest)
            continue;
        const uint32_t item_d = k_item[k + 1 + d], util_d = k_util[k + 1 + d];
        const uint32_t z      = p.rename[item_d];
        if (z == kNone)
            continue;
        add(p.subtree[z], remaining - before);
        before += util_d;
        hash ^= mix(z * 0x9e3779b9u + index++);
    }
    if (valid) {
        if (kept == 0)
            p.rep[i] = kNone;
        else {
            size.max_len = max(size.max_len, kept);
            for (uint32_t h = hash % p.table_size;; h = h + 1 == p.table_size ? 0 : h + 1) {
                const uint32_t other = compare_swap(p.table[h], kNone, i);
                if (other == kNone) {
                    p.rep[i] = i;
                    ++size.trans;
                    size.entries += kept;
                    break;
                }
                const Tx o = p.txs[other];
                if (same_kept_items(p.eitem + o.begin, o.length, k_item, tx.length, p.item, p.rename)) {
                    p.rep[i] = other;
                    break;
                }
            }
        }
    }
}

__device__ inline void subtree_finish(const Lanes& lanes, ChildSize& size, TaskState& state)
{
    size.trans   = lanes.sum(size.trans);
    size.entries = lanes.sum(size.entries);
    for (uint32_t offset = lanes.count >> 1; offset > 0; offset >>= 1)
        size.max_len = max(size.max_len, __shfl_xor_sync(lanes.mask, size.max_len, offset, lanes.count));
    if (lanes.leader() && size.trans != 0) {
        add(state.child_trans, size.trans);
        add(state.child_entries, size.entries);
        fetch_max(state.child_max_len, size.max_len);
    }
}

__device__ inline bool subtree_wants(const Projection& p, uint32_t i)
{
    return i < p.n_trans;
}

__device__ inline void scan_subtree(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step,
                                    TaskState& state)
{
    const GlobalSource src{p};
    ChildSize          size;
    for (uint32_t i0 = from; i0 < p.n_trans; i0 += step)
        subtree_tile(p, lanes, i0, src, size);
    subtree_finish(lanes, size, state);
}

} // namespace ghuim::bfs
