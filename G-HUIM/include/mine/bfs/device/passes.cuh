#pragma once
// Transaction passes of one task: local utilities, item renaming, subtree utilities and merging.

#include "mine/bfs/device/groups.cuh"
#include "mine/bfs/device/sources.cuh"
#include "mine/bfs/device/task_view.cuh"

namespace ghuim::inline GHUIM_WIDTHS::bfs {

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
    Util     utility = 0;
    uint64_t present = 0;
};

template <typename Src>
__device__ inline void local_tile(const Projection& p, const Lanes& lanes, uint32_t i0, const Src& src, LocalAcc& acc)
{
    const uint32_t  i      = i0 + lanes.rank;
    const Tx        tx     = i < p.n_trans ? src.tx(i, lanes.rank) : Tx{0, 0, 0};
    const auto*     k_item = src.keys(tx, lanes.rank);
    const Util*     k_util = src.utils(tx, lanes.rank);
    const int       k      = p.occ ? (i < p.n_trans ? static_cast<int>(p.occ[i].pos) : -1)
                                  : find_item(k_item, tx.length, p.item);
    if (i < p.n_trans)
        p.pos[i] = static_cast<uint32_t>(k + 1);
    const uint32_t  rest   = k >= 0 ? tx.length - k - 1 : 0;
    const uint32_t  span   = lanes.max(rest);
    const Util      prefix = k >= 0 ? tx.utility + k_util[k] : 0;
    Util            after  = 0;
    for (uint32_t d = 0; d < span; ++d)
        if (d < rest)
            after += k_util[k + 1 + d];
    for (uint32_t d = 0; d < span; ++d)
        if (d < rest)
            add(p.local[k_item[k + 1 + d]], prefix + after);
    acc.utility += prefix;
    acc.present += k >= 0;
}

__device__ inline void local_finish(const Lanes& lanes, LocalAcc& acc, TaskState& state)
{
#ifdef GHUIM_WORK_STATS
    const uint64_t present = lanes.sum(acc.present);
    if (lanes.leader() && present != 0)
        add(state.present, present);
#endif
    const Util     utility = lanes.sum(acc.utility);
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

// Refinement: items the rename dropped still count in every other item's local utility. Recomputing
// local utilities over the kept items only can push more items below the threshold, and so on until
// none drops. A pass reads each transaction from the position pass 1 recorded, lane per transaction,
// in either layout (entry d at begin + d * stride).
__device__ inline void refine_tile(const Projection& p, const Lanes& lanes, uint32_t i0)
{
    const uint32_t  i      = i0 + lanes.rank;
    const uint32_t  pos    = i < p.n_trans ? p.pos[i] : 0;
    const Tx        tx     = pos != 0 ? p.txs[p.occ ? p.occ[i].tid : i] : Tx{0, 0, 0};
    const Key*      k_item = p.eitem + tx.begin;
    const Util*     k_util = p.eutil + tx.begin;
    const uint32_t  s      = p.stride;
    const uint32_t  rest   = pos != 0 ? tx.length - pos : 0;
    const uint32_t  span   = lanes.max(rest);
    const Util      prefix = pos != 0 ? tx.utility + k_util[(pos - 1) * s] : 0;
    Util            after  = 0;
    for (uint32_t d = 0; d < span; ++d)
        if (d < rest && p.rename[k_item[(pos + d) * s]] != kNone)
            after += k_util[(pos + d) * s];
    for (uint32_t d = 0; d < span; ++d)
        if (d < rest && p.rename[k_item[(pos + d) * s]] != kNone)
            add(p.local[k_item[(pos + d) * s]], prefix + after);
}

__device__ inline void scan_refine(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step)
{
    for (uint32_t i0 = from; i0 < p.n_trans; i0 += step)
        refine_tile(p, lanes, i0);
}

// Zeroes the local utilities of the kept items before a refinement pass.
__device__ inline void clear_kept_local(const Projection& p, uint32_t first, uint32_t n, uint32_t rank, uint32_t stride)
{
    for (uint32_t i = first + rank; i < n; i += stride)
        if (p.rename[i] != kNone)
            p.local[i] = 0;
}

// rename_items over the kept items: keeps those whose refined local utility meets the threshold.
template <typename Group>
__device__ uint32_t refine_rename(const Group& g, const Projection& p, uint32_t first, uint32_t n, uint64_t minutil)
{
    uint32_t kept_items = 0;
    for (uint32_t chunk = first; chunk < n; chunk += g.size()) {
        const uint32_t i    = chunk + g.rank();
        const bool     keep = i < n && p.rename[i] != kNone && p.local[i] >= minutil;
        const uint32_t scan = g.scan(keep);
        if (i < n)
            p.rename[i] = keep ? kept_items + scan - 1 : kNone;
        kept_items += g.count(keep);
    }
    return kept_items;
}

// Up to `rounds` refinement rounds (-1: until none drops) after rename_items kept `kept` items; a
// round runs only if the previous rename dropped an item. Lane-per-transaction coverage from/step.
// Returns the kept count. All threads of the group must call it.
template <typename Group>
__device__ uint32_t refine_items(const Group& g, const Lanes& lanes, const Projection& p, uint32_t first, uint32_t n,
                                 uint32_t kept, uint64_t minutil, int32_t rounds, uint32_t from, uint32_t step)
{
    for (int32_t r = 0; kept != 0 && kept < n - first && (rounds < 0 || r < rounds); ++r) {
        g.sync();
        clear_kept_local(p, first, n, g.rank(), g.size());
        g.sync();
        scan_refine(p, lanes, from, step);
        g.sync();
        const uint32_t next = refine_rename(g, p, first, n, minutil);
        if (next == kept)
            break;
        kept = next;
    }
    return kept;
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
    const uint32_t  pos    = valid ? p.pos[i] : 0;
    const Tx        tx     = pos != 0 ? src.tx(i, lanes.rank) : Tx{0, 0, 0}; // absent: a staged tile holds no copy
    const auto*     k_item = src.keys(tx, lanes.rank);
    const Util*     k_util = src.utils(tx, lanes.rank);
    const int       k      = static_cast<int>(pos) - 1;
    const uint32_t  rest   = k >= 0 ? tx.length - k - 1 : 0;
    const uint32_t  span   = lanes.max(rest);
    // One backward sweep: subtree(z) = prefix + utility of z and every kept item after it, the kept count,
    // and a hash of the kept sequence (positions counted from its end; equal sequences hash equally).
    const Util     prefix = k >= 0 ? tx.utility + k_util[k] : 0;
    Util           after  = 0;
    uint32_t       kept = 0, hash = 0;
    for (uint32_t d = span; d-- > 0;) {
        if (d >= rest)
            continue;
        const uint32_t z = p.rename[k_item[k + 1 + d]];
        if (z == kNone)
            continue;
        after += k_util[k + 1 + d];
        add(p.subtree[z], prefix + after);
        hash ^= mix(z * 0x9e3779b9u + kept++);
    }
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
                    break;
                }
                const Tx o = p.txs[p.occ ? p.occ[other].tid : other];
                if (same_kept_items(p.eitem + o.begin, o.length, p.pos[other], k_item, tx.length, pos, p.rename)) {
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
    return i < p.n_trans && p.pos[i] != 0;
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

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
