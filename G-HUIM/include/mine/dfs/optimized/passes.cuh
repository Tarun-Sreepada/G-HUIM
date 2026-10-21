#pragma once
// Transaction passes over the parent database (adaptive-optimized kernels).

#include "mine/dfs/optimized/groups.cuh"
#include "mine/dfs/optimized/projection.cuh"

namespace ghuim::inline GHUIM_WIDTHS::dfs::opt {

// Transaction passes. A caller covers transactions i0 = from, from + step, ...; lane `rank` of the
// group handles transaction i0 + rank. After the search, a lane's work is the entries behind the
// extension item: `rest` of them, starting at e[k + 1].

// Pass 1: local utilities of later items, and the itemset's utility.
__device__ inline void scan_local(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step, TaskState& state)
{
    Util utility = 0;
    for (uint32_t i0 = from; i0 < p.n_trans; i0 += step) {
        const uint32_t i      = i0 + lanes.rank;
        const Tx       tx     = i < p.n_trans ? p.txs[i] : Tx{0, 0, 0};
        const Entry*   e      = p.entries + tx.begin;
        const int      k      = find_item(e, tx.length, p.item);
        const uint32_t rest   = k >= 0 ? tx.length - k - 1 : 0;
        const uint32_t span   = lanes.max(rest);
        const Util     prefix = k >= 0 ? tx.utility + e[k].utility : 0;
        Util           after  = 0;
        for (uint32_t d = 0; d < span; ++d)
            if (d < rest)
                after += e[k + 1 + d].utility;
        for (uint32_t d = 0; d < span; ++d)
            if (d < rest)
                add(p.local[e[k + 1 + d].item], prefix + after);
        utility += prefix;
    }
    utility = lanes.sum(utility);
    if (lanes.leader() && utility != 0)
        add(state.utility, utility);
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

// Refinement: local utilities again over the kept items only, so items the rename dropped no longer
// count towards the others (see refine_items). Lane per transaction.
__device__ inline void scan_refine(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step)
{
    for (uint32_t i0 = from; i0 < p.n_trans; i0 += step) {
        const uint32_t i      = i0 + lanes.rank;
        const Tx       tx     = i < p.n_trans ? p.txs[i] : Tx{0, 0, 0};
        const Entry*   e      = p.entries + tx.begin;
        const int      k      = find_item(e, tx.length, p.item);
        const uint32_t rest   = k >= 0 ? tx.length - k - 1 : 0;
        const uint32_t span   = lanes.max(rest);
        const Util     prefix = k >= 0 ? tx.utility + e[k].utility : 0;
        Util           after  = 0;
        for (uint32_t d = 0; d < span; ++d)
            if (d < rest && p.rename[e[k + 1 + d].item] != kNone)
                after += e[k + 1 + d].utility;
        for (uint32_t d = 0; d < span; ++d)
            if (d < rest && p.rename[e[k + 1 + d].item] != kNone)
                add(p.local[e[k + 1 + d].item], prefix + after);
    }
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
// round runs only if the previous rename dropped an item. Returns the kept count. All threads of the
// group must call it.
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
// size of the child database.
__device__ inline void scan_subtree(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step, TaskState& state)
{
    ChildSize size;
    for (uint32_t i0 = from; i0 < p.n_trans; i0 += step) {
        const uint32_t i     = i0 + lanes.rank;
        const bool     valid = i < p.n_trans;
        const Tx       tx    = valid ? p.txs[i] : Tx{0, 0, 0};
        const Entry*   e     = p.entries + tx.begin;
        const int      k     = find_item(e, tx.length, p.item);
        const uint32_t rest  = k >= 0 ? tx.length - k - 1 : 0;
        const uint32_t span  = lanes.max(rest);
        // Sweep 1: kept count and kept utility.
        Util     kept_sum = 0;
        uint32_t kept     = 0;
        for (uint32_t d = 0; d < span; ++d)
            if (d < rest && p.rename[e[k + 1 + d].item] != kNone) {
                ++kept;
                kept_sum += e[k + 1 + d].utility;
            }
        // Sweep 2: subtree(z) = prefix + utility of z and every kept item after it; hash the sequence.
        const Util     remaining = (k >= 0 ? tx.utility + e[k].utility : 0) + kept_sum;
        Util           before    = 0;
        uint32_t       index = 0, hash = 0;
        for (uint32_t d = 0; d < span; ++d) {
            if (d >= rest)
                continue;
            const Entry&   entry = e[k + 1 + d];
            const uint32_t z     = p.rename[entry.item];
            if (z == kNone)
                continue;
            add(p.subtree[z], remaining - before);
            before += entry.utility;
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
                    if (same_kept_items(p.txs[other], p.entries, tx, p.item, p.rename)) {
                        p.rep[i] = other;
                        break;
                    }
                }
            }
        }
    }
    // Add the group's share of the child size into the task state.
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

} // namespace ghuim::inline GHUIM_WIDTHS::dfs::opt
