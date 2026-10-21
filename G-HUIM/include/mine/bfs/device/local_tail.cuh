#pragma once

#include "mine/bfs/device/sources.cuh"
#include "mine/bfs/device/groups.cuh"
#include "mine/bfs/placement.cuh"

namespace ghuim::inline GHUIM_WIDTHS::bfs {


// A bounded cooperative DFS. Each lane owns one immutable parent transaction; the
// stack holds only that lane's prefix utility and matching position. Pending items
// are bitsets shared logically by all lanes. No descendant database is materialized.
template <typename Group>
__device__ __noinline__ void count_small_subtree(const Group& g, const BatchView& b, const TaskView& v)
{
    const Lanes lanes(g);
    const bool valid = g.rank() < v.p.n_trans;
    const Tx tx = valid ? GlobalSource{v.p}.tx(g.rank(), 0) : Tx{0, 0, 0};
    const Key* keys = v.p.eitem + tx.begin;
    const Util* utils = v.p.eutil + tx.begin;
    const uint32_t first = v.p.item + 1;
    Util utility[kMaxLocalTailItems + 1];
    int position[kMaxLocalTailItems + 1];
    uint32_t pending[kMaxLocalTailItems + 1];
    const uint32_t s = v.p.stride;
    position[0] = valid ? (v.p.occ ? static_cast<int>(v.p.occ[g.rank()].pos) : find_item_strided(keys, tx.length, v.p.item, s)) : -1;
    utility[0] = position[0] >= 0 ? tx.utility + utils[position[0] * s] : 0;
    uint64_t found = 0;
    uint32_t depth = 0;
    for (;;) {
        uint32_t suffix = 0;
        Util upper = utility[depth];
        if (position[depth] >= 0) {
            for (uint32_t pos = position[depth] + 1; pos < tx.length; ++pos) {
                suffix |= 1u << (keys[pos * s] - first);
                upper += utils[pos * s];
            }
        }
        const uint32_t candidates = __reduce_or_sync(lanes.mask, suffix);
        const Util total = lanes.sum(utility[depth]);
        const Util bound = lanes.sum(upper);
        // Transactions containing the whole remaining union also support every subset.
        // If their prefix utilities alone qualify, all 2^k completions qualify.
        const Util lower = lanes.sum(suffix == candidates ? utility[depth] : Util{0});
        pending[depth] = 0;
        if (lower >= b.minimum_utility)
            found += uint64_t{1} << __popc(candidates);
        else if (bound >= b.minimum_utility) {
            found += total >= b.minimum_utility;
            pending[depth] = candidates;
        }
        while (pending[depth] == 0 && depth != 0)
            --depth;
        if (pending[depth] == 0)
            break;
        const uint32_t item = first + (__ffs(pending[depth]) - 1);
        pending[depth] &= pending[depth] - 1;
        const int pos = position[depth] >= 0 ? find_item_strided(keys, tx.length, item, s) : -1;
        utility[depth + 1] = pos >= 0 ? utility[depth] + utils[pos * s] : 0;
        position[++depth] = pos;
    }
    if (g.rank() == 0 && found)
        add(b.control->patterns, found);
#ifdef GHUIM_WORK_STATS
    const uint32_t present = g.count(position[0] >= 0);
    if (g.rank() == 0) {
        add(b.control->work_total, static_cast<unsigned long long>(v.header.n_trans));
        add(b.control->work_present, static_cast<unsigned long long>(present));
    }
#endif
}

// A single (possibly merged) supporting transaction has the same support for every
// subset of its suffix. Count that subtree directly from the immutable parent entries.
// No child allocation, merging, item renaming or next-level tasks are needed. Utilities
// are nonnegative; when the prefix already qualifies, every remaining subset qualifies.
// The caller limits suffix length so partial-threshold enumeration remains bounded.
template <typename Group>
__device__ bool finish_local_tail(const Group& g, const BatchView& b, const TaskView& v)
{
    if (b.local_tail_items == 0 || v.p.n_trans > b.local_tail_trans)
        return false;
    // One transaction enumerates every subset of its suffix, so it takes the subset counter only while that
    // stays small; otherwise the pruned subtree walk handles it like several transactions.
    if (v.p.n_trans != 1 || v.header.max_len > min(b.local_tail_items, kMaxSubsetItems) + 1) {
        if (g.size() > 32 || v.p.n_trans > g.size() || v.header.n_items - v.p.item - 1 > b.local_tail_items)
            return false;
        count_small_subtree(g, b, v);
        return true;
    }
    if (g.rank() == 0) {
        const Tx tx = GlobalSource{v.p}.tx(0, 0);
        const Key* keys = v.p.eitem + tx.begin;
        const Util* utils = v.p.eutil + tx.begin;
        const uint32_t s = v.p.stride;
        const int pos = v.p.occ ? static_cast<int>(v.p.occ[0].pos) : find_item_strided(keys, tx.length, v.p.item, s);
        uint64_t count = 0;
        if (pos >= 0) {
            const uint32_t remaining = tx.length - pos - 1;
            const uint32_t subsets = 1u << remaining;
            Util utility = tx.utility + utils[pos * s];
            if (utility >= b.minimum_utility)
                count = subsets;
            else {
                Util upper = utility;
                for (uint32_t k = 0; k < remaining; ++k)
                    upper += utils[(pos + 1 + k) * s];
                if (upper >= b.minimum_utility) {
                    // Consecutive Gray-code subsets differ by exactly one entry.
                    uint32_t previous = 0;
                    for (uint32_t k = 1; k < subsets; ++k) {
                        const uint32_t next = k ^ (k >> 1);
                        const uint32_t changed = next ^ previous;
                        const Util value = utils[(pos + 1 + (__ffs(changed) - 1)) * s];
                        if (next & changed)
                            utility += value;
                        else
                            utility -= value;
                        count += utility >= b.minimum_utility;
                        previous = next;
                    }
                }
            }
        }
        if (count)
            add(b.control->patterns, count);
#ifdef GHUIM_WORK_STATS
        add(b.control->work_total, static_cast<unsigned long long>(v.header.n_trans));
        add(b.control->work_present, static_cast<unsigned long long>(pos >= 0));
#endif
    }
    return true;
}

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
