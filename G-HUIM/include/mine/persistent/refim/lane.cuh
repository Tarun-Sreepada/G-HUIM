#pragma once
// Lane mode: one thread mines the whole subtree of one primary item of a tiny database, sequentially, the way
// REFIM's worker does, with no barriers or atomics per node.
//
// A tiny database has at most 64 distinct items. They are renamed 0..m-1 in the same order, so a transaction is a
// 64-bit mask of its items plus their utilities in bit order: the items after x are mask & ~((2 << x) - 1), and a
// set of extensions is a mask. Each node: the itemset's utility and the local/subtree bounds of the items after it,
// REFIM's count_single, select, then a compacted child or REFIM's count_full. Children live in the thread's arena,
// a stack of levels popped with their DFS frames.
//
// Like REFIM's Level, every level carries an occurrence index of its extensions, so expanding item x visits only
// the transactions holding x, and knows x's position in each.
//
// Level layout (8-byte aligned, offsets from the arena start, all below 64 KiB): LaneLevel; per transaction a record
// (mask, prefix utility, the utilities of its items in bit order, padded to 8 bytes); then the index: begin[65] and
// the occurrences of local item b at occ[begin[b], begin[b + 1]), each record offset | position << 16.

#include "mine/persistent/refim/expand.cuh"

namespace ghuim::inline GHUIM_WIDTHS::refim {

constexpr uint32_t kLaneItems = 64;
constexpr uint64_t kLaneArena = 64 * 1024; // bytes per thread (record offsets must stay below 2^16)

struct LaneLevel
{
    uint32_t n_tx;
    uint32_t index; // offset of begin[65], then the occurrences
    uint64_t bytes;
};

__host__ __device__ inline uint64_t lane_record_bytes(uint32_t items)
{
    return 16 + ((uint64_t{items} * sizeof(Util) + 7) & ~uint64_t{7});
}

// Largest level a database of n_tx transactions and n_items entries can produce (its own conversion included:
// records, a transaction-to-record table, the index).
__host__ __device__ inline uint64_t lane_level_bound(uint64_t n_tx, uint64_t n_items)
{
    return sizeof(LaneLevel) + n_tx * 28 + n_items * (sizeof(Util) + 4) + 65 * 4 + 8;
}

__device__ inline uint64_t below(uint32_t x)
{
    return (uint64_t{1} << x) - 1;
}

__device__ inline uint64_t above(uint32_t x)
{
    return x >= 63 ? 0 : ~((uint64_t{2} << x) - 1);
}

struct LaneMiner
{
    const Settings& set;
    unsigned char*  arena;
    uint64_t        capacity;
    uint64_t        patterns = 0, nodes = 0, counted = 0, convert_cycles = 0, subtrees = 0;

    __device__ LaneLevel* level(uint64_t off) const
    {
        return reinterpret_cast<LaneLevel*>(arena + off);
    }
    __device__ uint32_t* index_of(uint64_t off) const
    {
        return reinterpret_cast<uint32_t*>(arena + level(off)->index);
    }

    // Converts D into level 0 (at offset 0), with an index holding only `x` (D's primary item k, local id x):
    // D's own occurrences of it, its positions unchanged. Item ids become positions in `items` (sorted, m).
    __device__ bool convert(const Db& D, uint32_t k, uint32_t x, const uint32_t* items, uint32_t m, uint64_t& top) const
    {
        if (!check(lane_level_bound(D.h->n_tx, D.h->n_items) <= capacity, set))
            return false;
        LaneLevel*     L      = level(0);
        unsigned char* p      = arena + sizeof(LaneLevel);
        uint32_t*      rec_of = nullptr; // record offset of each transaction, placed after the records
        for (uint32_t t = 0; t < D.h->n_tx; ++t) {
            const uint64_t at = D.tx_off[t], len = D.tx_len[t];
            if (!check(at + len <= D.h->n_items && len <= kLaneItems, set))
                return false;
            uint64_t mask  = 0;
            Util*    utils = reinterpret_cast<Util*>(p + 16);
            for (uint64_t i = 0; i < len; ++i) {
                // Binary search: items are sorted, and a transaction's items ascend, so bits come in order.
                const uint32_t y  = D.items[at + i];
                uint32_t       lo = 0, hi = m;
                while (lo < hi) {
                    const uint32_t mid = (lo + hi) / 2;
                    if (items[mid] < y)
                        lo = mid + 1;
                    else
                        hi = mid;
                }
                if (!check(lo < m && items[lo] == y, set))
                    return false;
                mask |= uint64_t{1} << lo;
                utils[i] = D.utils[at + i];
            }
            *reinterpret_cast<uint64_t*>(p)     = mask;
            *reinterpret_cast<uint64_t*>(p + 8) = D.prefix[t];
            p += lane_record_bytes(static_cast<uint32_t>(len));
        }
        // Record offsets, recomputed in a second walk (records are variable-length).
        rec_of      = reinterpret_cast<uint32_t*>(p);
        uint32_t at = sizeof(LaneLevel);
        for (uint32_t t = 0; t < D.h->n_tx; ++t) {
            rec_of[t] = at;
            at += static_cast<uint32_t>(lane_record_bytes(D.tx_len[t]));
        }
        uint32_t*      begin = rec_of + D.h->n_tx;
        uint32_t*      occ   = begin + kLaneItems + 1;
        const uint32_t ob = D.occ_begin[k], oe = D.occ_begin[k + 1];
        for (uint32_t b = 0; b <= kLaneItems; ++b)
            begin[b] = b <= x ? 0 : oe - ob;
        for (uint32_t o = ob; o < oe; ++o) {
            const uint32_t tid = D.occ[o] >> D.h->occ_shift, pos = D.occ[o] & ((1u << D.h->occ_shift) - 1);
            if (!check(tid < D.h->n_tx && pos < kLaneItems, set))
                return false;
            occ[o - ob] = rec_of[tid] | pos << 16;
        }
        L->n_tx  = D.h->n_tx;
        L->index = static_cast<uint32_t>(reinterpret_cast<unsigned char*>(begin) - arena);
        L->bytes = static_cast<uint64_t>(reinterpret_cast<unsigned char*>(occ + (oe - ob)) - arena);
        top      = (L->bytes + 7) & ~uint64_t{7};
        return check(top <= capacity, set);
    }

    // Expands item x of the level at `off`. Returns the child's extensions (0: nothing to descend into); the child
    // level is at `child`, and `top` past it.
    __device__ uint64_t expand_node(uint64_t off, uint32_t x, uint64_t& top, uint64_t& child)
    {
        const uint64_t     min_util = set.min_util;
        const uint32_t*    begin    = index_of(off);
        const uint32_t*    occ      = begin + kLaneItems + 1;
        const uint32_t     ob = begin[x], oe = begin[x + 1];
        unsigned long long loc[kLaneItems], sub[kLaneItems];
        uint64_t           seen = 0, single_mask = 0, single_curr = 0;
        const Util*        single_utils = nullptr;
        uint64_t           utility      = 0;
        uint32_t           nonempty     = 0;
        ++nodes;

        // The utility, and the bounds of the items after x: only the transactions holding x.
        for (uint32_t o = ob; o < oe; ++o) {
            const unsigned char* r     = arena + (occ[o] & 0xffff);
            const uint32_t       pos   = occ[o] >> 16;
            const uint64_t       mask  = *reinterpret_cast<const uint64_t*>(r);
            const Util*          utils = reinterpret_cast<const Util*>(r + 16);
            const uint64_t       curr  = *reinterpret_cast<const uint64_t*>(r + 8) + utils[pos];
            const uint64_t       sm    = mask & above(x);
            utility += curr;
            if (sm == 0)
                continue;
            ++nonempty;
            single_mask  = sm;
            single_curr  = curr;
            single_utils = utils + pos + 1;
            for (uint64_t fresh = sm & ~seen; fresh; fresh &= fresh - 1) // new items start at zero
                loc[__ffsll(fresh) - 1] = sub[__ffsll(fresh) - 1] = 0;
            seen |= sm;
            uint64_t suffix = 0;
            uint32_t idx    = __popcll(mask);
            for (uint64_t rest = sm; rest; rest &= ~(uint64_t{1} << (63 - __clzll(rest)))) {
                suffix += utils[--idx];
                sub[63 - __clzll(rest)] += curr + suffix;
            }
            for (uint64_t rest = sm; rest; rest &= rest - 1)
                loc[__ffsll(rest) - 1] += curr + suffix;
        }
        patterns += utility >= min_util;
        if (nonempty == 0)
            return 0;
        if (nonempty == 1) { // REFIM's count_single
            unsigned long long tail[kLaneItems];
            const uint32_t     n = __popcll(single_mask);
            for (uint32_t i = 0; i < n; ++i)
                tail[i] = single_utils[i];
            patterns += count_tail(tail, n, single_curr, min_util);
            ++counted;
            return 0;
        }
        // select, with REFIM's refinement: while some item is ruled out, recompute the kept items' bounds without the
        // ruled-out ones (the transactions holding x, restricted to the kept items) and select again.
        uint64_t sec = 0, prim = 0;
        for (int32_t round = 0;; ++round) {
            const uint64_t candidates = round == 0 ? seen : sec;
            sec = prim = 0;
            for (uint64_t rest = candidates; rest; rest &= rest - 1) {
                const uint32_t y = __ffsll(rest) - 1;
                if (loc[y] >= min_util) {
                    sec |= uint64_t{1} << y;
                    if (sub[y] >= min_util)
                        prim |= uint64_t{1} << y;
                }
            }
            if (sec == candidates || sec == 0 || (set.refine_rounds >= 0 && round >= set.refine_rounds))
                break;
            for (uint64_t rest = sec; rest; rest &= rest - 1)
                loc[__ffsll(rest) - 1] = sub[__ffsll(rest) - 1] = 0;
            for (uint32_t o = ob; o < oe; ++o) {
                const unsigned char* r     = arena + (occ[o] & 0xffff);
                const uint32_t       pos   = occ[o] >> 16;
                const uint64_t       mask  = *reinterpret_cast<const uint64_t*>(r);
                const Util*          utils = reinterpret_cast<const Util*>(r + 16);
                const uint64_t       sm    = mask & above(x);
                const uint64_t       cm    = sm & sec;
                if (cm == 0)
                    continue;
                const uint64_t curr   = *reinterpret_cast<const uint64_t*>(r + 8) + utils[pos];
                uint64_t       suffix = 0;
                uint32_t       idx    = __popcll(mask);
                for (uint64_t rest = sm; rest; rest &= ~(uint64_t{1} << (63 - __clzll(rest)))) {
                    const uint32_t y = 63 - __clzll(rest);
                    --idx;
                    if (cm >> y & 1) {
                        suffix += utils[idx];
                        sub[y] += curr + suffix;
                    }
                }
                for (uint64_t rest = cm; rest; rest &= rest - 1)
                    loc[__ffsll(rest) - 1] += curr + suffix;
            }
        }
        if (prim == 0)
            return 0;

        // The compacted child, and REFIM's count_full test.
        // The child's exact size first (records, then the index of its extensions), so writing it cannot overrun.
        const uint32_t n_sec = __popcll(sec);
        uint64_t       size  = sizeof(LaneLevel) + (kLaneItems + 1) * 4;
        for (uint32_t o = ob; o < oe; ++o) {
            const uint64_t cm = *reinterpret_cast<const uint64_t*>(arena + (occ[o] & 0xffff)) & above(x) & sec;
            if (cm != 0)
                size += lane_record_bytes(__popcll(cm)) + 4 * __popcll(cm & prim);
        }
        if (!check(top + size + 8 <= capacity, set))
            return 0;
        child                = top;
        LaneLevel*     C     = level(child);
        unsigned char* q     = arena + child + sizeof(LaneLevel);
        uint64_t       lower = 0;
        bool           all   = true;
        uint32_t       n_tx  = 0;
        for (uint32_t o = ob; o < oe; ++o) {
            const unsigned char* r     = arena + (occ[o] & 0xffff);
            const uint32_t       pos   = occ[o] >> 16;
            const uint64_t       mask  = *reinterpret_cast<const uint64_t*>(r);
            const Util*          utils = reinterpret_cast<const Util*>(r + 16);
            const uint64_t       cm    = mask & above(x) & sec;
            if (cm == 0)
                continue;
            const uint64_t curr = *reinterpret_cast<const uint64_t*>(r + 8) + utils[pos];
            Util*          out  = reinterpret_cast<Util*>(q + 16);
            uint32_t       w = 0, idx = pos + 1;
            for (uint64_t rest = mask & above(x); rest; rest &= rest - 1, ++idx)
                if (cm >> (__ffsll(rest) - 1) & 1)
                    out[w++] = utils[idx];
            *reinterpret_cast<uint64_t*>(q)     = cm;
            *reinterpret_cast<uint64_t*>(q + 8) = curr;
            q += lane_record_bytes(w);
            ++n_tx;
            if (cm == sec)
                lower += curr;
            else
                all = false;
        }
        if (lower >= min_util || all) { // REFIM's count_full
            if (lower >= min_util)
                patterns += (uint64_t{1} << n_sec) - 1;
            else {
                unsigned long long   tail[kLaneItems] = {};
                const unsigned char* r                = arena + child + sizeof(LaneLevel);
                for (uint32_t t = 0; t < n_tx; ++t) {
                    const Util* utils = reinterpret_cast<const Util*>(r + 16);
                    for (uint32_t i = 0; i < n_sec; ++i)
                        tail[i] += utils[i];
                    r += lane_record_bytes(n_sec);
                }
                patterns += count_tail(tail, n_sec, lower, min_util);
            }
            ++counted;
            return 0;
        }

        // The child's occurrence index for its extensions: count, prefix-sum, place.
        uint32_t* cbegin = reinterpret_cast<uint32_t*>(q);
        uint32_t* cocc   = cbegin + kLaneItems + 1;
        for (uint32_t b = 0; b <= kLaneItems; ++b)
            cbegin[b] = 0;
        const unsigned char* r = arena + child + sizeof(LaneLevel);
        for (uint32_t t = 0; t < n_tx; ++t) {
            const uint64_t cm = *reinterpret_cast<const uint64_t*>(r);
            for (uint64_t rest = cm & prim; rest; rest &= rest - 1)
                ++cbegin[__ffsll(rest)]; // counts at b + 1
            r += lane_record_bytes(__popcll(cm));
        }
        for (uint32_t b = 0; b < kLaneItems; ++b)
            cbegin[b + 1] += cbegin[b];
        uint32_t cursor[kLaneItems];
        for (uint32_t b = 0; b < kLaneItems; ++b)
            cursor[b] = cbegin[b];
        r = arena + child + sizeof(LaneLevel);
        for (uint32_t t = 0; t < n_tx; ++t) {
            const uint64_t cm = *reinterpret_cast<const uint64_t*>(r);
            const uint32_t at = static_cast<uint32_t>(r - arena);
            for (uint64_t rest = cm & prim; rest; rest &= rest - 1) {
                const uint32_t b  = __ffsll(rest) - 1;
                cocc[cursor[b]++] = at | __popcll(cm & below(b)) << 16;
            }
            r += lane_record_bytes(__popcll(cm));
        }
        C->n_tx  = n_tx;
        C->index = static_cast<uint32_t>(reinterpret_cast<unsigned char*>(cbegin) - arena);
        C->bytes = static_cast<uint64_t>(reinterpret_cast<unsigned char*>(cocc + cbegin[kLaneItems]) - arena) - child;
        top      = (child + C->bytes + 7) & ~uint64_t{7};
        if (!check(top <= capacity, set))
            return 0;
        return prim;
    }

    // Mines the subtree of D's primary item k: the itemset prefix + item and every extension of it.
    __device__ void mine(const Db& D, uint32_t k, const uint32_t* items, uint32_t m)
    {
        struct LaneFrame
        {
            uint64_t level, prim, free_to; // free_to: arena top to restore when the frame ends
        };
        // D's primary item k, as a local id.
        uint32_t lo = 0, hi = m;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if (items[mid] < D.primary[k])
                lo = mid + 1;
            else
                hi = mid;
        }
        if (!check(lo < m && items[lo] == D.primary[k], set))
            return;
        LaneFrame frames[kLaneItems + 1];
        uint64_t       top = 0;
        const uint64_t c0  = clock64();
        ++subtrees;
        const bool converted = convert(D, k, lo, items, m, top);
        convert_cycles += clock64() - c0;
        if (!converted)
            return;
        uint32_t nf  = 0;
        frames[nf++] = {0, uint64_t{1} << lo, 0};
        while (nf != 0) {
            LaneFrame& f = frames[nf - 1];
            if (f.prim == 0) {
                top = f.free_to;
                --nf;
                continue;
            }
            const uint32_t x = __ffsll(f.prim) - 1;
            f.prim &= f.prim - 1;
            const uint64_t before = top;
            uint64_t       child  = 0;
            const uint64_t prim   = expand_node(f.level, x, top, child);
            if (prim != 0 && check(nf <= kLaneItems, set))
                frames[nf++] = {child, prim, before};
            else
                top = before;
        }
    }
};

// Lane 0 of a warp: the sorted distinct items of D in `items` (at most kLaneItems); false if there are more, or if
// a full root-to-leaf path of lane levels might not fit in the arena.
__device__ inline bool lane_eligible(const Db& D, uint32_t* items, uint32_t& m)
{
    const uint64_t bound = lane_level_bound(D.h->n_tx, D.h->n_items);
    if (bound * 2 > kLaneArena)
        return false;
    m = 0;
    for (uint64_t i = 0; i < D.h->n_items; ++i) {
        const uint32_t y  = D.items[i];
        uint32_t       lo = 0, hi = m;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if (items[mid] < y)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo < m && items[lo] == y)
            continue;
        if (m == kLaneItems)
            return false;
        for (uint32_t j = m; j > lo; --j)
            items[j] = items[j - 1];
        items[lo] = y;
        ++m;
    }
    // Every level along a path is at most the root's size, and a path has at most m + 1 levels.
    return (uint64_t{m} + 1) * bound <= kLaneArena;
}

} // namespace ghuim::inline GHUIM_WIDTHS::refim
