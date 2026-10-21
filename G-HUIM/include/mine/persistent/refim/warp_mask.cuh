#pragma once
// Warp mask mode: a warp mines the subtrees of a tiny database's primary items one after another, over lane mode's
// representation (a transaction is a 64-bit mask of its items plus their utilities in bit order), with the items
// spread over the lanes: lane j owns items j and j + 32 and keeps their local and subtree bounds in registers.
// Every lane takes the same branches: a transaction is read once by the whole warp (its mask and prefix broadcast,
// each lane loading its own items' utilities) and suffix sums are shuffles. The same search as LaneMiner, node for
// node, without lane mode's divergent per-thread search; the database is converted once for all the siblings.
//
// Level layout (8-byte aligned, offsets from the warp's arena): LaneLevel; per transaction a record (mask, prefix
// utility, the utilities of its items in bit order, padded to 8 bytes); then the index: begin[65] and the
// occurrences of local item b at occ[begin[b], begin[b + 1]), each the record's offset. Level 0 indexes every item,
// a child only its extensions.

#include "mine/persistent/refim/lane.cuh"

namespace ghuim::inline GHUIM_WIDTHS::refim {

struct WarpMaskMiner
{
    const Settings&     set;
    unsigned char*      arena;    // the warp's
    uint64_t            capacity; // bytes for levels
    unsigned long long* tail;     // kLaneItems values for count_tail (lane 0), after the levels
    uint64_t            base_top = 0;
    uint64_t            patterns = 0, nodes = 0, counted = 0; // lane 0's are the warp's

    __device__ static uint32_t lane()
    {
        return threadIdx.x & 31;
    }

    // Sum over lanes >= this lane.
    __device__ static uint64_t suffix_scan(uint64_t v)
    {
        for (uint32_t d = 1; d < 32; d <<= 1) {
            const uint64_t y = __shfl_down_sync(~0u, v, d);
            if (lane() + d < 32)
                v += y;
        }
        return v;
    }

    // Sum over lanes < this lane.
    __device__ static uint32_t exclusive_scan(uint32_t v)
    {
        uint32_t x = v;
        for (uint32_t d = 1; d < 32; d <<= 1) {
            const uint32_t y = __shfl_up_sync(~0u, x, d);
            if (lane() >= d)
                x += y;
        }
        return x - v;
    }

    __device__ static uint64_t ballot64(bool lo, bool hi)
    {
        return uint64_t{__ballot_sync(~0u, lo)} | uint64_t{__ballot_sync(~0u, hi)} << 32;
    }

    __device__ static uint32_t local_id(const uint32_t* items, uint32_t m, uint32_t y)
    {
        uint32_t lo = 0, hi = m;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if (items[mid] < y)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    }

    __device__ uint64_t mask_at(uint64_t off) const
    {
        return *reinterpret_cast<const uint64_t*>(arena + off);
    }
    __device__ uint64_t prefix_at(uint64_t off) const
    {
        return *reinterpret_cast<const uint64_t*>(arena + off + 8);
    }
    __device__ const Util* utils_at(uint64_t off) const
    {
        return reinterpret_cast<const Util*>(arena + off + 16);
    }

    // Item y's utility in the record at `off` with mask `mask` (y in mask).
    __device__ uint64_t util_of(uint64_t off, uint64_t mask, uint32_t y) const
    {
        return utils_at(off)[__popcll(mask & below(y))];
    }

    // Every lane: converts D into level 0 (at offset 0), indexing all its items. Item ids become positions in
    // `items` (sorted, m). False (the same in every lane) when D does not fit or is inconsistent.
    __device__ bool prepare(const Db& D, const uint32_t* items, uint32_t m)
    {
        if (!check(lane_level_bound(D.h->n_tx, D.h->n_items) <= capacity, set))
            return false;
        const uint32_t l    = lane();
        uint64_t       off  = sizeof(LaneLevel);
        uint32_t       cnt0 = 0, cnt1 = 0;
        for (uint32_t t = 0; t < D.h->n_tx; ++t) {
            const uint64_t at = D.tx_off[t], len = D.tx_len[t];
            if (!check(at + len <= D.h->n_items && len <= kLaneItems, set))
                return false;
            Util*    utils = reinterpret_cast<Util*>(arena + off + 16);
            uint64_t bits  = 0;
            bool     ok    = true;
            for (uint32_t i = l; i < len; i += 32) {
                const uint32_t y = D.items[at + i], lo = local_id(items, m, y);
                ok               = ok && lo < m && items[lo] == y;
                bits |= lo < 64 ? uint64_t{1} << lo : 0;
                utils[i] = D.utils[at + i];
            }
            if (!check(__all_sync(~0u, ok), set))
                return false;
            const uint64_t mask = uint64_t{__reduce_or_sync(~0u, static_cast<uint32_t>(bits))} |
                                  uint64_t{__reduce_or_sync(~0u, static_cast<uint32_t>(bits >> 32))} << 32;
            if (l == 0) {
                *reinterpret_cast<uint64_t*>(arena + off)     = mask;
                *reinterpret_cast<uint64_t*>(arena + off + 8) = D.prefix[t];
            }
            cnt0 += mask >> l & 1;
            cnt1 += mask >> (l + 32) & 1;
            off += lane_record_bytes(static_cast<uint32_t>(len));
        }
        uint32_t* const begin = reinterpret_cast<uint32_t*>(arena + off);
        uint32_t* const occ   = begin + kLaneItems + 1;
        const uint32_t  e0    = exclusive_scan(cnt0);
        const uint32_t  tot0  = __shfl_sync(~0u, e0 + cnt0, 31);
        const uint32_t  e1    = tot0 + exclusive_scan(cnt1);
        const uint32_t  tot   = __shfl_sync(~0u, e1 + cnt1, 31);
        begin[l]              = e0;
        begin[l + 32]         = e1;
        if (l == 0)
            begin[kLaneItems] = tot;
        __syncwarp();
        uint32_t c0 = e0, c1 = e1;
        for (uint64_t r = sizeof(LaneLevel); r < off;) {
            const uint64_t mask = mask_at(r);
            if (mask >> l & 1)
                occ[c0++] = static_cast<uint32_t>(r);
            if (mask >> (l + 32) & 1)
                occ[c1++] = static_cast<uint32_t>(r);
            r += lane_record_bytes(__popcll(mask));
        }
        if (l == 0) {
            LaneLevel* L = reinterpret_cast<LaneLevel*>(arena);
            L->n_tx      = D.h->n_tx;
            L->index     = static_cast<uint32_t>(off);
            L->bytes     = off + (kLaneItems + 1 + tot) * 4;
        }
        base_top = (off + (kLaneItems + 1 + tot) * 4 + 7) & ~uint64_t{7};
        __syncwarp();
        return check(base_top <= capacity, set);
    }

    // Every lane: f(record offset, mask, curr) for each occurrence of item x in occ[ob, oe) (curr: the record's
    // prefix plus x's utility), in order.
    template <typename F> __device__ void for_each_occ(const uint32_t* occ, uint32_t ob, uint32_t oe, uint32_t x, F&& f) const
    {
        for (uint32_t o = ob; o < oe; ++o) {
            const uint64_t r = occ[o], mask = mask_at(r);
            f(r, mask, prefix_at(r) + util_of(r, mask, x));
        }
    }

    // Adds a transaction's contribution to the lane's items in `cm` (its kept items after x): local bound curr +
    // all of them, subtree bound curr + those from the item on.
    __device__ void add_bounds(uint64_t off, uint64_t mask, uint64_t cm, uint64_t curr, uint64_t& loc0,
                               uint64_t& loc1, uint64_t& sub0, uint64_t& sub1) const
    {
        const uint32_t l   = lane();
        const bool     in0 = cm >> l & 1, in1 = cm >> (l + 32) & 1;
        const uint64_t u0 = in0 ? util_of(off, mask, l) : 0, u1 = in1 ? util_of(off, mask, l + 32) : 0;
        const uint64_t s1 = suffix_scan(u1);
        const uint64_t s0 = suffix_scan(u0) + __shfl_sync(~0u, s1, 0);
        const uint64_t all = __shfl_sync(~0u, s0, 0);
        if (in0) {
            sub0 += curr + s0;
            loc0 += curr + all;
        }
        if (in1) {
            sub1 += curr + s1;
            loc1 += curr + all;
        }
    }

    // Expands item x of the level at `off` (LaneMiner::expand_node). Returns the child's extensions (0: nothing to
    // descend into); the child level is at `child`, and `top` past it. Every lane returns the same.
    __device__ uint64_t expand_node(uint64_t off, uint32_t x, uint64_t& top, uint64_t& child)
    {
        const uint64_t  min_util = set.min_util;
        const uint32_t  l        = lane();
        const uint32_t* begin    = reinterpret_cast<const uint32_t*>(arena + reinterpret_cast<LaneLevel*>(arena + off)->index);
        const uint32_t* occ      = begin + kLaneItems + 1;
        const uint32_t  ob = begin[x], oe = begin[x + 1];
        uint64_t        loc0 = 0, loc1 = 0, sub0 = 0, sub1 = 0;
        uint64_t        seen = 0, utility = 0, single = 0, single_curr = 0;
        uint32_t        nonempty = 0;
        ++nodes;

        // The utility, and the bounds of the items after x: only the transactions holding x.
        for_each_occ(occ, ob, oe, x, [&](uint64_t r, uint64_t mask, uint64_t curr) {
            const uint64_t sm = mask & above(x);
            utility += curr;
            if (sm == 0)
                return;
            ++nonempty;
            single      = r;
            single_curr = curr;
            seen |= sm;
            add_bounds(r, mask, sm, curr, loc0, loc1, sub0, sub1);
        });
        patterns += utility >= min_util;
        if (nonempty == 0)
            return 0;
        if (nonempty == 1) { // REFIM's count_single
            const uint64_t mask = mask_at(single);
            const uint32_t n = __popcll(mask & above(x)), first = __popcll(mask & below(x)) + 1;
            if (l == 0) {
                for (uint32_t i = 0; i < n; ++i)
                    tail[i] = utils_at(single)[first + i];
                patterns += count_tail(tail, n, single_curr, min_util);
            }
            ++counted;
            __syncwarp();
            return 0;
        }
        // select, with REFIM's refinement (LaneMiner's rounds).
        uint64_t sec = 0, prim = 0;
        for (int32_t round = 0;; ++round) {
            const uint64_t candidates = round == 0 ? seen : sec;
            const bool     k0 = (candidates >> l & 1) && loc0 >= min_util;
            const bool     k1 = (candidates >> (l + 32) & 1) && loc1 >= min_util;
            sec               = ballot64(k0, k1);
            prim              = ballot64(k0 && sub0 >= min_util, k1 && sub1 >= min_util);
            if (sec == candidates || sec == 0 || (set.refine_rounds >= 0 && round >= set.refine_rounds))
                break;
            loc0 = loc1 = sub0 = sub1 = 0;
            for_each_occ(occ, ob, oe, x, [&](uint64_t r, uint64_t mask, uint64_t curr) {
                const uint64_t cm = mask & above(x) & sec;
                if (cm != 0)
                    add_bounds(r, mask, cm, curr, loc0, loc1, sub0, sub1);
            });
        }
        if (prim == 0)
            return 0;

        // The child's size, its extensions' occurrence counts, and REFIM's count_full test.
        const uint32_t n_sec = __popcll(sec);
        uint64_t       records = 0, lower = 0, tail0 = 0, tail1 = 0;
        uint32_t       n_tx = 0, cnt0 = 0, cnt1 = 0;
        bool           all = true;
        for_each_occ(occ, ob, oe, x, [&](uint64_t r, uint64_t mask, uint64_t curr) {
            const uint64_t cm = mask & above(x) & sec;
            if (cm == 0)
                return;
            records += lane_record_bytes(__popcll(cm));
            ++n_tx;
            cnt0 += (cm & prim) >> l & 1;
            cnt1 += (cm & prim) >> (l + 32) & 1;
            if (cm == sec) {
                lower += curr;
                if (cm >> l & 1)
                    tail0 += util_of(r, mask, l);
                if (cm >> (l + 32) & 1)
                    tail1 += util_of(r, mask, l + 32);
            }
            else
                all = false;
        });
        const uint32_t e0   = exclusive_scan(cnt0);
        const uint32_t tot0 = __shfl_sync(~0u, e0 + cnt0, 31);
        const uint32_t e1   = tot0 + exclusive_scan(cnt1);
        const uint32_t tot  = __shfl_sync(~0u, e1 + cnt1, 31);
        const uint64_t size = sizeof(LaneLevel) + records + (kLaneItems + 1 + uint64_t{tot}) * 4;
        if (!check(top + size + 8 <= capacity, set))
            return 0;
        if (lower >= min_util || all) { // REFIM's count_full
            if (lower >= min_util)
                patterns += (uint64_t{1} << n_sec) - 1;
            else {
                if (sec >> l & 1)
                    tail[__popcll(sec & below(l))] = tail0;
                if (sec >> (l + 32) & 1)
                    tail[__popcll(sec & below(l + 32))] = tail1;
                __syncwarp();
                if (l == 0)
                    patterns += count_tail(tail, n_sec, lower, min_util);
                __syncwarp();
            }
            ++counted;
            return 0;
        }

        // The compacted child: records, then the index of its extensions.
        child                  = top;
        const uint64_t index   = child + sizeof(LaneLevel) + records;
        uint32_t* const cbegin = reinterpret_cast<uint32_t*>(arena + index);
        uint32_t* const cocc   = cbegin + kLaneItems + 1;
        cbegin[l]              = e0;
        cbegin[l + 32]         = e1;
        if (l == 0)
            cbegin[kLaneItems] = tot;
        uint32_t c0 = e0, c1 = e1;
        uint64_t q  = child + sizeof(LaneLevel);
        for_each_occ(occ, ob, oe, x, [&](uint64_t r, uint64_t mask, uint64_t curr) {
            const uint64_t cm = mask & above(x) & sec;
            if (cm == 0)
                return;
            if (l == 0) {
                *reinterpret_cast<uint64_t*>(arena + q)     = cm;
                *reinterpret_cast<uint64_t*>(arena + q + 8) = curr;
            }
            Util* const out = reinterpret_cast<Util*>(arena + q + 16);
            if (cm >> l & 1) {
                out[__popcll(cm & below(l))] = static_cast<Util>(util_of(r, mask, l));
                if (prim >> l & 1)
                    cocc[c0++] = static_cast<uint32_t>(q);
            }
            if (cm >> (l + 32) & 1) {
                out[__popcll(cm & below(l + 32))] = static_cast<Util>(util_of(r, mask, l + 32));
                if (prim >> (l + 32) & 1)
                    cocc[c1++] = static_cast<uint32_t>(q);
            }
            q += lane_record_bytes(__popcll(cm));
        });
        if (l == 0) {
            LaneLevel* C = reinterpret_cast<LaneLevel*>(arena + child);
            C->n_tx      = n_tx;
            C->index     = static_cast<uint32_t>(index);
            C->bytes     = size;
        }
        top = (child + size + 7) & ~uint64_t{7};
        __syncwarp();
        return prim;
    }

    // Every lane: mines the subtree of the prepared database's primary item `item` (a global id among `items`).
    __device__ void mine(uint32_t item, const uint32_t* items, uint32_t m)
    {
        struct Frame
        {
            uint64_t level, prim, free_to;
        };
        const uint32_t x = local_id(items, m, item);
        if (!check(x < m && items[x] == item, set))
            return;
        Frame    frames[kLaneItems + 1];
        uint64_t top = base_top;
        uint32_t nf  = 0;
        frames[nf++] = {0, uint64_t{1} << x, base_top};
        while (nf != 0) {
            Frame& f = frames[nf - 1];
            if (f.prim == 0) {
                top = f.free_to;
                --nf;
                continue;
            }
            const uint32_t y = __ffsll(f.prim) - 1;
            f.prim &= f.prim - 1;
            const uint64_t before = top;
            uint64_t       child  = 0;
            const uint64_t prim   = expand_node(f.level, y, top, child);
            if (prim != 0 && check(nf <= kLaneItems, set))
                frames[nf++] = {child, prim, before};
            else
                top = before;
        }
    }
};

} // namespace ghuim::inline GHUIM_WIDTHS::refim
