#pragma once
// Vertical mode (after G-HUIM's vertical subtrees, src/mine/bfs/vertical_kernels.cu): a warp mines the subtrees of
// some primary items of a database with at most 32 distinct items and 1024 transactions from per-item transaction
// bitmaps and utility columns, without projecting child databases.
//
// Build (once per call): transaction t is bit t % 32 of lane t / 32's words. For every local item y (D's items
// renamed 0..m-1 in order): pres[y][lane], which of the lane's transactions hold y, and u[y][t], y's utility in t;
// per transaction its item mask tmask[t] and U_0[t], the utility of D's itemset in t (its prefix).
//
// Search, depth first, with REFIM's pruning: a node X has per transaction its utility U_X[t], (lane words) whether
// t supports X, and its secondary items `cand`. For every candidate y, over the transactions supporting X and y,
// with rest(t) = utility in t of the candidates after y:
//   utility(X + y) = sum U_X + u[y]                 -> counted when it reaches min_util
//   subtree bound  = sum U_X + u[y] + rest          -> X + y is searched only when it reaches min_util
//   local bound of a later candidate z below X + y = sum over t holding z of U_{X+y} + rest
//                                                   -> z stays secondary below X + y only when it does
// refined until no candidate drops (rest counts the survivors only), then REFIM's count_single (one supporting
// transaction) and count_full (the transactions holding every candidate qualify them all, or all transactions
// hold them) before searching X + y's children.

#include "mine/persistent/refim/expand.cuh"

namespace ghuim::inline GHUIM_WIDTHS::refim {

constexpr uint32_t kVertItems = 32;
constexpr uint32_t kVertTrans = 1024;

// Scratch one warp needs: pres, u, tmask, and U for every depth (at most kVertItems + 1).
__host__ __device__ constexpr uint64_t vert_bytes()
{
    return uint64_t{kVertItems} * 32 * 4 + uint64_t{kVertItems} * kVertTrans * sizeof(Util) + kVertTrans * 4 +
           uint64_t{kVertItems + 1} * kVertTrans * sizeof(Util);
}

__device__ inline uint64_t warp_sum64(uint64_t v)
{
    for (int d = 16; d > 0; d /= 2)
        v += __shfl_xor_sync(~0u, v, d);
    return v;
}

__device__ inline uint32_t after_bits(uint32_t y)
{
    return y >= 31 ? 0u : ~((2u << y) - 1);
}

// Lane 0: D's sorted distinct items in `items` (at most kVertItems) when D fits vertical mode.
__device__ inline bool vert_eligible(const Db& D, uint32_t* items, uint32_t& m)
{
    if (D.h->n_tx > kVertTrans || D.h->n_tx == 0)
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
        if (m == kVertItems)
            return false;
        for (uint32_t j = m; j > lo; --j)
            items[j] = items[j - 1];
        items[lo] = y;
        ++m;
    }
    return true;
}

// The whole warp: mines the subtrees of D's primary items [k_begin, k_end). `items` holds D's m sorted items;
// `scratch` has vert_bytes(), 16-byte aligned.
__device__ inline void vert_mine(const Settings& set, const Db& D, const uint32_t* items, uint32_t m, uint32_t k_begin,
                                 uint32_t k_end, unsigned char* scratch, BlockStats& stats)
{
    const uint32_t lane     = threadIdx.x & 31;
    const uint64_t min_util = set.min_util;
    uint32_t*      pres     = reinterpret_cast<uint32_t*>(scratch);                     // [y][lane]
    Util*          u        = reinterpret_cast<Util*>(scratch + kVertItems * 32 * 4);   // [y][t]
    uint32_t*      tmask    = reinterpret_cast<uint32_t*>(u + kVertItems * kVertTrans); // [t]
    Util*          U        = reinterpret_cast<Util*>(tmask + kVertTrans);              // [depth][t]
    const auto     local_id = [&](uint32_t y) {
        uint32_t lo = 0, hi = m;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if (items[mid] < y)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    };
    // Utility in transaction t of its items in `among` (a set of local items).
    const auto sum_items = [&](uint32_t t, uint32_t among) {
        uint64_t s = 0;
        for (uint32_t r = tmask[t] & among; r; r &= r - 1)
            s += u[(__ffs(r) - 1) * kVertTrans + t];
        return s;
    };

    // Build: each lane its 32 transactions.
    uint32_t words[kVertItems];
    for (uint32_t y = 0; y < kVertItems; ++y)
        words[y] = 0;
    uint32_t support = 0; // transactions that exist
    for (uint32_t b = 0; b < 32; ++b) {
        const uint32_t t = lane * 32 + b;
        if (t >= D.h->n_tx)
            break;
        const uint64_t at = D.tx_off[t], len = D.tx_len[t];
        uint32_t       mask = 0;
        if (check(at + len <= D.h->n_items, set)) {
            support |= 1u << b;
            for (uint64_t i = 0; i < len; ++i) {
                const uint32_t y = local_id(D.items[at + i]);
                if (!check(y < m, set))
                    continue;
                words[y] |= 1u << b;
                mask |= 1u << y;
                u[y * kVertTrans + t] = D.utils[at + i];
            }
        }
        tmask[t] = mask;
        U[t]     = D.prefix[t];
    }
    for (uint32_t y = 0; y < m; ++y)
        pres[y * 32 + lane] = words[y];
    __syncwarp();

    // Search. Per depth (replicated in every lane): secondary items, the ones still to try, the support word.
    uint32_t cand[kVertItems + 1], todo[kVertItems + 1], sup[kVertItems + 1];
    uint64_t patterns = 0, nodes = 0, counted = 0;
    uint32_t root_todo = 0;
    for (uint32_t k = k_begin; k < k_end; ++k) {
        const uint32_t y = local_id(D.primary[k]);
        if (check(y < m, set))
            root_todo |= 1u << y;
    }
    cand[0]   = m == 32 ? ~0u : (1u << m) - 1; // every item of D is secondary
    todo[0]   = root_todo;
    sup[0]    = support;
    int depth = 0;
    while (depth >= 0) {
        if (todo[depth] == 0) {
            --depth;
            continue;
        }
        const uint32_t y = __ffs(todo[depth]) - 1;
        todo[depth] &= todo[depth] - 1;
        ++nodes;
        const Util*    Ud    = U + depth * kVertTrans;
        const uint32_t sy    = sup[depth] & pres[y * 32 + lane];
        const uint32_t later = cand[depth] & after_bits(y);
        uint64_t       util = 0, sub = 0;
        for (uint32_t r = sy; r; r &= r - 1) {
            const uint32_t t  = lane * 32 + __ffs(r) - 1;
            const uint64_t uu = uint64_t{Ud[t]} + u[y * kVertTrans + t];
            util += uu;
            sub += uu + sum_items(t, later);
        }
        util = warp_sum64(util);
        sub  = warp_sum64(sub);
        if (sub < min_util)
            continue; // not a primary extension: X + y and everything below it fall short
        patterns += util >= min_util;
        if (later == 0 || depth >= static_cast<int>(kVertItems))
            continue;

        // U for X + y, then its secondary items, refined until none drops.
        Util* Un = U + (depth + 1) * kVertTrans;
        for (uint32_t r = sy; r; r &= r - 1) {
            const uint32_t t = lane * 32 + __ffs(r) - 1;
            Un[t]            = static_cast<Util>(uint64_t{Ud[t]} + u[y * kVertTrans + t]);
        }
        uint32_t next = later;
        for (int32_t round = 0;; ++round) {
            uint32_t kept = 0;
            for (uint32_t rest = next; rest; rest &= rest - 1) {
                const uint32_t z  = __ffs(rest) - 1;
                const uint32_t sz = sy & pres[z * 32 + lane];
                uint64_t       lb = 0;
                for (uint32_t r2 = sz; r2; r2 &= r2 - 1) {
                    const uint32_t t = lane * 32 + __ffs(r2) - 1;
                    lb += uint64_t{Un[t]} + sum_items(t, next);
                }
                if (warp_sum64(lb) >= min_util)
                    kept |= 1u << z;
            }
            const bool dropped = kept != next;
            next               = kept;
            if (!dropped || next == 0 || (set.refine_rounds >= 0 && round >= set.refine_rounds))
                break;
        }
        if (next == 0)
            continue;

        // REFIM's count_single and count_full: X + y's subtree counted without searching it.
        const uint32_t n_sec = __popc(next);
        uint32_t       n_tx = 0, n_full = 0;
        uint64_t       lower = 0;
        for (uint32_t r = sy; r; r &= r - 1) {
            const uint32_t t = lane * 32 + __ffs(r) - 1;
            if ((tmask[t] & next) == 0)
                continue;
            ++n_tx;
            if ((tmask[t] & next) == next) {
                ++n_full;
                lower += Un[t];
            }
        }
        n_tx   = __reduce_add_sync(~0u, n_tx);
        n_full = __reduce_add_sync(~0u, n_full);
        lower  = warp_sum64(lower);
        if (n_tx == 1 || lower >= min_util || n_full == n_tx) {
            unsigned long long tail[kVertItems] = {};
            uint64_t           base             = lower;
            uint32_t           n_single         = 0; // candidates the one transaction holds
            if (n_tx == 1) {                         // the one transaction: its candidates' utilities over its prefix
                base = 0;
                for (uint32_t r = sy; r; r &= r - 1) {
                    const uint32_t t = lane * 32 + __ffs(r) - 1;
                    if ((tmask[t] & next) == 0)
                        continue;
                    base     = Un[t];
                    n_single = __popc(tmask[t] & next);
                    for (uint32_t r2 = tmask[t] & next, i = 0; r2; r2 &= r2 - 1, ++i)
                        tail[i] = u[(__ffs(r2) - 1) * kVertTrans + t];
                }
            }
            else if (lower < min_util) // every transaction holds every candidate: one merged transaction
                for (uint32_t r = sy; r; r &= r - 1) {
                    const uint32_t t = lane * 32 + __ffs(r) - 1;
                    if ((tmask[t] & next) != next)
                        continue;
                    uint32_t i = 0;
                    for (uint32_t r2 = next; r2; r2 &= r2 - 1, ++i)
                        tail[i] += u[(__ffs(r2) - 1) * kVertTrans + t];
                }
            // Combine every lane's contribution into lane 0.
            // count_single counts the subsets of the items its transaction holds (not of all candidates).
            const uint32_t n = n_tx == 1 ? __reduce_add_sync(~0u, n_single) : n_sec;
            for (uint32_t i = 0; i < n; ++i)
                tail[i] = warp_sum64(tail[i]);
            base = n_tx == 1 ? warp_sum64(base) : base;
            if (lane == 0) {
                if (n_tx != 1 && lower >= min_util)
                    patterns += (uint64_t{1} << n_sec) - 1;
                else
                    patterns += count_tail(tail, n, base, min_util);
            }
            ++counted;
            continue;
        }
        ++depth;
        cand[depth] = next;
        todo[depth] = next;
        sup[depth]  = sy;
    }
    if (lane == 0) {
        add_stat(stats.work[0], patterns);
        add_stat(stats.work[1], nodes);
        add_stat(stats.work[2], counted);
    }
}

} // namespace ghuim::inline GHUIM_WIDTHS::refim
