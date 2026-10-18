// Vertical subtrees: a warp mines a task's whole subtree depth first from per-item transaction bitmaps and
// utility columns, without projecting child databases (count-only).
//
// Build: the warp scans the task's transactions once. The n <= vert_trans transactions holding the extension
// item get slots 0..n-1; lane l owns slots l, l + 32, ... and slot c is bit c / 32 of its lane's words. For
// every remaining item y (local name first + y, y < vert_items) it records
//   pres[y][lane]  which of the lane's slots contain y (shared memory),
//   u[y][c]        y's utility in slot c,
//   rem[y][c]      utility of the remaining items after y in slot c,
// plus U_0[c], the utility of the task's itemset in slot c, and re_0[c], all remaining items' utility.
//
// Search: a node X has, per slot, its utility U_X and (lane bits) whether it supports X. For every candidate y,
// over the slots supporting X and y:
//   utility(X + y) = sum U_X + u[y]                     -> counted when it reaches minutil
//   subtree bound  = sum U_X + u[y] + rem[y]            -> X + y is extended only when it reaches minutil
//   local bound    = sum U_X + re_X                     -> y stays a candidate below X only when it does
// (EFIM's bounds; re_X is rem of X's last item). Every sum is at most the database's total utility, which
// load_dataset checks fits 32 bits, so sums are 32-bit warp reductions.

#include "cuda/buffer.cuh"
#include "mine/bfs/device/task_view.cuh"
#include "mine/bfs/kernels.cuh"
#include "mine/primitives.cuh"

#include <algorithm>

namespace ghuim::bfs {
namespace {

constexpr uint32_t kVertBlockWarps = 4;

__device__ __forceinline__ uint32_t warp_sum(uint32_t value)
{
    return __reduce_add_sync(kFullMask, value);
}

__global__ void __launch_bounds__(kVertBlockWarps * 32)
        vertical_tasks(BatchView b, const uint32_t* order, uint32_t count)
{
    __shared__ uint32_t pres_all[kVertBlockWarps][kMaxVertItems * 32];
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const uint32_t global_warp = blockIdx.x * kVertBlockWarps + warp, warps = gridDim.x * kVertBlockWarps;
    uint32_t*      pres        = pres_all[warp];
    const uint32_t tp          = b.vert_slots; // slots per column (vert_trans rounded up to 32)
    const uint64_t minutil     = b.minimum_utility;
    // Per-warp scratch: u[V][tp] | rem[V][tp] | re0[tp] | U[V + 1][tp].
    uint32_t* const u_col   = b.vert_scratch + uint64_t{global_warp} * b.vert_warp_words;
    uint32_t* const rem_col = u_col + uint64_t{kMaxVertItems} * tp;
    uint32_t* const re0     = rem_col + uint64_t{kMaxVertItems} * tp;
    uint32_t* const ustack  = re0 + tp;

    for (uint32_t ti = global_warp; ti < count; ti += warps) {
        const TaskView v     = open_task(b, order[ti]);
        const uint32_t first = v.task.item + 1, items = v.header.n_items - first;
        for (uint32_t w = lane; w < items * 32; w += 32)
            pres[w] = 0;
        __syncwarp();

        // Build the columns.
        uint32_t n = 0;
        for (uint32_t base = 0; base < v.p.n_trans; base += 32) {
            const uint32_t i  = base + lane;
            const Tx       tx = i < v.p.n_trans ? v.p.txs[v.p.occ ? v.p.occ[i].tid : i] : Tx{0, 0, 0};
            const Key*     keys  = v.p.eitem + tx.begin;
            const uint32_t* util = v.p.eutil + tx.begin;
            const int k = i < v.p.n_trans ? (v.p.occ ? static_cast<int>(v.p.occ[i].pos) : find_item(keys, tx.length, v.p.item)) : -1;
            const uint32_t has = __ballot_sync(kFullMask, k >= 0);
            const uint32_t c   = n + __popc(has & ((1u << lane) - 1));
            n += __popc(has);
            if (k < 0)
                continue;
            ustack[c]  = tx.utility + util[k];
            uint32_t s = 0;
            for (int e = static_cast<int>(tx.length) - 1; e > k; --e) {
                const uint32_t y = keys[e] - first;
                u_col[y * tp + c]   = util[e];
                rem_col[y * tp + c] = s;
                s += util[e];
                atomicOr(&pres[y * 32 + (c & 31)], 1u << (c >> 5));
            }
            re0[c] = s;
        }
        __syncwarp();

        // Depth-first search. Per depth: the lane's slot bits, candidates, extensions still to visit, and the row
        // holding re_X.
        uint32_t  bits[kMaxVertItems + 1], cand[kMaxVertItems + 1], todo[kMaxVertItems + 1], lower[kMaxVertItems + 1];
        const uint32_t* re[kMaxVertItems + 1];
        const uint32_t  slots = (n + 31) / 32;
        bits[0] = 0;
        for (uint32_t k = 0; k < slots; ++k)
            if (lane + 32 * k < n)
                bits[0] |= 1u << k;
        re[0]   = re0;
        cand[0] = items == 32 ? ~0u : (1u << items) - 1;
        uint64_t found = 0;
        {
            uint32_t total = 0;
            for (uint32_t m = bits[0]; m; m &= m - 1)
                total += ustack[lane + 32 * (__ffs(m) - 1)];
            found += warp_sum(total) >= minutil;
        }
        uint32_t d = 0;
        for (;;) {
            // Evaluate every candidate of the node at depth d.
            const uint32_t* U = ustack + uint64_t{d} * tp;
            uint32_t extend = 0, keep = 0;
            for (uint32_t c = cand[d]; c; c &= c - 1) {
                const uint32_t y = __ffs(c) - 1;
                uint32_t       ut = 0, st = 0, lt = 0;
                for (uint32_t m = bits[d] & pres[y * 32 + lane]; m; m &= m - 1) {
                    const uint32_t slot = lane + 32 * (__ffs(m) - 1);
                    const uint32_t a    = U[slot] + u_col[y * tp + slot];
                    ut += a;
                    st += a + rem_col[y * tp + slot];
                    lt += U[slot] + re[d][slot];
                }
                ut = warp_sum(ut);
                st = warp_sum(st);
                lt = warp_sum(lt);
                found += ut >= minutil;
                extend |= st >= minutil ? 1u << y : 0u;
                keep |= lt >= minutil ? 1u << y : 0u;
            }
            todo[d]  = extend;
            lower[d] = keep;
            // Next node: the lowest extension left at the deepest depth that has one with candidates below it.
            bool descended = false;
            for (;;) {
                while (todo[d] == 0 && d > 0)
                    --d;
                if (todo[d] == 0)
                    break;
                const uint32_t y = __ffs(todo[d]) - 1;
                todo[d] &= todo[d] - 1;
                const uint32_t below = y == 31 ? 0u : lower[d] & ~((2u << y) - 1);
                if (below == 0)
                    continue; // X + y has no extensions; its utility is counted already
                const uint32_t  child = bits[d] & pres[y * 32 + lane];
                const uint32_t* U_in  = ustack + uint64_t{d} * tp;
                uint32_t*       U_out = ustack + uint64_t{d + 1} * tp;
                for (uint32_t m = child; m; m &= m - 1) {
                    const uint32_t slot = lane + 32 * (__ffs(m) - 1);
                    U_out[slot]         = U_in[slot] + u_col[y * tp + slot];
                }
                bits[d + 1] = child;
                cand[d + 1] = below;
                re[d + 1]   = rem_col + uint64_t{y} * tp;
                ++d;
                descended = true;
                break;
            }
            if (!descended)
                break;
        }
        if (lane == 0 && found != 0)
            add(b.control->patterns, static_cast<unsigned long long>(found));
        __syncwarp();
    }
}

} // namespace

void launch_vertical_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, const AutoTune& tune,
                           cudaStream_t stream)
{
    const uint32_t blocks = std::max(1u, std::min(ceil_div(count, kVertBlockWarps), tune.vert_warps / kVertBlockWarps));
    vertical_tasks<<<blocks, kVertBlockWarps * 32, 0, stream>>>(batch, order, count);
    check_launch("launch vertical-task kernel");
}

} // namespace ghuim::bfs
