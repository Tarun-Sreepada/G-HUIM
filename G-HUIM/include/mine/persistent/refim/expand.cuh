#pragma once
// REFIM's expand() for one node: project database D on its primary item k, record the itemset, choose the child's
// extensions and build the compacted child, all as passes over D's occurrences of the item by a cooperating group
// of threads: a warp (small databases), a block (work-stealing phase) or the whole grid (wide phase). The group
// syncs between passes; every thread takes the same branches, from values read after a sync.
//
// Each occurrence (a transaction holding the item) is handled by a tile of W lanes, W a power of two up to 32 sized
// to D's average transaction: the lanes walk the transaction's suffix in chunks of W consecutive entries (coalesced),
// with shuffles for suffix sums and ballots for the positions of kept items.
//
//   A   per occurrence: the itemset's utility, and the local/subtree bound of every item after it
//   -   the child has one transaction: count its patterns directly (REFIM's count_single); done
//   select   items with local >= min_util are kept (secondary), those with subtree >= min_util are extended
//   B1  per occurrence: kept items, child transactions/entries/occurrences, and REFIM's count_full test
//   -   count_full holds: count the child's whole subtree directly; done
//   alloc    on the group's stack, or in the heap when the child is split into tasks (or the stack is full)
//   scan     occurrence offsets of the child's primary items
//   B2  per occurrence: write the child transaction and its occurrences
//
// Work counters (BlockStats::work, added atomically: a block's warps share them): [0] patterns, [1] nodes
// (projections), [2] children counted without search, [3] children split or offered as tasks, [4] nodes expanded by
// a warp, [5] nodes mined by a lane, [6] ns blocks spent in warp mode; with timers (kTimers), cycles of [7] warps in lane mode, [8] lanes mining
// (summed per thread), [9] warps expanding, [10] lane 0 checking lane eligibility, [11] lanes converting (per
// thread), [12] warps waiting for work; [13] lane subtrees mined, [14] lane-mode batches, [15]
// warps waiting at the end of warp mode for the block's other warps, [16] warps in warp mode, [20] lane jobs taken
// from the pool, [21] batches pooled, [22] siblings
// borrowed from the block's loops.

#include "cuda/atomics.cuh"
#include "mine/persistent/control.cuh"
#include "mine/persistent/heap.cuh"
#include "mine/persistent/refim/db.cuh"

#include <cooperative_groups.h>

namespace ghuim::inline GHUIM_WIDTHS::refim {

namespace cg = cooperative_groups;
using persistent::BlockStats;
using persistent::Heap;

__device__ inline void add_stat(uint64_t& counter, uint64_t v)
{
    atomicAdd(reinterpret_cast<unsigned long long*>(&counter), static_cast<unsigned long long>(v));
}

// Timing counters (the cycle counts in BlockStats::work), compiled in with -DGHUIM_TIMERS=ON.
#ifdef GHUIM_TIMERS
constexpr bool kTimers = true;
#else
constexpr bool kTimers = false;
#endif

__device__ inline uint64_t timer()
{
    if constexpr (kTimers)
        return clock64();
    else
        return 0;
}

// Adds the cycles since `start` (a timer() value) to counter.
__device__ inline void add_time(uint64_t& counter, uint64_t start)
{
    if constexpr (kTimers)
        add_stat(counter, clock64() - start);
}

// Exclusive scan of in[0, n) into out[0, n] and cursor[0, n), by one whole block.
__device__ inline void block_exclusive_scan(const uint32_t* in, uint32_t* out, uint32_t* cursor, uint32_t n);

// Exclusive scan of in[0, n) into out[0, n] and cursor[0, n), by one warp.
__device__ inline void warp_exclusive_scan(const uint32_t* in, uint32_t* out, uint32_t* cursor, uint32_t n)
{
    const uint32_t lane  = threadIdx.x & 31;
    uint32_t       carry = 0;
    for (uint32_t base = 0; base < n; base += 32) {
        const uint32_t i = base + lane;
        const uint32_t v = i < n ? in[i] : 0;
        uint32_t       x = v;
        for (uint32_t d = 1; d < 32; d <<= 1) {
            const uint32_t y = __shfl_up_sync(~0u, x, d);
            if (lane >= d)
                x += y;
        }
        if (i < n) {
            out[i]    = carry + x - v;
            cursor[i] = carry + x - v;
        }
        carry += __shfl_sync(~0u, x, 31);
    }
    if (lane == 0)
        out[n] = carry;
    __syncwarp();
}

struct WarpGroup
{
    static constexpr bool kGrid = false, kWarp = true;
    __device__ uint32_t   rank() const
    {
        return threadIdx.x & 31;
    }
    __device__ uint32_t size() const
    {
        return 32;
    }
    __device__ bool leader() const
    {
        return (threadIdx.x & 31) == 0;
    }
    __device__ void sync() const
    {
        __syncwarp();
    }
    __device__ void exclusive_scan(const uint32_t* in, uint32_t* out, uint32_t* cursor, uint32_t n) const
    {
        warp_exclusive_scan(in, out, cursor, n);
    }
};

struct BlockGroup
{
    static constexpr bool kGrid = false, kWarp = false;
    __device__ uint32_t   rank() const
    {
        return threadIdx.x;
    }
    __device__ uint32_t size() const
    {
        return blockDim.x;
    }
    __device__ bool leader() const
    {
        return threadIdx.x == 0;
    }
    __device__ void sync() const
    {
        __syncthreads();
    }
    __device__ void exclusive_scan(const uint32_t* in, uint32_t* out, uint32_t* cursor, uint32_t n) const
    {
        block_exclusive_scan(in, out, cursor, n);
    }
};

struct GridGroup
{
    static constexpr bool kGrid = true, kWarp = false;
    cg::grid_group*       grid;
    __device__ uint32_t   rank() const
    {
        return static_cast<uint32_t>(grid->thread_rank());
    }
    __device__ uint32_t size() const
    {
        return static_cast<uint32_t>(grid->size());
    }
    __device__ bool leader() const
    {
        return grid->thread_rank() == 0;
    }
    __device__ void sync() const
    {
        grid->sync();
    }
    // One block scans; the caller's sync publishes the result.
    __device__ void exclusive_scan(const uint32_t* in, uint32_t* out, uint32_t* cursor, uint32_t n) const
    {
        if (blockIdx.x == 0)
            block_exclusive_scan(in, out, cursor, n);
    }
};

// Shared values of one expansion: in shared memory for a block or warp, in global memory for the grid.
struct Counters
{
    unsigned long long utility, lower, n_items, item_cursor;
    unsigned long long tail[64];
    unsigned long long child; // address of the child database
    uint32_t           n_touched, n_sec, n_prim, top, n_tx, n_occ, all, tx_cursor;
    uint32_t           n_nonempty, single; // A: occurrences with items after the item, and one of them
    uint32_t           outcome;
    uint32_t           n_dropped; // select: items ruled out so far (all rounds)
    // Merging identical child transactions: a hash table (keys, then the child transaction of each slot) and the
    // state of each occurrence (kNoTx, a representative's slot, or kDup | its representative's slot).
    unsigned long long table;     // address; 0: no merging at this node
    uint32_t           table_cap; // slots, a power of two
    uint32_t           table_cls; // heap class of the table (grid), else kStackDb
    unsigned long long table_bytes;
};

enum Outcome : uint32_t
{
    kNone   = 0, // nothing to descend into
    kStack  = 1, // child on the group's stack
    kOwned  = 2, // child in the heap, used by the expanding frame only (refs = 1)
    kShared = 3, // child in the heap, to be split into tasks (refs set by the caller)
    kFailed = 4, // out of memory: the run aborts
};

// A group's stack of databases: [base, base + capacity), `top` bytes in use.
struct Stack
{
    char*     base;
    uint64_t  capacity;
    uint64_t* top;
};

struct Settings
{
    uint64_t             min_util;
    uint32_t             num_items; // item ids are below this
    persistent::Control* control;   // device-side checks abort through it
    // REFIM's split rule: a child is split into tasks when its work (occurrences x primary items) is large near the
    // top of the search, or moderately large while some block is idle.
    uint32_t max_split_depth, max_hungry_depth;
    uint64_t split_work, hungry_work;
    uint32_t split_chunks;  // tasks a split child becomes (each a range of its primary items)
    int32_t  refine_rounds; // REFIM's refinement: rescans of a child without the items it rules out (-1: until none)
    // Databases below these sizes (entries) skip refinement / merging: on tiny nodes the extra passes cost more
    // than they save.
    uint64_t refine_min_items, merge_min_items;
};

// ---- small helpers (one thread) ------------------------------------------------------------------------

// Nonempty subsets of u[0, n) (any order) whose sum reaches min_util - prefix; n < 64. REFIM's count_tail.
__device__ inline uint64_t count_tail(unsigned long long* u, uint32_t n, uint64_t prefix, uint64_t min_util)
{
    for (uint32_t i = 1; i < n; ++i) // descending
        for (uint32_t j = i; j > 0 && u[j - 1] < u[j]; --j) {
            const unsigned long long t = u[j - 1];
            u[j - 1]                   = u[j];
            u[j]                       = t;
        }
    uint64_t rest[65];
    rest[n] = 0;
    for (uint32_t i = n; i-- > 0;)
        rest[i] = rest[i + 1] + u[i];
    const uint64_t need = prefix >= min_util ? 0 : min_util - prefix;
    if (need == 0)
        return (uint64_t{1} << n) - 1; // the empty subset is the prefix itself, already recorded
    struct Frame
    {
        uint32_t i;
        uint64_t need;
    };
    Frame    stack[64];
    int      top   = 0;
    uint64_t found = 0;
    stack[top++]   = {0, need};
    while (top > 0) {
        Frame f = stack[--top];
        for (uint32_t i = f.i; rest[i] >= f.need; ++i) {
            if (u[i] >= f.need)
                found += uint64_t{1} << (n - i - 1);
            else
                stack[top++] = {i + 1, f.need - u[i]};
        }
    }
    return found;
}

// A device-side check: false (after aborting the run) when `ok` is false; the caller skips the access.
__device__ inline bool check(bool ok, const Settings& set, uint32_t line = __builtin_LINE())
{
    if (!ok && atomicCAS(&set.control->abort, 0u, static_cast<uint32_t>(persistent::Abort::kCorrupt)) == 0)
        set.control->failed_line = line;
    return ok;
}

// Occurrence o of a database, validated: transaction `tid` at entry offset `at` with `len` entries, item at `pos`.
struct Occ
{
    uint32_t tid, pos;
    uint64_t at, len;
};

__device__ inline bool read_occ(const Db& D, uint32_t o, const Settings& set, Occ& r)
{
    const uint32_t oc = D.occ[o];
    r.tid             = oc >> D.h->occ_shift;
    r.pos             = oc & ((1u << D.h->occ_shift) - 1);
    if (!check(r.tid < D.h->n_tx, set))
        return false;
    r.at  = D.tx_off[r.tid];
    r.len = D.tx_len[r.tid];
    return check(r.pos < r.len && r.at + r.len <= D.h->n_items, set);
}

__device__ inline void record(uint64_t utility, uint64_t min_util, BlockStats& stats)
{
    if (utility >= min_util)
        add_stat(stats.work[0], 1);
}

// W lanes of a group working on one occurrence. Lanes of a tile are consecutive lanes of one warp.
struct Tile
{
    uint32_t w, lane, index, count; // width, lane in the tile, tile index in the group, tiles in the group
    uint32_t base;                  // first lane of the tile in its warp
    unsigned mask;                  // the tile's lanes, for shuffles and ballots

    template <typename G> __device__ Tile(const G& g, uint32_t width)
    {
        w     = width;
        lane  = g.rank() % w;
        index = g.rank() / w;
        count = g.size() / w;
        base  = (threadIdx.x & 31) & ~(w - 1);
        mask  = w == 32 ? ~0u : ((1u << w) - 1) << base;
    }
    template <typename T> __device__ T sum(T v) const
    {
        for (uint32_t d = w / 2; d > 0; d /= 2)
            v += __shfl_xor_sync(mask, v, d, w);
        return v;
    }
    template <typename T> __device__ T from(T v, uint32_t src) const
    {
        return __shfl_sync(mask, v, src, w);
    }
    // Inclusive scan over the tile's lanes.
    template <typename T> __device__ T scan(T v) const
    {
        for (uint32_t d = 1; d < w; d *= 2) {
            const T y = __shfl_up_sync(mask, v, d, w);
            if (lane >= d)
                v += y;
        }
        return v;
    }
    // Ballot of the tile, lane 0 in bit 0.
    __device__ uint32_t ballot(bool flag) const
    {
        const uint32_t b = __ballot_sync(mask, flag) >> base;
        return w == 32 ? b : b & ((1u << w) - 1);
    }
};

// Tile width for database D: its average transaction length, rounded up to a power of two in [1, 32].
__device__ inline uint32_t tile_width(const Db& D)
{
    const uint64_t avg = D.h->n_tx == 0 ? 1 : (D.h->n_items + D.h->n_tx - 1) / D.h->n_tx;
    uint32_t       w   = 1;
    while (w < 32 && w < avg)
        w *= 2;
    return w;
}

__device__ inline void block_exclusive_scan(const uint32_t* in, uint32_t* out, uint32_t* cursor, uint32_t n)
{
    __shared__ uint32_t warp_sum[32];
    __shared__ uint32_t carry;
    if (threadIdx.x == 0)
        carry = 0;
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
    for (uint32_t base = 0; base < n; base += blockDim.x) {
        const uint32_t i = base + threadIdx.x;
        const uint32_t v = i < n ? in[i] : 0;
        uint32_t       x = v;
        for (uint32_t d = 1; d < 32; d <<= 1) {
            const uint32_t y = __shfl_up_sync(~0u, x, d);
            if (lane >= d)
                x += y;
        }
        if (lane == 31)
            warp_sum[warp] = x;
        __syncthreads();
        if (warp == 0) {
            uint32_t s = lane < warps ? warp_sum[lane] : 0;
            for (uint32_t d = 1; d < 32; d <<= 1) {
                const uint32_t y = __shfl_up_sync(~0u, s, d);
                if (lane >= d)
                    s += y;
            }
            if (lane < warps)
                warp_sum[lane] = s; // inclusive
        }
        __syncthreads();
        const uint32_t exclusive = carry + (warp > 0 ? warp_sum[warp - 1] : 0) + x - v;
        if (i < n) {
            out[i]    = exclusive;
            cursor[i] = exclusive;
        }
        __syncthreads();
        if (threadIdx.x == 0)
            carry += warp_sum[warps - 1];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[n] = carry;
    __syncthreads();
}

// ---- expand -----------------------------------------------------------------------------------------------

// Expands D's primary item k (D is at depth `depth`). On return (all threads) c->outcome says what was built and
// c->child where. `hungry`: some block is idle (REFIM's split rule); `stack` is null for the grid.
template <typename G>
__device__ void expand(const G& g, const Settings& set, const Heap& heap, const Scratch& s, Counters* c, const Db& D,
                       uint32_t k, uint32_t depth, bool hungry, const Stack* stack, BlockStats& stats)
{
    const uint64_t min_util = set.min_util;
    if (g.leader()) {
        c->utility = c->lower = c->n_items = c->item_cursor = 0;
        c->n_touched = c->n_sec = c->n_prim = c->top = c->n_tx = c->n_occ = c->tx_cursor = c->n_nonempty = 0;
        c->n_dropped                                                                                     = 0;
        c->all                                                                                           = 1;
        c->outcome                                                                                       = kNone;
        for (int i = 0; i < 64; ++i)
            c->tail[i] = 0;
        add_stat(stats.work[1], 1);
        if constexpr (G::kWarp)
            add_stat(stats.work[4], 1);
    }
    g.sync();
    // Every thread reads the same header, so all return together.
    if (!check(k < D.h->n_primary && D.occ_begin[k] <= D.occ_begin[k + 1] && D.occ_begin[k + 1] <= D.h->n_occ, set))
        return;
    const uint32_t ob = D.occ_begin[k], oe = D.occ_begin[k + 1];
    const Tile     T(g, tile_width(D));

    // A: utility and bounds. A transaction's items after position pos are all secondary in D. The suffix is walked
    // from its end in chunks of W (lane j takes the j-th entry from the end), so a lane's subtree increment is its
    // inclusive scan plus the chunks before.
    for (uint32_t o = ob + T.index; o < oe; o += T.count) {
        Occ r;
        if (!read_occ(D, o, set, r))
            continue;
        const uint64_t curr = uint64_t{D.utils[r.at + r.pos]} + D.prefix[r.tid];
        if (T.lane == 0)
            atomicAdd(&c->utility, curr);
        if (r.pos + 1 == r.len)
            continue;
        if (T.lane == 0 && atomicAdd(&c->n_nonempty, 1u) == 0)
            c->single = o; // only read when this is the only one
        const uint64_t first = r.at + r.pos + 1, n = r.len - r.pos - 1;
        uint64_t       carry = 0;
        for (uint64_t c0 = 0; c0 < n; c0 += T.w) {
            const uint64_t j     = c0 + T.lane;
            const bool     valid = j < n;
            const uint64_t i     = first + n - 1 - j;
            const uint64_t u     = valid ? uint64_t{D.utils[i]} : 0;
            const uint64_t x     = T.scan(u);
            if (valid && check(D.items[i] < set.num_items, set))
                atomicAdd(&s.bound[D.items[i]].subtree, curr + carry + x);
            carry += T.from(x, T.w - 1);
        }
        for (uint64_t j = T.lane; j < n; j += T.w) {
            const uint32_t y = D.items[first + j];
            if (!check(y < set.num_items, set))
                continue;
            atomicAdd(&s.bound[y].local, curr + carry);
            if (atomicCAS(&s.mark[y], 0u, 1u) == 0) {
                const uint32_t t = atomicAdd(&c->n_touched, 1u);
                if (check(t < set.num_items, set))
                    s.touched[t] = y;
            }
        }
    }
    g.sync();
    if (g.leader())
        record(c->utility, min_util, stats);
    const uint32_t n_touched = min(c->n_touched, set.num_items);
    if (n_touched == 0)
        return; // no item after the extension: a leaf

    // The child holds a single transaction: its patterns are the subsets of that transaction's items reaching
    // min_util with the prefix (REFIM's count_single). Its items are all secondary in D, and those select would
    // drop cannot be in a high-utility itemset, so counting before select is exact.
    bool single = false;
    Occ  one;
    if (c->n_nonempty == 1 && c->single >= ob && c->single < oe && read_occ(D, c->single, set, one)) {
        const uint32_t n = static_cast<uint32_t>(one.len) - one.pos - 1;
        if (n < 64) {
            single = true;
            if (g.leader()) {
                for (uint32_t i = 0; i < n; ++i)
                    c->tail[i] = D.utils[one.at + one.pos + 1 + i];
                add_stat(stats.work[0],
                         count_tail(c->tail, n, uint64_t{D.utils[one.at + one.pos]} + D.prefix[one.tid], min_util));
                add_stat(stats.work[2], 1);
            }
        }
    }

    const auto kept_item = [&](uint32_t y) { return y < set.num_items && s.mark[y] == 2; };

    // select, with REFIM's refinement. Each round first rules out the items whose local bound is below min_util
    // (mark 0) and keeps the rest (mark 2). If some were ruled out, the kept items' bounds still count utility no
    // extension can use: they are recomputed from D's suffixes without the ruled-out items, and the round repeats
    // until none drops (or refine_rounds rounds). The last round's bounds choose the extensions.
    uint32_t dropped_before = 0;
    for (int32_t round = 0;; ++round) {
        for (uint32_t t = g.rank(); t < n_touched; t += g.size()) {
            const uint32_t y = s.touched[t];
            if (!check(y < set.num_items, set) || s.mark[y] == 0)
                continue;
            if (single || s.bound[y].local < min_util) {
                s.mark[y]  = 0;
                s.bound[y] = {0, 0};
                atomicAdd(&c->n_dropped, 1u);
            }
            else
                s.mark[y] = 2;
        }
        g.sync();
        const uint32_t dropped = c->n_dropped - dropped_before;
        dropped_before         = c->n_dropped;
        if (single || dropped == 0 || (set.refine_rounds >= 0 && round >= set.refine_rounds) ||
            D.h->n_items < set.refine_min_items)
            break;
        for (uint32_t t = g.rank(); t < n_touched; t += g.size())
            if (kept_item(s.touched[t]))
                s.bound[s.touched[t]] = {0, 0};
        g.sync();
        // The bounds again, counting kept items only.
        for (uint32_t o = ob + T.index; o < oe; o += T.count) {
            Occ r;
            if (!read_occ(D, o, set, r) || r.pos + 1 == r.len)
                continue;
            const uint64_t curr  = uint64_t{D.utils[r.at + r.pos]} + D.prefix[r.tid];
            const uint64_t first = r.at + r.pos + 1, n = r.len - r.pos - 1;
            uint64_t       carry = 0;
            for (uint64_t c0 = 0; c0 < n; c0 += T.w) {
                const uint64_t j    = c0 + T.lane;
                const uint64_t i    = first + n - 1 - j;
                const bool     keep = j < n && kept_item(D.items[i]);
                const uint64_t x    = T.scan(keep ? uint64_t{D.utils[i]} : 0);
                if (keep)
                    atomicAdd(&s.bound[D.items[i]].subtree, curr + carry + x);
                carry += T.from(x, T.w - 1);
            }
            for (uint64_t j = T.lane; j < n; j += T.w)
                if (kept_item(D.items[first + j]))
                    atomicAdd(&s.bound[D.items[first + j]].local, curr + carry);
        }
        g.sync();
    }

    // The kept items are the child's secondary items; those whose subtree bound reaches min_util get a slot.
    for (uint32_t t = g.rank(); t < n_touched; t += g.size()) {
        const uint32_t y = s.touched[t];
        if (!kept_item(y))
            continue;
        const Bound b = s.bound[y];
        s.bound[y]    = {0, 0};
        atomicAdd(&c->n_sec, 1u);
        atomicMax(&c->top, y);
        if (b.subtree >= min_util) {
            const uint32_t j = atomicAdd(&c->n_prim, 1u);
            if (check(j < set.num_items, set)) {
                s.chosen[j] = y;
                s.slot[y]   = static_cast<int32_t>(j);
            }
        }
    }
    g.sync();
    const uint32_t n_sec = c->n_sec, n_prim = min(c->n_prim, set.num_items);
    const auto     clear_marks = [&] {
        for (uint32_t t = g.rank(); t < n_touched; t += g.size())
            if (const uint32_t y = s.touched[t]; y < set.num_items) {
                s.mark[y] = 0;
                s.slot[y] = -1;
            }
        for (uint32_t j = g.rank(); j < n_prim; j += g.size())
            s.count[j] = 0;
        g.sync();
    };
    if (single || n_prim == 0) {
        clear_marks();
        return;
    }
    // Merging identical child transactions (REFIM's Level::add merges equal neighbours; here every duplicate):
    // a table of 2+ slots per occurrence, at the far end of the group's stack (the heap for the grid), only for
    // this expansion. Without room the child is simply not merged.
    const uint32_t n_par = oe - ob;
    if (g.leader()) {
        c->table       = 0;
        c->table_bytes = 0;
        if (n_par >= 2 && D.h->n_items >= set.merge_min_items) {
            uint32_t cap = 64;
            while (cap < 2 * n_par)
                cap *= 2;
            const uint64_t bytes = align16(uint64_t{cap} * 12 + uint64_t{n_par} * 4);
            if constexpr (G::kGrid) {
                uint32_t       cls;
                const uint64_t off = heap.tight(bytes) ? Heap::kNull : heap.allocate(bytes, cls);
                if (off != Heap::kNull) {
                    c->table     = reinterpret_cast<unsigned long long>(heap.base + off);
                    c->table_cls = cls;
                }
            }
            else if (*stack->top + bytes + 16 <= stack->capacity) {
                c->table     = reinterpret_cast<unsigned long long>(stack->base + (stack->capacity - bytes) / 16 * 16);
                c->table_cls = kStackDb;
            }
            c->table_cap   = cap;
            c->table_bytes = c->table ? bytes + 16 : 0;
        }
    }
    g.sync();
    constexpr unsigned long long kEmpty = ~0ull;
    constexpr uint32_t           kNoTx = ~0u, kDup = 1u << 31;
    unsigned long long* const    keys      = reinterpret_cast<unsigned long long*>(c->table);
    const uint32_t               cap       = keys ? c->table_cap : 0;
    uint32_t* const              slot_tx   = keys ? reinterpret_cast<uint32_t*>(keys + cap) : nullptr;
    uint32_t* const              occ_state = keys ? slot_tx + cap : nullptr;
    if (keys) { // (every thread reads the same c->table)
        for (uint32_t i = g.rank(); i < cap; i += g.size())
            keys[i] = kEmpty;
        g.sync();
    }

    // Lane 0 of a tile: whether occurrences a and b (D's transactions after their positions) keep the same items.
    const auto same_kept = [&](uint32_t a, uint32_t b) {
        Occ ra, rb;
        if (!read_occ(D, a, set, ra) || !read_occ(D, b, set, rb))
            return false;
        uint64_t       i = ra.at + ra.pos + 1, j = rb.at + rb.pos + 1;
        const uint64_t ie = ra.at + ra.len, je = rb.at + rb.len;
        for (;;) {
            while (i < ie && !kept_item(D.items[i]))
                ++i;
            while (j < je && !kept_item(D.items[j]))
                ++j;
            if (i == ie || j == je)
                return i == ie && j == je;
            if (D.items[i] != D.items[j])
                return false;
            ++i, ++j;
        }
    };

    // B1a: per occurrence, its kept items (count and a position-sensitive hash), count_full's test, and with merging
    // whether it is a representative (first of its content in the table) or a duplicate.
    for (uint32_t o = ob + T.index; o < oe; o += T.count) {
        Occ r;
        if (!read_occ(D, o, set, r))
            continue;
        const uint64_t first = r.at + r.pos + 1, n = r.len - r.pos - 1;
        uint32_t       kept = 0, occs = 0;
        uint64_t       hash = 0;
        for (uint64_t c0 = 0; c0 < n; c0 += T.w) {
            const uint64_t j    = c0 + T.lane;
            const uint32_t y    = j < n ? D.items[first + j] : ~0u;
            const bool     keep = j < n && kept_item(y);
            const uint32_t b    = T.ballot(keep);
            if (keep) {
                const uint64_t p = kept + __popc(b & ((1u << T.lane) - 1));
                const uint64_t h = (uint64_t{y} << 32 | p) * 0x9E3779B97F4A7C15ull;
                hash += h ^ (h >> 29);
                // Without merging, every transaction is its own: size the child here (B1b is skipped).
                if (!keys)
                    if (const int32_t sl = s.slot[y]; sl >= 0 && check(static_cast<uint32_t>(sl) < n_prim, set)) {
                        atomicAdd(&s.count[sl], 1u);
                        ++occs;
                    }
            }
            kept += __popc(b);
        }
        hash = T.sum(hash);
        if (!keys)
            occs = T.sum(occs);
        if (T.lane != 0)
            continue;
        if (!keys && kept != 0) {
            atomicAdd(&c->n_tx, 1u);
            atomicAdd(&c->n_items, static_cast<unsigned long long>(kept));
            atomicAdd(&c->n_occ, occs);
        }
        if (kept == n_sec)
            atomicAdd(&c->lower, uint64_t{D.utils[r.at + r.pos]} + D.prefix[r.tid]);
        else if (kept != 0)
            c->all = 0;
        if (!keys)
            continue;
        const uint32_t rel = o - ob;
        if (kept == 0) {
            occ_state[rel] = kNoTx;
            continue;
        }
        const unsigned long long key   = (hash ^ (hash >> 32) ^ kept) & 0xffffffffull;
        uint32_t                 index = static_cast<uint32_t>(key) & (cap - 1);
        uint32_t                 state = kNoTx;
        for (uint32_t probe = 0; probe < cap && state == kNoTx; ++probe, index = (index + 1) & (cap - 1)) {
            unsigned long long cur = keys[index];
            if (cur == kEmpty) {
                cur = atomicCAS(&keys[index], kEmpty, key << 32 | rel);
                if (cur == kEmpty) {
                    state = index; // the representative of this content
                    break;
                }
            }
            if ((cur >> 32) == key && same_kept(o, ob + static_cast<uint32_t>(cur)))
                state = kDup | index;
        }
        check(state != kNoTx, set); // the table has twice the slots needed
        occ_state[rel] = state;
    }
    g.sync();

    // B1b (merging only): sizes of the child, counting each distinct transaction once.
    for (uint32_t o = ob + T.index; keys && o < oe; o += T.count) {
        if (occ_state[o - ob] == kNoTx || (occ_state[o - ob] & kDup))
            continue;
        Occ r;
        if (!read_occ(D, o, set, r))
            continue;
        const uint64_t first = r.at + r.pos + 1, n = r.len - r.pos - 1;
        uint32_t       kept = 0, occs = 0;
        for (uint64_t j = T.lane; j < n; j += T.w) {
            const uint32_t y = D.items[first + j];
            if (!kept_item(y))
                continue;
            ++kept;
            if (const int32_t sl = s.slot[y]; sl >= 0 && check(static_cast<uint32_t>(sl) < n_prim, set)) {
                atomicAdd(&s.count[sl], 1u);
                ++occs;
            }
        }
        kept = T.sum(kept);
        occs = T.sum(occs);
        if (kept == 0 || T.lane != 0)
            continue;
        atomicAdd(&c->n_tx, 1u);
        atomicAdd(&c->n_items, static_cast<unsigned long long>(kept));
        atomicAdd(&c->n_occ, occs);
    }
    if (keys)
        g.sync();
    const auto free_table = [&] {
        if constexpr (G::kGrid)
            if (g.leader() && c->table)
                heap.release(static_cast<uint64_t>(reinterpret_cast<char*>(c->table) - heap.base), c->table_cls);
    };

    // count_full: every extension qualifies through the transactions holding all kept items, or every transaction
    // holds all of them (then the child is one merged transaction).
    if (n_sec < 64 && (c->lower >= min_util || c->all)) {
        const bool exact = c->lower < min_util;
        if (exact)
            for (uint32_t o = ob + T.index; o < oe; o += T.count) {
                Occ r;
                if (!read_occ(D, o, set, r))
                    continue;
                const uint64_t first = r.at + r.pos + 1, n = r.len - r.pos - 1;
                uint32_t       carry = 0;
                for (uint64_t c0 = 0; c0 < n; c0 += T.w) {
                    const uint64_t j    = c0 + T.lane;
                    const bool     keep = j < n && kept_item(D.items[first + j]);
                    const uint32_t b    = T.ballot(keep);
                    const uint32_t at   = carry + __popc(b & ((1u << T.lane) - 1));
                    if (keep && check(at < 64, set))
                        atomicAdd(&c->tail[at], static_cast<unsigned long long>(D.utils[first + j]));
                    carry += __popc(b);
                }
            }
        g.sync();
        if (g.leader()) {
            add_stat(stats.work[0],
                     exact ? count_tail(c->tail, n_sec, c->lower, min_util) : (uint64_t{1} << n_sec) - 1);
            add_stat(stats.work[2], 1);
        }
        free_table();
        clear_marks();
        return;
    }

    // Placement. The grid always builds in the heap and splits; a warp never splits; a block keeps the child on its
    // stack unless REFIM's rule splits it. A full stack spills into the heap.
    const DbLayout L = db_layout(c->n_tx, c->n_items, n_prim, c->n_occ);
    if (g.leader()) {
        const uint64_t work = uint64_t{c->n_occ} * n_prim;
        const uint32_t next = depth + 1;
        // A split child lives in the heap until its last task ends: none while the heap is tight.
        const bool split = G::kGrid || (!G::kWarp && !heap.tight(L.bytes) &&
                                        ((next <= set.max_split_depth && work >= set.split_work) ||
                                         (next <= set.max_hungry_depth && work >= set.hungry_work && hungry)));
        char*      base  = nullptr;
        uint32_t   cls   = kStackDb;
        // (A merge table on the stack sits at its far end until this expansion ends.)
        if (!split && *stack->top + L.bytes + c->table_bytes <= stack->capacity) {
            base = stack->base + *stack->top;
            *stack->top += L.bytes;
            c->outcome = kStack;
        }
        else {
            const uint64_t off = heap.allocate(L.bytes, cls);
            if (off != Heap::kNull) {
                base       = heap.base + off;
                c->outcome = split ? kShared : kOwned;
            }
            else
                c->outcome = kFailed;
        }
        if (base) {
            DbHeader* h  = reinterpret_cast<DbHeader*>(base);
            h->n_tx      = c->n_tx;
            h->n_primary = n_prim;
            h->n_occ     = c->n_occ;
            h->top       = c->top;
            h->n_items   = c->n_items;
            h->bytes     = L.bytes;
            h->refs      = 1;
            h->cls       = cls;
            h->occ_shift = D.h->occ_shift;
        }
        c->child = reinterpret_cast<unsigned long long>(base);
        if (split)
            add_stat(stats.work[3], 1);
    }
    g.sync();
    if (c->outcome == kFailed) {
        free_table();
        clear_marks();
        return;
    }
    const Db C = Db::view(reinterpret_cast<char*>(c->child));

    // Occurrence offsets, and the primary items.
    g.exclusive_scan(s.count, C.occ_begin, s.count, n_prim);
    for (uint32_t j = g.rank(); j < n_prim; j += g.size())
        C.primary[j] = s.chosen[j];
    g.sync();

    // B2: write the child. Kept entries go to consecutive slots in order (ballot prefix counts), coalesced.
    for (uint32_t o = ob + T.index; o < oe; o += T.count) {
        if (keys && (occ_state[o - ob] == kNoTx || (occ_state[o - ob] & kDup)))
            continue;
        Occ r;
        if (!read_occ(D, o, set, r))
            continue;
        const uint64_t first = r.at + r.pos + 1, n = r.len - r.pos - 1;
        uint32_t       kept = 0;
        for (uint64_t j = T.lane; j < n; j += T.w)
            kept += kept_item(D.items[first + j]);
        kept = T.sum(kept);
        if (kept == 0)
            continue;
        uint32_t t   = 0;
        uint64_t out = 0;
        if (T.lane == 0) {
            t   = atomicAdd(&c->tx_cursor, 1u);
            out = atomicAdd(&c->item_cursor, static_cast<unsigned long long>(kept));
        }
        t   = T.from(t, 0);
        out = T.from(out, 0);
        if (!check(t < C.h->n_tx && out + kept <= C.h->n_items, set))
            continue;
        if (T.lane == 0) {
            C.tx_off[t] = static_cast<uint32_t>(out);
            C.tx_len[t] = kept;
            C.prefix[t] = static_cast<Util>(uint64_t{D.utils[r.at + r.pos]} + D.prefix[r.tid]);
            if (keys)
                slot_tx[occ_state[o - ob]] = t;
        }
        uint32_t carry = 0;
        for (uint64_t c0 = 0; c0 < n; c0 += T.w) {
            const uint64_t j    = c0 + T.lane;
            const uint32_t y    = j < n ? D.items[first + j] : ~0u;
            const bool     keep = j < n && kept_item(y);
            const uint32_t b    = T.ballot(keep);
            const uint32_t p    = carry + __popc(b & ((1u << T.lane) - 1));
            if (keep && p < kept) {
                C.items[out + p] = static_cast<Key>(y);
                C.utils[out + p] = D.utils[first + j];
                if (const int32_t sl = s.slot[y]; sl >= 0 && check(static_cast<uint32_t>(sl) < n_prim, set)) {
                    const uint32_t at_occ = atomicAdd(&s.count[sl], 1u);
                    if (check(at_occ < C.h->n_occ && at_occ < C.occ_begin[sl + 1], set))
                        C.occ[at_occ] = make_occ(t, p, C.h->occ_shift);
                }
            }
            carry += __popc(b);
        }
    }
    g.sync();

    // B3: each duplicate adds its prefix and utilities into its representative's transaction (same items, same
    // order), as Level::add does for equal neighbours.
    if (keys) {
        for (uint32_t o = ob + T.index; o < oe; o += T.count) {
            const uint32_t state = occ_state[o - ob];
            if (state == kNoTx || !(state & kDup))
                continue;
            Occ r;
            if (!read_occ(D, o, set, r))
                continue;
            const uint32_t t = slot_tx[state & ~kDup];
            if (!check(t < C.h->n_tx, set))
                continue;
            const uint64_t out = C.tx_off[t], len = C.tx_len[t];
            if (T.lane == 0)
                add(C.prefix[t], uint64_t{D.utils[r.at + r.pos]} + D.prefix[r.tid]);
            const uint64_t first = r.at + r.pos + 1, n = r.len - r.pos - 1;
            uint32_t       carry = 0;
            for (uint64_t c0 = 0; c0 < n; c0 += T.w) {
                const uint64_t j    = c0 + T.lane;
                const bool     keep = j < n && kept_item(D.items[first + j]);
                const uint32_t b    = T.ballot(keep);
                const uint32_t p    = carry + __popc(b & ((1u << T.lane) - 1));
                if (keep && check(p < len && out + p < C.h->n_items, set))
                    add(C.utils[out + p], D.utils[first + j]);
                carry += __popc(b);
            }
        }
        g.sync();
    }
    free_table();
    clear_marks();
}

} // namespace ghuim::inline GHUIM_WIDTHS::refim
