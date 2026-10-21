#pragma once
// The persistent REFIM search as a megakernel Work type.
//
// A task is a range [k, end) of a heap database's primary items. Its block expands them depth first, like REFIM's
// worker: each child that is not split goes on the block's stack of databases and becomes a frame (an open loop
// over its primary items), popped when its loop ends. A child that REFIM's rule splits is built in the heap and
// handed out as range tasks. While some block is idle, the block also offers the unstarted siblings of its
// shallowest open loop (REFIM's offer), copying that database to the heap if it lives on the stack.
//
// Heap databases are reference counted: one reference per task, plus one for a frame that owns one (a child built
// in the heap because the stack was full). The last release frees it.
//
// Wide phase: the grid expands the frontier's tasks (single items) one after another, each child in the heap and
// its primary items appended to the next frontier as single-item tasks.

#include "mine/persistent/megakernel.cuh"
#include "mine/persistent/refim/expand.cuh"
#include "mine/persistent/refim/lane.cuh"
#include "mine/persistent/refim/vertical.cuh"

namespace ghuim::inline GHUIM_WIDTHS::refim {

using persistent::Abort;
using persistent::Control;
using persistent::Spawn;
using persistent::WideLevel;

struct Task
{
    uint64_t db; // address of a heap (or the static root) database
    uint32_t k, end;
    uint32_t depth;
    uint32_t pad;
};

struct Frame
{
    char*    db;
    uint32_t next, end, depth;
    uint32_t on_stack; // 1: db is on the stack, popped back to `mark`; 0: the frame holds a heap reference;
                       // 2: borrowed (a warp's view of its block's frame: nothing to release)
    uint64_t mark;
    uint32_t no_lane; // lane mode was ruled out for this loop's database
    uint32_t vert;    // vertical mode for this loop's database: 0 not checked, 1 ruled out, 2 eligible
};

constexpr uint32_t kMaxChunks    = 64; // most tasks a split or an offer creates
constexpr uint32_t kMaxWarps     = 32;
constexpr uint32_t kVertSiblings = 4; // siblings one vertical call mines
constexpr uint32_t kPoolBatches  = 32; // warp mode's lane-job pool: batches (one bit each in a mask) ...
constexpr uint32_t kPoolJobs     = 256; // ... and jobs

// Warp mode's lane-job pool (see run_warps), in dynamic shared memory.
struct LanePool
{
    char*    batch_db[kPoolBatches];
    uint32_t batch_items[kPoolBatches][kLaneItems], batch_m[kPoolBatches], batch_left[kPoolBatches];
    uint32_t batch_ref[kPoolBatches]; // 1: the batch holds a heap reference to batch_db
    uint32_t batch_free, pool_n, jobs[kPoolJobs];
    uint32_t wjobs[kMaxWarps][32], wjobs_n[kMaxWarps], wsrc[kMaxWarps], wpushed[kMaxWarps];
};

// Bytes of one scratch region for `items` items (Bound, then slot/mark/touched/count/chosen), 16-byte aligned.
__host__ __device__ inline uint64_t scratch_bytes(uint64_t items)
{
    return align16(items * (sizeof(Bound) + 5 * sizeof(uint32_t)));
}

__device__ inline Scratch scratch_in(unsigned char* base, uint32_t items)
{
    uint32_t* w = reinterpret_cast<uint32_t*>(base + uint64_t{items} * sizeof(Bound));
    return {reinterpret_cast<Bound*>(base),
            reinterpret_cast<int32_t*>(w),
            w + items,
            w + 2 * items,
            w + 3 * items,
            w + 4 * items};
}

struct Work
{
    using Task                       = refim::Task;
    static constexpr int kMaxThreads = 512;

    Settings  set;
    Heap      heap;
    Scratch*  scratch;       // one per block, then the grid's
    Counters* grid_counters; // the grid's
    char*     stacks;        // stack_bytes per block
    uint64_t  stack_bytes;
    Frame*    frames; // max_frames per block
    uint32_t  max_frames;
    Control*  control;
    char*     root;
    uint64_t  root_bytes;
    // Warp mode: a loop whose database has at most warp_items entries runs on the block's warps, one sibling
    // subtree per warp (0: off).
    uint32_t warp_items;
    uint32_t borrow_items; // idle warps take siblings of the block's open loops on databases this small (0: off)
    Scratch* warp_scratch; // kMaxWarps per block, used when the warps' scratch is not in shared memory
    Frame*   warp_frames;  // kMaxWarps * max_frames per block
    // Scratch in dynamic shared memory, at smem_offset: 0 none, 1 the block's, 2 the block's and each warp's.
    uint32_t smem_scratch;
    uint32_t smem_offset;
    uint32_t pool_offset; // the LanePool in dynamic shared memory
    // Lane mode: kLaneArena bytes per thread (null: off).
    unsigned char* lane_arena;
    // Vertical mode (refim/vertical.cuh) for small dense databases.
    bool vert_on;

    __device__ static unsigned char* dynamic_smem()
    {
        extern __shared__ __align__(16) unsigned char smem[];
        return smem;
    }

    __device__ Scratch block_scratch() const
    {
        return smem_scratch ? scratch_in(dynamic_smem() + smem_offset, set.num_items) : scratch[blockIdx.x];
    }

    __device__ Scratch warp_scratch_of(uint32_t warp) const
    {
        return smem_scratch == 2 ? scratch_in(dynamic_smem() + smem_offset + (1 + warp) * scratch_bytes(set.num_items),
                                              set.num_items)
                                 : warp_scratch[blockIdx.x * kMaxWarps + warp];
    }

    // Every thread, once per block before the first task: shared-memory scratch starts cleared (slot = -1).
    __device__ void init_block() const
    {
        if (smem_scratch == 0)
            return;
        const uint32_t regions = smem_scratch == 2 ? 1 + blockDim.x / 32 : 1;
        for (uint32_t r = 0; r < regions; ++r) {
            const Scratch sc = r == 0 ? block_scratch() : warp_scratch_of(r - 1);
            for (uint32_t i = threadIdx.x; i < set.num_items; i += blockDim.x) {
                sc.bound[i] = {0, 0};
                sc.slot[i]  = -1;
                sc.mark[i] = sc.touched[i] = sc.count[i] = sc.chosen[i] = 0;
            }
        }
    }

    // Leader: whether db is a whole database inside the root, the heap or this block's stack (an [lo, hi) range),
    // with a header consistent with its size and class. Aborts the run (kCorrupt) when it is not.
    __device__ bool valid_db(const char* db, const char* stack_lo, const char* stack_hi) const
    {
        const auto inside = [&](const char* lo, const char* hi) {
            return db >= lo && db + sizeof(DbHeader) <= hi && reinterpret_cast<uintptr_t>(db) % 16 == 0;
        };
        const char* hi = nullptr;
        if (inside(root, root + root_bytes))
            hi = root + root_bytes;
        else if (inside(heap.base, heap.base + heap.capacity))
            hi = heap.base + heap.capacity;
        else if (inside(stack_lo, stack_hi))
            hi = stack_hi;
        if (!check(hi != nullptr, set))
            return false;
        const DbHeader* h = reinterpret_cast<const DbHeader*>(db);
        if (h->bytes != db_layout(h->n_tx, h->n_items, h->n_primary, h->n_occ).bytes &&
            atomicCAS(reinterpret_cast<unsigned long long*>(&control->debug[0]), 0ull, 1ull) == 0) {
            control->debug[1] = uint64_t{h->n_tx} << 32 | h->n_primary;
            control->debug[2] = uint64_t{h->n_occ} << 32 | h->top;
            control->debug[3] = h->n_items;
            control->debug[4] = h->bytes;
            control->debug[5] = uint64_t{h->refs} << 32 | h->cls;
        }
        return check(h->bytes == db_layout(h->n_tx, h->n_items, h->n_primary, h->n_occ).bytes, set) &&
               check(db + h->bytes <= hi, set) && check(h->top < set.num_items, set) &&
               check(h->n_occ <= h->n_items, set) &&
               check(h->cls == kStaticDb  ? db == root
                     : h->cls == kStackDb ? db >= stack_lo && db < stack_hi
                                          : h->cls < Heap::kClasses && Heap::class_bytes(h->cls) >= h->bytes,
                     set);
    }

    // Leader: drops one reference to a heap database.
    __device__ void release(char* db) const
    {
        DbHeader* h = reinterpret_cast<DbHeader*>(db);
        if (h->cls == kStaticDb || h->cls == kStackDb)
            return;
        const uint32_t before = persistent::fetch_add_acq_rel(&h->refs, ~0u); // -1, ordered after our reads
        if (check(before != 0, set) && before == 1)
            heap.release(static_cast<uint64_t>(db - heap.base), h->cls);
    }

    // Tasks covering [begin, end) of db's primary items, at most `most` (>= 1) of them. Returns how many.
    __device__ static uint32_t chunk(char* db, uint32_t begin, uint32_t end, uint32_t depth, Task* out, uint32_t most)
    {
        const uint32_t n = end - begin, m = min(n, max(1u, most));
        for (uint32_t i = 0; i < m; ++i)
            out[i] = {reinterpret_cast<uint64_t>(db), begin + static_cast<uint32_t>(uint64_t{n} * i / m),
                      begin + static_cast<uint32_t>(uint64_t{n} * (i + 1) / m), depth, 0};
        return m;
    }

    static constexpr int kNothing = -1, kOffered = -2;

    // Tasks one offer to other blocks creates: one per idle block (REFIM hands each sibling to whoever asks; tiny
    // tasks cost a steal each here), within the deques' room. 0: no room.
    __device__ uint32_t offer_chunks(const Spawn<Task>& spawn) const
    {
        return min(min(spawn.idle(), spawn.room()), min(set.split_chunks, kMaxChunks));
    }

    // Leader: hands the unstarted siblings of the shallowest open loop to other blocks (kOffered), or returns that
    // loop's frame index when its database is on the stack and must be copied to the heap first. Loops on the stack
    // are skipped when `no_copy`.
    __device__ int offer(Frame* frames, uint32_t nf, bool no_copy, const Spawn<Task>& spawn, Task* out,
                         BlockStats& stats) const
    {
        if (offer_chunks(spawn) == 0)
            return kNothing;
        for (uint32_t i = 0; i < nf; ++i) {
            Frame& f = frames[i];
            if (f.next >= f.end || (f.on_stack && no_copy))
                continue;
            if (f.on_stack)
                return static_cast<int>(i);
            const uint32_t m = chunk(f.db, f.next, f.end, f.depth, out, offer_chunks(spawn));
            DbHeader*      h = reinterpret_cast<DbHeader*>(f.db);
            if (h->cls != kStaticDb)
                atomicAdd(&h->refs, m);
            f.end = f.next;
            spawn.push(out, m);
            add_stat(stats.work[3], 1);
            return kOffered;
        }
        return kNothing;
    }

    __device__ void run(const Task& task, const Spawn<Task>& spawn, BlockStats& stats) const
    {
        enum : uint32_t
        {
            kDone,
            kAgain,
            kExpand,
            kCopy,
            kWarps
        };
        __shared__ Counters c;
        __shared__ uint64_t stack_top;
        __shared__ uint32_t nf, action, copy_frame, copy_cls;
        __shared__ Frame    cur;
        __shared__ char*    copy_to;
        __shared__ Task     out[kMaxChunks];
        __shared__ bool     hungry, no_copy;
        __shared__ uint32_t warp_next, warp_end;              // warp mode: siblings [warp_next, warp_end) of cur.db
        __shared__ uint32_t block_items[kVertItems], block_m; // scratch of the block's vertical check
        Frame* const        frames = this->frames + static_cast<size_t>(blockIdx.x) * max_frames;
        const Scratch       s      = block_scratch();
        const Stack         stk{stacks + static_cast<size_t>(blockIdx.x) * stack_bytes, stack_bytes, &stack_top};
        const bool          leader = threadIdx.x == 0;
        const BlockGroup    g;
        char* const         stack_lo = stk.base;
        char* const         stack_hi = stk.base + stk.capacity;
        if (leader) {
            stack_top = 0;
            frames[0] = {reinterpret_cast<char*>(task.db), task.k, task.end, task.depth, 0, 0};
            no_copy   = false;
            const bool ok =
                    valid_db(frames[0].db, stack_lo, stack_hi) &&
                    check(task.k < task.end && task.end <= reinterpret_cast<const DbHeader*>(task.db)->n_primary &&
                                  reinterpret_cast<const DbHeader*>(task.db)->cls != kStackDb,
                          set);
            nf = ok ? 1 : 0;
        }
        for (;;) {
            __syncthreads();
            if (leader) {
                action = kDone;
                if (nf != 0 && !persistent::aborted(control)) {
                    Frame& f = frames[nf - 1];
                    action   = kAgain;
                    hungry   = gridDim.x > 1 && spawn.hungry();
                    if (f.next == f.end) {
                        if (f.on_stack == 1)
                            stack_top = f.mark;
                        else if (f.on_stack == 0)
                            release(f.db);
                        --nf;
                    }
                    else if (warp_items != 0 && blockDim.x >= 64 && !hungry && f.end - f.next >= 2 &&
                             (reinterpret_cast<const DbHeader*>(f.db)->n_items <= warp_items ||
                              (vert_on && f.vert != 1 &&
                               (f.vert = vert_eligible(Db::view(f.db), block_items, block_m) ? 2 : 1) == 2))) {
                        // A small database: its remaining siblings go to the warps, a subtree each.
                        cur       = f;
                        warp_next = f.next;
                        warp_end  = f.end;
                        f.next    = f.end;
                        action    = kWarps;
                    }
                    else {
                        // Take the next sibling first (REFIM: next = c + 1 before offering), so a block
                        // always makes progress; then, if some block is idle, offer the shallowest open loop.
                        cur = f;
                        ++f.next;
                        action = kExpand;
                        if (const int o = hungry ? offer(frames, nf, no_copy, spawn, out, stats) : kNothing; o >= 0) {
                            copy_frame = static_cast<uint32_t>(o);
                            action     = kCopy;
                        }
                    }
                }
            }
            __syncthreads();
            if (action == kDone)
                break;
            if (action == kAgain)
                continue;
            if (action == kWarps) {
                const uint64_t t0 = persistent::global_ns();
                run_warps(cur, &warp_next, warp_end, frames, nf - 1, stk, spawn, stats);
                if (leader)
                    add_stat(stats.work[6], persistent::global_ns() - t0);
                continue;
            }
            if (action == kCopy) { // the offered loop's database is on the stack: copy it to the heap first
                const Frame&    f = frames[copy_frame];
                const DbHeader* h = reinterpret_cast<const DbHeader*>(f.db);
                if (leader) {
                    const uint64_t off = heap.tight(h->bytes) ? Heap::kNull : heap.allocate(h->bytes, copy_cls);
                    copy_to            = off == Heap::kNull ? nullptr : heap.base + off;
                    no_copy            = off == Heap::kNull; // the heap is full: stop offering stack loops
                }
                __syncthreads();
                if (copy_to != nullptr) {
                    const uint4* src = reinterpret_cast<const uint4*>(f.db);
                    uint4*       dst = reinterpret_cast<uint4*>(copy_to);
                    for (uint64_t i = threadIdx.x; i < h->bytes / 16; i += blockDim.x)
                        dst[i] = src[i];
                    __syncthreads();
                    if (leader) {
                        Frame&         of = frames[copy_frame];
                        DbHeader*      ch = reinterpret_cast<DbHeader*>(copy_to);
                        const uint32_t m  = chunk(copy_to, of.next, of.end, of.depth, out, offer_chunks(spawn));
                        ch->cls           = copy_cls;
                        ch->refs          = m;
                        of.end            = of.next;
                        spawn.push(out, m);
                        add_stat(stats.work[3], 1);
                    }
                }
            }

            // Expand cur.db's primary item cur.next.
            const uint64_t mark = stack_top;
            expand(g, set, heap, s, &c, Db::view(cur.db), cur.next, cur.depth, hungry, &stk, stats);
            if (leader) {
                char* const    child  = reinterpret_cast<char*>(c.child);
                const uint32_t n_prim = c.n_prim;
                const bool     built  = c.outcome == kStack || c.outcome == kOwned || c.outcome == kShared;
                if (built && !(valid_db(child, stack_lo, stack_hi) &&
                               check(n_prim == reinterpret_cast<const DbHeader*>(child)->n_primary && n_prim > 0 &&
                                             (c.outcome == kShared || nf < max_frames),
                                     set)))
                    c.outcome = kNone; // (the run is aborting)
                switch (c.outcome) {
                case kStack:
                    frames[nf++] = {child, 0, n_prim, cur.depth + 1, 1, mark};
                    break;
                case kOwned:
                    frames[nf++] = {child, 0, n_prim, cur.depth + 1, 0, 0};
                    break;
                case kShared:
                    if (nf >= max_frames || spawn.room() >= min(n_prim, min(set.split_chunks, kMaxChunks))) {
                        const uint32_t m =
                                chunk(child, 0, n_prim, cur.depth + 1, out, min(set.split_chunks, kMaxChunks));
                        reinterpret_cast<DbHeader*>(child)->refs = m;
                        spawn.push(out, m);
                    }
                    else // the deques are full: keep it (the frame holds its one reference)
                        frames[nf++] = {child, 0, n_prim, cur.depth + 1, 0, 0};
                    break;
                case kFailed:
                    persistent::raise_abort(control, Abort::kMemory);
                    break;
                default:
                    break;
                }
            }
        }
        // An abort can leave frames open; their databases are not released (the run is over).
    }

    // Warp mode, every thread of the block: REFIM's worker loop with warps as workers. A warp takes the next sibling
    // of `parent` ([*next, end)), else a task another warp offered to the block's warp queue, and runs its whole
    // subtree with its own counters, frames, scratch and slice of the block's free stack. Between siblings, a busy
    // warp offers the unstarted siblings of its shallowest open loop: to the warp queue while some warp of the block
    // waits, else to the block's deques while some other block is idle (REFIM's offer). Warp mode ends when no
    // sibling or queued task is left and no warp is busy.
    // A warp that finds nothing else also takes the next sibling of an open ancestor loop of the block (ancestors[0,
    // n_ancestors), deepest first) whose database has at most borrow_items entries: the block's frames stay put
    // during warp mode, so their databases outlive it.
    __device__ void run_warps(const Frame& parent, uint32_t* next, uint32_t end, Frame* ancestors, uint32_t n_ancestors,
                              const Stack& block_stack,
                              const Spawn<Task>& spawn, BlockStats& stats) const
    {
        constexpr uint32_t  kQueue = 64, kWarpChunks = 8;
        __shared__ Counters wc[kMaxWarps];
        __shared__ uint64_t wtop[kMaxWarps];
        __shared__ uint32_t wnf[kMaxWarps], wact[kMaxWarps], wcls[kMaxWarps], wcopy[kMaxWarps];
        __shared__ char*    wdst[kMaxWarps];
        __shared__ Frame    wcur[kMaxWarps];
        __shared__ Task     wout[kMaxWarps][kWarpChunks];
        __shared__ Task     queue[kQueue];
        __shared__ uint32_t lock, deque_lock, queued, busy, waiting;
        __shared__ uint32_t lane_items[kMaxWarps][kLaneItems], lane_m[kMaxWarps], lane_next[kMaxWarps],
                lane_end[kMaxWarps];
        __shared__ char*    lane_db[kMaxWarps];
        __shared__ uint32_t vert_items[kMaxWarps][kVertItems], vert_m[kMaxWarps], vert_k[kMaxWarps],
                vert_end[kMaxWarps];
        __shared__ char*       vert_db[kMaxWarps];
        __shared__ const char* vert_items_of[kMaxWarps]; // the database vert_items[warp] describes
        // The lane-job pool: batches of a tiny database's siblings, each sibling a job one lane mines alone. A batch
        // holds its database (a heap reference, or borrowed from the block's frame), its item list and its unfinished
        // jobs; the last job to finish releases it. Jobs (batch << 16 | primary index) are a stack under `lock`.
        // (In dynamic shared memory: static shared memory is full.)
        LanePool& P           = *reinterpret_cast<LanePool*>(dynamic_smem() + pool_offset);
        auto&     batch_items = P.batch_items;
        auto&     batch_m     = P.batch_m;
        auto&     batch_left  = P.batch_left;
        auto&     batch_ref   = P.batch_ref;
        auto&     batch_db    = P.batch_db;
        auto&     batch_free  = P.batch_free;
        auto&     pool_n      = P.pool_n;
        auto&     jobs        = P.jobs;
        auto&     wjobs       = P.wjobs;
        auto&     wjobs_n     = P.wjobs_n;
        auto&     wsrc        = P.wsrc;
        auto&     wpushed     = P.wpushed;
        enum : uint32_t
        {
            kWDone,
            kWAgain,
            kWExpand,
            kWCopy,
            kWLane,
            kWVert,
            kWPool
        };
        const WarpGroup g;
        const uint32_t  warp = threadIdx.x / 32, warps = blockDim.x / 32;
        const bool      leader = g.leader();
        Frame* const    frames = warp_frames + (static_cast<size_t>(blockIdx.x) * kMaxWarps + warp) * max_frames;
        const Scratch   s      = warp_scratch_of(warp);
        const uint64_t  slice  = (block_stack.capacity - *block_stack.top) / warps / 256 * 256;
        const Stack     stk{block_stack.base + *block_stack.top + warp * slice, slice, &wtop[warp]};
        char* const     lo      = stk.base;
        char* const     hi      = stk.base + stk.capacity;
        const auto      acquire = [](uint32_t* l) {
            while (atomicCAS(l, 0u, 1u) != 0)
                __nanosleep(32);
            __threadfence_block();
        };
        const auto release_lock = [](uint32_t* l) {
            __threadfence_block();
            atomicExch(l, 0u);
        };
        // Lane 0, holding `lock`: moves up to 32 jobs from the pool to this warp's lanes.
        const auto take_jobs = [&] {
            const uint32_t n = min(pool_n, 32u);
            for (uint32_t i = 0; i < n; ++i)
                wjobs[warp][i] = jobs[pool_n - n + i];
            pool_n -= n;
            wjobs_n[warp] = n;
            return n;
        };
        // Every lane: mines the job it was given (if any); the last job of a batch releases it.
        const auto mine_jobs = [&] {
            __syncwarp();
            const uint32_t n    = wjobs_n[warp];
            const uint32_t lane = g.rank();
            const uint64_t c0   = clock64();
            if (lane < n) {
                const uint32_t job = wjobs[warp][lane], b = job >> 16, k = job & 0xffff;
                LaneMiner      miner{set,
                                lane_arena + (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kLaneArena,
                                kLaneArena};
                if (check(b < kPoolBatches, set)) {
                    const Db D = Db::view(batch_db[b]);
                    if (!persistent::aborted(control) && check(k < D.h->n_primary, set))
                        miner.mine(D, k, batch_items[b], batch_m[b]);
                    add_stat(stats.work[0], miner.patterns);
                    add_stat(stats.work[1], miner.nodes);
                    add_stat(stats.work[2], miner.counted);
                    add_stat(stats.work[5], miner.nodes);
                    add_stat(stats.work[11], miner.convert_cycles);
                    add_stat(stats.work[13], miner.subtrees);
                    __threadfence_block(); // done reading the batch before it can be released
                    if (atomicSub(&batch_left[b], 1u) == 1) {
                        if (batch_ref[b])
                            release(batch_db[b]);
                        __threadfence_block();
                        atomicOr(&batch_free, 1u << b);
                    }
                }
            }
            add_stat(stats.work[8], clock64() - c0);
            __syncwarp();
            if (leader) {
                add_stat(stats.work[7], clock64() - c0);
                add_stat(stats.work[14], 1);
                add_stat(stats.work[20], n);
            }
        };
        const uint64_t c_begin = clock64();
        if (threadIdx.x == 0) {
            lock = deque_lock = queued = busy = waiting = 0;
            batch_free = kPoolBatches == 32 ? ~0u : (1u << kPoolBatches) - 1;
            pool_n     = 0;
        }
        if ((threadIdx.x & 31) == 0)
            vert_items_of[warp] = nullptr;
        __syncthreads(); // and everyone has read the block's stack top

        bool counted_waiting = false;
        for (;;) {
            // Lane 0 takes work: 1 a sibling, 2 a queued task, 4 lane jobs, 5 an ancestor's sibling, 3 none yet
            // (others are busy), 0 warp mode is over.
            uint32_t       got    = 0;
            Task           t{};
            const uint64_t c_take = clock64();
            if (leader) {
                acquire(&lock);
                if (pool_n >= 32) { // a full warp's worth of lane jobs first: they keep the pool moving
                    take_jobs();
                    got = 4;
                    ++busy;
                }
                else if (*next < end) {
                    t   = {reinterpret_cast<uint64_t>(parent.db), *next, *next + 1, parent.depth, 0};
                    got = 1;
                    ++*next;
                    ++busy;
                }
                else if (queued != 0) {
                    t   = queue[--queued];
                    got = 2;
                    ++busy;
                }
                else if (pool_n != 0) {
                    take_jobs();
                    got = 4;
                    ++busy;
                }
                else if (const int a = open_ancestor(ancestors, n_ancestors); a >= 0) {
                    Frame& o = ancestors[a];
                    t        = {reinterpret_cast<uint64_t>(o.db), o.next, o.next + 1, o.depth, 0};
                    got      = 5;
                    ++o.next;
                    ++busy;
                    add_stat(stats.work[22], 1);
                }
                else
                    got = busy == 0 || persistent::aborted(control) ? 0 : 3;
                release_lock(&lock);
                if (got == 3 && !counted_waiting) {
                    atomicAdd(&waiting, 1u);
                    counted_waiting = true;
                }
                if (got != 3 && counted_waiting) {
                    atomicSub(&waiting, 1u);
                    counted_waiting = false;
                }
            }
            got = __shfl_sync(~0u, got, 0);
            if (leader)
                add_stat(stats.work[got == 3 ? 19 : 17], clock64() - c_take);
            if (got == 0)
                break;
            if (got == 4) {
                mine_jobs();
                if (leader) {
                    acquire(&lock);
                    --busy;
                    release_lock(&lock);
                }
                continue;
            }
            if (got == 3) {
                const uint64_t c0 = clock64();
                __nanosleep(256);
                if (leader)
                    add_stat(stats.work[12], clock64() - c0);
                continue;
            }
            if (leader) {
                // A sibling borrows the parent's database (possibly on the block's stack: not checked against the
                // heap); a queued task holds a heap reference.
                frames[0] = {reinterpret_cast<char*>(t.db), t.k, t.end, t.depth, got == 1 || got == 5 ? 2u : 0u, 0};
                if (got == 1) { // what the block found about the parent's database holds for its siblings
                    frames[0].no_lane = parent.no_lane;
                    frames[0].vert    = parent.vert;
                }
                wnf[warp]  = got == 1 || got == 5 || valid_db(frames[0].db, nullptr, nullptr) ? 1 : 0;
                wtop[warp] = 0;
            }
            for (;;) {
                __syncwarp();
                const uint64_t c_decide = clock64();
                if (leader) {
                    wact[warp] = kWDone;
                    if (wnf[warp] != 0 && !persistent::aborted(control)) {
                        Frame& f   = frames[wnf[warp] - 1];
                        wact[warp] = kWAgain;
                        if (f.next == f.end) {
                            if (f.on_stack == 1)
                                wtop[warp] = f.mark;
                            else if (f.on_stack == 0)
                                release(f.db);
                            --wnf[warp];
                        }
                        else {
                            // Lane mode for a tiny database, else vertical mode for a small dense one, else the
                            // warp's passes. Each check runs once per loop.
                            if (lane_arena != nullptr && !f.no_lane) {
                                const uint64_t c0       = clock64();
                                const bool     eligible = lane_eligible(Db::view(f.db), lane_items[warp], lane_m[warp]);
                                add_stat(stats.work[10], clock64() - c0);
                                if (eligible) {
                                    // Its remaining siblings go to the lanes, a subtree each: through the pool
                                    // when it has room (a database on this warp's stack is copied to the heap),
                                    // else to this warp's own lanes.
                                    const bool pool =
                                            load_relaxed_u32(&batch_free) != 0 &&
                                            load_relaxed_u32(&pool_n) + (f.end - f.next) <= kPoolJobs &&
                                            !(f.on_stack == 1 &&
                                              heap.tight(reinterpret_cast<const DbHeader*>(f.db)->bytes));
                                    lane_db[warp]   = f.db;
                                    lane_next[warp] = f.next;
                                    lane_end[warp]  = f.end;
                                    wsrc[warp]      = f.on_stack;
                                    f.next          = f.end;
                                    wact[warp]      = pool ? kWPool : kWLane;
                                }
                                else
                                    f.no_lane = 1;
                            }
                            if (wact[warp] != kWLane && vert_on && f.vert != 1) {
                                const bool room = *stk.top + vert_bytes() + 16 <= stk.capacity;
                                if (vert_items_of[warp] == f.db)
                                    f.vert = 2; // this warp's item list already describes it
                                else if (room) {
                                    f.vert = vert_eligible(Db::view(f.db), vert_items[warp], vert_m[warp]) ? 2 : 1;
                                    vert_items_of[warp] = f.vert == 2 ? f.db : nullptr;
                                }
                                if (f.vert == 2 && room) {
                                    // A few siblings per call: the build is amortized, and the rest can still be
                                    // offered.
                                    vert_db[warp]  = f.db;
                                    vert_k[warp]   = f.next;
                                    vert_end[warp] = min(f.end, f.next + kVertSiblings);
                                    f.next         = vert_end[warp];
                                    wact[warp]     = kWVert;
                                }
                            }
                        }
                        // Neither lane nor vertical mode took it, and the frame was not just popped: expand.
                        if (wact[warp] == kWAgain && f.next != f.end) {
                            wcur[warp] = f;
                            ++f.next;
                            wact[warp] = kWExpand;
                            // Offer the shallowest open loop to a waiting warp, or to an idle block.
                            const bool     to_warps = load_relaxed_u32(&waiting) != 0;
                            const uint32_t most     = to_warps ? min(load_relaxed_u32(&waiting), kWarpChunks)
                                                               : min(offer_chunks(spawn), kWarpChunks);
                            if (most != 0 && (to_warps || (gridDim.x > 1 && spawn.hungry())))
                                for (uint32_t i = 0; i < wnf[warp]; ++i) {
                                    Frame& o = frames[i];
                                    if (o.next >= o.end || o.on_stack == 2)
                                        continue;
                                    if (o.on_stack == 1) { // copy it to the heap first
                                        if (heap.tight(reinterpret_cast<const DbHeader*>(o.db)->bytes))
                                            break;
                                        wcopy[warp] = i;
                                        wact[warp]  = kWCopy;
                                        break;
                                    }
                                    offer_range(o, o.db, most, to_warps, queue, &queued, &lock, &deque_lock, spawn,
                                                wout[warp], stats, acquire, release_lock);
                                    break;
                                }
                        }
                    }
                }
                __syncwarp();
                if (leader)
                    add_stat(stats.work[18], clock64() - c_decide);
                uint32_t act = wact[warp];
                if (act == kWPool) {
                    // The database, in the heap or borrowed: the batch can outlive this warp's frame.
                    const DbHeader* h = reinterpret_cast<const DbHeader*>(lane_db[warp]);
                    if (leader) {
                        wdst[warp] = nullptr;
                        if (wsrc[warp] == 1) {
                            const uint64_t off = heap.allocate(h->bytes, wcls[warp]);
                            wdst[warp]         = off == Heap::kNull ? nullptr : heap.base + off;
                        }
                    }
                    __syncwarp();
                    char* const copy = wdst[warp];
                    if (copy != nullptr) {
                        const uint4* src = reinterpret_cast<const uint4*>(lane_db[warp]);
                        for (uint64_t i = g.rank(); i < h->bytes / 16; i += 32)
                            reinterpret_cast<uint4*>(copy)[i] = src[i];
                        __syncwarp();
                    }
                    if (leader) {
                        bool pushed = false;
                        if (wsrc[warp] != 1 || copy != nullptr) {
                            char* const    db = copy != nullptr ? copy : lane_db[warp];
                            const uint32_t n  = lane_end[warp] - lane_next[warp];
                            if (copy != nullptr) {
                                DbHeader* ch = reinterpret_cast<DbHeader*>(copy);
                                ch->cls      = wcls[warp];
                                ch->refs     = 1; // the batch's
                            }
                            acquire(&lock);
                            if (load_relaxed_u32(&batch_free) != 0 && pool_n + n <= kPoolJobs) {
                                const uint32_t b = __ffs(load_relaxed_u32(&batch_free)) - 1;
                                atomicAnd(&batch_free, ~(1u << b));
                                for (uint32_t i = 0; i < lane_m[warp]; ++i)
                                    batch_items[b][i] = lane_items[warp][i];
                                batch_m[b]    = lane_m[warp];
                                batch_left[b] = n;
                                batch_db[b]   = db;
                                batch_ref[b]  = wsrc[warp] != 2;
                                if (wsrc[warp] == 0 && reinterpret_cast<const DbHeader*>(db)->cls != kStaticDb)
                                    atomicAdd(&reinterpret_cast<DbHeader*>(db)->refs, 1u); // the frame keeps its own
                                for (uint32_t k = lane_end[warp]; k-- > lane_next[warp];) // lowest k on top
                                    jobs[pool_n++] = b << 16 | k;
                                pushed = true;
                                add_stat(stats.work[21], 1);
                            }
                            wjobs_n[warp] = pushed && pool_n >= 32 ? take_jobs() : 0; // a full warp: mine now
                            release_lock(&lock);
                            if (!pushed && copy != nullptr)
                                heap.release(static_cast<uint64_t>(copy - heap.base), wcls[warp]);
                        }
                        wpushed[warp] = pushed;
                    }
                    __syncwarp();
                    if (wpushed[warp]) {
                        if (wjobs_n[warp] != 0)
                            mine_jobs();
                        continue;
                    }
                    act = kWLane; // no room after all: this warp's lanes mine them
                }
                if (act == kWDone)
                    break;
                if (act == kWAgain)
                    continue;
                if (act == kWLane) {
                    LaneMiner miner{set,
                                    lane_arena +
                                            (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kLaneArena,
                                    kLaneArena};
                    const Db       D  = Db::view(lane_db[warp]);
                    const uint64_t c0 = clock64();
                    for (;;) {
                        const uint32_t k = atomicAdd(&lane_next[warp], 1u);
                        if (k >= lane_end[warp] || persistent::aborted(control))
                            break;
                        miner.mine(D, k, lane_items[warp], lane_m[warp]);
                    }
                    add_stat(stats.work[8], clock64() - c0);
                    __syncwarp();
                    if (leader)
                        add_stat(stats.work[7], clock64() - c0);
                    add_stat(stats.work[0], miner.patterns);
                    add_stat(stats.work[1], miner.nodes);
                    add_stat(stats.work[2], miner.counted);
                    add_stat(stats.work[5], miner.nodes);
                    add_stat(stats.work[11], miner.convert_cycles);
                    add_stat(stats.work[13], miner.subtrees);
                    if (leader)
                        add_stat(stats.work[14], 1);
                    continue;
                }
                if (act == kWVert) {
                    unsigned char* scratch = reinterpret_cast<unsigned char*>(
                            (reinterpret_cast<uintptr_t>(stk.base + *stk.top) + 15) & ~uintptr_t{15});
                    vert_mine(set, Db::view(vert_db[warp]), vert_items[warp], vert_m[warp], vert_k[warp],
                              vert_end[warp], scratch, stats);
                    continue;
                }
                if (act == kWCopy) { // the whole warp copies the loop's database, then lane 0 offers it
                    const Frame&    o = frames[wcopy[warp]];
                    const DbHeader* h = reinterpret_cast<const DbHeader*>(o.db);
                    if (leader) {
                        const uint64_t off = heap.allocate(h->bytes, wcls[warp]);
                        wdst[warp]         = off == Heap::kNull ? nullptr : heap.base + off;
                    }
                    __syncwarp();
                    if (char* const dst = wdst[warp]) {
                        const uint4* src = reinterpret_cast<const uint4*>(o.db);
                        for (uint64_t i = g.rank(); i < h->bytes / 16; i += 32)
                            reinterpret_cast<uint4*>(dst)[i] = src[i];
                        __syncwarp();
                        if (leader) {
                            DbHeader* ch            = reinterpret_cast<DbHeader*>(dst);
                            ch->cls                 = wcls[warp];
                            ch->refs                = 0; // offer_range adds one per task
                            const bool     to_warps = load_relaxed_u32(&waiting) != 0;
                            const uint32_t most     = to_warps ? min(load_relaxed_u32(&waiting), kWarpChunks)
                                                               : min(offer_chunks(spawn), kWarpChunks);
                            offer_range(frames[wcopy[warp]], dst, max(1u, most), to_warps, queue, &queued, &lock,
                                        &deque_lock, spawn, wout[warp], stats, acquire, release_lock);
                            if (ch->refs == 0) // nothing was taken after all
                                heap.release(static_cast<uint64_t>(dst - heap.base), ch->cls);
                        }
                    }
                }
                const Frame    cur  = wcur[warp];
                const uint64_t mark = wtop[warp];
                const uint64_t c0   = clock64();
                expand(g, set, heap, s, &wc[warp], Db::view(cur.db), cur.next, cur.depth, false, &stk, stats);
                if (leader)
                    add_stat(stats.work[9], clock64() - c0);
                if (leader) {
                    Counters&      c      = wc[warp];
                    char* const    child  = reinterpret_cast<char*>(c.child);
                    const uint32_t n_prim = c.n_prim;
                    const bool     built  = c.outcome == kStack || c.outcome == kOwned;
                    if (built && !(valid_db(child, lo, hi) &&
                                   check(n_prim == reinterpret_cast<const DbHeader*>(child)->n_primary && n_prim > 0 &&
                                                 wnf[warp] < max_frames,
                                         set)))
                        c.outcome = kNone; // (the run is aborting)
                    if (c.outcome == kStack)
                        frames[wnf[warp]++] = {child, 0, n_prim, cur.depth + 1, 1, mark};
                    else if (c.outcome == kOwned)
                        frames[wnf[warp]++] = {child, 0, n_prim, cur.depth + 1, 0, 0};
                    else if (c.outcome == kFailed || c.outcome == kShared)
                        persistent::raise_abort(control, Abort::kMemory);
                }
            }
            if (leader) {
                acquire(&lock);
                --busy;
                release_lock(&lock);
            }
        }
        const uint64_t c_end = clock64();
        __syncthreads(); // every warp is done before the block goes on (its stack slices are free again)
        if (leader) {
            add_stat(stats.work[15], clock64() - c_end);
            add_stat(stats.work[16], clock64() - c_begin);
        }
    }

    // Lane 0, holding warp mode's lock: the deepest open ancestor loop small enough for a warp, or -1.
    __device__ int open_ancestor(const Frame* ancestors, uint32_t n) const
    {
        for (uint32_t i = n; i-- > 0;) {
            const Frame& o = ancestors[i];
            if (o.next < o.end && reinterpret_cast<const DbHeader*>(o.db)->n_items <= borrow_items)
                return static_cast<int>(i);
        }
        return -1;
    }

    __device__ static uint32_t load_relaxed_u32(const uint32_t* p)
    {
        return *reinterpret_cast<const volatile uint32_t*>(p);
    }

    // Lane 0 of a warp: hands siblings [o.next, o.end) of loop `o` (whose database, in the heap, is `db`) to the
    // warp queue or the block's deques, at most `most` tasks, one heap reference each. Leaves o.end = o.next on
    // success.
    template <typename Lock, typename Unlock>
    __device__ void offer_range(Frame& o, char* db, uint32_t most, bool to_warps, Task* queue, uint32_t* queued,
                                uint32_t* lock, uint32_t* deque_lock, const Spawn<Task>& spawn, Task* out,
                                BlockStats& stats, const Lock& acquire, const Unlock& release_lock) const
    {
        constexpr uint32_t kQueue = 64;
        DbHeader*          h      = reinterpret_cast<DbHeader*>(db);
        if (to_warps) {
            acquire(lock);
            const uint32_t m = chunk(db, o.next, o.end, o.depth, out, min(most, kQueue - *queued));
            if (*queued + m <= kQueue && m != 0) {
                if (h->cls != kStaticDb)
                    atomicAdd(&h->refs, m);
                for (uint32_t i = m; i-- > 0;)
                    queue[(*queued)++] = out[i];
                o.end = o.next;
            }
            release_lock(lock);
            return;
        }
        acquire(deque_lock); // the deques have one owner: warps push one at a time
        if (spawn.room() >= 1) {
            const uint32_t m = chunk(db, o.next, o.end, o.depth, out, min(most, spawn.room()));
            if (h->cls != kStaticDb)
                atomicAdd(&h->refs, m);
            o.end = o.next;
            spawn.push(out, m);
            add_stat(stats.work[3], 1);
        }
        release_lock(deque_lock);
    }

    // Every thread of the grid: a frontier task on a valid database (same answer everywhere).
    __device__ bool valid_task(const Task& t) const
    {
        return valid_db(reinterpret_cast<const char*>(t.db), nullptr, nullptr) &&
               check(t.k < t.end && t.end <= reinterpret_cast<const DbHeader*>(t.db)->n_primary, set);
    }

    __device__ static uint64_t task_size(const Task& t)
    {
        const Db d = Db::view(reinterpret_cast<char*>(t.db));
        return d.occ_begin[t.end] - d.occ_begin[t.k];
    }

    // One wide level (every thread of the grid).
    __device__ void wide(cg::grid_group& grid, const WideLevel<Task>& w, BlockStats& stats) const
    {
        const GridGroup g{&grid};
        Counters* const c = grid_counters;
        const Scratch   s = scratch[gridDim.x];
        const auto chosen = [&](const Task& t) { return valid_task(t) && (w.need_tasks || task_size(t) > w.grain); };

        // Room for the worst case: every expanded task's child extends every larger item.
        if (g.leader())
            c->lower = 0;
        g.sync();
        for (uint32_t i = g.rank(); i < w.n; i += g.size()) {
            const Task& t = w.frontier[i];
            const Db    d = Db::view(reinterpret_cast<char*>(t.db));
            atomicAdd(&c->lower,
                      chosen(t) ? static_cast<unsigned long long>(max(1u, d.h->top - d.primary[t.k])) : 1ull);
        }
        g.sync();
        const bool room = c->lower <= w.next_capacity;
        g.sync(); // everyone has read c->lower before expand() resets it

        // Tasks not expanded move to the next frontier. Decided before any expansion: an expanded task releases
        // its database, which may then be freed and reused, so its header must not be read again.
        for (uint32_t i = g.rank(); i < w.n; i += g.size()) {
            const Task& t = w.frontier[i];
            if (!valid_task(t) || (room && chosen(t)))
                continue;
            w.next[atomicAdd(w.next_n, 1u)] = t;
            atomicMax(reinterpret_cast<unsigned long long*>(w.largest), static_cast<unsigned long long>(task_size(t)));
        }
        g.sync();

        for (uint32_t i = 0; room && i < w.n; ++i) {
            const Task t = w.frontier[i];
            if (!chosen(t))
                continue;
            // Children of the wide phase all live at once: stop expanding (carry the task instead) while the heap is
            // past half full. A child is never larger than its parent. Read after a sync, so every thread agrees.
            const DbHeader* th = reinterpret_cast<const DbHeader*>(t.db);
            if (persistent::load_relaxed(heap.live) + th->bytes > heap.capacity / 2) {
                if (g.leader()) {
                    w.next[atomicAdd(w.next_n, 1u)] = t;
                    atomicMax(reinterpret_cast<unsigned long long*>(w.largest),
                              static_cast<unsigned long long>(task_size(t)));
                }
                continue;
            }
            expand(g, set, heap, s, c, Db::view(reinterpret_cast<char*>(t.db)), t.k, t.depth, false, nullptr, stats);
            const uint32_t outcome = c->outcome;
            if (g.leader()) {
                ++stats.wide_tasks;
                if (outcome == kShared) {
                    reinterpret_cast<DbHeader*>(c->child)->refs = c->n_prim;
                    c->item_cursor                              = atomicAdd(w.next_n, c->n_prim);
                }
                if (outcome == kFailed)
                    persistent::raise_abort(control, Abort::kMemory);
                release(reinterpret_cast<char*>(t.db));
            }
            g.sync();
            if (outcome == kFailed)
                break; // every thread: the outcome was read after a sync
            if (outcome == kShared && valid_db(reinterpret_cast<char*>(c->child), nullptr, nullptr) &&
                check(c->item_cursor + c->n_prim <= w.next_capacity, set)) {
                char* const    child = reinterpret_cast<char*>(c->child);
                const Db       C     = Db::view(child);
                const uint64_t base  = c->item_cursor;
                for (uint32_t j = g.rank(); j < c->n_prim; j += g.size()) {
                    w.next[base + j] = {reinterpret_cast<uint64_t>(child), j, j + 1, t.depth + 1, 0};
                    atomicMax(reinterpret_cast<unsigned long long*>(w.largest),
                              static_cast<unsigned long long>(C.occ_begin[j + 1] - C.occ_begin[j]));
                }
                g.sync(); // c is reset by the next expand
            }
        }
    }
};

} // namespace ghuim::inline GHUIM_WIDTHS::refim
