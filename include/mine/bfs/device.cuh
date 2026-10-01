#pragma once
// Device code of the BFS optimized miner: atomics, thread groups, and the passes that process one
// task (project its database on the extension item and build the child database).

#include "mine/bfs/layout.cuh"

#include <cuda/atomic>

namespace bfs {

constexpr unsigned kFullMask = 0xffffffffu;

// ---------------------------------------------------------------------------------------------
// Atomics
// ---------------------------------------------------------------------------------------------

template <cuda::thread_scope Scope = cuda::thread_scope_device, typename T, typename V>
__device__ inline T fetch_add(T& target, V value)
{
    return cuda::atomic_ref<T, Scope>(target).fetch_add(static_cast<T>(value), cuda::std::memory_order_relaxed);
}

template <cuda::thread_scope Scope = cuda::thread_scope_device, typename T, typename V>
__device__ inline T fetch_max(T& target, V value)
{
    return cuda::atomic_ref<T, Scope>(target).fetch_max(static_cast<T>(value), cuda::std::memory_order_relaxed);
}

template <cuda::thread_scope Scope = cuda::thread_scope_device, typename T, typename V>
__device__ inline T fetch_or(T& target, V value)
{
    return cuda::atomic_ref<T, Scope>(target).fetch_or(static_cast<T>(value), cuda::std::memory_order_relaxed);
}

// Fire-and-forget add: nvcc turns an unused atomicAdd into a RED instruction.
template <typename T, typename V> __device__ inline void add(T& target, V value)
{
    if constexpr (sizeof(T) == 8)
        atomicAdd(reinterpret_cast<unsigned long long*>(&target), static_cast<unsigned long long>(value));
    else
        atomicAdd(&target, static_cast<T>(value));
}

// Stores `desired` if `target` holds `expected`; returns the value seen (like atomicCAS).
template <cuda::thread_scope Scope = cuda::thread_scope_device, typename T>
__device__ inline T compare_swap(T& target, T expected, T desired)
{
    cuda::atomic_ref<T, Scope>(target).compare_exchange_strong(expected, desired, cuda::std::memory_order_relaxed);
    return expected;
}

// ---------------------------------------------------------------------------------------------
// Scans
// ---------------------------------------------------------------------------------------------

template <typename T> __device__ __forceinline__ T warp_inclusive_scan(T value)
{
    const unsigned lane = threadIdx.x & 31;
#pragma unroll
    for (unsigned offset = 1; offset < 32; offset <<= 1) {
        const T other = __shfl_up_sync(kFullMask, value, offset);
        if (lane >= offset)
            value += other;
    }
    return value;
}

// Inclusive scan across the block (blockDim.x a multiple of 32). `shared` holds 32 values.
template <typename T> __device__ T block_inclusive_scan(T value, T* shared)
{
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x / 32, warps = blockDim.x / 32;
    value = warp_inclusive_scan(value);
    if (lane == 31)
        shared[warp] = value;
    __syncthreads();
    if (warp == 0) {
        T total      = lane < warps ? shared[lane] : T{0};
        shared[lane] = warp_inclusive_scan(total);
    }
    __syncthreads();
    if (warp > 0)
        value += shared[warp - 1];
    __syncthreads();
    return value;
}

// ---------------------------------------------------------------------------------------------
// Thread groups. Every group exposes the lanes of its warp that work together (`lanes()`, a power
// of two <= 32, and their `mask()`); collectives are over those lanes only, so several groups can
// share a warp and finish at different times.
// ---------------------------------------------------------------------------------------------

struct WarpGroup
{
    __device__ uint32_t rank() const { return threadIdx.x & 31; }
    __device__ uint32_t size() const { return 32; }
    __device__ uint32_t lanes() const { return 32; }
    __device__ unsigned mask() const { return kFullMask; }
    __device__ void     sync() const { __syncwarp(); }
    __device__ uint32_t scan(uint32_t value) const { return warp_inclusive_scan(value); }
    __device__ uint32_t count(bool flag) const { return __popc(__ballot_sync(kFullMask, flag)); }
};

// `lanes` consecutive lanes of a warp (a power of two) form a group.
struct SubGroup
{
    uint32_t n;
    __device__ explicit SubGroup(uint32_t lanes) : n(lanes) {}
    __device__ uint32_t rank() const { return threadIdx.x & (n - 1); }
    __device__ uint32_t size() const { return n; }
    __device__ uint32_t lanes() const { return n; }
    __device__ unsigned mask() const
    {
        return n == 32 ? kFullMask : ((1u << n) - 1) << (threadIdx.x & 31 & ~(n - 1));
    }
    __device__ void     sync() const { __syncwarp(mask()); }
    __device__ uint32_t scan(uint32_t value) const
    {
        for (uint32_t offset = 1; offset < n; offset <<= 1) {
            const uint32_t other = __shfl_up_sync(mask(), value, offset, n);
            if (rank() >= offset)
                value += other;
        }
        return value;
    }
    __device__ uint32_t count(bool flag) const { return __popc(__ballot_sync(mask(), flag)); }
};

struct BlockGroup
{
    __device__ uint32_t rank() const { return threadIdx.x; }
    __device__ uint32_t size() const { return blockDim.x; }
    __device__ uint32_t lanes() const { return 32; }
    __device__ unsigned mask() const { return kFullMask; }
    __device__ void     sync() const { __syncthreads(); }
    __device__ uint32_t scan(uint32_t value) const
    {
        __shared__ uint32_t carry[32];
        return block_inclusive_scan<uint32_t>(value, carry);
    }
    __device__ uint32_t count(bool flag) const { return static_cast<uint32_t>(__syncthreads_count(flag)); }
};

// The warp lanes of a group: lane `rank` handles transactions rank, rank + step, ... Loops bounded
// by `max` run in lockstep, and every lane of the group must call a reduction.
struct Lanes
{
    uint32_t count, rank;
    unsigned mask;

    template <typename Group>
    __device__ explicit Lanes(const Group& g)
        : count(g.lanes()), rank((threadIdx.x & 31) & (g.lanes() - 1)), mask(g.mask())
    {
    }
    __device__ uint32_t max(uint32_t value) const { return count == 1 ? value : __reduce_max_sync(mask, value); }
    template <typename T> __device__ T sum(T value) const
    {
        for (uint32_t offset = count >> 1; offset > 0; offset >>= 1)
            value += __shfl_xor_sync(mask, value, offset, count);
        return value;
    }
    __device__ bool leader() const { return rank == 0; }
};

// ---------------------------------------------------------------------------------------------
// Per-chunk view and per-task view
// ---------------------------------------------------------------------------------------------

// Device pointers shared by every kernel processing one chunk of a level.
struct BatchView
{
    const char*     src;            // region holding the parent databases
    char*           dst;            // region receiving child databases
    uint64_t        dst_capacity;
    Control*        control;
    const Task*     tasks;          // the whole level; kernels index it with order[]
    const uint64_t* scratch_prefix; // exclusive prefix of scratch_bytes over the level
    uint64_t        scratch_base;   // scratch_prefix of the chunk's first task
    char*           scratch;
    Task*           next;           // the next level
    uint32_t        next_capacity;
    Node*           nodes;          // one per task of the level
    uint64_t        minimum_utility;
};

__device__ __forceinline__ uint32_t mix(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Position of `item` in a sorted transaction, or -1.
__device__ inline int find_item(const Entry* entries, uint32_t length, uint32_t item)
{
    int low = 0, high = static_cast<int>(length) - 1;
    while (low <= high) {
        const int      mid = (low + high) >> 1;
        const uint32_t key = entries[mid].item;
        if (key == item)
            return mid;
        if (key < item)
            low = mid + 1;
        else
            high = mid - 1;
    }
    return -1;
}

// Whether two parent transactions project to the same kept items after `item`.
__device__ inline bool same_kept_items(Tx a, const Entry* entries, Tx b, uint32_t item, const uint32_t* rename)
{
    const Entry* ea = entries + a.begin;
    const Entry* eb = entries + b.begin;
    uint32_t     i  = find_item(ea, a.length, item) + 1;
    uint32_t     j  = find_item(eb, b.length, item) + 1;
    for (;;) {
        while (i < a.length && rename[ea[i].item] == kNone)
            ++i;
        while (j < b.length && rename[eb[j].item] == kNone)
            ++j;
        if (i == a.length || j == b.length)
            return i == a.length && j == b.length;
        if (ea[i].item != eb[j].item)
            return false;
        ++i;
        ++j;
    }
}

// The arrays the transaction passes work on.
struct Projection
{
    const Tx*           txs;
    const Entry*        entries;
    uint32_t            n_trans;
    uint32_t            item;    // extension item (local name in the parent)
    unsigned long long* local;   // by parent item name
    unsigned long long* subtree; // by child item name
    uint32_t*           rename;  // parent item -> child item, or kNone
    uint32_t*           table;   // merge hash table over kept-item sequences
    uint32_t            table_size;
    uint32_t*           rep;     // representative of each projected transaction
    uint32_t*           slot_of; // child transaction written by a representative
};

struct TaskView
{
    Task        task;
    uint32_t    index; // position in the level
    const char* base;
    DbHeader    header;
    DbLayout    layout;
    TaskState*  state;
    Projection  p;
};

// Scratch layout: TaskState | local[n] | subtree[n] | rename[n] | table[2t] | rep[t] | slot[t].
__device__ __forceinline__ TaskView open_task(const BatchView& b, uint32_t t)
{
    TaskView v;
    v.task        = b.tasks[t];
    v.index       = t;
    v.base        = b.src + v.task.db_offset();
    v.header      = *reinterpret_cast<const DbHeader*>(v.base);
    v.layout      = DbLayout::of(v.header.n_trans, v.header.n_entries, v.header.n_items);
    char* scratch = b.scratch + (b.scratch_prefix[t] - b.scratch_base);
    v.state       = reinterpret_cast<TaskState*>(scratch);

    const uint32_t n = v.header.n_items, trans = v.header.n_trans;
    Projection&    p = v.p;
    p.txs            = reinterpret_cast<const Tx*>(v.base + v.layout.tx);
    p.entries        = reinterpret_cast<const Entry*>(v.base + v.layout.entries);
    p.n_trans        = trans;
    p.item           = v.task.item;
    p.local          = reinterpret_cast<unsigned long long*>(scratch + sizeof(TaskState));
    p.subtree        = p.local + n;
    p.rename         = reinterpret_cast<uint32_t*>(p.subtree + n);
    p.table_size     = 2 * trans;
    p.table          = p.rename + n;
    p.rep            = p.table + p.table_size;
    p.slot_of        = p.rep + trans;
    return v;
}

// Zeroes the per-item accumulators and the task state; `rank`/`stride` split the work.
__device__ __forceinline__ void clear_task(const TaskView& v, uint32_t rank, uint32_t stride)
{
    const uint32_t first = v.task.item + 1, n = v.header.n_items;
    if (rank == 0)
        *v.state = TaskState{};
    for (uint32_t i = first + rank; i < n; i += stride)
        v.p.local[i] = 0;
    for (uint32_t i = rank; i < n - first; i += stride)
        v.p.subtree[i] = 0;
    for (uint32_t i = rank; i < v.p.table_size; i += stride)
        v.p.table[i] = kNone;
}

// Records the task's itemset; one thread per task calls it once the utility is complete.
__device__ inline void finish_task(const BatchView& b, const TaskView& v)
{
    const uint64_t  utility = v.state->utility;
    const uint32_t* globals = reinterpret_cast<const uint32_t*>(v.base + v.layout.items);
    b.nodes[v.index]        = Node{v.task.parent, globals[v.task.item], utility};
    if (utility >= b.minimum_utility)
        add(b.control->patterns, 1ull);
}

// ---------------------------------------------------------------------------------------------
// Transaction passes. A caller covers transactions i0 = from, from + step, ...; lane `rank` of the
// group handles transaction i0 + rank. After the search, a lane's work is the entries behind the
// extension item: `rest` of them, starting at e[k + 1].
// ---------------------------------------------------------------------------------------------

// Pass 1: local utilities of later items, and the itemset's utility.
__device__ inline void scan_local(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step,
                                  TaskState& state)
{
    uint64_t utility = 0;
    for (uint32_t i0 = from; i0 < p.n_trans; i0 += step) {
        const uint32_t i      = i0 + lanes.rank;
        const Tx       tx     = i < p.n_trans ? p.txs[i] : Tx{0, 0, 0};
        const Entry*   e      = p.entries + tx.begin;
        const int      k      = find_item(e, tx.length, p.item);
        const uint32_t rest   = k >= 0 ? tx.length - k - 1 : 0;
        const uint32_t span   = lanes.max(rest);
        const uint64_t prefix = k >= 0 ? uint64_t{tx.utility} + e[k].utility : 0;
        uint64_t       after  = 0;
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

struct ChildSize
{
    uint32_t trans = 0, entries = 0, max_len = 0;
};

// Pass 2: subtree utilities of kept items, merging of identical projected transactions, and the
// exact size of the child database.
__device__ inline void scan_subtree(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step,
                                    TaskState& state)
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
        uint64_t kept_sum = 0;
        uint32_t kept     = 0;
        for (uint32_t d = 0; d < span; ++d)
            if (d < rest && p.rename[e[k + 1 + d].item] != kNone) {
                ++kept;
                kept_sum += e[k + 1 + d].utility;
            }
        // Sweep 2: subtree(z) = prefix + utility of z and every kept item after it; hash the sequence.
        const uint64_t remaining = (k >= 0 ? uint64_t{tx.utility} + e[k].utility : 0) + kept_sum;
        uint64_t       before    = 0;
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

// ---------------------------------------------------------------------------------------------
// Child database
// ---------------------------------------------------------------------------------------------

// Counts the promising extensions; one thread then allocates the exact-size child database,
// records the task's itemset, and reserves next-level slots. All threads of the group must call it.
template <typename Group>
__device__ void allocate_child(const Group& g, const BatchView& b, const TaskView& v, uint32_t child_items)
{
    TaskState& state     = *v.state;
    uint32_t   primaries = 0;
    for (uint32_t z0 = 0; z0 < child_items; z0 += g.size()) {
        const uint32_t z = z0 + g.rank();
        primaries += g.count(z < child_items && v.p.subtree[z] >= b.minimum_utility);
    }
    if (g.rank() == 0) {
        if (primaries != 0) {
            const DbLayout child = DbLayout::of(state.child_trans, state.child_entries, child_items);
            const uint64_t at    = fetch_add(b.control->arena_top, child.bytes);
            if (at + child.bytes > b.dst_capacity) {
                fetch_or(b.control->error, kArenaFull);
                primaries = 0;
            }
            else {
                *reinterpret_cast<DbHeader*>(b.dst + at) =
                        DbHeader{state.child_trans, state.child_entries, child_items, state.child_max_len, child.bytes};
                state.child_db = at;
                state.reserved = fetch_add(b.control->next_count, primaries);
                if (uint64_t{state.reserved} + primaries > b.next_capacity)
                    fetch_or(b.control->error, kNextLevelFull);
            }
        }
        finish_task(b, v);
        state.child_primaries = primaries;
    }
    g.sync();
}

struct ChildView
{
    uint64_t  bytes;
    uint32_t* global;
    Tx*       txs;
    Entry*    entries;
};

__device__ __forceinline__ ChildView child_view(const BatchView& b, const TaskState& state, uint32_t child_items)
{
    ChildView      c;
    const DbLayout layout = DbLayout::of(state.child_trans, state.child_entries, child_items);
    char*          at     = b.dst + state.child_db;
    c.bytes               = layout.bytes;
    c.global              = reinterpret_cast<uint32_t*>(at + layout.items);
    c.txs                 = reinterpret_cast<Tx*>(at + layout.tx);
    c.entries             = reinterpret_cast<Entry*>(at + layout.entries);
    return c;
}

// The child's item names: parent global ids of the kept items.
__device__ __forceinline__ void copy_child_items(const TaskView& v, const ChildView& c, uint32_t rank,
                                                 uint32_t stride)
{
    const auto*    globals = reinterpret_cast<const uint32_t*>(v.base + v.layout.items);
    const uint32_t n       = v.header.n_items;
    for (uint32_t i = v.task.item + 1 + rank; i < n; i += stride)
        if (v.p.rename[i] != kNone)
            c.global[v.p.rename[i]] = globals[i];
}

// Pass 3a: representatives write their child transaction and entries.
__device__ inline void write_representatives(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step,
                                             TaskState& state, const ChildView& c)
{
    for (uint32_t i0 = from; i0 < p.n_trans; i0 += step) {
        const uint32_t i     = i0 + lanes.rank;
        const bool     write = i < p.n_trans && p.rep[i] == i;
        const Tx       tx    = write ? p.txs[i] : Tx{0, 0, 0};
        const Entry*   e     = p.entries + tx.begin;
        const int      k     = write ? find_item(e, tx.length, p.item) : -1;
        const uint32_t rest  = k >= 0 ? tx.length - k - 1 : 0;
        const uint32_t span  = lanes.max(rest);
        uint32_t       kept  = 0;
        for (uint32_t d = 0; d < span; ++d)
            kept += d < rest && p.rename[e[k + 1 + d].item] != kNone;
        Entry* out = nullptr;
        if (write) {
            const uint32_t slot  = fetch_add(state.tx_cursor, 1u);
            const uint32_t begin = fetch_add(state.entry_cursor, kept);
            p.slot_of[i]         = slot;
            c.txs[slot]          = Tx{begin, kept, tx.utility + e[k].utility};
            out                  = c.entries + begin;
        }
        for (uint32_t d = 0; d < span; ++d) {
            if (d >= rest)
                continue;
            const uint32_t z = p.rename[e[k + 1 + d].item];
            if (z != kNone)
                *out++ = Entry{z, e[k + 1 + d].utility};
        }
    }
}

// Pass 3b: duplicates add their utilities into their representative's child transaction.
__device__ inline void merge_duplicates(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step,
                                        const ChildView& c)
{
    for (uint32_t i0 = from; i0 < p.n_trans; i0 += step) {
        const uint32_t i     = i0 + lanes.rank;
        const uint32_t r     = i < p.n_trans ? p.rep[i] : kNone;
        const bool     merge = r != kNone && r != i;
        const Tx       tx    = merge ? p.txs[i] : Tx{0, 0, 0};
        const Entry*   e     = p.entries + tx.begin;
        const int      k     = merge ? find_item(e, tx.length, p.item) : -1;
        const uint32_t rest  = k >= 0 ? tx.length - k - 1 : 0;
        const uint32_t span  = lanes.max(rest);
        Entry*         out   = nullptr;
        if (merge) {
            const uint32_t dst = p.slot_of[r];
            add(c.txs[dst].utility, tx.utility + e[k].utility);
            out = c.entries + c.txs[dst].begin;
        }
        for (uint32_t d = 0; d < span; ++d) {
            if (d >= rest)
                continue;
            if (p.rename[e[k + 1 + d].item] != kNone)
                add((out++)->utility, e[k + 1 + d].utility);
        }
    }
}

// Writes one next-level task per promising extension into the reserved slots (group scan).
template <typename Group>
__device__ void emit_children(const Group& g, const BatchView& b, const TaskView& v, uint32_t child_items,
                              const ChildView& c)
{
    const TaskState& state   = *v.state;
    uint32_t         emitted = 0;
    for (uint32_t z0 = 0; z0 < child_items; z0 += g.size()) {
        const uint32_t z    = z0 + g.rank();
        const bool     emit = z < child_items && v.p.subtree[z] >= b.minimum_utility;
        const uint32_t scan = g.scan(emit);
        const uint32_t slot = state.reserved + emitted + scan - 1;
        if (emit && slot < b.next_capacity)
            b.next[slot] = Task{static_cast<uint32_t>(state.child_db / kArenaUnit),
                                static_cast<uint32_t>(c.bytes / kArenaUnit), state.child_trans, child_items, z,
                                v.index};
        emitted += g.count(emit);
    }
}

// One group (lanes of a warp, a warp, or a block whose warps split the transactions) processes one
// whole task.
template <typename Group>
__device__ void process_task(const Group& g, const BatchView& b, uint32_t t, uint32_t warp, uint32_t warps)
{
    const TaskView v     = open_task(b, t);
    const uint32_t first = v.task.item + 1, n = v.header.n_items;
    const Lanes    lanes(g);
    const uint32_t from = warp * lanes.count, step = warps * lanes.count;

    clear_task(v, g.rank(), g.size());
    g.sync();
    scan_local(v.p, lanes, from, step, *v.state);
    g.sync();

    const uint32_t child_items = rename_items(g, v.p, first, n, b.minimum_utility);
    if (child_items == 0) {
        if (g.rank() == 0)
            finish_task(b, v);
        return;
    }
    g.sync();
    scan_subtree(v.p, lanes, from, step, *v.state);
    g.sync();

    allocate_child(g, b, v, child_items);
    if (v.state->child_primaries == 0)
        return;

    const ChildView c = child_view(b, *v.state, child_items);
    copy_child_items(v, c, g.rank(), g.size());
    write_representatives(v.p, lanes, from, step, *v.state, c);
    g.sync();
    merge_duplicates(v.p, lanes, from, step, c);
    g.sync();
    emit_children(g, b, v, child_items, c);
}

} // namespace bfs
