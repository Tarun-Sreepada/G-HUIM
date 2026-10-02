#pragma once
// The per-chunk view of the level and the per-task view of its database and scratch state.

#include "cuda/atomics.cuh"
#include "mine/bfs/layout.cuh"

namespace ghuim::bfs {

// ---------------------------------------------------------------------------------------------
// Per-chunk view and per-task view
// ---------------------------------------------------------------------------------------------

// Device pointers shared by every kernel processing one chunk of a level.
struct BatchView
{
    const char*     src; // region holding the parent databases
    char*           dst; // region receiving child databases
    uint64_t        dst_capacity;
    Control*        control;
    const Task*     tasks;          // the whole level; kernels index it with order[]
    const uint64_t* scratch_prefix; // exclusive prefix of scratch_bytes over the level
    uint64_t        scratch_base;   // scratch_prefix of the chunk's first task
    char*           scratch;
    Task*           next; // the next level
    uint32_t        next_capacity;
    Node*           nodes; // one per task of the level
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
__device__ inline int find_item(const uint32_t* keys, uint32_t length, uint32_t item)
{
    int low = 0, high = static_cast<int>(length) - 1;
    while (low <= high) {
        const int      mid = (low + high) >> 1;
        const uint32_t key = keys[mid];
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
__device__ inline bool same_kept_items(Tx a, const uint32_t* keys, Tx b, uint32_t item, const uint32_t* rename)
{
    const uint32_t* ea = keys + a.begin;
    const uint32_t* eb = keys + b.begin;
    uint32_t     i  = find_item(ea, a.length, item) + 1;
    uint32_t     j  = find_item(eb, b.length, item) + 1;
    for (;;) {
        while (i < a.length && rename[ea[i]] == kNone)
            ++i;
        while (j < b.length && rename[eb[j]] == kNone)
            ++j;
        if (i == a.length || j == b.length)
            return i == a.length && j == b.length;
        if (ea[i] != eb[j])
            return false;
        ++i;
        ++j;
    }
}

// The arrays the transaction passes work on.
struct Projection
{
    const Tx*           txs;
    const uint32_t*     eitem;   // entry keys (local item names)
    const uint32_t*     eutil;   // entry utilities
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
    p.eitem          = reinterpret_cast<const uint32_t*>(v.base + v.layout.eitem);
    p.eutil          = reinterpret_cast<const uint32_t*>(v.base + v.layout.eutil);
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

} // namespace ghuim::bfs
