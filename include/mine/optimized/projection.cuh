#pragma once
// One task's view of its database and its global-memory scratch.

#include "mine/device/common.cuh"

namespace ghuim::opt {

__device__ __forceinline__ uint32_t mix(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// The arrays the transaction passes work on.
struct Projection
{
    const Tx*           txs;
    const Entry*        entries;
    uint32_t            n_trans;
    uint32_t            item;       // extension item (local name in the parent)
    unsigned long long* local;      // by parent item name
    unsigned long long* subtree;    // by child item name
    uint32_t*           rename;     // parent item -> child item, or kNone
    uint32_t*           table;      // merge hash table over kept-item sequences
    uint32_t            table_size;
    uint32_t*           rep;        // representative of each projected transaction
    uint32_t*           slot_of;    // child transaction written by a representative
};

struct TaskView
{
    Task        task;
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
    v.task   = b.tasks[t];
    v.base   = b.arena + v.task.db_offset();
    v.header = *reinterpret_cast<const DbHeader*>(v.base);
    v.layout = DbLayout::of(v.header.n_trans, v.header.n_entries, v.header.n_items);
    char* scratch = b.scratch + b.scratch_offset[t];
    v.state       = reinterpret_cast<TaskState*>(scratch);

    const uint32_t n = v.header.n_items, trans = v.header.n_trans;
    Projection&    p = v.p;
    p.txs        = reinterpret_cast<const Tx*>(v.base + v.layout.tx);
    p.entries    = reinterpret_cast<const Entry*>(v.base + v.layout.entries);
    p.n_trans    = trans;
    p.item       = v.task.item;
    p.local      = reinterpret_cast<unsigned long long*>(scratch + sizeof(TaskState));
    p.subtree    = p.local + n;
    p.rename     = reinterpret_cast<uint32_t*>(p.subtree + n);
    p.table_size = 2 * trans;
    p.table      = p.rename + n;
    p.rep        = p.table + p.table_size;
    p.slot_of    = p.rep + trans;
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

} // namespace ghuim::opt
