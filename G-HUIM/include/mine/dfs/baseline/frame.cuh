#pragma once
// One task's database and scratch pointers (static-parallelism kernels).

#include "mine/dfs/device/common.cuh"

namespace ghuim::dfs::baseline {

// Scratch layout: TaskState | local[n] | subtree[n] | rename[n] | table[2t] | rep[t] | slot[t].
// Identical projected transactions are merged; tables are indexed by parent transaction.
struct Frame
{
    Task                task;
    const char*         base;
    DbHeader            header;
    DbLayout            layout;
    const Tx*           txs;
    const Entry*        entries;
    uint32_t            n;       // parent items
    uint32_t            first;   // only later items can extend the itemset
    uint64_t            minutil;
    TaskState*          state;
    unsigned long long* local;   // by parent name
    unsigned long long* subtree; // by child name
    uint32_t*           rename;  // parent name -> child name
    uint32_t            table_size;
    uint32_t*           table;   // open addressing over kept-item sequences
    uint32_t*           rep;     // representative of each projected transaction
    uint32_t*           slot_of; // child transaction written by a representative
};

__device__ inline Frame open_frame(const BatchView& b, uint32_t t)
{
    Frame f;
    f.task    = b.tasks[t];
    f.base    = b.arena + f.task.db_offset();
    f.header  = *reinterpret_cast<const DbHeader*>(f.base);
    f.layout  = DbLayout::of(f.header.n_trans, f.header.n_entries, f.header.n_items);
    f.txs     = reinterpret_cast<const Tx*>(f.base + f.layout.tx);
    f.entries = reinterpret_cast<const Entry*>(f.base + f.layout.entries);
    f.n       = f.header.n_items;
    f.first   = f.task.item + 1;
    f.minutil = b.minimum_utility;

    char* scratch = b.scratch + b.scratch_offset[t];
    f.state       = reinterpret_cast<TaskState*>(scratch);
    f.local       = reinterpret_cast<unsigned long long*>(scratch + sizeof(TaskState));
    f.subtree     = f.local + f.n;
    f.rename      = reinterpret_cast<uint32_t*>(f.subtree + f.n);
    f.table_size  = 2 * f.header.n_trans;
    f.table       = f.rename + f.n;
    f.rep         = f.table + f.table_size;
    f.slot_of     = f.rep + f.header.n_trans;
    return f;
}

} // namespace ghuim::dfs::baseline
