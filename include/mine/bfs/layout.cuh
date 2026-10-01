#pragma once
// Memory layout of the level-synchronous (BFS) miner.
//
// A projected database lives in one contiguous arena block and refers to its own parts by
// offsets only:
//
//   DbHeader | items[n_items] | Tx[n_trans] | Entry[n_entries]
//
// Items inside a database use local names 0..n_items-1; items[] maps them to global ids
// (ascending, so local order equals global order). Databases start on 16-byte boundaries and
// tasks address them in 16-byte units.
//
// Level L reads the databases written while mining level L-1 and writes its children into the
// other of two arena regions, so a region is reused as soon as the level reading it finishes.

#include <cuda/std/bit>

#include <cstdint>

namespace ghuim::bfs {

constexpr uint64_t kArenaUnit = 16;
constexpr uint32_t kNone      = 0xffffffffu;

struct DbHeader
{
    uint32_t n_trans;
    uint32_t n_entries;
    uint32_t n_items;
    uint32_t reserved;
    uint64_t bytes;
};

struct Tx
{
    uint32_t begin;   // first entry
    uint32_t length;  // entry count
    uint32_t utility; // utility of the database's prefix itemset in this transaction
};

struct Entry
{
    uint32_t item; // local name
    uint32_t utility;
};

__host__ __device__ inline uint64_t align_up(uint64_t value, uint64_t alignment = kArenaUnit)
{
    return (value + alignment - 1) / alignment * alignment;
}

struct DbLayout
{
    uint64_t items, tx, entries, bytes;

    __host__ __device__ static DbLayout of(uint64_t n_trans, uint64_t n_entries, uint64_t n_items)
    {
        DbLayout l;
        l.items   = sizeof(DbHeader);
        l.tx      = align_up(l.items + n_items * sizeof(uint32_t), 8);
        l.entries = align_up(l.tx + n_trans * sizeof(Tx), 8);
        l.bytes   = align_up(l.entries + n_entries * sizeof(Entry));
        return l;
    }
};

// One unit of work: extend the prefix of database `db` with its local item `item`.
// The parent's sizes are copied in so scheduling never has to read the database.
struct Task
{
    uint32_t db;       // offset of the parent database in the source region, in kArenaUnit units
    uint32_t db_units; // size of that database
    uint32_t n_trans;  // parent transaction count
    uint32_t n_items;  // parent item count
    uint32_t item;     // local name in the parent database
    uint32_t parent;   // index of the task (in the previous level) that built the parent database

    __host__ __device__ uint64_t db_offset() const
    {
        return uint64_t{db} * kArenaUnit;
    }
};

// One mined itemset: its last item and a link to its prefix in the previous level.
struct Node
{
    uint32_t parent; // index into the previous level's nodes, kNone at level 1
    uint32_t item;   // global item id
    uint64_t utility;
};

// Per-task working state in the scratch buffer. With n parent items and t parent transactions
// it is followed by local[n], subtree[n] (u64), rename[n], table[2t], rep[t], slot[t] (u32).
struct TaskState
{
    uint64_t utility;
    uint64_t child_db;
    uint32_t child_trans;
    uint32_t child_entries;
    uint32_t child_items;
    uint32_t child_primaries;
    uint32_t tx_cursor;
    uint32_t entry_cursor;
    uint32_t child_max_len;
    uint32_t reserved; // first slot of this task's children in the next level
};

__host__ __device__ inline uint64_t scratch_bytes(uint32_t n_items, uint32_t n_trans)
{
    return align_up(sizeof(TaskState) + uint64_t{n_items} * (2 * sizeof(uint64_t) + sizeof(uint32_t)) +
                    uint64_t{n_trans} * 4 * sizeof(uint32_t));
}

constexpr uint32_t kMaxBins = 16;

enum ControlError : uint32_t
{
    kNextLevelFull = 1u << 0,
    kArenaFull     = 1u << 1,
};

// Scheduler state shared by every kernel; lives in device memory and is read back once per chunk.
struct Control
{
    unsigned long long arena_top;  // bump allocator for child databases in the destination region
    unsigned long long patterns;   // high-utility itemsets found so far
    unsigned long long batch_work; // sum of transactions over the chunk's tasks
    uint32_t           next_count; // tasks written to the next level
    uint32_t           error;      // ControlError bits
    uint32_t           bin_count[kMaxBins];
    uint32_t           bin_cursor[kMaxBins];
    uint32_t           bin_max_slices[kMaxBins]; // most blocks any task of the bin is split into
};

} // namespace ghuim::bfs
