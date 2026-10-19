#pragma once
// Memory layout shared by the host scheduler and the mining kernels.
//
// A projected database lives in one contiguous arena block and refers to its own parts by
// offsets only, so it can be moved during compaction:
//
//   DbHeader | items[n_items] | Tx[n_trans] | Entry[n_entries]
//
// Items inside a database use local names 0..n_items-1; items[] maps them to global names
// (ascending, so local order equals global order). Databases start on 16-byte boundaries and
// tasks address them in 16-byte units.

#include <cstdint>
#include <cuda/std/bit>

#ifdef __CUDACC__
#define GHUIM_HD __host__ __device__
#else
#define GHUIM_HD
#endif

namespace ghuim::dfs {

constexpr uint64_t kArenaUnit = 16;

struct DbHeader
{
    uint32_t n_trans;
    uint32_t n_entries;
    uint32_t n_items;
    uint32_t max_tx_len; // longest transaction
    uint64_t bytes;
    uint32_t length_bin; // length bin holding the median entry (see median_length_bin)
    uint32_t reserved;
};

// Transaction lengths are grouped in power-of-two bins: 1, 2, 3-4, 5-8, 9-16, 17-32, longer.
constexpr uint32_t kLengthBins = 7;

GHUIM_HD inline uint32_t length_bin(uint32_t length)
{
    const uint32_t bin = cuda::std::countr_zero(cuda::std::bit_ceil(length > 0 ? length : 1u));
    return bin < kLengthBins - 1 ? bin : kLengthBins - 1;
}

// The length bin that holds the median entry: weighting by entries follows where the work is,
// not the longest or the average transaction.
GHUIM_HD inline uint32_t median_length_bin(const uint32_t* entries_by_length)
{
    uint64_t total = 0;
    for (uint32_t b = 0; b < kLengthBins; ++b)
        total += entries_by_length[b];
    uint64_t seen = 0;
    for (uint32_t b = 0; b < kLengthBins; ++b) {
        seen += entries_by_length[b];
        if (2 * seen >= total)
            return b;
    }
    return kLengthBins - 1;
}

struct Tx
{
    uint32_t begin;   // first entry
    uint32_t length;  // entry count
    uint64_t utility; // utility of the database's prefix itemset in this transaction
};

struct Entry
{
    uint32_t item; // local name
    uint64_t utility;
};

GHUIM_HD inline uint64_t align_up(uint64_t value, uint64_t alignment = kArenaUnit)
{
    return (value + alignment - 1) / alignment * alignment;
}

struct DbLayout
{
    uint64_t items, tx, entries, bytes;

    GHUIM_HD static DbLayout of(uint64_t n_trans, uint64_t n_entries, uint64_t n_items)
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
    uint32_t db;       // arena offset of the parent database, in kArenaUnit units
    uint32_t db_units; // size of that database; bounds the size of the child database
    uint32_t n_trans;  // parent transaction count
    uint32_t n_items;  // parent item count; bounds scratch and the number of children
    uint32_t item;     // local name in the parent database
    uint32_t depth;    // length of the itemset this task evaluates (prefix + item)

    GHUIM_HD uint64_t db_offset() const { return uint64_t{db} * kArenaUnit; }
    GHUIM_HD uint64_t db_bytes() const { return uint64_t{db_units} * kArenaUnit; }
    // Arena bytes the task may allocate: a child is never larger than its parent, and a
    // task extending with the parent's last item has no children.
    GHUIM_HD uint64_t arena_bound() const { return item + 1 < n_items ? db_bytes() : 0; }
    GHUIM_HD uint32_t child_bound() const { return n_items - item - 1; }
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
    uint32_t reserved;
    uint32_t refined; // huge tasks: refinement has converged (or is off)
    uint32_t entries_by_length[kLengthBins + 1]; // child entries per transaction-length bin (+ padding)
};

GHUIM_HD inline uint64_t scratch_bytes(uint32_t n_items, uint32_t n_trans)
{
    return align_up(sizeof(TaskState) + uint64_t{n_items} * (2 * sizeof(uint64_t) + sizeof(uint32_t)) +
                    uint64_t{n_trans} * 4 * sizeof(uint32_t));
}

constexpr uint32_t kMaxBins  = 32;
constexpr uint32_t kMaxDepth = 64; // deeper itemsets are counted in the last slot

enum ControlError : uint32_t
{
    kStackExhausted = 1u << 0,
};

// Scheduler state shared by every kernel; lives in device memory and is read back once per batch.
struct Control
{
    unsigned long long arena_top; // bump allocator for child databases (bytes)
    unsigned long long patterns;  // high-utility itemsets found so far
    uint32_t           stack_top; // tasks waiting on the device stack
    uint32_t           batch_count;
    uint32_t           deferred;  // tasks of this batch pushed back because their child did not fit
    uint32_t           error;     // ControlError bits
    uint32_t           forced;    // the batch exceeded the admission bounds to make progress
    uint32_t           reserved;
    uint32_t           bin_count[kMaxBins];
    uint32_t           bin_cursor[kMaxBins];
    uint32_t           bin_max_slices[kMaxBins]; // most blocks any task of the bin is split into
    unsigned long long batch_work;               // sum of transactions over the batch's tasks
    unsigned long long depth_tasks[kMaxDepth];    // completed tasks by itemset length
    unsigned long long depth_patterns[kMaxDepth]; // high-utility itemsets by length
};

} // namespace ghuim::dfs
