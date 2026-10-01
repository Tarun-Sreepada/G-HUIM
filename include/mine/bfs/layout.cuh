#pragma once
// Memory layout of the level-synchronous (BFS) optimized miner.
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

#include <cstdint>
#include <cuda/std/bit>

namespace bfs {

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

    __host__ __device__ uint64_t db_offset() const { return uint64_t{db} * kArenaUnit; }
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

// Task placement. Bins 0..kOptBins-1 are size classes: class 0 packs one task per warp into
// blocks, class s > 0 gives each task a block of (32 << s) threads. kOptHugeBin splits a task over
// several blocks. Tasks with at most a warp's worth of transactions run on a group of
// bit_ceil(n_trans) lanes: bins kOptSubBin + log2(group size).
constexpr uint32_t kOptBlockSizes   = 6;
constexpr uint32_t kOptBins         = kOptBlockSizes;
constexpr uint32_t kOptHugeBin      = kOptBins;
constexpr uint32_t kOptSubBin       = kOptBins + 1;
constexpr uint32_t kOptSubBins      = 6;
constexpr uint32_t kBinCount        = kOptSubBin + kOptSubBins;
constexpr uint32_t kWarpSize        = 32;
constexpr uint32_t kMinTransPerWarp = 8; // below this, extra warps only add overhead
constexpr uint32_t kHugeKernels     = 8;

static_assert(kBinCount <= kMaxBins);

// Device profile measured at startup from kernel occupancy.
struct AutoTune
{
    uint32_t resident_warps;      // warps the whole GPU keeps resident
    uint32_t max_block_warps;     // largest block (in warps) that still reaches full occupancy
    uint32_t packed_block_warps;  // block size (in warps) for the one-task-per-warp kernel
    uint32_t subwarp_block_warps; // block size (in warps) for the lane-group-per-task kernel
    uint32_t max_slices;          // blocks one huge task may use
};

struct Placement
{
    uint32_t bin;
    uint32_t slices; // blocks sharing the task (huge bin only)
};

__host__ __device__ inline uint32_t ceil_div(uint64_t a, uint64_t b)
{
    return static_cast<uint32_t>((a + b - 1) / b);
}

__host__ __device__ inline uint32_t size_class(uint32_t warps)
{
    return cuda::std::countr_zero(cuda::std::bit_ceil(warps > 0 ? warps : 1u));
}

// Where a task runs: the chunk's total work is spread over the resident warps, and each task gets
// enough warps for its share, rounded up to a power of two.
__host__ __device__ inline Placement place_task(const Task& task, const AutoTune& tune, uint64_t batch_work)
{
    if (task.n_trans <= kWarpSize)
        return {kOptSubBin + size_class(task.n_trans), 1};
    const uint64_t share    = batch_work / tune.resident_warps;
    const uint64_t per_warp = share > kMinTransPerWarp ? share : kMinTransPerWarp;
    const uint32_t warps    = ceil_div(task.n_trans, per_warp);
    if (warps > tune.max_block_warps) {
        const uint32_t slices = ceil_div(warps, tune.max_block_warps);
        return {kOptHugeBin, slices < tune.max_slices ? slices : tune.max_slices};
    }
    return {size_class(warps), 1};
}

} // namespace bfs
