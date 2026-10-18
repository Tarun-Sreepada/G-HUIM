#pragma once
// Memory layout of the level-synchronous (BFS) miner.
//
// A projected database lives in one contiguous arena block and refers to its own parts by
// offsets only:
//
//   DbHeader | items[n_items] | Tx[n_trans] | entry_item[n_entries] | entry_utility[n_entries]
//
// Keys and values of the entries are separate arrays: passes that only test items read 4 bytes per
// entry, and utilities are read only for the entries that need them. Transaction t owns entries
// [begin, begin + length) of both arrays.
//
// Items inside a database use local names 0..n_items-1; items[] maps them to global ids
// (ascending, so local order equals global order). Databases start on 16-byte boundaries and
// tasks address them in 16-byte units.
//
// Tile method (--method tile): the entry arrays are stored in tiles of kTile transactions, column-major,
// so entry k of the kTile transactions of a tile is one contiguous row. Transaction slot s sits in
// tile s / kTile at lane s % kTile; Tx::begin is then tile_base + lane, and entry k of the transaction is
// at begin + k * kTile. Transactions are sorted by length into tiles (counting sort by kept length), and
// a tile is as long as its longest transaction. The Tx array is padded to whole tiles and n_entries
// counts the padded entry slots (rows beyond a transaction's length are never read).
//
// Level L reads the databases written while mining level L-1 and writes its children into the
// other of two arena regions, so a region is reused as soon as the level reading it finishes.

#include <cuda/std/bit>

#include <cstdint>

namespace ghuim::bfs {

constexpr uint64_t kArenaUnit = 16;
constexpr uint32_t kTile        = 32; // transactions per tile (tile method)
constexpr uint32_t kTileBuckets = 32; // length classes of the tile counting sort: 0..30 exact, 31 = 31 or longer
constexpr uint32_t kNone      = 0xffffffffu;

// Transactions per tile of a database with n_trans transactions: a small database is one narrow tile, so it
// is not padded to kTile transactions. Readers derive it from the header's n_trans.
__host__ __device__ inline uint32_t tile_stride(uint64_t n_trans)
{
    return n_trans >= kTile ? kTile : n_trans <= 1 ? 1u : cuda::std::bit_ceil(static_cast<uint32_t>(n_trans));
}

struct DbHeader
{
    uint32_t n_trans;
    uint32_t n_entries;
    uint32_t n_items;
    uint32_t max_len; // longest transaction
    uint64_t bytes;
    uint32_t indexed = 0; // occurrence offsets and (transaction, position) pairs follow the entry arrays
};

struct Tx
{
    uint32_t begin;   // first entry
    uint32_t length;  // entry count
    uint32_t utility; // utility of the database's prefix itemset in this transaction
};

struct Occurrence
{
    uint32_t tid;
    uint32_t pos;
};

// Whether a database gets an occurrence index: at least `min_trans` transactions (0 = never) and, with
// `sparsity` s != 0, at most 1/s of its transaction-item pairs present. Then an extension visits few of the
// transactions; in a dense database the index only duplicates the entries.
__host__ __device__ inline bool wants_index(uint64_t n_trans, uint64_t n_entries, uint64_t n_items, uint32_t min_trans,
                                            uint32_t sparsity)
{
    return min_trans != 0 && n_trans >= min_trans && (sparsity == 0 || n_entries * sparsity <= n_trans * n_items);
}

__host__ __device__ inline uint64_t align_up(uint64_t value, uint64_t alignment = kArenaUnit)
{
    return (value + alignment - 1) / alignment * alignment;
}

struct DbLayout
{
    uint64_t items, tx, eitem, eutil, bytes, occ_begin = 0, occ = 0;

    __host__ __device__ static DbLayout of(uint64_t n_trans, uint64_t n_entries, uint64_t n_items, bool tiled = false,
                                          bool indexed = false)
    {
        DbLayout l;
        l.items   = sizeof(DbHeader);
        l.tx      = align_up(l.items + n_items * sizeof(uint32_t), 8);
        if (tiled)
            n_trans = align_up(n_trans, tile_stride(n_trans));
        l.eitem   = align_up(l.tx + n_trans * sizeof(Tx), 16);
        l.eutil   = align_up(l.eitem + n_entries * sizeof(uint32_t), 16);
        l.bytes   = align_up(l.eutil + n_entries * sizeof(uint32_t));
        if (indexed) {
            l.occ_begin = l.bytes;
            l.occ = align_up(l.occ_begin + (n_items + 1) * sizeof(uint32_t));
            l.bytes = align_up(l.occ + n_entries * sizeof(Occurrence));
        }
        return l;
    }
};

// One unit of work: extend the prefix of database `db` with its local item `item`.
// The parent's sizes are copied in so scheduling never has to read the database.
struct Task
{
    uint32_t db;       // offset of the parent database in the source region, in kArenaUnit units
    uint32_t db_units; // size of that database
    uint32_t n_trans;  // transactions visited: occurrence count when the parent has an index
    uint32_t n_items;  // parent item count
    uint32_t item;     // local name in the parent database
    uint32_t parent;   // index of the task (in the previous level) that built the parent database
    uint32_t max_len;  // parent's longest transaction (shape of the task)
    uint32_t n_entries; // parent's entry count

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
    uint64_t present; // GHUIM_WORK_STATS: transactions of the parent that contain the extension item
    uint64_t child_db;
    uint32_t child_trans;
    uint32_t child_entries;
    uint32_t child_items;
    uint32_t child_primaries;
    uint32_t tx_cursor;
    uint32_t entry_cursor;
    uint32_t child_max_len;
    uint32_t reserved; // first slot of this task's children in the next level
    uint32_t refined;  // huge tasks: refinement has converged (or is off)
};

// Tile method: hist[kTileBuckets], base[kTileBuckets], cursor[kTileBuckets], then tile_len[t / kTile + 1]
// and tile_off[t / kTile + 1] (u32), after the arrays above (a child has at most that many tiles).
__host__ __device__ inline uint64_t tile_scratch_words(uint32_t n_trans)
{
    return 3 * uint64_t{kTileBuckets} + 2 * (uint64_t{n_trans} / kTile + 1);
}

__host__ __device__ inline uint64_t scratch_bytes(uint32_t n_items, uint32_t n_trans, bool tiled = false)
{
    return align_up(sizeof(TaskState) + uint64_t{n_items} * (2 * sizeof(uint64_t) + sizeof(uint32_t)) +
                    uint64_t{n_trans} * 6 * sizeof(uint32_t) + (tiled ? tile_scratch_words(n_trans) * sizeof(uint32_t) : 0));
}

constexpr uint32_t kMaxCells = 64; // scheduling cells (placement.cuh)

// Staged method: ring slots and the shared memory ahead of the tiles (two counters per slot).
constexpr uint32_t kRingMaxSlots      = 32;
constexpr uint32_t kStageBarrierBytes = 2 * kRingMaxSlots * sizeof(uint32_t);

enum ControlError : uint32_t
{
    kNextLevelFull = 1u << 0,
    kArenaFull     = 1u << 1,
    kStageTimeout  = 1u << 2, // a staging wait exceeded its bound
    kStageOverflow = 1u << 3, // a transaction was longer than the database's recorded longest
};

// Scheduler state shared by every kernel; lives in device memory and is read back once per chunk.
struct Control
{
    unsigned long long arena_top;  // bump allocator for child databases in the destination region
    unsigned long long patterns;   // high-utility itemsets found so far
    unsigned long long batch_work; // sum of transactions over the chunk's tasks
    unsigned long long work_total;   // GHUIM_WORK_STATS: transactions scanned by pass 1, over all tasks
    unsigned long long work_present; // GHUIM_WORK_STATS: of those, the ones containing the extension item
    uint32_t           next_count; // tasks written to the next level
    uint32_t           error;      // ControlError bits
    uint32_t           cell_count[kMaxCells];
    uint32_t           cell_start[kMaxCells];
    uint32_t           cell_cursor[kMaxCells];
    uint32_t           cell_max_slices[kMaxCells]; // most blocks any task of the cell is split into
    // Staging statistics (GHUIM_STAGE_STATS): warp-cycles spent waiting.
    unsigned long long stage_full_wait;       // compute warps waiting for an IO warp
    unsigned long long stage_empty_wait;      // IO warps waiting for a compute warp to free a slot
    unsigned long long stage_consumer_cycles; // compute warps, whole staged passes
};

} // namespace ghuim::bfs
