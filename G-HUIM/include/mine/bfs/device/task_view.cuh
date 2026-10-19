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
    uint32_t        io_warps;   // staged method: IO warps per block (0 = unstaged)
    uint32_t        ring_bytes; // staged method: shared memory for the tile ring
    uint32_t        tiled;        // tile method: databases use the tile layout
    uint32_t        index_min_trans;
    uint32_t        index_sparsity;
    uint32_t        local_tail_items;
    uint32_t        local_tail_trans;
    int32_t         refine_rounds; // bound refinement rounds per task (-1 = until none drops, 0 = off)
    uint32_t        leaf_shortcut; // count-only: count the last child item's itemset instead of emitting its task
    uint32_t*       vert_scratch;  // vertical subtrees: per-warp columns and utility stack
    uint32_t        vert_slots;    // vertical subtrees: slots per column (most transactions, rounded up to 32)
    uint64_t        vert_warp_words; // vertical subtrees: scratch words per warp
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
template <typename K> __device__ inline int find_item(const K* keys, uint32_t length, uint32_t item)
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

// Whether two parent transactions (their key arrays and lengths) project to the same kept items after `item`.
// `i`, `j` are the positions after the extension item (the passes' `pos`).
template <typename K>
__device__ inline bool same_kept_items(const Key* ea, uint32_t a_length, uint32_t i, const K* eb,
                                       uint32_t b_length, uint32_t j, const uint32_t* rename)
{
    for (;;) {
        while (i < a_length && rename[ea[i]] == kNone)
            ++i;
        while (j < b_length && rename[eb[j]] == kNone)
            ++j;
        if (i == a_length || j == b_length)
            return i == a_length && j == b_length;
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
    const Occurrence*   occ; // this extension's occurrences, or null for a full database scan
    const Key*          eitem;   // entry keys (local item names)
    const Util*         eutil;   // entry utilities
    uint32_t            n_trans;
    uint32_t            max_len; // longest transaction
    uint32_t            item;    // extension item (local name in the parent)
    unsigned long long* local;   // by parent item name
    unsigned long long* subtree; // by child item name
    uint32_t*           rename;  // parent item -> child item, or kNone
    uint32_t*           table;   // merge hash table over kept-item sequences
    uint32_t            table_size;
    uint32_t*           rep;     // representative of each projected transaction
    uint32_t*           slot_of; // child transaction written by a representative
    uint32_t*           pos;     // pass 1: position after the extension item in each transaction (k + 1), 0 = absent
    uint32_t*           kept;    // pass 2: kept entries of each transaction behind the extension item
    // Tile method only: tile widths of this database and of the child (set once pass 2 has sized it).
    uint32_t            stride;
    uint32_t            child_stride;
    // Tile method only (null otherwise): the child's counting sort by length.
    uint32_t*           hist;     // pass 2: representatives per length class
    uint32_t*           base;     // first child slot of each class
    uint32_t*           cursor;   // next free slot of each class, relative to base
    uint32_t*           tile_len; // longest transaction of each child tile
    uint32_t*           tile_off; // first entry of each child tile
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

// Scratch layout: TaskState | local[n] | subtree[n] | rename[n] | table[2t] | rep[t] | slot[t] | pos[t] | kept[t].
__device__ __forceinline__ TaskView open_task(const BatchView& b, uint32_t t)
{
    TaskView v;
    v.task        = b.tasks[t];
    v.index       = t;
    v.base        = b.src + v.task.db_offset();
    v.header      = *reinterpret_cast<const DbHeader*>(v.base);
    v.layout      = DbLayout::of(v.header.n_trans, v.header.n_entries, v.header.n_items, b.tiled != 0, v.header.indexed != 0);
    char* scratch = b.scratch + (b.scratch_prefix[t] - b.scratch_base);
    v.state       = reinterpret_cast<TaskState*>(scratch);

    const uint32_t n = v.header.n_items, trans = v.task.n_trans;
    Projection&    p = v.p;
    p.txs            = reinterpret_cast<const Tx*>(v.base + v.layout.tx);
    p.occ            = v.header.indexed ? reinterpret_cast<const Occurrence*>(v.base + v.layout.occ) +
                            reinterpret_cast<const uint32_t*>(v.base + v.layout.occ_begin)[v.task.item] : nullptr;
    p.eitem          = reinterpret_cast<const Key*>(v.base + v.layout.eitem);
    p.eutil          = reinterpret_cast<const Util*>(v.base + v.layout.eutil);
    p.n_trans        = trans;
    p.max_len        = v.header.max_len;
    p.item           = v.task.item;
    p.local          = reinterpret_cast<unsigned long long*>(scratch + sizeof(TaskState));
    p.subtree        = p.local + n;
    p.rename         = reinterpret_cast<uint32_t*>(p.subtree + n);
    p.table_size     = 2 * trans;
    p.table          = p.rename + n;
    p.rep            = p.table + p.table_size;
    p.slot_of        = p.rep + trans;
    p.pos            = p.slot_of + trans;
    p.kept           = p.pos + trans;
    p.hist = p.base = p.cursor = p.tile_len = p.tile_off = nullptr;
    p.stride = p.child_stride = 1;
    if (b.tiled) {
        p.stride   = tile_stride(trans);
        p.hist     = p.kept + trans;
        p.base     = p.hist + kTileBuckets;
        p.cursor   = p.base + kTileBuckets;
        p.tile_len = p.cursor + kTileBuckets;
        p.tile_off = p.tile_len + trans / kTile + 1;
    }
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
    if (v.p.hist != nullptr) {
        for (uint32_t i = rank; i < kTileBuckets; i += stride)
            v.p.hist[i] = v.p.cursor[i] = 0;
        for (uint32_t i = rank; i < v.p.n_trans / kTile + 1; i += stride)
            v.p.tile_len[i] = 0;
    }
}

// Records the task's itemset; one thread per task calls it once the utility is complete.
__device__ inline void finish_task(const BatchView& b, const TaskView& v)
{
    const uint64_t  utility = v.state->utility;
    const uint32_t* globals = reinterpret_cast<const uint32_t*>(v.base + v.layout.items);
    b.nodes[v.index]        = Node{v.task.parent, globals[v.task.item], utility};
    if (utility >= b.minimum_utility)
        add(b.control->patterns, 1ull);
#ifdef GHUIM_WORK_STATS
    add(b.control->work_total, static_cast<unsigned long long>(v.header.n_trans));
    add(b.control->work_present, static_cast<unsigned long long>(v.state->present));
#endif
}

} // namespace ghuim::bfs
