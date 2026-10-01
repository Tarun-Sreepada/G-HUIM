#pragma once
// One task processed by one thread block (static method, and the adaptive method's larger tasks):
// project the database on the task's primary item, filter and merge the projected transactions into
// the child database, and push one task per promising extension.
//
// Every phase ends with a barrier. After a phase that allocates, a null pointer means the arena is
// full: the error flag is set and all threads of the block leave the kernel together.

#include "mine/dfs/hash.cuh"
#include "mine/primitives.cuh"

namespace ghuim::dfs {

// Most items a projected transaction may keep (per-thread buffer of the merge phase).
constexpr uint32_t kItemBuffer = 256;
// Adaptive: tasks with at most this many transactions run on a single thread.
constexpr uint32_t kSingleThreadTransactions = 16;

struct BlockState
{
    Task         current, child;
    Utilities*   table;           // per-item utilities of the projected database
    uint32_t     slots;           // table size
    Transaction* projected;       // transactions that continue after the primary item
    uint32_t     projected_count; // their number
    uint32_t     projected_items; // their total length
    uint32_t     cursor;          // next free `projected` slot
    int32_t*     hashes;          // merge table over projected transactions
    uint32_t     hash_slots;
    uint32_t     new_unique; // items of the child database that meet the threshold
};

__device__ inline bool failed_open(const BlockState& s)
{
    return s.table == nullptr || s.child.pattern == nullptr;
}

// Loads the task for this block and allocates its utility table and the child's pattern.
__device__ inline void open_task(Control& control, const Stack<Task>& stack, BlockState& s)
{
    if (threadIdx.x == 0) {
        stack.get(s.current, blockIdx.x);
        s.slots           = table_slots(s.current.unique_items);
        s.table           = static_cast<Utilities*>(control.allocate(sizeof(Utilities) * s.slots));
        s.child           = Task{};
        s.child.pattern   = static_cast<uint32_t*>(control.allocate(sizeof(uint32_t) * (s.current.pattern_length + 1)));
        s.projected       = nullptr;
        s.projected_count = s.projected_items = s.cursor = s.new_unique = 0;
        if (s.child.pattern != nullptr) {
            s.child.pattern_length                    = s.current.pattern_length + 1;
            s.child.pattern[s.current.pattern_length] = s.current.primary;
        }
    }
    __syncthreads();
    if (s.table == nullptr || s.child.pattern == nullptr)
        return;
    for (uint32_t i = threadIdx.x; i < s.current.pattern_length; i += blockDim.x)
        s.child.pattern[i] = s.current.pattern[i];
    for (uint32_t i = threadIdx.x; i < s.slots; i += blockDim.x)
        s.table[i].clear();
    __syncthreads();
}

// Phase 1: collect the transactions containing the primary item, the child's utility and the local
// utility of every item that follows it.
__device__ inline void project(Control& control, BlockState& s)
{
    const Database& db = *s.current.db;
    for (uint32_t i = threadIdx.x; i < db.num_transactions; i += blockDim.x) {
        const Transaction& t   = db.transactions[i];
        const int          idx = t.binary_search(s.current.primary);
        if (idx >= 0 && t.length - idx - 1 > 0)
            add(s.projected_count, 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0)
        s.projected = static_cast<Transaction*>(control.allocate(sizeof(Transaction) * s.projected_count));
    __syncthreads();
    if (s.projected == nullptr)
        return;

    for (uint32_t i = threadIdx.x; i < db.num_transactions; i += blockDim.x) {
        const Transaction& t   = db.transactions[i];
        const int          idx = t.binary_search(s.current.primary);
        if (idx < 0)
            continue;
        const uint32_t prefix = t.utility + t[idx].util;
        add(s.child.utility, prefix);
        const uint32_t remain = t.length - idx - 1;
        if (remain == 0)
            continue;
        const uint32_t slot = fetch_add(s.cursor, 1u);
        if (slot < s.projected_count) {
            add(s.projected_items, remain);
            s.projected[slot] = Transaction{&t.data[idx + 1], remain, prefix};
        }
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < s.projected_count; i += blockDim.x) {
        const Transaction& t     = s.projected[i];
        const uint32_t     total = t.utility + t.util_sum();
        for (uint32_t j = 0; j < t.length; ++j)
            accumulate<Utility::Local>(control, s.table, s.slots, t[j].key, total);
    }
    __syncthreads();
}

// Allocates the merge table and the child database (empty, with room for every projected item).
__device__ inline void allocate_child(Control& control, BlockState& s)
{
    if (threadIdx.x == 0) {
        s.hash_slots         = table_slots(s.projected_count);
        s.hashes             = static_cast<int32_t*>(control.allocate(sizeof(int32_t) * s.hash_slots));
        const uint64_t bytes = sizeof(Database) + sizeof(Item) * uint64_t{s.projected_items} +
                               sizeof(Transaction) * uint64_t{s.projected_count};
        auto* db = static_cast<Database*>(control.allocate(bytes));
        if (db != nullptr) {
            db->data             = reinterpret_cast<Item*>(db + 1);
            db->transactions     = reinterpret_cast<Transaction*>(db->data + s.projected_items);
            db->num_transactions = 0;
            db->num_items        = 0;
        }
        s.child.db = db;
    }
    __syncthreads();
    if (s.hashes == nullptr || s.child.db == nullptr)
        return;
    for (uint32_t i = threadIdx.x; i < s.hash_slots; i += blockDim.x)
        s.hashes[i] = -1;
    __syncthreads();
}

// Folds transaction `t` into the transaction already stored in merge slot `owner`.
__device__ inline void merge_into(Database& db, uint32_t owner, uint32_t t, uint32_t length)
{
    Transaction&       target = db.transactions[owner];
    const Transaction& source = db.transactions[t];
    add(target.utility, source.utility);
    for (uint32_t j = 0; j < length; ++j)
        add(target.data[j].util, source.data[j].util);
}

// Phase 2: keep only the items whose local utility meets the threshold, add the transactions to the
// child database, accumulate subtree utilities, and merge transactions that became identical.
__device__ inline void write_merge(Control& control, BlockState& s, uint32_t minimum_utility)
{
    Item      kept[kItemBuffer];
    Database& db = *s.child.db;

    for (uint32_t i = threadIdx.x; i < s.projected_count; i += blockDim.x) {
        const Transaction& t = s.projected[i];

        uint32_t n = 0, cumulative = t.utility;
        for (uint32_t j = 0; j < t.length; ++j) {
            const Item it    = t.data[j];
            const int  found = find_slot(s.table, s.slots, it.key);
            if (found < 0 || s.table[found].local_utility < minimum_utility)
                continue;
            if (n == kItemBuffer) {
                control.fail(kLongTransaction);
                break;
            }
            kept[n++] = it;
            cumulative += it.util;
        }
        if (n == 0)
            continue;

        const uint32_t slot = fetch_add(db.num_transactions, 1u);
        const uint32_t from = fetch_add(db.num_items, n);
        Transaction&   nt   = db.transactions[slot];
        nt                  = Transaction{&db.data[from], n, t.utility};
        for (uint32_t j = 0; j < n; ++j)
            nt[j] = kept[j];

        uint64_t hash   = 0;
        uint32_t before = 0;
        for (uint32_t j = 0; j < n; ++j) {
            accumulate<Utility::Subtree>(control, s.table, s.slots, kept[j].key, cumulative - before);
            hash ^= fasthash(kept[j].key + j);
            before += kept[j].util;
        }

        __threadfence_block(); // publish the transaction before other threads can find it
        uint32_t h = hash % s.hash_slots;
        for (uint32_t probes = 0; probes < s.hash_slots; ++probes) {
            const int32_t owner = compare_swap(s.hashes[h], int32_t{-1}, static_cast<int32_t>(slot));
            if (owner == -1)
                break;
            __threadfence_block();
            if (db.same_items(owner, slot)) {
                merge_into(db, owner, slot, n);
                nt.data = nullptr; // marks the duplicate for trim()
                break;
            }
            h = h + 1 == s.hash_slots ? 0 : h + 1;
        }
    }
    __syncthreads();
}

// Phase 3: drops the duplicates marked by write_merge, keeping the order of the rest.
__device__ inline void trim(Database& db)
{
    __shared__ Transaction tile[1024];
    __shared__ uint32_t    scan_shared[32];
    __shared__ uint32_t    kept_total, write_base;

    if (threadIdx.x == 0)
        kept_total = 0;
    __syncthreads();

    for (uint32_t base = 0; base < db.num_transactions; base += blockDim.x) {
        const uint32_t index = base + threadIdx.x;
        bool           valid = false;
        if (index < db.num_transactions) {
            tile[threadIdx.x] = db.transactions[index];
            valid             = tile[threadIdx.x].data != nullptr;
        }
        const uint32_t rank = block_inclusive_scan<uint32_t>(valid, scan_shared);
        if (threadIdx.x == blockDim.x - 1)
            write_base = rank > 0 ? fetch_add(kept_total, rank) : 0;
        __syncthreads();
        if (valid)
            db.transactions[write_base + rank - 1] = tile[threadIdx.x];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        db.num_transactions = kept_total;
    __syncthreads();
}

// Where a pushed task goes in the adaptive method: tiny ones to the single-thread stack, the rest to
// the block stack, with the size histogram the host uses to choose the next block size.
struct Routing
{
    bool         adaptive;
    uint32_t*    distribution; // 8 buckets: <=32, 64, ..., 1024, larger
    Stack<Task>* singles;
};

// Phase 4: counts the items that stay promising and pushes one task per item whose subtree utility
// meets the threshold.
__device__ inline void push_children(Control& control, BlockState& s, Stack<Task>& next, const Routing& routing,
                                     uint32_t minimum_utility)
{
    for (uint32_t i = threadIdx.x; i < s.slots; i += blockDim.x)
        if (s.table[i].local_utility >= minimum_utility)
            add(s.new_unique, 1u);
    __syncthreads();
    if (threadIdx.x == 0)
        s.child.unique_items = s.new_unique;
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < s.slots; i += blockDim.x) {
        if (s.table[i].subtree_utility < minimum_utility)
            continue;
        Task task    = s.child;
        task.primary = s.table[i].key;

        bool pushed = true;
        if (!routing.adaptive)
            pushed = next.push(task);
        else if (task.db->num_transactions <= kSingleThreadTransactions)
            pushed = routing.singles->push(task);
        else {
            const uint32_t n      = task.db->num_transactions;
            uint32_t       bucket = 7;
            if (n <= 1024) {
                bucket = 0;
                for (uint32_t size = 32; size < n; size <<= 1)
                    ++bucket;
            }
            add(routing.distribution[bucket], 1u);
            pushed = next.push(task);
        }
        if (!pushed)
            control.fail(kStackFull);
    }
    __syncthreads();
}

// The whole pipeline for the block's task.
__device__ inline void mine_block_task(Control& control, const Stack<Task>& current, Stack<Task>& next,
                                       const Routing& routing, uint32_t minimum_utility)
{
    __shared__ BlockState s;

    open_task(control, current, s);
    if (failed_open(s))
        return;
    project(control, s);
    if (s.projected == nullptr)
        return;

    if (threadIdx.x == 0)
        control.record(s.child, minimum_utility);
    __syncthreads();
    if (s.projected_count == 0)
        return;

    allocate_child(control, s);
    if (s.hashes == nullptr || s.child.db == nullptr)
        return;
    write_merge(control, s, minimum_utility);
    trim(*s.child.db);
    push_children(control, s, next, routing, minimum_utility);
}

} // namespace ghuim::dfs
