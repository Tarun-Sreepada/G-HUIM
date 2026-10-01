#pragma once
// Data structures of the depth-first (stack based) miners: projected databases, tasks, stacks and the
// device-side control block (arena allocator, itemset buffer, error flags).

#include "cuda/atomics.cuh"

#include <cstdint>

namespace ghuim::dfs {

struct Item
{
    uint32_t key; // item name, 1-based: 0 marks an empty hash slot
    uint32_t util;
};

struct Transaction
{
    Item*    data    = nullptr;
    uint32_t length  = 0;
    uint32_t utility = 0; // utility of the database's prefix itemset in this transaction

    __host__ __device__ Item& operator[](uint32_t index)
    {
        return data[index];
    }
    __host__ __device__ const Item& operator[](uint32_t index) const
    {
        return data[index];
    }

    // Position of `item` in the (key-sorted) transaction, or -1.
    __device__ int binary_search(uint32_t item) const
    {
        int low = 0, high = static_cast<int>(length) - 1;
        while (low <= high) {
            const int      mid = low + ((high - low) >> 1);
            const uint32_t key = data[mid].key;
            if (key == item)
                return mid;
            if (key > item)
                high = mid - 1;
            else
                low = mid + 1;
        }
        return -1;
    }

    // Sum of the item utilities.
    __device__ uint32_t util_sum() const
    {
        uint32_t total = 0;
        for (uint32_t i = 0; i < length; ++i)
            total += data[i].util;
        return total;
    }
};

struct Database
{
    Item*        data;
    Transaction* transactions;
    uint32_t     num_transactions;
    uint32_t     num_items;

    // Whether two transactions hold the same item keys.
    __device__ bool same_items(uint32_t a, uint32_t b) const
    {
        const Transaction& x = transactions[a];
        const Transaction& y = transactions[b];
        if (x.length != y.length)
            return false;
        for (uint32_t i = 0; i < x.length; ++i)
            if (x.data[i].key != y.data[i].key)
                return false;
        return true;
    }
};

// One slot of the open-addressing table that accumulates per-item utilities.
struct Utilities
{
    uint32_t key;
    uint32_t local_utility;
    uint32_t subtree_utility;

    __device__ void clear()
    {
        key = local_utility = subtree_utility = 0;
    }
};

// Slots of the utility table for `unique` items: one more than 1.5x, so a probe always meets a free slot.
__host__ __device__ constexpr uint32_t table_slots(uint32_t unique)
{
    return unique + unique / 2 + 1;
}

// Extend `pattern` with `primary` inside `db`.
struct Task
{
    uint32_t* pattern        = nullptr;
    uint32_t  pattern_length = 0;
    uint32_t  utility        = 0;
    Database* db             = nullptr;
    uint32_t  unique_items   = 0; // items of `db` whose local utility meets the threshold
    uint32_t  primary        = 0;
};

// A stack of tasks in device memory; the struct itself lives in managed memory so the host can read `top`.
template <typename T> struct Stack
{
    T*       data     = nullptr;
    uint32_t top      = 0;
    uint32_t capacity = 0;

    __device__ bool push(const T& item)
    {
        const uint32_t index = fetch_add(top, 1u);
        if (index >= capacity) {
            fetch_sub(top, 1u);
            return false;
        }
        data[index] = item;
        return true;
    }

    __device__ bool get(T& item, uint32_t index) const
    {
        if (index < top && index < capacity) {
            item = data[index];
            return true;
        }
        return false;
    }
};

enum ControlError : uint32_t
{
    kArenaFull       = 1u << 0, // out of projected-database memory
    kStackFull       = 1u << 1, // a task stack overflowed
    kPatternsFull    = 1u << 2, // the itemset buffer overflowed
    kLongTransaction = 1u << 3, // a projected transaction exceeds the per-thread item buffer
    kTableFull       = 1u << 4, // a utility hash table filled up
    kLimitExceeded   = 1u << 5, // a task exceeds a fixed per-thread limit
};

// State shared by every kernel. Out-of-memory conditions never touch memory outside the arena:
// they set `error` and the kernels bail out; the host reports it after the launch.
struct Control
{
    char*              arena;
    unsigned long long arena_limit;  // allocations must end below this offset
    unsigned long long arena_offset; // bump pointer
    uint32_t*          patterns;     // itemset buffer: items..., utility, 0 per itemset (null = count only)
    unsigned long long pattern_capacity;
    unsigned long long pattern_offset;
    unsigned long long pattern_count;
    uint32_t           error;

    // Reserves `bytes` (8-byte aligned) from the arena; null, with the error flag set, if it does not fit.
    __device__ void* allocate(uint64_t bytes)
    {
        bytes             = (bytes + 7) & ~uint64_t{7};
        const uint64_t at = fetch_add(arena_offset, bytes);
        if (at + bytes > arena_limit) {
            fail(kArenaFull);
            return nullptr;
        }
        return arena + at;
    }

    __device__ void fail(uint32_t bits)
    {
        fetch_or(error, bits);
    }

    // Counts the task's itemset if it is high-utility and stores it when a buffer is attached.
    // One thread calls it per itemset.
    __device__ void record(const Task& task, uint32_t minimum_utility)
    {
        if (task.utility < minimum_utility)
            return;
        add(pattern_count, 1ull);
        if (patterns == nullptr)
            return;
        const unsigned long long words = task.pattern_length + 2ull;
        const unsigned long long at    = fetch_add(pattern_offset, words);
        if (at + words > pattern_capacity) {
            fail(kPatternsFull);
            return;
        }
        for (uint32_t i = 0; i < task.pattern_length; ++i)
            patterns[at + i] = task.pattern[i];
        patterns[at + task.pattern_length]     = task.utility;
        patterns[at + task.pattern_length + 1] = 0;
    }
};

} // namespace ghuim::dfs
