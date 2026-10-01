#pragma once
// One small task (adaptive method) processed by one thread. Same steps as block_task.cuh without
// atomics or barriers; the projected transactions of all threads of a block live in shared memory.

#include "mine/dfs/block_task.cuh"

namespace ghuim::dfs {

constexpr uint32_t kSingleBlock = 64; // threads per block of the single-thread kernel
// Tables up to this many slots live on the thread's stack, larger ones in the arena.
constexpr uint32_t kLocalSlots = kItemBuffer * 3 / 2;

__device__ inline void mine_single_task(Control& control, const Task& task, Stack<Task>& next, Transaction* projected,
                                        uint32_t minimum_utility)
{
    const Database& db    = *task.db;
    const uint32_t  slots = table_slots(task.unique_items);

    Utilities  local[kLocalSlots];
    Utilities* table = local;
    if (slots > kLocalSlots) {
        table = static_cast<Utilities*>(control.allocate(sizeof(Utilities) * slots));
        if (table == nullptr)
            return;
    }
    for (uint32_t i = 0; i < slots; ++i)
        table[i].clear();

    Task child{};
    child.pattern_length = task.pattern_length + 1;
    child.pattern        = static_cast<uint32_t*>(control.allocate(sizeof(uint32_t) * child.pattern_length));
    if (child.pattern == nullptr)
        return;
    for (uint32_t i = 0; i < task.pattern_length; ++i)
        child.pattern[i] = task.pattern[i];
    child.pattern[task.pattern_length] = task.primary;

    // Project on the primary item.
    uint32_t count = 0, items = 0;
    for (uint32_t i = 0; i < db.num_transactions; ++i) {
        const Transaction& t   = db.transactions[i];
        const int          idx = t.binary_search(task.primary);
        if (idx < 0)
            continue;
        const uint32_t prefix = t.utility + t[idx].util;
        child.utility += prefix;
        const uint32_t remain = t.length - idx - 1;
        if (remain == 0)
            continue;
        if (count == kSingleThreadTransactions) {
            control.fail(kLimitExceeded);
            return;
        }
        projected[count++] = Transaction{&t.data[idx + 1], remain, prefix};
        items += remain;
    }
    control.record(child, minimum_utility);
    if (count == 0)
        return;

    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t total = projected[i].utility + projected[i].util_sum();
        for (uint32_t j = 0; j < projected[i].length; ++j)
            accumulate_private<Utility::Local>(control, table, slots, projected[i][j].key, total);
    }

    const uint64_t bytes    = sizeof(Database) + sizeof(Item) * uint64_t{items} + sizeof(Transaction) * uint64_t{count};
    auto*          child_db = static_cast<Database*>(control.allocate(bytes));
    if (child_db == nullptr)
        return;
    child_db->data             = reinterpret_cast<Item*>(child_db + 1);
    child_db->transactions     = reinterpret_cast<Transaction*>(child_db->data + items);
    child_db->num_transactions = child_db->num_items = 0;
    child.db                                         = child_db;

    // Child database: items that still meet the threshold, with their subtree utilities.
    Item kept[kItemBuffer];
    for (uint32_t i = 0; i < count; ++i) {
        const Transaction& t = projected[i];
        uint32_t           n = 0, cumulative = t.utility;
        for (uint32_t j = 0; j < t.length; ++j) {
            const int found = find_slot(table, slots, t.data[j].key);
            if (found < 0 || table[found].local_utility < minimum_utility)
                continue;
            if (n == kItemBuffer) {
                control.fail(kLimitExceeded | kLongTransaction);
                return;
            }
            kept[n++] = t.data[j];
            cumulative += t.data[j].util;
        }
        if (n == 0)
            continue;
        Transaction& nt = child_db->transactions[child_db->num_transactions++];
        nt              = Transaction{&child_db->data[child_db->num_items], n, t.utility};
        child_db->num_items += n;
        uint32_t before = 0;
        for (uint32_t j = 0; j < n; ++j) {
            nt[j] = kept[j];
            accumulate_private<Utility::Subtree>(control, table, slots, kept[j].key, cumulative - before);
            before += kept[j].util;
        }
    }

    for (uint32_t i = 0; i < slots; ++i)
        child.unique_items += table[i].local_utility >= minimum_utility;
    for (uint32_t i = 0; i < slots; ++i) {
        if (table[i].subtree_utility < minimum_utility)
            continue;
        child.primary = table[i].key;
        if (!next.push(child))
            control.fail(kStackFull);
    }
}

} // namespace ghuim::dfs
