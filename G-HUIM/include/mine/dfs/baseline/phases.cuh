#pragma once
// The passes of one search-tree node (static-parallelism kernels). Each takes the group `g` that
// cooperates on the task; the caller synchronizes the group between passes.

#include "mine/dfs/baseline/frame.cuh"

namespace ghuim::inline GHUIM_WIDTHS::dfs::baseline {

template <typename Group> __device__ void clear_state(const Group& g, const Frame& f)
{
    if (g.rank() == 0)
        *f.state = TaskState{};
    for (uint32_t i = f.first + g.rank(); i < f.n; i += g.size())
        f.local[i] = 0;
    for (uint32_t i = g.rank(); i < f.n - f.first; i += g.size())
        f.subtree[i] = 0;
}

// 1. Itemset utility and local utilities of later items.
template <typename Group> __device__ void scan_local(const Group& g, const Frame& f)
{
    Util utility = 0;
    for (uint32_t i = g.rank(); i < f.header.n_trans; i += g.size()) {
        const Tx     tx = f.txs[i];
        const Entry* e  = f.entries + tx.begin;
        const int    k  = find_item(e, tx.length, f.task.item);
        if (k < 0)
            continue;
        const Util prefix = tx.utility + e[k].utility;
        utility += prefix;
        Util rest = 0;
        for (uint32_t j = k + 1; j < tx.length; ++j)
            rest += e[j].utility;
        for (uint32_t j = k + 1; j < tx.length; ++j)
            add(f.local[e[j].item], prefix + rest);
    }
    if (utility != 0)
        add(f.state->utility, utility);
}

// Keeps the items whose local utility meets the threshold and renames them 0, 1, ... (one thread).
template <typename Group> __device__ void rename_items(const Group& g, const Frame& f)
{
    if (g.rank() == 0) {
        uint32_t kept = 0;
        for (uint32_t i = f.first; i < f.n; ++i)
            f.rename[i] = f.local[i] >= f.minutil ? kept++ : kNone;
        f.state->child_items = kept;
    }
}

template <typename Group> __device__ void clear_table(const Group& g, const Frame& f)
{
    for (uint32_t i = g.rank(); i < f.table_size; i += g.size())
        f.table[i] = kNone;
}

// 2. Subtree utilities over the kept items, and merging: each projected transaction finds the
//    first one with the same kept items (its representative). Only representatives count towards
//    the child database's size.
template <typename Group> __device__ void scan_subtree(const Group& g, const Frame& f)
{
    TaskState& state = *f.state;
    uint32_t my_trans = 0, my_entries = 0;
    for (uint32_t i = g.rank(); i < f.header.n_trans; i += g.size()) {
        f.rep[i]        = kNone;
        const Tx     tx = f.txs[i];
        const Entry* e  = f.entries + tx.begin;
        const int    k  = find_item(e, tx.length, f.task.item);
        if (k < 0)
            continue;
        Util remaining = tx.utility + e[k].utility;
        uint32_t kept = 0, hash = 2166136261u;
        for (uint32_t j = k + 1; j < tx.length; ++j)
            if (f.rename[e[j].item] != kNone) {
                remaining += e[j].utility;
                ++kept;
                hash = (hash ^ f.rename[e[j].item]) * 16777619u;
            }
        if (kept != 0)
            fetch_max(state.child_max_len, kept);
        if (kept == 0)
            continue;
        // subtree(z) = prefix utility + utility of z and every kept item after it.
        Util before = 0;
        for (uint32_t j = k + 1; j < tx.length; ++j) {
            const uint32_t z = f.rename[e[j].item];
            if (z == kNone)
                continue;
            add(f.subtree[z], remaining - before);
            before += e[j].utility;
        }
        for (uint32_t h = hash % f.table_size;; h = h + 1 == f.table_size ? 0 : h + 1) {
            const uint32_t other = compare_swap(f.table[h], kNone, i);
            if (other == kNone) {
                f.rep[i] = i;
                ++my_trans;
                my_entries += kept;
                add(state.entries_by_length[length_bin(kept)], kept);
                break;
            }
            if (same_kept_items(f.txs[other], f.entries, f.txs[i], f.task.item, f.rename)) {
                f.rep[i] = other;
                break;
            }
        }
    }
    if (my_trans != 0) {
        add(state.child_trans, my_trans);
        add(state.child_entries, my_entries);
    }
}

// One thread counts the promising extensions, allocates the child database and records the task.
template <typename Group> __device__ void allocate_child(const Group& g, const BatchView& b, const Frame& f)
{
    if (g.rank() != 0)
        return;
    TaskState& state       = *f.state;
    const uint32_t child_items = state.child_items;
    bool     deferred  = false;
    uint32_t primaries = 0;
    for (uint32_t z = 0; z < child_items; ++z)
        primaries += f.subtree[z] >= f.minutil;
    if (primaries != 0) {
        const DbLayout child = DbLayout::of(state.child_trans, state.child_entries, child_items);
        const uint64_t at    = fetch_add(b.control->arena_top, child.bytes);
        deferred = at + child.bytes > b.arena_capacity;
        if (deferred) {
            // The child does not fit: push the task back so a later batch retries it.
            add(b.control->deferred, 1u);
            push_task(b, f.task);
        }
        else {
            *reinterpret_cast<DbHeader*>(b.arena + at) =
                    DbHeader{state.child_trans, state.child_entries, child_items, state.child_max_len, child.bytes,
                             median_length_bin(state.entries_by_length), 0};
            state.child_db = at;
        }
    }
    if (!deferred)
        finish_task(b, f.task, state.utility >= f.minutil);
    state.child_primaries = deferred ? 0 : primaries;
}

// The projected child database in the arena.
struct Child
{
    uint64_t  bytes;
    uint32_t* global;
    Tx*       txs;
    Entry*    entries;
};

__device__ inline Child child_of(const BatchView& b, const Frame& f)
{
    const TaskState& state  = *f.state;
    const DbLayout   layout = DbLayout::of(state.child_trans, state.child_entries, state.child_items);
    char*            at     = b.arena + state.child_db;
    return Child{layout.bytes, reinterpret_cast<uint32_t*>(at + layout.items), reinterpret_cast<Tx*>(at + layout.tx),
                 reinterpret_cast<Entry*>(at + layout.entries)};
}

// 3. Materialize the projected database: item names, then representatives, then duplicates add
//    their utilities into their representative.
template <typename Group> __device__ void write_items(const Group& g, const Frame& f, const Child& c)
{
    const auto* globals = reinterpret_cast<const uint32_t*>(f.base + f.layout.items);
    for (uint32_t i = f.first + g.rank(); i < f.n; i += g.size())
        if (f.rename[i] != kNone)
            c.global[f.rename[i]] = globals[i];
}

template <typename Group> __device__ void write_representatives(const Group& g, const Frame& f, const Child& c)
{
    TaskState& state = *f.state;
    for (uint32_t i = g.rank(); i < f.header.n_trans; i += g.size()) {
        if (f.rep[i] != i)
            continue;
        const Tx       tx    = f.txs[i];
        const Entry*   e     = f.entries + tx.begin;
        const int      k     = find_item(e, tx.length, f.task.item);
        uint32_t       kept  = 0;
        for (uint32_t j = k + 1; j < tx.length; ++j)
            kept += f.rename[e[j].item] != kNone;
        const uint32_t slot  = fetch_add(state.tx_cursor, 1u);
        const uint32_t begin = fetch_add(state.entry_cursor, kept);
        f.slot_of[i]         = slot;
        c.txs[slot]          = Tx{begin, kept, static_cast<Util>(tx.utility + e[k].utility)};
        Entry* out           = c.entries + begin;
        for (uint32_t j = k + 1; j < tx.length; ++j) {
            const uint32_t z = f.rename[e[j].item];
            if (z != kNone)
                *out++ = Entry{static_cast<Key>(z), e[j].utility};
        }
    }
}

template <typename Group> __device__ void merge_duplicates(const Group& g, const Frame& f, const Child& c)
{
    for (uint32_t i = g.rank(); i < f.header.n_trans; i += g.size()) {
        if (f.rep[i] == kNone || f.rep[i] == i)
            continue;
        const Tx     tx  = f.txs[i];
        const Entry* e   = f.entries + tx.begin;
        const int    k   = find_item(e, tx.length, f.task.item);
        Tx&          dst = c.txs[f.slot_of[f.rep[i]]];
        add(dst.utility, tx.utility + e[k].utility);
        Entry* out = c.entries + dst.begin;
        for (uint32_t j = k + 1; j < tx.length; ++j)
            if (f.rename[e[j].item] != kNone)
                add((out++)->utility, e[j].utility);
    }
}

// Pushes one child task per promising extension.
template <typename Group> __device__ void emit_children(const Group& g, const BatchView& b, const Frame& f, const Child& c)
{
    const TaskState& state = *f.state;
    for (uint32_t z = g.rank(); z < state.child_items; z += g.size()) {
        if (f.subtree[z] < f.minutil)
            continue;
        push_task(b, Task{static_cast<uint32_t>(state.child_db / kArenaUnit), static_cast<uint32_t>(c.bytes / kArenaUnit),
                          state.child_trans, state.child_items, z, f.task.depth + 1});
    }
}

} // namespace ghuim::inline GHUIM_WIDTHS::dfs::baseline
