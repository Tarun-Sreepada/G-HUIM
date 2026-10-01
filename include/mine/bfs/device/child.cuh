#pragma once
// The child database: allocation, item names, transaction writing, merging and the next-level tasks.

#include "mine/bfs/device/passes.cuh"

namespace ghuim::bfs {

// ---------------------------------------------------------------------------------------------
// Child database
// ---------------------------------------------------------------------------------------------

// Counts the promising extensions; one thread then allocates the exact-size child database,
// records the task's itemset, and reserves next-level slots. All threads of the group must call it.
template <typename Group>
__device__ void allocate_child(const Group& g, const BatchView& b, const TaskView& v, uint32_t child_items)
{
    TaskState& state     = *v.state;
    uint32_t   primaries = 0;
    for (uint32_t z0 = 0; z0 < child_items; z0 += g.size()) {
        const uint32_t z = z0 + g.rank();
        primaries += g.count(z < child_items && v.p.subtree[z] >= b.minimum_utility);
    }
    if (g.rank() == 0) {
        if (primaries != 0) {
            const DbLayout child = DbLayout::of(state.child_trans, state.child_entries, child_items);
            const uint64_t at    = fetch_add(b.control->arena_top, child.bytes);
            if (at + child.bytes > b.dst_capacity) {
                fetch_or(b.control->error, kArenaFull);
                primaries = 0;
            }
            else {
                *reinterpret_cast<DbHeader*>(b.dst + at) =
                        DbHeader{state.child_trans, state.child_entries, child_items, state.child_max_len, child.bytes};
                state.child_db = at;
                state.reserved = fetch_add(b.control->next_count, primaries);
                if (uint64_t{state.reserved} + primaries > b.next_capacity)
                    fetch_or(b.control->error, kNextLevelFull);
            }
        }
        finish_task(b, v);
        state.child_primaries = primaries;
    }
    g.sync();
}

struct ChildView
{
    uint64_t  bytes;
    uint32_t* global;
    Tx*       txs;
    Entry*    entries;
};

__device__ __forceinline__ ChildView child_view(const BatchView& b, const TaskState& state, uint32_t child_items)
{
    ChildView      c;
    const DbLayout layout = DbLayout::of(state.child_trans, state.child_entries, child_items);
    char*          at     = b.dst + state.child_db;
    c.bytes               = layout.bytes;
    c.global              = reinterpret_cast<uint32_t*>(at + layout.items);
    c.txs                 = reinterpret_cast<Tx*>(at + layout.tx);
    c.entries             = reinterpret_cast<Entry*>(at + layout.entries);
    return c;
}

// The child's item names: parent global ids of the kept items.
__device__ __forceinline__ void copy_child_items(const TaskView& v, const ChildView& c, uint32_t rank, uint32_t stride)
{
    const auto*    globals = reinterpret_cast<const uint32_t*>(v.base + v.layout.items);
    const uint32_t n       = v.header.n_items;
    for (uint32_t i = v.task.item + 1 + rank; i < n; i += stride)
        if (v.p.rename[i] != kNone)
            c.global[v.p.rename[i]] = globals[i];
}

// Pass 3a: representatives write their child transaction and entries.
__device__ inline void write_representatives(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step,
                                             TaskState& state, const ChildView& c)
{
    for (uint32_t i0 = from; i0 < p.n_trans; i0 += step) {
        const uint32_t i     = i0 + lanes.rank;
        const bool     write = i < p.n_trans && p.rep[i] == i;
        const Tx       tx    = write ? p.txs[i] : Tx{0, 0, 0};
        const Entry*   e     = p.entries + tx.begin;
        const int      k     = write ? find_item(e, tx.length, p.item) : -1;
        const uint32_t rest  = k >= 0 ? tx.length - k - 1 : 0;
        const uint32_t span  = lanes.max(rest);
        uint32_t       kept  = 0;
        for (uint32_t d = 0; d < span; ++d)
            kept += d < rest && p.rename[e[k + 1 + d].item] != kNone;
        Entry* out = nullptr;
        if (write) {
            const uint32_t slot  = fetch_add(state.tx_cursor, 1u);
            const uint32_t begin = fetch_add(state.entry_cursor, kept);
            p.slot_of[i]         = slot;
            c.txs[slot]          = Tx{begin, kept, tx.utility + e[k].utility};
            out                  = c.entries + begin;
        }
        for (uint32_t d = 0; d < span; ++d) {
            if (d >= rest)
                continue;
            const uint32_t z = p.rename[e[k + 1 + d].item];
            if (z != kNone)
                *out++ = Entry{z, e[k + 1 + d].utility};
        }
    }
}

// Pass 3b: duplicates add their utilities into their representative's child transaction.
__device__ inline void merge_duplicates(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step,
                                        const ChildView& c)
{
    for (uint32_t i0 = from; i0 < p.n_trans; i0 += step) {
        const uint32_t i     = i0 + lanes.rank;
        const uint32_t r     = i < p.n_trans ? p.rep[i] : kNone;
        const bool     merge = r != kNone && r != i;
        const Tx       tx    = merge ? p.txs[i] : Tx{0, 0, 0};
        const Entry*   e     = p.entries + tx.begin;
        const int      k     = merge ? find_item(e, tx.length, p.item) : -1;
        const uint32_t rest  = k >= 0 ? tx.length - k - 1 : 0;
        const uint32_t span  = lanes.max(rest);
        Entry*         out   = nullptr;
        if (merge) {
            const uint32_t dst = p.slot_of[r];
            add(c.txs[dst].utility, tx.utility + e[k].utility);
            out = c.entries + c.txs[dst].begin;
        }
        for (uint32_t d = 0; d < span; ++d) {
            if (d >= rest)
                continue;
            if (p.rename[e[k + 1 + d].item] != kNone)
                add((out++)->utility, e[k + 1 + d].utility);
        }
    }
}

// Writes one next-level task per promising extension into the reserved slots (group scan).
template <typename Group>
__device__ void emit_children(const Group& g, const BatchView& b, const TaskView& v, uint32_t child_items,
                              const ChildView& c)
{
    const TaskState& state   = *v.state;
    uint32_t         emitted = 0;
    for (uint32_t z0 = 0; z0 < child_items; z0 += g.size()) {
        const uint32_t z    = z0 + g.rank();
        const bool     emit = z < child_items && v.p.subtree[z] >= b.minimum_utility;
        const uint32_t scan = g.scan(emit);
        const uint32_t slot = state.reserved + emitted + scan - 1;
        if (emit && slot < b.next_capacity)
            b.next[slot] = Task{static_cast<uint32_t>(state.child_db / kArenaUnit),
                                static_cast<uint32_t>(c.bytes / kArenaUnit),
                                state.child_trans,
                                child_items,
                                z,
                                v.index};
        emitted += g.count(emit);
    }
}

} // namespace ghuim::bfs
