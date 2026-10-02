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
    uint32_t* eitem;
    uint32_t* eutil;
};

__device__ __forceinline__ ChildView child_view(const BatchView& b, const TaskState& state, uint32_t child_items)
{
    ChildView      c;
    const DbLayout layout = DbLayout::of(state.child_trans, state.child_entries, child_items);
    char*          at     = b.dst + state.child_db;
    c.bytes               = layout.bytes;
    c.global              = reinterpret_cast<uint32_t*>(at + layout.items);
    c.txs                 = reinterpret_cast<Tx*>(at + layout.tx);
    c.eitem               = reinterpret_cast<uint32_t*>(at + layout.eitem);
    c.eutil               = reinterpret_cast<uint32_t*>(at + layout.eutil);
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
template <typename Src>
__device__ inline void write_tile(const Projection& p, const Lanes& lanes, uint32_t i0, const Src& src, TaskState& state,
                                  const ChildView& c)
{
    const uint32_t  i      = i0 + lanes.rank;
    const bool      write  = i < p.n_trans && p.rep[i] == i;
    const Tx        tx     = write ? src.tx(i, lanes.rank) : Tx{0, 0, 0};
    const uint32_t* k_item = src.keys(tx, lanes.rank);
    const uint32_t* k_util = src.utils(tx, lanes.rank);
    const int       k      = write ? static_cast<int>(p.pos[i]) - 1 : -1;
    const uint32_t  rest   = k >= 0 ? tx.length - k - 1 : 0;
    const uint32_t  span   = lanes.max(rest);
    const uint32_t  kept   = write ? p.kept[i] : 0;
    uint32_t* out_item = nullptr;
    uint32_t* out_util = nullptr;
    if (write) {
        const uint32_t slot  = fetch_add(state.tx_cursor, 1u);
        const uint32_t begin = fetch_add(state.entry_cursor, kept);
        p.slot_of[i]         = slot;
        c.txs[slot]          = Tx{begin, kept, tx.utility + k_util[k]};
        out_item             = c.eitem + begin;
        out_util             = c.eutil + begin;
    }
    for (uint32_t d = 0; d < span; ++d) {
        if (d >= rest)
            continue;
        const uint32_t z = p.rename[k_item[k + 1 + d]];
        if (z != kNone) {
            *out_item++ = z;
            *out_util++ = k_util[k + 1 + d];
        }
    }
}

__device__ inline bool write_wants(const Projection& p, uint32_t i)
{
    return i < p.n_trans && p.rep[i] == i;
}

__device__ inline void write_representatives(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step,
                                             TaskState& state, const ChildView& c)
{
    const GlobalSource src{p};
    for (uint32_t i0 = from; i0 < p.n_trans; i0 += step)
        write_tile(p, lanes, i0, src, state, c);
}

// Pass 3b: duplicates add their utilities into their representative's child transaction.
template <typename Src>
__device__ inline void merge_tile(const Projection& p, const Lanes& lanes, uint32_t i0, const Src& src, const ChildView& c)
{
    const uint32_t  i      = i0 + lanes.rank;
    const uint32_t  r      = i < p.n_trans ? p.rep[i] : kNone;
    const bool      merge  = r != kNone && r != i;
    const Tx        tx     = merge ? src.tx(i, lanes.rank) : Tx{0, 0, 0};
    const uint32_t* k_item = src.keys(tx, lanes.rank);
    const uint32_t* k_util = src.utils(tx, lanes.rank);
    const int       k      = merge ? static_cast<int>(p.pos[i]) - 1 : -1;
    const uint32_t  rest   = k >= 0 ? tx.length - k - 1 : 0;
    const uint32_t  span   = lanes.max(rest);
    uint32_t*       out    = nullptr;
    if (merge) {
        const uint32_t dst = p.slot_of[r];
        add(c.txs[dst].utility, tx.utility + k_util[k]);
        out = c.eutil + c.txs[dst].begin;
    }
    for (uint32_t d = 0; d < span; ++d) {
        if (d >= rest)
            continue;
        if (p.rename[k_item[k + 1 + d]] != kNone)
            add(*out++, k_util[k + 1 + d]);
    }
}

__device__ inline bool merge_wants(const Projection& p, uint32_t i)
{
    if (i >= p.n_trans)
        return false;
    const uint32_t r = p.rep[i];
    return r != kNone && r != i;
}

__device__ inline void merge_duplicates(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step,
                                        const ChildView& c)
{
    const GlobalSource src{p};
    for (uint32_t i0 = from; i0 < p.n_trans; i0 += step)
        merge_tile(p, lanes, i0, src, c);
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
                                v.index,
                                state.child_max_len,
                                state.child_entries};
        emitted += g.count(emit);
    }
}

} // namespace ghuim::bfs
