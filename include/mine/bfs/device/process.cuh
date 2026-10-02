#pragma once
// All passes of one task on one thread group.

#include "mine/bfs/device/staging.cuh"

namespace ghuim::bfs {

// One group (lanes of a warp, a warp, or a block whose warps split the transactions) processes one
// whole task.
template <typename Mode, typename Group>
__device__ void process_task(const Group& g, const BatchView& b, uint32_t t, uint32_t warp, uint32_t warps)
{
    const TaskView v     = open_task(b, t);
    const uint32_t first = v.task.item + 1, n = v.header.n_items;
    const Lanes    lanes(g);
    const uint32_t unit = Mode::unit(v.p, lanes);
    const uint32_t from = warp * unit, step = warps * unit;
    const auto     ctx  = Mode::make_ctx(b, v.p, warps, 1, 0);

    clear_task(v, g.rank(), g.size());
    g.sync();
    Mode::scan_local(v.p, lanes, from, step, *v.state, ctx);
    g.sync();

    const uint32_t child_items = rename_items(g, v.p, first, n, b.minimum_utility);
    if (child_items == 0) {
        if (g.rank() == 0)
            finish_task(b, v);
        return;
    }
    g.sync();
    Mode::scan_subtree(v.p, lanes, from, step, *v.state, ctx);
    g.sync();

    allocate_child(g, b, v, child_items);
    if (v.state->child_primaries == 0)
        return;

    const ChildView c = child_view(b, *v.state, child_items);
    copy_child_items(v, c, g.rank(), g.size());
    Mode::write_representatives(v.p, lanes, from, step, *v.state, c, ctx);
    g.sync();
    Mode::merge_duplicates(v.p, lanes, from, step, c, ctx);
    g.sync();
    emit_children(g, b, v, child_items, c);
}

} // namespace ghuim::bfs
