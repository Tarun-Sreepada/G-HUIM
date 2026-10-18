#pragma once
// One group (a warp, or a block whose warps split the transactions) processes one whole task.

#include "mine/dfs/optimized/child.cuh"
#include "mine/dfs/optimized/passes.cuh"

namespace ghuim::dfs::opt {

template <typename Group>
__device__ void process_task(const Group& g, const BatchView& b, uint32_t t, uint32_t warp, uint32_t warps)
{
    const TaskView v      = open_task(b, t);
    const uint32_t first  = v.task.item + 1, n = v.header.n_items;
    const Lanes    lanes(g);
    const uint32_t from = warp * lanes.count, step = warps * lanes.count;

    clear_task(v, g.rank(), g.size());
    g.sync();
    scan_local(v.p, lanes, from, step, *v.state);
    g.sync();

    uint32_t child_items = rename_items(g, v.p, first, n, b.minimum_utility);
    if (b.refine_rounds != 0)
        child_items = refine_items(g, lanes, v.p, first, n, child_items, b.minimum_utility, b.refine_rounds, from, step);
    if (child_items == 0) {
        if (g.rank() == 0)
            finish_task(b, v.task, v.state->utility >= b.minimum_utility);
        return;
    }
    g.sync();
    scan_subtree(v.p, lanes, from, step, *v.state);
    g.sync();

    allocate_child(g, b, v, child_items);
    if (v.state->child_primaries == 0)
        return;

    const ChildView c = child_view(b, *v.state, child_items);
    copy_child_items(v, c, g.rank(), g.size());
    write_representatives(v.p, lanes, from, step, *v.state, c);
    g.sync();
    merge_duplicates(v.p, lanes, from, step, c);
    g.sync(); // the child's items and transactions are complete
    emit_children(g, b, v, child_items, c);
}

} // namespace ghuim::dfs::opt
