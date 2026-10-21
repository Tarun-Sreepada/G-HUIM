#pragma once
// Extends the parent database's prefix with task.item (one EFIM search-tree node):
//  1. scan transactions containing the item: itemset utility and local utilities of later items
//  2. keep items whose local utility meets the threshold; compute their subtree utilities
//  3. if any kept item has a promising subtree, materialize the projected database and
//     emit one child task per such item

#include "mine/dfs/baseline/groups.cuh"
#include "mine/dfs/baseline/phases.cuh"

namespace ghuim::inline GHUIM_WIDTHS::dfs::baseline {

template <typename Group> __device__ void process_task(Group g, const BatchView& b, uint32_t t)
{
    const Frame f = open_frame(b, t);

    clear_state(g, f);
    g.sync();
    scan_local(g, f);
    g.sync();

    rename_items(g, f);
    g.sync();
    if (f.state->child_items == 0) {
        if (g.rank() == 0)
            finish_task(b, f.task, f.state->utility >= f.minutil);
        return;
    }

    clear_table(g, f);
    g.sync();
    scan_subtree(g, f);
    g.sync();

    allocate_child(g, b, f);
    g.sync();
    if (f.state->child_primaries == 0)
        return;

    const Child c = child_of(b, f);
    write_items(g, f, c);
    write_representatives(g, f, c);
    g.sync();
    merge_duplicates(g, f, c);
    g.sync(); // the child's item names must be complete before the tasks are emitted
    emit_children(g, b, f, c);
}

} // namespace ghuim::inline GHUIM_WIDTHS::dfs::baseline
