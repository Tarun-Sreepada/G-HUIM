#pragma once

#include "mine/bfs/device/child.cuh"

namespace ghuim::inline GHUIM_WIDTHS::bfs {

// ---------------------------------------------------------------------------------------------
// Transaction modes: how a task's passes map transactions onto lanes. process_task and the huge-task
// kernels are templates over a mode. `unit` is the number of consecutive transactions one warp covers
// per step, so a warp starts at warp_index * unit and the whole group of warps strides by their sum.
// ---------------------------------------------------------------------------------------------

// No per-task context: the plain mode needs none.
struct NoCtx
{
};

// Lane per transaction: a warp covers 32 consecutive transactions per step.
struct LaneTx
{
    using Ctx = NoCtx;
    static __device__ Ctx make_ctx(const BatchView&, const Projection&, uint32_t, uint32_t, uint32_t) { return {}; }
    static __device__ uint32_t unit(const Projection&, const Lanes& lanes) { return lanes.count; }
    static __device__ void scan_local(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step, TaskState& s,
                                      const Ctx&)
    {
        bfs::scan_local(p, lanes, from, step, s);
    }
    static __device__ void scan_subtree(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step, TaskState& s,
                                        const Ctx&)
    {
        bfs::scan_subtree(p, lanes, from, step, s);
    }
    static __device__ void write_representatives(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step,
                                                 TaskState& s, const ChildView& c, const Ctx&)
    {
        bfs::write_representatives(p, lanes, from, step, s, c);
    }
    static __device__ void merge_duplicates(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step,
                                            const ChildView& c, const Ctx&)
    {
        bfs::merge_duplicates(p, lanes, from, step, c);
    }
};

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
