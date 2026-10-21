#pragma once
// Lane per transaction, with the transactions staged by IO warps when the task's tiles fit the ring.

#include "mine/bfs/device/staging/pipeline.cuh"

namespace ghuim::inline GHUIM_WIDTHS::bfs {

// Lane per transaction, with the transactions staged by IO warps when the task's tiles fit the ring.
struct StagedTx
{
    using Ctx = StageCfg;
    static __device__ Ctx make_ctx(const BatchView& b, const Projection& p, uint32_t warps, uint32_t slices, uint32_t slice)
    {
        return make_stage_cfg(b, p, warps, slices, slice);
    }
    static __device__ uint32_t unit(const Projection&, const Lanes& lanes) { return lanes.count; }

    static __device__ void scan_local(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step, TaskState& s,
                                      const Ctx& c)
    {
        if (!c.staged) {
            bfs::scan_local(p, lanes, from, step, s);
            return;
        }
        LocalAcc acc;
        run_staged_pass(p, c, local_wants, [&](uint32_t i0, const TileSource& src) { local_tile(p, lanes, i0, src, acc); });
        if (threadIdx.x / 32 >= c.io_warps)
            local_finish(lanes, acc, s);
    }
    static __device__ void scan_subtree(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step, TaskState& s,
                                        const Ctx& c)
    {
        if (!c.staged) {
            bfs::scan_subtree(p, lanes, from, step, s);
            return;
        }
        ChildSize size;
        run_staged_pass(p, c, subtree_wants, [&](uint32_t i0, const TileSource& src) { subtree_tile(p, lanes, i0, src, size); });
        if (threadIdx.x / 32 >= c.io_warps)
            subtree_finish(lanes, size, s);
    }
    static __device__ void write_representatives(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step,
                                                 TaskState& s, const ChildView& cv, const Ctx& c)
    {
        if (!c.staged) {
            bfs::write_representatives(p, lanes, from, step, s, cv);
            return;
        }
        run_staged_pass(p, c, write_wants, [&](uint32_t i0, const TileSource& src) { write_tile(p, lanes, i0, src, s, cv); });
    }
    static __device__ void merge_duplicates(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step,
                                            const ChildView& cv, const Ctx& c)
    {
        if (!c.staged) {
            bfs::merge_duplicates(p, lanes, from, step, cv);
            return;
        }
        run_staged_pass(p, c, merge_wants, [&](uint32_t i0, const TileSource& src) { merge_tile(p, lanes, i0, src, cv); });
    }
};

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
