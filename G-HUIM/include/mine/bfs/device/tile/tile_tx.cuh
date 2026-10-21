#pragma once
// The tile transaction mode.

#include "mine/bfs/device/child.cuh"
#include "mine/bfs/device/modes.cuh"
#include "mine/bfs/placement.cuh"
#include "mine/bfs/device/tile/child_layout.cuh"
#include "mine/bfs/device/tile/emit.cuh"
#include "mine/bfs/device/tile/scan.cuh"

namespace ghuim::inline GHUIM_WIDTHS::bfs {

// Lane per transaction over tiled databases: a warp covers one tile of the parent per step.
struct TileTx
{
    using Ctx                  = NoCtx;
    static constexpr bool kTiled = true;
    static __device__ Ctx make_ctx(const BatchView&, const Projection&, uint32_t, uint32_t, uint32_t) { return {}; }
    static __device__ uint32_t unit(const Projection&, const Lanes& lanes) { return lanes.count; }
    static __device__ void scan_local(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step, TaskState& s,
                                      const Ctx&)
    {
        LocalAcc acc;
        for (uint32_t i0 = from; i0 < p.n_trans; i0 += step)
            tile_local(p, lanes, i0, acc);
        local_finish(lanes, acc, s);
    }
    static __device__ void scan_subtree(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step, TaskState& s,
                                        const Ctx&)
    {
        ChildSize size;
        for (uint32_t i0 = from; i0 < p.n_trans; i0 += step)
            tile_subtree(p, lanes, i0, size);
        subtree_finish(lanes, size, s);
    }
    static __device__ void assign_slots(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step, const Ctx&)
    {
        for (uint32_t i0 = from; i0 < p.n_trans; i0 += step)
            tile_assign(p, lanes, i0);
    }
    static __device__ void write_representatives(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step,
                                                 TaskState&, const ChildView& c, const Ctx&)
    {
        for (uint32_t i0 = from; i0 < p.n_trans; i0 += step)
            tile_write(p, lanes, i0, c);
    }
    static __device__ void merge_duplicates(const Projection& p, const Lanes& lanes, uint32_t from, uint32_t step,
                                            const ChildView& c, const Ctx&)
    {
        for (uint32_t i0 = from; i0 < p.n_trans; i0 += step)
            tile_merge(p, lanes, i0, c);
    }
};

template <typename Mode> concept TiledMode = requires { Mode::kTiled; };

// Calls `f` with the transaction mode a launch uses: TileTx for the tile method, else LaneTx.
template <typename F> void with_mode(const AutoTune& tune, F&& f)
{
    if (tune.tiled)
        f(TileTx{});
    else
        f(LaneTx{});
}

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
