#pragma once
// Where a pass reads a lane's transaction from. The passes are written once over a Source: a
// GlobalSource reads the database in global memory, a TileSource (staging.cuh) reads a copy that an
// IO warp placed in shared memory.

#include "mine/bfs/device/task_view.cuh"

namespace ghuim::inline GHUIM_WIDTHS::bfs {

struct GlobalSource
{
    const Projection& p;

    // Transaction `i` (global index); `lane` is the lane's position inside the 32-transaction step.
    __device__ Tx tx(uint32_t i, uint32_t) const { return p.txs[p.occ ? p.occ[i].tid : i]; }
    __device__ const Key*      keys(const Tx& t, uint32_t) const { return p.eitem + t.begin; }
    __device__ const Util*     utils(const Tx& t, uint32_t) const { return p.eutil + t.begin; }
};

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
