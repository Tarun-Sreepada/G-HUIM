#pragma once
// Ring geometry for one task: slots, counters, and the tile view of a slot.

#include "mine/bfs/device/modes.cuh"

#include <cuda/atomic>

namespace ghuim::inline GHUIM_WIDTHS::bfs {

constexpr uint32_t kTileTx            = 32;
constexpr uint32_t kTileMetaBytes     = kTileTx * sizeof(Tx);
constexpr uint32_t kStageSpinLimit    = 1u << 26;

// Dynamic shared memory of the staged kernels: barriers, then the tiles.
extern __shared__ __align__(16) unsigned char stage_smem[];


struct StageCfg
{
    bool     staged; // false: the task runs the plain passes with every warp computing
    uint32_t io_warps, compute_warps;
    uint32_t slots, stride, tile_bytes;
    uint32_t slices, slice; // this block's share of the task (huge tasks: blockDim.y slices)
    uint32_t n_tiles;
    Control* control;
};

__device__ __forceinline__ StageCfg make_stage_cfg(const BatchView& b, const Projection& p, uint32_t warps,
                                                   uint32_t slices, uint32_t slice)
{
    StageCfg c;
    c.io_warps      = b.io_warps;
    c.compute_warps = warps > b.io_warps ? warps - b.io_warps : 0;
    c.stride        = p.max_len | 1u;
    c.tile_bytes    = kTileMetaBytes + kTileTx * c.stride * (sizeof(Key) + sizeof(Util));
    c.slots         = min(kRingMaxSlots, b.ring_bytes / c.tile_bytes);
    c.staged        = b.io_warps > 0 && c.compute_warps > 0 && c.slots >= 2;
    c.slices        = slices;
    c.slice         = slice;
    c.n_tiles       = (p.n_trans + kTileTx - 1) / kTileTx;
    c.control       = b.control;
    return c;
}

// The tile this block consumes as its q-th: with C compute warps the block covers tiles
// (r * slices + slice) * C + c in round r, which is the coverage of the unstaged loops.
__device__ __forceinline__ uint32_t tile_of(const StageCfg& c, uint32_t q)
{
    return ((q / c.compute_warps) * c.slices + c.slice) * c.compute_warps + q % c.compute_warps;
}

using Flag = cuda::atomic_ref<uint32_t, cuda::thread_scope_block>;

__device__ __forceinline__ uint32_t* full_counter(uint32_t s)
{
    return reinterpret_cast<uint32_t*>(stage_smem) + s;
}
__device__ __forceinline__ uint32_t* empty_counter(uint32_t s)
{
    return reinterpret_cast<uint32_t*>(stage_smem) + kRingMaxSlots + s;
}
__device__ __forceinline__ unsigned char* tile_base(const StageCfg& c, uint32_t s)
{
    return stage_smem + kStageBarrierBytes + size_t{s} * c.tile_bytes;
}

// A tile in shared memory as a pass Source: lane `l` of the step reads slot `l`.
struct TileSource
{
    const Tx*       meta;
    const Key*      keys_base;
    const Util*     utils_base;
    uint32_t        stride;

    __device__ Tx tx(uint32_t, uint32_t lane) const { return meta[lane]; }
    __device__ const Key*      keys(const Tx&, uint32_t lane) const { return keys_base + lane * stride; }
    __device__ const Util*     utils(const Tx&, uint32_t lane) const { return utils_base + lane * stride; }
};

__device__ __forceinline__ TileSource tile_source(const StageCfg& c, uint32_t s)
{
    unsigned char* t = tile_base(c, s);
    auto*          k = reinterpret_cast<Key*>(t + kTileMetaBytes);
    return TileSource{reinterpret_cast<const Tx*>(t), k, reinterpret_cast<const Util*>(k + kTileTx * c.stride), c.stride};
}

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
