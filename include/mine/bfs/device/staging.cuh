#pragma once
// Asynchronous staging: IO warps copy transactions from global memory into a ring of shared-memory
// tiles with cp.async while compute warps run the lane-per-transaction passes on the tiles.
//
// A tile is 32 consecutive transactions (what one compute warp covers per step). Every transaction
// owns a slot of `stride` entries (the database's longest transaction, rounded up to an odd number so
// the 32 lanes of a compute warp, each reading its own slot, fall on different shared-memory banks),
// as separate key and utility arrays. Each ring slot has two sequence counters: `full[s]` counts the
// uses of the slot whose data has landed, `empty[s]` counts the uses that compute warps have finished.
// Use n of slot s (tile q = n * S + s) is readable once full[s] > n and the slot may be refilled for
// use n + 1 once empty[s] > n. Exact counters, not phase parities, because a compute warp can run a
// whole lap ahead of a slow IO warp and a parity would then look complete too early.
//
// An IO warp keeps up to two tiles of cp.async copies in flight (commit groups) and publishes a tile
// once its group has landed. Tiles are numbered q = 0, 1, ... in the order a block consumes them;
// compute warp c takes q = c (mod C), IO warp p fills q = p (mod P), the slot is q mod S. Every wait
// depends on a strictly smaller q, so the ring cannot deadlock, and an IO warp publishes what it has
// in flight before it blocks. All waits are bounded: a protocol bug ends the kernel with kStageTimeout
// instead of hanging the GPU.

#include "mine/bfs/device/warp_passes.cuh"

#include <cuda/atomic>

namespace ghuim::bfs {

constexpr uint32_t kTileTx            = 32;
constexpr uint32_t kTileMetaBytes     = kTileTx * sizeof(Tx);
constexpr uint32_t kStageSpinLimit    = 1u << 26;

// Dynamic shared memory of the staged kernels: barriers, then the tiles.
extern __shared__ __align__(16) unsigned char stage_smem[];

// ---------------------------------------------------------------------------------------------
// cp.async (sm_80+)
// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ uint32_t smem_address(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async_4(void* dst, const void* src)
{
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem_address(dst)), "l"(src) : "memory");
}

__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;" ::: "memory");
}

// Waits until at most N of this thread's most recent groups are still in flight.
template <int N> __device__ __forceinline__ void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

// ---------------------------------------------------------------------------------------------
// Ring geometry for one task
// ---------------------------------------------------------------------------------------------

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
    c.tile_bytes    = kTileMetaBytes + 2 * kTileTx * c.stride * sizeof(uint32_t);
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
    const uint32_t* keys_base;
    const uint32_t* utils_base;
    uint32_t        stride;

    __device__ Tx tx(uint32_t, uint32_t lane) const { return meta[lane]; }
    __device__ const uint32_t* keys(const Tx&, uint32_t lane) const { return keys_base + lane * stride; }
    __device__ const uint32_t* utils(const Tx&, uint32_t lane) const { return utils_base + lane * stride; }
};

__device__ __forceinline__ TileSource tile_source(const StageCfg& c, uint32_t s)
{
    unsigned char* t = tile_base(c, s);
    auto*          k = reinterpret_cast<uint32_t*>(t + kTileMetaBytes);
    return TileSource{reinterpret_cast<const Tx*>(t), k, k + kTileTx * c.stride, c.stride};
}

// ---------------------------------------------------------------------------------------------
// Waiting, with a bound and optional statistics
// ---------------------------------------------------------------------------------------------

__device__ __noinline__ inline void stage_fail(const StageCfg& c, uint32_t error)
{
    atomicOr(&c.control->error, error);
    __trap();
}

__device__ __forceinline__ bool counter_reached(uint32_t* counter, uint32_t value)
{
    return Flag(*counter).load(cuda::std::memory_order_acquire) >= value;
}

// Spins until `*counter >= value`, within the bound.
__device__ __forceinline__ void stage_wait(const StageCfg& c, uint32_t* counter, uint32_t value,
                                           unsigned long long* wait_counter)
{
#ifdef GHUIM_STAGE_STATS
    const long long t0 = clock64();
#endif
    uint32_t spins = 0;
    while (!counter_reached(counter, value)) {
        if (++spins > kStageSpinLimit)
            stage_fail(c, kStageTimeout);
        if (spins > 32)
            __nanosleep(32);
    }
#ifdef GHUIM_STAGE_STATS
    if ((threadIdx.x & 31) == 0)
        add(*wait_counter, static_cast<unsigned long long>(clock64() - t0));
#else
    (void)wait_counter;
#endif
}

// ---------------------------------------------------------------------------------------------
// IO warp and compute warps
// ---------------------------------------------------------------------------------------------

template <typename Wants>
__device__ void stage_produce(const Projection& p, const StageCfg& c, uint32_t io, Wants wants)
{
    const uint32_t lane        = threadIdx.x & 31;
    bool           pending     = false; // a tile whose copies are committed but not yet published
    uint32_t       pending_s   = 0, pending_use = 0;
    auto publish = [&] {
        cp_async_wait<0>();
        __syncwarp();
        if (lane == 0)
            Flag(*full_counter(pending_s)).store(pending_use + 1, cuda::std::memory_order_release);
        pending = false;
    };
    for (uint32_t q = io;; q += c.io_warps) {
        const uint32_t tile = tile_of(c, q);
        if (tile >= c.n_tiles)
            break;
        const uint32_t s = q % c.slots, use = q / c.slots;
        // The slot is free once its previous use has been consumed. Publish what is in flight before blocking.
        if (use > 0 && !counter_reached(empty_counter(s), use)) {
            if (pending)
                publish();
            stage_wait(c, empty_counter(s), use, &c.control->stage_empty_wait);
        }
        unsigned char* t     = tile_base(c, s);
        auto*          meta  = reinterpret_cast<Tx*>(t);
        auto*          keys  = reinterpret_cast<uint32_t*>(t + kTileMetaBytes);
        uint32_t*      utils = keys + kTileTx * c.stride;
        const uint32_t i     = tile * kTileTx + lane;
        const bool     want  = wants(p, i);
        const Tx       tx    = want ? p.txs[i] : Tx{0, 0, 0};
        if (want) {
            cp_async_4(&meta[lane].begin, &p.txs[i].begin);
            cp_async_4(&meta[lane].length, &p.txs[i].length);
            cp_async_4(&meta[lane].utility, &p.txs[i].utility);
        }
        uint32_t mask = __ballot_sync(kFullMask, want);
        while (mask != 0) {
            const uint32_t j     = __ffs(mask) - 1;
            mask &= mask - 1;
            const uint32_t begin = __shfl_sync(kFullMask, tx.begin, j);
            const uint32_t len   = __shfl_sync(kFullMask, tx.length, j);
            if (len > c.stride)
                stage_fail(c, kStageOverflow);
            for (uint32_t e = lane; e < len; e += 32) {
                cp_async_4(&keys[j * c.stride + e], &p.eitem[begin + e]);
                cp_async_4(&utils[j * c.stride + e], &p.eutil[begin + e]);
            }
        }
        cp_async_commit();
        // Publish the previous tile (at most one other group stays in flight).
        if (pending) {
            cp_async_wait<1>();
            __syncwarp();
            if (lane == 0)
                Flag(*full_counter(pending_s)).store(pending_use + 1, cuda::std::memory_order_release);
        }
        pending     = true;
        pending_s   = s;
        pending_use = use;
    }
    if (pending)
        publish();
}

// `body(i0, source)` processes the tile that starts at transaction i0.
template <typename Body>
__device__ void stage_consume(const StageCfg& c, uint32_t cw, Body body)
{
    const uint32_t lane = threadIdx.x & 31;
#ifdef GHUIM_STAGE_STATS
    const long long t0 = clock64();
#endif
    for (uint32_t q = cw;; q += c.compute_warps) {
        const uint32_t tile = tile_of(c, q);
        if (tile >= c.n_tiles)
            break;
        const uint32_t s = q % c.slots, use = q / c.slots;
        stage_wait(c, full_counter(s), use + 1, &c.control->stage_full_wait);
        body(tile * kTileTx, tile_source(c, s));
        __syncwarp();
        if (lane == 0)
            Flag(*empty_counter(s)).store(use + 1, cuda::std::memory_order_release);
    }
#ifdef GHUIM_STAGE_STATS
    if (lane == 0)
        add(c.control->stage_consumer_cycles, static_cast<unsigned long long>(clock64() - t0));
#endif
}

// One staged pass: the counters are reset, then warps below `io_warps` produce and the rest consume.
// Every thread of the block must call it.
template <typename Wants, typename Body>
__device__ void run_staged_pass(const Projection& p, const StageCfg& c, Wants wants, Body body)
{
    __syncthreads();
    if (threadIdx.x == 0)
        for (uint32_t s = 0; s < c.slots; ++s) {
            *full_counter(s)  = 0;
            *empty_counter(s) = 0;
        }
    __syncthreads();
    const uint32_t warp = threadIdx.x / 32;
    if (warp < c.io_warps)
        stage_produce(p, c, warp, wants);
    else
        stage_consume(c, warp - c.io_warps, body);
}

// ---------------------------------------------------------------------------------------------
// Staged transaction mode
// ---------------------------------------------------------------------------------------------

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

} // namespace ghuim::bfs
