#pragma once
// IO warps fill the ring, compute warps drain it.

#include "mine/bfs/device/staging/cp_async.cuh"
#include "mine/bfs/device/staging/ring.cuh"
#include "mine/bfs/device/staging/wait.cuh"

namespace ghuim::inline GHUIM_WIDTHS::bfs {

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
        Key*           keys  = reinterpret_cast<Key*>(t + kTileMetaBytes);
        Util*          utils = reinterpret_cast<Util*>(keys + kTileTx * c.stride);
        const uint32_t i     = tile * kTileTx + lane;
        const bool     want  = wants(p, i);
        const Tx       tx    = want ? p.txs[i] : Tx{0, 0, 0};
        if (want) {
            cp_async_4(&meta[lane].begin, &p.txs[i].begin);
            cp_async_4(&meta[lane].length, &p.txs[i].length);
            stage_copy(&meta[lane].utility, &p.txs[i].utility);
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
                stage_copy(&keys[j * c.stride + e], &p.eitem[begin + e]);
                stage_copy(&utils[j * c.stride + e], &p.eutil[begin + e]);
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

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
