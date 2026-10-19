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

#include "mine/bfs/device/staging/cp_async.cuh"
#include "mine/bfs/device/staging/pipeline.cuh"
#include "mine/bfs/device/staging/ring.cuh"
#include "mine/bfs/device/staging/staged_tx.cuh"
#include "mine/bfs/device/staging/wait.cuh"
