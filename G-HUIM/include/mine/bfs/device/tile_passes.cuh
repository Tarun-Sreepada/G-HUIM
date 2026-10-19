#pragma once
// Tile method: lane-per-transaction passes over databases in the tile layout (layout.cuh). Entry k of
// a transaction is at begin + k * kTile, so the lanes of a warp step (one tile of the parent) read
// neighbouring words whenever they read the same row.
//
// The child is built in tile order: pass 2 counts the representatives of each length class, a class
// scan gives every class its first slot, every representative takes a slot of its class, the longest
// transaction of each child tile sets the tile's length, and a scan over the tiles gives their offsets.
// Only then is the child allocated, at its exact size, and written.

#include "mine/bfs/device/tile/child_layout.cuh"
#include "mine/bfs/device/tile/emit.cuh"
#include "mine/bfs/device/tile/scan.cuh"
#include "mine/bfs/device/tile/search.cuh"
#include "mine/bfs/device/tile/tile_tx.cuh"
