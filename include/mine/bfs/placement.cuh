#pragma once
// Where each task of a chunk runs: bins by task size, and the device profile that sizes them.

#include "mine/bfs/layout.cuh"

#include <cuda/std/bit>

namespace ghuim::bfs {

// Task placement. Bins 0..kOptBins-1 are size classes: class 0 packs one task per warp into
// blocks, class s > 0 gives each task a block of (32 << s) threads. kOptHugeBin splits a task over
// several blocks. Tasks with at most a warp's worth of transactions run on a group of
// bit_ceil(n_trans) lanes: bins kOptSubBin + log2(group size).
constexpr uint32_t kOptBlockSizes   = 6;
constexpr uint32_t kOptBins         = kOptBlockSizes;
constexpr uint32_t kOptHugeBin      = kOptBins;
constexpr uint32_t kOptSubBin       = kOptBins + 1;
constexpr uint32_t kOptSubBins      = 6;
constexpr uint32_t kBinCount        = kOptSubBin + kOptSubBins;
constexpr uint32_t kWarpSize        = 32;
constexpr uint32_t kMinTransPerWarp = 8; // below this, extra warps only add overhead
constexpr uint32_t kHugeKernels     = 8;
constexpr uint32_t kWarpMinTrans    = 2; // warp method: fewest transactions worth giving a warp

static_assert(kBinCount <= kMaxBins);

// Device profile measured at startup from kernel occupancy.
struct AutoTune
{
    uint32_t resident_warps;      // warps the whole GPU keeps resident
    uint32_t max_block_warps;     // largest block (in warps) that still reaches full occupancy
    uint32_t packed_block_warps;  // block size (in warps) for the one-task-per-warp kernel
    uint32_t subwarp_block_warps; // block size (in warps) for the lane-group-per-task kernel
    uint32_t max_slices;          // blocks one huge task may use
    uint32_t fixed_bin;           // static method: the one bin every task runs in; kNone = adaptive
    uint32_t warp_mode;           // warp method: lane groups own one transaction at a time (WarpTx)
};

struct Placement
{
    uint32_t bin;
    uint32_t slices; // blocks sharing the task (huge bin only)
};

__host__ __device__ inline uint32_t ceil_div(uint64_t a, uint64_t b)
{
    return static_cast<uint32_t>((a + b - 1) / b);
}

__host__ __device__ inline uint32_t size_class(uint32_t warps)
{
    return cuda::std::countr_zero(cuda::std::bit_ceil(warps > 0 ? warps : 1u));
}

// Where a task runs: the chunk's total work is spread over the resident warps, and each task gets
// enough warps for its share, rounded up to a power of two.
__host__ __device__ inline Placement place_task(const Task& task, const AutoTune& tune, uint64_t batch_work)
{
    if (tune.fixed_bin != kNone)
        return {tune.fixed_bin, 1};
    // Warp method: fewer than 32 transactions run on one warp (packed into small blocks); larger
    // tasks get one warp per few transactions, up to full blocks and then slices.
    if (tune.warp_mode && task.n_trans < kWarpSize)
        return {0, 1};
    if (!tune.warp_mode && task.n_trans <= kWarpSize)
        return {kOptSubBin + size_class(task.n_trans), 1};
    const uint64_t floor    = tune.warp_mode ? kWarpMinTrans : kMinTransPerWarp;
    const uint64_t share    = batch_work / tune.resident_warps;
    const uint64_t per_warp = share > floor ? share : floor;
    const uint32_t warps    = ceil_div(task.n_trans, per_warp);
    if (warps > tune.max_block_warps) {
        const uint32_t slices = ceil_div(warps, tune.max_block_warps);
        return {kOptHugeBin, slices < tune.max_slices ? slices : tune.max_slices};
    }
    return {size_class(warps), 1};
}

} // namespace ghuim::bfs
