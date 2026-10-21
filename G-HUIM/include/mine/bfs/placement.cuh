#pragma once
// Where each task of a chunk runs: bins by task size, and the device profile that sizes them.

#include "mine/bfs/layout.cuh"

#include <cuda/std/bit>

namespace ghuim::inline GHUIM_WIDTHS::bfs {

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

// A scheduling cell is a size bin and a kind: 0 projects, 1 is a local subtree (sub-warp bins), 2 a vertical
// subtree (bin 0). The cells are counted as tasks are created and every non-empty cell gets its own kernel.
constexpr uint32_t kCellKinds = 3;
constexpr uint32_t kTailKind  = 1;
constexpr uint32_t kVertKind  = 2;
constexpr uint32_t kCells     = kBinCount * kCellKinds;

static_assert(kCells <= kMaxCells);

// Device profile measured at startup from kernel occupancy.
struct AutoTune
{
    uint32_t resident_warps;      // warps the whole GPU keeps resident
    uint32_t oversubscribe;       // tasks are sized for this many times the resident warps (more, smaller blocks balance better)
    uint32_t max_block_warps;     // largest block (in warps) that still reaches full occupancy
    uint32_t packed_block_warps;  // block size (in warps) for the one-task-per-warp kernel
    uint32_t max_slices;          // blocks one huge task may use
    uint32_t subwarp_block_warps; // block size (in warps) for the lane-group-per-task kernel
    uint32_t fixed_bin;           // static method: the one bin every task runs in; kNone = adaptive
    uint32_t tiled;               // tile method: databases are stored in tiles (layout.cuh), lane-per-transaction passes
    uint32_t staged;              // staged method: IO warps stage transactions in shared memory (StagedTx)
    uint32_t io_warps;            // staged method: IO warps per block
    uint32_t ring_bytes;          // staged method: shared memory for the tile ring
    uint32_t tail_items;          // local subtrees (local_tail.cuh): most remaining items, 0 = off
    uint32_t tail_trans;          // local subtrees: most supporting transactions
    uint32_t vert_items;          // vertical subtrees (vertical_kernels.cu): most remaining items, 0 = off
    uint32_t vert_trans;          // vertical subtrees: most transactions
    uint32_t vert_warps;          // warps the vertical kernel runs (each has its own scratch)
};

// Vertical subtrees: remaining items are a 32-bit set, and a lane's slots one 32-bit word.
constexpr uint32_t kMaxVertItems = 32;
constexpr uint32_t kMaxVertTrans = 1024;

// Local subtrees: most remaining items, and the longest suffix the single-transaction subset counter
// enumerates (2^12 subsets on one lane; longer suffixes take the pruned subtree walk).
constexpr uint32_t kMaxLocalTailItems = 31;
constexpr uint32_t kMaxSubsetItems    = 12;

// Whether finish_local_tail completes the task's whole subtree. Such tasks get their own cell, so the larger
// local-subtree kernel runs only on them.
__host__ __device__ inline bool local_tail_task(const Task& task, const AutoTune& tune)
{
    if (tune.tail_items == 0 || task.n_trans > tune.tail_trans || task.n_trans > kWarpSize)
        return false;
    const uint32_t subset = tune.tail_items < kMaxSubsetItems ? tune.tail_items : kMaxSubsetItems;
    return (task.n_trans == 1 && task.max_len <= subset + 1) || task.n_items - task.item - 1 <= tune.tail_items;
}

// Whether a task's subtree is mined vertically: few enough remaining items and transactions.
__host__ __device__ inline bool vertical_task(const Task& task, const AutoTune& tune)
{
    return tune.vert_items != 0 && task.n_trans <= tune.vert_trans && task.n_items - task.item - 1 <= tune.vert_items;
}

struct Placement
{
    uint32_t bin;
    uint32_t slices; // blocks sharing the task (huge bin only)
    uint32_t tail;   // cell kind (kTailKind, kVertKind; 0 = projected)
    __host__ __device__ uint32_t cell() const { return bin * kCellKinds + tail; }
};

__host__ __device__ inline uint32_t ceil_div(uint64_t a, uint64_t b)
{
    return static_cast<uint32_t>((a + b - 1) / b);
}

__host__ __device__ inline uint32_t size_class(uint32_t warps)
{
    return cuda::std::countr_zero(cuda::std::bit_ceil(warps > 0 ? warps : 1u));
}

// Where a task runs: the chunk's total work (transactions) is spread over the resident warps, and each task
// gets enough warps for its share, rounded up to a power of two. At most a warp's worth of transactions run
// on a lane group.
__host__ __device__ inline Placement place_task(const Task& task, const AutoTune& tune, uint64_t batch_work)
{
    if (tune.fixed_bin != kNone)
        return {tune.fixed_bin, 1, 0};
    if (local_tail_task(task, tune))
        return {kOptSubBin + size_class(task.n_trans), 1, kTailKind};
    if (vertical_task(task, tune))
        return {0, 1, kVertKind};
    if (task.n_trans <= kWarpSize)
        return {kOptSubBin + size_class(task.n_trans), 1, 0};
    const uint64_t share    = batch_work / (uint64_t{tune.resident_warps} * tune.oversubscribe);
    const uint64_t per_warp = share > kMinTransPerWarp ? share : kMinTransPerWarp;
    const uint32_t warps    = ceil_div(task.n_trans, per_warp);
    if (warps > tune.max_block_warps) {
        const uint32_t slices = ceil_div(warps, tune.max_block_warps);
        return {kOptHugeBin, slices < tune.max_slices ? slices : tune.max_slices, 0};
    }
    return {size_class(warps), 1, 0};
}

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
