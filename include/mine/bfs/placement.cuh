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
constexpr uint32_t kWarpMinTrans    = 2; // lane-group tasks: fewest transactions worth giving a warp

// Task shape: the length class of the task's longest transaction. Shape s is served by lane groups of
// 4 << s lanes (4, 8, 16, 32). A scheduling cell is a size bin and a shape; the matrix of cells is
// counted as tasks are created and every non-empty cell gets its own kernel.
constexpr uint32_t kShapes = 4;
constexpr uint32_t kCells  = kBinCount * kShapes;

static_assert(kCells <= kMaxCells);

__host__ __device__ inline uint32_t shape_of(uint32_t max_len)
{
    return max_len <= 4 ? 0u : max_len <= 8 ? 1u : max_len <= 16 ? 2u : 3u;
}


// Per-shape kernel profile (occupancy depends on the kernels the shape runs).
struct ShapeTune
{
    uint32_t group;              // kLaneMode, kGroupMode or a fixed group size
    uint32_t max_block_warps;    // largest block (in warps) that still reaches full occupancy
    uint32_t packed_block_warps; // block size (in warps) for the one-task-per-warp kernel
    uint32_t max_slices;         // blocks one huge task may use
    uint32_t rep;                // first shape of the run of neighbours with the same group: they share cells
};

// Device profile measured at startup from kernel occupancy.
struct AutoTune
{
    uint32_t resident_warps;      // warps the whole GPU keeps resident
    uint32_t oversubscribe;       // tasks are sized for this many times the resident warps (more, smaller blocks balance better)
    ShapeTune shape[kShapes];
    uint32_t subwarp_block_warps; // block size (in warps) for the lane-group-per-task kernel
    uint32_t fixed_bin;           // static method: the one bin every task runs in; kNone = adaptive
    uint32_t warp_mode;           // staged method: lane groups own one transaction at a time (StagedWarpTx)
    uint32_t staged;              // staged method: IO warps stage transactions in shared memory (StagedTx)
    uint32_t io_warps;            // staged method: IO warps per block
    uint32_t ring_bytes;          // staged method: shared memory for the tile ring
    uint32_t table_bytes;         // shared memory for the per-item tables of block-sized tasks (0 = global tables)
    uint32_t min_entries_per_warp; // size tasks by entries, not transactions: fewest entries worth a warp (0 = by transactions)
};

// Work of a task for sizing: transactions, or entries when the tune sizes by entries.
__host__ __device__ inline uint64_t task_work(const Task& task, const AutoTune& tune)
{
    return tune.min_entries_per_warp != 0 ? task.n_entries : task.n_trans;
}

struct Placement
{
    uint32_t bin;
    uint32_t slices; // blocks sharing the task (huge bin only)
    uint32_t shape;  // representative shape of the task's cell
    __host__ __device__ uint32_t cell() const { return bin * kShapes + shape; }
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
    const uint32_t   shape = tune.shape[shape_of(task.max_len)].rep;
    const ShapeTune& st    = tune.shape[shape];
    if (tune.fixed_bin != kNone)
        return {tune.fixed_bin, 1, 0};
    // Lane-group shapes: fewer than 32 transactions run on one warp (packed into small blocks); larger
    // tasks get one warp per few transactions, up to full blocks and then slices.
    const bool group = st.group != kLaneMode;
    if (group && task.n_trans < kWarpSize)
        return {0, 1, shape};
    if (!group && task.n_trans <= kWarpSize)
        return {kOptSubBin + size_class(task.n_trans), 1, 0};
    const uint64_t floor    = tune.min_entries_per_warp != 0 ? tune.min_entries_per_warp : group ? kWarpMinTrans : kMinTransPerWarp;
    const uint64_t share    = batch_work / (uint64_t{tune.resident_warps} * tune.oversubscribe);
    const uint64_t per_warp = share > floor ? share : floor;
    const uint32_t warps    = ceil_div(task_work(task, tune), per_warp);
    if (warps > st.max_block_warps) {
        const uint32_t slices = ceil_div(warps, st.max_block_warps);
        return {kOptHugeBin, slices < st.max_slices ? slices : st.max_slices, shape};
    }
    return {size_class(warps), 1, shape};
}

} // namespace ghuim::bfs
