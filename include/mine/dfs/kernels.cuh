#pragma once

#include "mine/dfs/layout.cuh"

#include <cuda/std/bit>
#include <cuda_runtime.h>

namespace ghuim::dfs {

// adaptive-optimized bins are size classes: class 0 packs one task per warp into blocks, class s > 0
// gives each task a block of (32 << s) threads, and the last bin splits a task over several blocks.
constexpr uint32_t kOptBlockSizes = 6;
constexpr uint32_t kOptBins       = kOptBlockSizes;
constexpr uint32_t kOptHugeBin    = kOptBins;
// Tasks with at most a warp's worth of transactions run on a group of bit_ceil(n_trans) lanes: bins
// kOptSubBin + log2(group size).
constexpr uint32_t kOptSubBin     = kOptBins + 1;
constexpr uint32_t kOptSubBins    = 6;
constexpr uint32_t kWarpSize      = 32;
// A warp needs a few transactions to hide memory latency; below this, extra warps only add overhead.
constexpr uint32_t kMinTransPerWarp = 8;

// Device profile measured at startup (see make_auto_tune in miner.cu).
struct AutoTune
{
    uint32_t resident_warps;     // warps the whole GPU keeps resident
    uint32_t max_block_warps;    // largest block (in warps) that still reaches full occupancy
    uint32_t packed_block_warps;  // block size (in warps) for the one-task-per-warp kernel
    uint32_t subwarp_block_warps; // block size (in warps) for the lane-group-per-task kernel
    uint32_t max_slices;         // blocks one huge task may use: all resident blocks of that size
};

// Device pointers shared by every kernel processing one batch of tasks.
struct BatchView
{
    char*           arena;
    uint64_t        arena_capacity;
    Control*        control;        // arena bump pointer, pattern count, task stack top
    const Task*     tasks;          // the batch
    const uint64_t* scratch_offset; // per batch task, into `scratch`
    char*           scratch;
    Task*           stack;          // children and deferred tasks are pushed here
    uint32_t        stack_capacity;
    uint64_t        minimum_utility;
};

// Each launcher processes tasks order[0..count) of the batch on `stream`.
void launch_block_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, uint32_t block_size,
                        cudaStream_t stream);

// adaptive-optimized: every launch parameter is derived from the device and the batch.
//
// Per batch, the total work W (transactions) is spread over the resident warps of the GPU: a task
// gets enough warps for its share of W, rounded up to a power of two, and a task needing more warps
// than one block holds is split into slices. Tasks with at most a warp's worth of transactions get a
// group of lanes (one per transaction) sharing a warp with other such tasks. All per-task state lives in global scratch.
struct Placement
{
    uint32_t bin;
    uint32_t slices; // blocks sharing the task (huge bin only)
};

GHUIM_HD inline uint32_t ceil_div(uint64_t a, uint64_t b)
{
    return static_cast<uint32_t>((a + b - 1) / b);
}

// 0: one warp, packed with other tasks; s > 0: a block of (32 << s) threads.
GHUIM_HD inline uint32_t size_class(uint32_t warps)
{
    return cuda::std::countr_zero(cuda::std::bit_ceil(warps > 0 ? warps : 1u));
}

// Where and how a task runs, given the batch's total work (sum of transactions).
GHUIM_HD inline Placement place_task(const Task& task, const AutoTune& tune, uint64_t batch_work)
{
    if (task.n_trans <= kWarpSize) // fewer transactions than lanes: a lane group per task
        return {kOptSubBin + size_class(task.n_trans), 1};
    const uint64_t share    = batch_work / tune.resident_warps;
    const uint64_t per_warp = share > kMinTransPerWarp ? share : kMinTransPerWarp;
    const uint32_t warps    = ceil_div(task.n_trans, per_warp);
    if (warps > tune.max_block_warps) {
        const uint32_t slices = ceil_div(warps, tune.max_block_warps);
        return {kOptHugeBin, slices < tune.max_slices ? slices : tune.max_slices};
    }
    return {size_class(warps), 1};
}

// Launches the packed-warp and block-per-task kernels for tasks order[0..count) of size class `bin`.
void launch_optimized_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, uint32_t bin,
                            const AutoTune& tune, cudaStream_t stream);
// Subwarp bins: `group` lanes (a power of two <= 32) per task.
void launch_subwarp_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, uint32_t group,
                          const AutoTune& tune, cudaStream_t stream);
// Huge bin: each task is split into its own number of slices (at most max_slices blocks).
void launch_huge_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, uint32_t max_slices,
                       const AutoTune& tune, cudaStream_t stream);
// Fills `tune` for a device with `sms` multiprocessors from kernel occupancy; resident_warps must be set.
void measure_optimized_occupancy(AutoTune& tune, uint32_t sms);

} // namespace ghuim::dfs
