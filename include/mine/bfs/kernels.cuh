#pragma once
// Launchers and kernel handles of the BFS miner. Each launcher processes tasks order[0..count) of
// the chunk on `stream`.

#include "mine/bfs/device/task_view.cuh"
#include "mine/bfs/placement.cuh"

#include <cuda_runtime.h>

namespace ghuim::bfs {

// Bin 0: one task per warp, packed into blocks of tune.packed_block_warps warps.
void launch_warp_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, uint32_t shape,
                       const AutoTune& tune, cudaStream_t stream);
// Bins 1..kOptHugeBin-1: one block of (32 << bin) threads per task.
void launch_block_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, uint32_t bin, uint32_t shape,
                        const AutoTune& tune, cudaStream_t stream);
// Sub-warp bins: `group` lanes (a power of two <= 32) per task.
void launch_subwarp_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, uint32_t group,
                          const AutoTune& tune, cudaStream_t stream);
// Huge bin: each task is sliced over up to max_slices blocks, one launch per pass.
void launch_huge_tasks(const BatchView& batch, const uint32_t* order, uint32_t count, uint32_t max_slices, uint32_t shape,
                       const AutoTune& tune, cudaStream_t stream);

// Kernel handles for occupancy queries from another translation unit.
// `group` is a ShapeTune::group value.
const void* block_kernel(uint32_t group);
const void* warp_kernel(uint32_t group);
const void* subwarp_kernel();
const void* block_kernel_staged(bool warp_mode);
// The pass kernels of a huge task, in launch order.
void huge_kernels(const void* (&kernels)[kHugeKernels], uint32_t group);
void huge_kernels_staged(const void* (&kernels)[kHugeKernels], bool warp_mode);
void prepare_staged_block_kernel(uint32_t dynamic_bytes, bool warp_mode);
// Raises the dynamic shared-memory limit of the staged kernels; call before querying occupancy or launching.
void prepare_staged_kernels(uint32_t dynamic_bytes, bool warp_mode);

// Fills the block-size fields of `tune` for a device with `sms` multiprocessors from kernel occupancy.
void measure_occupancy(AutoTune& tune, uint32_t sms);

// Scheduling kernels of a chunk [begin, end) of the level.
void launch_scratch_sizes(const Task* tasks, uint32_t count, uint64_t* sizes, cudaStream_t stream);
void launch_bin_chunk(Control* control, const Task* tasks, uint32_t begin, uint32_t end, uint32_t* order,
                      const AutoTune& tune, cudaStream_t stream);

} // namespace ghuim::bfs
