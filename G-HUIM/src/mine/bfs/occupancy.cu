// Device profile for the BFS miner: block sizes derived from kernel occupancy.

#include "cuda/buffer.cuh"
#include "mine/bfs/device/staging.cuh"
#include "mine/bfs/kernels.cuh"

#include <algorithm>
#include <spdlog/spdlog.h>

namespace ghuim::inline GHUIM_WIDTHS::bfs {
namespace {

uint32_t occupancy(const void* kernel, uint32_t threads, uint32_t dynamic_bytes = 0)
{
    int count = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&count, kernel, static_cast<int>(threads), dynamic_bytes),
               "query occupancy");
    return static_cast<uint32_t>(count);
}

// Warp-sized kernels: the smallest block whose resident warps are the best the kernel reaches. Their
// tasks are independent, so smaller blocks only finish more evenly (1024-thread blocks were 25% slower
// on BMS than 128-thread ones at the same occupancy).
uint32_t warp_kernel_block(const void* kernel)
{
    uint32_t best_warps = 0, block_warps = 1;
    for (uint32_t size = 0; size < kOptBlockSizes; ++size) {
        const uint32_t resident = occupancy(kernel, 32u << size) << size;
        if (resident != 0 && resident > best_warps) {
            best_warps  = resident;
            block_warps = 1u << size;
        }
    }
    return block_warps;
}

// Block sizes from the occupancy of the kernels the method runs.
void measure_blocks(AutoTune& tune, uint32_t sms)
{
    const uint32_t block_dynamic = tune.staged ? kStageBarrierBytes + tune.ring_bytes : 0;
    const void*    block_kernels[1 + kHugeKernels + kTileHugeKernels];
    const void*    huge[kHugeKernels];
    uint32_t       kernel_count = 1 + kHugeKernels;
    if (tune.staged) {
        block_kernels[0] = block_kernel_staged();
        huge_kernels_staged(huge);
    }
    else {
        block_kernels[0] = block_kernel(tune.tiled != 0);
        huge_kernels(huge, tune.tiled != 0);
    }
    std::copy(std::begin(huge), std::end(huge), block_kernels + 1);
    if (tune.tiled) {
        const void* extra[kTileHugeKernels];
        tile_huge_kernels(extra);
        std::copy(std::begin(extra), std::end(extra), block_kernels + kernel_count);
        kernel_count += kTileHugeKernels;
    }
    // Only the staged kernels carry dynamic shared memory: the block kernel and the local/subtree/write/merge
    // passes.
    const bool dynamic_kernel[1 + kHugeKernels + kTileHugeKernels] = {true, false, true, false, true, false, true, true, false, false, true};
    auto       block_occupancy = [&](uint32_t threads) {
        uint32_t blocks = ~0u;
        for (uint32_t k = 0; k < kernel_count; ++k)
            blocks = std::min(blocks, occupancy(block_kernels[k], threads, dynamic_kernel[k] ? block_dynamic : 0));
        return blocks;
    };

    // The largest block size whose resident warps per SM equal the best any size achieves.
    uint32_t best = 0, best_size = 1;
    for (uint32_t size = 1; size < kOptBlockSizes; ++size) {
        const uint32_t resident = block_occupancy(32u << size) << size;
        spdlog::debug("Occupancy: {} threads -> {} blocks per SM, {} warps", 32u << size, block_occupancy(32u << size), resident);
        if (resident != 0 && resident >= best) {
            best      = resident;
            best_size = size;
        }
    }
    tune.max_block_warps    = 1u << best_size;
    tune.max_slices         = sms * std::max(1u, block_occupancy(32u << best_size));
    tune.packed_block_warps = warp_kernel_block(warp_kernel(tune.tiled != 0));
}

} // namespace

void measure_occupancy(AutoTune& tune, uint32_t sms)
{
    if (tune.staged)
        prepare_staged_kernels(kStageBarrierBytes + tune.ring_bytes);
    measure_blocks(tune, sms);
    // One block size serves both sub-warp kernels, so it must fit the local-subtree one too.
    tune.subwarp_block_warps = warp_kernel_block(subwarp_kernel(tune.tiled != 0));
    if (tune.tail_items != 0)
        tune.subwarp_block_warps = std::min(tune.subwarp_block_warps, warp_kernel_block(subwarp_kernel(false, true)));
}

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
