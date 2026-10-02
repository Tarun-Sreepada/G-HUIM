// Device profile for the BFS miner: block sizes derived from kernel occupancy.

#include "cuda/buffer.cuh"
#include "mine/bfs/device/staging.cuh"
#include "mine/bfs/kernels.cuh"

#include <algorithm>
#include <spdlog/spdlog.h>

namespace ghuim::bfs {
namespace {

uint32_t occupancy(const void* kernel, uint32_t threads, uint32_t dynamic_bytes = 0)
{
    int count = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&count, kernel, static_cast<int>(threads), dynamic_bytes),
               "query occupancy");
    return static_cast<uint32_t>(count);
}

// Warp-sized kernels: the largest block whose resident warps are the best the kernel reaches.
uint32_t warp_kernel_block(const void* kernel)
{
    uint32_t best_warps = 0, block_warps = 1;
    for (uint32_t size = 0; size < kOptBlockSizes; ++size) {
        const uint32_t resident = occupancy(kernel, 32u << size) << size;
        if (resident != 0 && resident >= best_warps) {
            best_warps  = resident;
            block_warps = 1u << size;
        }
    }
    return block_warps;
}

} // namespace

void measure_occupancy(AutoTune& tune, uint32_t sms)
{
    // Kernels launched with the same block size: a size is usable only if all of them fit.
    const bool     warp_mode = tune.warp_mode != 0;
    const uint32_t dynamic   = tune.staged ? kStageBarrierBytes + tune.ring_bytes : 0;
    const uint32_t block_dynamic = dynamic + tune.table_bytes;
    if (tune.staged)
        prepare_staged_kernels(block_dynamic);
    const void* block_kernels[1 + kHugeKernels] = {tune.staged ? block_kernel_staged() : block_kernel(warp_mode)};
    const void* huge[kHugeKernels];
    if (tune.staged)
        huge_kernels_staged(huge);
    else
        huge_kernels(huge, warp_mode);
    std::copy(std::begin(huge), std::end(huge), block_kernels + 1);
    // Only the staged kernels are launched with dynamic shared memory: the block kernel and the
    // local/subtree/write/merge passes (huge kernel indices 1, 3, 5, 6).
    const bool staged_kernel[1 + kHugeKernels] = {true, false, true, false, true, false, true, true, false};
    auto block_occupancy = [&](uint32_t threads) {
        uint32_t blocks = occupancy(block_kernels[0], threads, block_dynamic);
        for (uint32_t k = 1; k < 1 + kHugeKernels; ++k)
            blocks = std::min(blocks, occupancy(block_kernels[k], threads, staged_kernel[k] ? dynamic : 0));
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
    tune.max_block_warps = 1u << best_size;
    tune.max_slices      = sms * std::max(1u, block_occupancy(32u << best_size));

    tune.packed_block_warps  = warp_kernel_block(warp_kernel(warp_mode));
    tune.subwarp_block_warps = warp_kernel_block(subwarp_kernel());
}

} // namespace ghuim::bfs
