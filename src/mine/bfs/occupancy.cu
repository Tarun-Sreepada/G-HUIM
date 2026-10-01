// Device profile for the BFS miner: block sizes derived from kernel occupancy.

#include "cuda/buffer.cuh"
#include "mine/bfs/kernels.cuh"

#include <algorithm>

namespace ghuim::bfs {
namespace {

uint32_t occupancy(const void* kernel, uint32_t threads)
{
    int count = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&count, kernel, static_cast<int>(threads), 0),
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
    const void* block_kernels[1 + kHugeKernels] = {block_kernel()};
    const void* huge[kHugeKernels];
    huge_kernels(huge);
    std::copy(std::begin(huge), std::end(huge), block_kernels + 1);
    auto block_occupancy = [&](uint32_t threads) {
        uint32_t blocks = ~0u;
        for (const void* kernel : block_kernels)
            blocks = std::min(blocks, occupancy(kernel, threads));
        return blocks;
    };

    // The largest block size whose resident warps per SM equal the best any size achieves.
    uint32_t best = 0, best_size = 1;
    for (uint32_t size = 1; size < kOptBlockSizes; ++size) {
        const uint32_t resident = block_occupancy(32u << size) << size;
        if (resident != 0 && resident >= best) {
            best      = resident;
            best_size = size;
        }
    }
    tune.max_block_warps = 1u << best_size;
    tune.max_slices      = sms * std::max(1u, block_occupancy(32u << best_size));

    tune.packed_block_warps  = warp_kernel_block(warp_kernel());
    tune.subwarp_block_warps = warp_kernel_block(subwarp_kernel());
}

} // namespace ghuim::bfs
