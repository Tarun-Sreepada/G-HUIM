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

// Block sizes of one shape from the occupancy of the kernels its mode runs.
void measure_shape(ShapeTune& st, const AutoTune& tune, uint32_t sms)
{
    const bool     group_mode    = st.group != kLaneMode;
    const bool     warp_mode     = tune.warp_mode != 0;
    const uint32_t block_dynamic = (tune.staged ? kStageBarrierBytes + tune.ring_bytes : 0) + tune.table_bytes;
    const void*    block_kernels[1 + kHugeKernels + kTileHugeKernels];
    const void*    huge[kHugeKernels];
    uint32_t       kernel_count = 1 + kHugeKernels;
    if (tune.staged) {
        block_kernels[0] = block_kernel_staged(warp_mode);
        huge_kernels_staged(huge, warp_mode);
    }
    else {
        block_kernels[0] = block_kernel(st.group, tune.tiled != 0);
        huge_kernels(huge, st.group, tune.tiled != 0);
    }
    std::copy(std::begin(huge), std::end(huge), block_kernels + 1);
    if (tune.tiled) {
        const void* extra[kTileHugeKernels];
        tile_huge_kernels(extra);
        std::copy(std::begin(extra), std::end(extra), block_kernels + kernel_count);
        kernel_count += kTileHugeKernels;
    }
    // Only the kernels launched with dynamic shared memory carry it: the staged ones, and the lane
    // kernels when they keep their item tables there (the block kernel and the local/subtree/write/merge
    // passes).
    const bool dynamic_kernel[1 + kHugeKernels + kTileHugeKernels] = {true, false, true, false, true, false, true, true, false, false, true};
    const bool carries_dynamic                  = tune.staged || !group_mode;
    auto       block_occupancy                  = [&](uint32_t threads) {
        uint32_t blocks = ~0u;
        for (uint32_t k = 0; k < kernel_count; ++k)
            blocks = std::min(blocks, occupancy(block_kernels[k], threads, carries_dynamic && dynamic_kernel[k] ? block_dynamic : 0));
        return blocks;
    };

    // The largest block size whose resident warps per SM equal the best any size achieves.
    uint32_t best = 0, best_size = 1;
    for (uint32_t size = 1; size < kOptBlockSizes; ++size) {
        const uint32_t resident = block_occupancy(32u << size) << size;
        spdlog::debug("Occupancy (group {}): {} threads -> {} blocks per SM, {} warps", st.group, 32u << size,
                      block_occupancy(32u << size), resident);
        if (resident != 0 && resident >= best) {
            best      = resident;
            best_size = size;
        }
    }
    st.max_block_warps    = 1u << best_size;
    st.max_slices         = sms * std::max(1u, block_occupancy(32u << best_size));
    st.packed_block_warps = warp_kernel_block(warp_kernel(st.group, tune.tiled != 0));
}

} // namespace

void measure_occupancy(AutoTune& tune, uint32_t sms)
{
    const uint32_t dynamic = tune.staged ? kStageBarrierBytes + tune.ring_bytes : 0;
    if (tune.staged)
        prepare_staged_kernels(dynamic + tune.table_bytes, tune.warp_mode != 0);
    // Shapes that run the same mode share one profile and, when adjacent, one set of cells.
    for (uint32_t s = 0; s < kShapes; ++s) {
        ShapeTune& st = tune.shape[s];
        uint32_t   same = s;
        for (uint32_t r = 0; r < s; ++r)
            if (tune.shape[r].group == st.group)
                same = r;
        if (same != s) {
            st.max_block_warps    = tune.shape[same].max_block_warps;
            st.max_slices         = tune.shape[same].max_slices;
            st.packed_block_warps = tune.shape[same].packed_block_warps;
        }
        else
            measure_shape(st, tune, sms);
        st.rep = s > 0 && tune.shape[s - 1].group == st.group ? tune.shape[s - 1].rep : s;
    }
    tune.subwarp_block_warps = warp_kernel_block(subwarp_kernel(tune.tiled != 0));
}

} // namespace ghuim::bfs
