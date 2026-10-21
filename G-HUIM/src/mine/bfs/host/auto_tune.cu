#include "mine/bfs/host/auto_tune.hpp"

#include "cuda/buffer.cuh"
#include "mine/bfs/kernels.cuh"

#include <algorithm>
#include <bit>
#include <limits>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace ghuim::inline GHUIM_WIDTHS::bfs {

// Device profile of the current GPU.
AutoTune make_auto_tune(const MiningOptions& options, const Dataset& data)
{
    int device = 0;
    check_cuda(cudaGetDevice(&device), "get device");
    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device), "query device");
    const uint32_t sms = static_cast<uint32_t>(props.multiProcessorCount);

    AutoTune tune{};
    tune.tiled          = options.method == Method::Tile;
    tune.tail_items     = options.local_tail_items;
    tune.tail_trans     = options.local_tail_trans;
    tune.staged         = options.method == Method::Staged;
    tune.io_warps       = options.io_warps;
    tune.oversubscribe  = options.oversubscribe;
    tune.ring_bytes     = options.ring_kib * 1024;
    tune.resident_warps = sms * (props.maxThreadsPerMultiProcessor / 32);
    measure_occupancy(tune, sms);
    if (options.vert_items != 0 && data.max_twu > std::numeric_limits<uint32_t>::max())
        spdlog::warn("Vertical subtrees off: their columns are 32-bit and the largest TWU {} exceeds that", data.max_twu);
    else if (options.vert_items != 0) {
        // Every vertical warp has its own scratch: at most the resident warps, within an eighth of the budget.
        tune.vert_items = options.vert_items;
        tune.vert_trans = options.vert_trans;
        size_t free_bytes = 0, total_bytes = 0;
        check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "query device memory");
        const uint64_t budget   = options.memory_bytes == 0 ? free_bytes : std::min<uint64_t>(options.memory_bytes, free_bytes);
        const uint64_t per_warp = vertical_warp_words(vertical_slots(tune)) * sizeof(uint32_t);
        tune.vert_warps = static_cast<uint32_t>(std::min<uint64_t>(tune.resident_warps, budget / 8 / per_warp)) / 4 * 4;
        if (tune.vert_warps == 0)
            throw std::runtime_error("not enough device memory for vertical subtrees; lower --vert-trans");
        spdlog::info("Vertical subtrees: up to {} items and {} transactions, {} warps ({} MiB)", tune.vert_items,
                     tune.vert_trans, tune.vert_warps, (uint64_t{tune.vert_warps} * per_warp) >> 20);
    }
    // Static: every task runs in the block-per-task bin of --block-size threads (bin 0 is one warp).
    tune.fixed_bin = options.method == Method::Static ? static_cast<uint32_t>(std::countr_zero(options.block_size / 32))
                                                      : kNone;
    spdlog::info("Auto-tune ({}): {} ({} SMs), {} resident warps, max block {} threads, packed-warp block {} threads, "
                 "lane-group block {} threads",
                 options.method == Method::Tile ? "tiled lanes" : options.method == Method::Staged ? "staged lanes"
                                                                                                  : "lane per transaction",
                 props.name, sms, tune.resident_warps, tune.max_block_warps * 32, tune.packed_block_warps * 32,
                 tune.subwarp_block_warps * 32);
    return tune;
}

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
