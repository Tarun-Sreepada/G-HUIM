#include "mine/bfs/host/level_miner.hpp"

#include "cuda/nvtx.hpp"
#include "mine/host/transfer.hpp"
#include "mine/bfs/root_image.hpp"
#include "mine/primitives.cuh"

#include <chrono>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace ghuim::inline GHUIM_WIDTHS::bfs {

void LevelMiner::allocate()
{
    GHUIM_RANGE("bfs: build + upload root database");
    auto lap = std::chrono::steady_clock::now();
    auto stage = [&](const char* name) {
        spdlog::debug("  {:<28} {:9.1f} ms", name, ms_since(lap));
        lap = std::chrono::steady_clock::now();
    };
    RootDatabase root = upload_root(data_, tune_.tiled != 0, options_.index_min_trans, options_.index_sparsity);
    stage("build + upload root");
    root_bytes_   = root.header.bytes;
    root_trans_   = root.header.n_trans;
    root_items_   = root.header.n_items;
    root_max_len_ = root.header.max_len;
    root_entries_ = root.header.n_entries;
    if (root_bytes_ / kArenaUnit > 0xffffffffull)
        throw std::runtime_error("the input database is too large for 32-bit arena offsets");
    root_            = std::move(root.buffer);
    root_occ_counts_ = std::move(root.occ_counts);
    if (root.header.indexed)
        spdlog::info("Occurrence indexes: databases with at least {} transactions{}", options_.index_min_trans,
                     options_.index_sparsity ? fmt::format(" and at most 1/{} filled", options_.index_sparsity) : "");

    const uint64_t capacity = budget_.capacity;
    level_a_                = DeviceBuffer<Task>(capacity);
    level_b_                = DeviceBuffer<Task>(capacity);
    nodes_                  = DeviceBuffer<Node>(capacity);
    prefix_                 = DeviceBuffer<uint64_t>(capacity + 1);
    scan_sums_              = DeviceBuffer<uint64_t>(scan_block_count(capacity + 1));
    scan_total_             = DeviceBuffer<uint64_t>(1);
    order_                  = DeviceBuffer<uint32_t>(capacity);
    scratch_                = DeviceBuffer<char>(budget_.scratch_capacity);
    regions_                = {ArenaBuffer(budget_.region_bytes, options_.compress_arena),
                               ArenaBuffer(budget_.region_bytes, options_.compress_arena)};
    if (options_.compress_arena)
        spdlog::info("Arena compression: {}", regions_[0].compressed() && regions_[1].compressed() ? "on" : "unavailable");
    control_                = DeviceBuffer<Control>(1);
    vertical_ = DeviceBuffer<uint32_t>(uint64_t{tune_.vert_warps} * vertical_warp_words(vertical_slots(tune_)));
    chunk_plan_             = DeviceBuffer<ChunkPlan>(1);
    stage("allocate buffers");
}

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
