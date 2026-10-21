#include "mine/miner.hpp"

#include "mine/widths.hpp"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ghuim {

// The searches of every built width pair (src/mine/dfs/dfs_miner.cu) and triple (src/mine/bfs/bfs_miner.cu).
#define GHUIM_DECLARE_DFS(key, util)                                                                                   \
    inline namespace GHUIM_WIDTH_NAME(key, util) {                                                                     \
    MiningResult mine_dfs(const Dataset& data, const MiningOptions& options);                                         \
    }
#define GHUIM_DECLARE_BFS(key, util, occ)                                                                              \
    inline namespace GHUIM_BFS_WIDTH_NAME(key, util, occ) {                                                            \
    MiningResult mine_bfs(const Dataset& data, const MiningOptions& options);                                         \
    }
GHUIM_FOR_EACH_BUILT_WIDTH(GHUIM_DECLARE_DFS)
GHUIM_FOR_EACH_BUILT_BFS_WIDTH(GHUIM_DECLARE_BFS)
#undef GHUIM_DECLARE_DFS
#undef GHUIM_DECLARE_BFS

namespace {

struct Widths
{
    uint32_t key_bits, util_bits, occ_bits; // occ_bits: BFS only, 0 for DFS
};

// Local item names run 0..items-1; every utility is at most the largest TWU (mine/widths.hpp).
uint32_t key_bits_for(uint64_t items)
{
    return items <= uint64_t{1} << 8 ? 8 : items <= uint64_t{1} << 16 ? 16 : 32;
}

uint32_t util_bits_for(uint64_t max_twu)
{
    return max_twu <= UINT16_MAX ? 16 : max_twu <= UINT32_MAX ? 32 : 64;
}

// --key-bits/--util-bits/--occ-bits when given (they must hold the dataset), else the narrowest built miner that
// holds it: fewest bytes per entry, then the narrower utilities, then (BFS) the narrower occurrences.
Widths choose_widths(const Dataset& data, const MiningOptions& options, bool bfs)
{
    const uint64_t items       = data.original_item.size();
    const uint32_t key_needed  = key_bits_for(items);
    const uint32_t util_needed = util_bits_for(data.max_twu);
    if (options.key_bits != 0 && options.key_bits < key_needed)
        throw std::runtime_error("--key-bits " + std::to_string(options.key_bits) + " is too narrow: " +
                                 std::to_string(items) + " items need " + std::to_string(key_needed) + "-bit keys");
    if (options.util_bits != 0 && options.util_bits < util_needed)
        throw std::runtime_error("--util-bits " + std::to_string(options.util_bits) + " is too narrow: the largest TWU " +
                                 std::to_string(data.max_twu) + " needs " + std::to_string(util_needed) + "-bit utilities");
    const uint32_t occ_needed = bfs ? occurrence_bits_for(data.transaction_count(), data.max_length) : 0;
    if (bfs && options.occ_bits != 0 && options.occ_bits < occ_needed)
        throw std::runtime_error("--occ-bits " + std::to_string(options.occ_bits) + " is too narrow: " +
                                 std::to_string(data.transaction_count()) + " transactions of up to " +
                                 std::to_string(data.max_length) + " items need " + std::to_string(occ_needed) +
                                 "-bit occurrences");
    Widths best{0, 0, 0};
    const auto consider = [&](uint32_t key, uint32_t util, uint32_t occ) {
        if (options.key_bits != 0 ? key != options.key_bits : key < key_needed)
            return;
        if (options.util_bits != 0 ? util != options.util_bits : util < util_needed)
            return;
        if (bfs && (options.occ_bits != 0 ? occ != options.occ_bits : occ < occ_needed))
            return;
        const uint32_t bits = key + util, best_bits = best.key_bits + best.util_bits;
        if (best.key_bits == 0 || bits < best_bits || (bits == best_bits && util < best.util_bits) ||
            (bits == best_bits && util == best.util_bits && occ < best.occ_bits))
            best = {key, util, occ};
    };
#define GHUIM_CONSIDER_DFS(key, util) consider(key, util, 0);
#define GHUIM_CONSIDER_BFS(key, util, occ) consider(key, util, occ);
    if (bfs) {
        GHUIM_FOR_EACH_BUILT_BFS_WIDTH(GHUIM_CONSIDER_BFS)
    }
    else {
        GHUIM_FOR_EACH_BUILT_WIDTH(GHUIM_CONSIDER_DFS)
    }
#undef GHUIM_CONSIDER_DFS
#undef GHUIM_CONSIDER_BFS
    if (best.key_bits == 0)
        throw std::runtime_error("no miner was built for " + std::to_string(options.key_bits ? options.key_bits : key_needed) +
                                 "-bit keys and " + std::to_string(options.util_bits ? options.util_bits : util_needed) +
                                 "-bit utilities" +
                                 (bfs ? " and " + std::to_string(options.occ_bits ? options.occ_bits : occ_needed) +
                                                "-bit occurrences"
                                      : std::string()) +
                                 " (see CMake options GHUIM_WIDTHS, GHUIM_OCC_WIDTHS)");
    return best;
}

} // namespace

void initialize_device()
{
    setenv("CUDA_VISIBLE_DEVICES", "0", /*overwrite=*/0); // read by the driver at its first initialization
    const cudaError_t error = cudaFree(nullptr); // the usual way to force context creation
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("create CUDA context: ") + cudaGetErrorString(error));
}

MiningResult mine(const Dataset& data, const MiningOptions& options)
{
    const bool   bfs    = options.search == Search::Bfs;
    const Widths widths = choose_widths(data, options, bfs);
    spdlog::info("Widths: {}-bit keys ({} items), {}-bit utilities (largest TWU {}){}", widths.key_bits,
                 data.original_item.size(), widths.util_bits, data.max_twu,
                 bfs ? fmt::format(", {}-bit occurrences ({} transactions, longest {})", widths.occ_bits,
                                   data.transaction_count(), data.max_length)
                     : std::string());
#define GHUIM_DISPATCH_DFS(key, util)                                                                                  \
    if (!bfs && widths.key_bits == key && widths.util_bits == util)                                                    \
        return GHUIM_WIDTH_NAME(key, util)::mine_dfs(data, options);
#define GHUIM_DISPATCH_BFS(key, util, occ)                                                                             \
    if (bfs && widths.key_bits == key && widths.util_bits == util && widths.occ_bits == occ)                           \
        return GHUIM_BFS_WIDTH_NAME(key, util, occ)::mine_bfs(data, options);
    GHUIM_FOR_EACH_BUILT_WIDTH(GHUIM_DISPATCH_DFS)
    GHUIM_FOR_EACH_BUILT_BFS_WIDTH(GHUIM_DISPATCH_BFS)
#undef GHUIM_DISPATCH_DFS
#undef GHUIM_DISPATCH_BFS
    throw std::logic_error("unreachable: chosen widths were not built");
}

} // namespace ghuim
