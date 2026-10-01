#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace ghuim {

// One level (BFS) or batch (DFS) of the search.
struct IterationStats
{
    uint64_t tasks        = 0; // tasks processed
    uint64_t patterns     = 0; // high-utility itemsets found
    uint64_t arena_bytes  = 0; // projected-database memory the iteration produced
    double   milliseconds = 0;
};

struct MiningResult
{
    uint64_t                    patterns   = 0;
    uint64_t                    tasks      = 0;
    uint64_t                    peak_arena = 0; // largest live projected-database memory
    double                      kernel_ms  = 0;
    double                      setup_ms   = 0; // building and uploading the root database
    std::vector<IterationStats> iterations;

    // Filled by the DFS search only.
    uint64_t                 compactions = 0; // arena compactions
    uint64_t                 deferrals   = 0; // tasks pushed back because their child did not fit
    std::array<uint64_t, 64> depth_tasks{};    // tasks evaluated, by itemset length (last slot: 63+)
    std::array<uint64_t, 64> depth_patterns{}; // high-utility itemsets, by itemset length
};

} // namespace ghuim
