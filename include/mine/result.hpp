#pragma once

#include <cstdint>
#include <vector>

namespace ghuim {

// One iteration (static/adaptive) or level (BFS) of the search.
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
};

} // namespace ghuim
