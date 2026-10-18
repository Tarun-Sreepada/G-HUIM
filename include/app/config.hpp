#pragma once

#include <cstdint>
#include <string>

namespace ghuim {

enum class Search
{
    Bfs, // level-synchronous: every task of a level runs before any task of the next
    Dfs  // task stack: batches are popped from the top, so few projected databases are live
};

enum class Method
{
    Static,  // every task gets one block of --block-size threads
    Adaptive, // size-binned kernels: each task gets lanes, a warp, a block or several blocks to fit its work
    Staged,   // BFS only: adaptive kernels with IO warps staging transactions in shared memory
    Tile      // BFS only: adaptive scheduling over databases stored in warp tiles (entries column-major)
};

inline const char* search_name(Search search)
{
    return search == Search::Bfs ? "bfs" : "dfs";
}

inline const char* method_name(Method method)
{
    switch (method) {
    case Method::Static: return "static";
    case Method::Staged: return "staged";
    case Method::Tile: return "tile";
    default: return "adaptive";
    }
}

struct MiningOptions
{
    uint64_t    minimum_utility = 0;
    Search      search          = Search::Bfs;
    Method      method          = Method::Adaptive;
    uint32_t    block_size      = 256;          // static: threads per task
    uint32_t    io_warps        = 1;            // staged: IO warps per block
    uint32_t    ring_kib        = 64;           // staged: shared memory for the tile ring
    uint32_t    oversubscribe   = 12;           // BFS: tasks are sized for this many times the resident warps
    bool        bin_times       = false;        // BFS: serialize bins and log the time of each
    uint32_t    index_min_trans = 4096;         // BFS lane kernels: index databases at least this large; 0 = off
    uint32_t    index_sparsity  = 32;           // ... and at most 1/this filled (0 = any density)
    uint32_t    local_tail_items = 8;           // BFS count-only: finish bounded small subtrees locally; 0 = off
    uint32_t    local_tail_trans = 32;          // maximum supporting transactions for local subtree processing
    int32_t     refine_rounds   = 1;            // bound refinement rounds per task: -1 = until no item drops, 0 = off
    uint64_t    memory_bytes    = 0;            // projected-database arena; 0 = most of the free device memory
    uint64_t    stack_capacity  = 128ull << 20; // most tasks one level may hold (BFS)
    std::string output;                         // itemset file; empty = only count the itemsets (BFS)
};

struct Config
{
    std::string   input;
    char          separator = ' ';
    MiningOptions mining;
    std::string   log_level = "info";
    std::string   log_file;
};

// Parses a positive count/size with an optional binary K/M/G (or KiB/MiB/GiB) suffix.
uint64_t parse_size(const std::string& text);

} // namespace ghuim
