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
    Adaptive // size-binned kernels: each task gets lanes, a warp, a block or several blocks to fit its work
};

inline const char* search_name(Search search)
{
    return search == Search::Bfs ? "bfs" : "dfs";
}

inline const char* method_name(Method method)
{
    return method == Method::Static ? "static" : "adaptive";
}

struct MiningOptions
{
    uint64_t    minimum_utility = 0;
    Search      search          = Search::Bfs;
    Method      method          = Method::Adaptive;
    uint32_t    block_size      = 256;          // static: threads per task
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
