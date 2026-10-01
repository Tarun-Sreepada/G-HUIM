#pragma once

#include <cstdint>
#include <string>

namespace ghuim {

enum class Method
{
    Static,       // DFS: every task gets one block of a fixed size
    Adaptive,     // DFS: block size follows the task sizes; tiny tasks run on single threads
    BfsOptimized  // level-synchronous: every task of a level runs before any task of the next
};

inline const char* method_name(Method method)
{
    switch (method) {
    case Method::Static: return "static";
    case Method::Adaptive: return "adaptive";
    default: return "bfs-optimized";
    }
}

struct MiningOptions
{
    uint64_t minimum_utility = 0;
    Method   method          = Method::Adaptive;
    uint32_t block_size      = 1024; // DFS static: threads per block-sized task
    uint64_t memory_bytes    = 0;    // projected-database arena; 0 = most of the free device memory
    uint64_t pattern_bytes   = 1ull << 30; // DFS: device buffer collecting itemsets for --output
    uint64_t stack_capacity  = 128ull << 20; // most tasks one stack (DFS) or level (BFS) may hold
    std::string output;              // itemset file; empty = only count the itemsets
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
