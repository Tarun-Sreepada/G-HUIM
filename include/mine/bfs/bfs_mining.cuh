#pragma once
// Level-synchronous (BFS) optimized miner: every task of a level runs before any task of the next.

#include "file/build.cuh"
#include "mine/bfs/layout.cuh"
#include "timer.hpp"

#include <cstdint>
#include <string>
#include <vector>

// Default cap on tasks per level for the optimized miner (each costs 76 bytes of task arrays).
constexpr uint32_t kBfsDefaultTaskCapacity = 32u * 1024u * 1024u;

struct BfsOptions
{
    uint64_t    minimum_utility;
    uint64_t    memory_bytes;  // device memory for arenas, scratch and task arrays
    uint32_t    task_capacity; // most tasks one level may hold
    std::string output;        // "/dev/null" skips collecting itemsets
};

struct BfsResult
{
    uint64_t                            patterns = 0;
    std::vector<std::vector<bfs::Node>> levels; // per level, when options.output is a file
};

// Mines `bf`; records one timer point per level.
BfsResult bfs_optimized_mining(const built_file& bf, const Item_mapping* item_map, uint32_t max_item,
                               const BfsOptions& options, Timer& timer);

// Writes the high-utility itemsets in the original output format ("items... UTIL u").
void write_bfs_itemsets(const std::string& path, const BfsResult& result, const Item_mapping* item_map,
                        uint32_t max_item, uint64_t minimum_utility);
