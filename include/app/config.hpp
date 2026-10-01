#pragma once

#include <cstdint>
#include <string>

namespace ghuim {

struct MiningOptions
{
    uint64_t    minimum_utility = 0;
    uint64_t    memory_bytes    = 0;            // projected-database arena; 0 = most of the free device memory
    uint64_t    stack_capacity  = 128ull << 20; // most tasks one level may hold
    std::string output;                         // itemset file; empty = only count the itemsets
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
