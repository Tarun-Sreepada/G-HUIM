#pragma once

#include <cstdint>
#include <string>

namespace ghuim {

enum class Method
{
    Static,  // every task gets one block of a fixed size
    Adaptive, // each task gets a thread, a warp, or a block depending on its database size
    AdaptiveOptimized // size-binned kernels, lane-per-entry transaction scans
};

inline const char* method_name(Method method)
{
    switch (method) {
    case Method::Static: return "static";
    case Method::Adaptive: return "adaptive";
    default: return "adaptive-optimized";
    }
}

struct MiningOptions
{
    uint64_t minimum_utility = 0;
    Method   method          = Method::Adaptive;
    uint32_t block_size      = 256;
    uint64_t memory_bytes    = 0; // projected-database arena; 0 = most of the free device memory
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
