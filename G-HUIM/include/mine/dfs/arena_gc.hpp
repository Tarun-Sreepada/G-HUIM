#pragma once
// Arena reclamation: slides every database still referenced by the task stack to the bottom of
// the arena (databases use internal offsets only) and re-points the stack.

#include "cuda/buffer.cuh"
#include "mine/dfs/layout.cuh"

namespace ghuim::inline GHUIM_WIDTHS::dfs {

struct ArenaBuffers
{
    DeviceBuffer<char>&     arena;
    DeviceBuffer<char>&     scratch;
    DeviceBuffer<Task>&     stack;
    DeviceBuffer<Control>&  control;
    DeviceBuffer<uint32_t>& bitmap;
    DeviceBuffer<uint32_t>& counts;
    DeviceBuffer<uint32_t>& block_sums;
    DeviceBuffer<uint32_t>& scan_total;
    uint32_t                words; // bitmap words covering the arena
};

struct Compaction
{
    uint32_t live_databases;
    uint64_t arena_top; // bytes in use after compaction
};

Compaction compact_arena(const ArenaBuffers& buffers, uint32_t stack_size, cudaStream_t stream);

} // namespace ghuim::inline GHUIM_WIDTHS::dfs
