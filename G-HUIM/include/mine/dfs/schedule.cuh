#pragma once
// Device-side scheduling: batch selection from the task stack, binning, and arena compaction.

#include "mine/dfs/kernels.cuh"
#include "mine/dfs/layout.cuh"

#include <cuda_runtime.h>

namespace ghuim::inline GHUIM_WIDTHS::dfs {

struct SelectArgs
{
    Control*    control;
    const Task* stack;
    Task*       batch;
    uint64_t*   scratch_offset;
    uint64_t    arena_capacity;
    uint64_t    scratch_capacity;
    uint32_t    stack_capacity;
    uint32_t    max_batch;
    uint32_t    arena_divisor;    // optimistic batches assume children are this many times smaller
    uint32_t    headroom_divisor; // and use only 1/headroom_divisor of the free arena
};

// Tasks go to the first bin whose transaction limit they fit; the last bin takes the rest.
struct BinRule
{
    bool      optimized; // use place_task() instead of the transaction limits
    uint32_t bins;
    uint32_t  max_trans[kMaxBins];
    AutoTune  tune; // adaptive-optimized
};

// Pops a batch from the stack top within the arena/scratch/stack bounds (at least one task),
// then partitions it into bins: order[] lists batch indices bin by bin.
void select_batch(const SelectArgs& args, cudaStream_t stream);
void bin_batch(Control* control, const Task* batch, uint32_t* order, const BinRule& rule, uint32_t max_batch,
               cudaStream_t stream);

// Arena compaction helpers. `bitmap`/`counts` have one word per 32 arena units.
void mark_live(const Control* control, const Task* stack, uint32_t stack_size, uint32_t* bitmap, uint32_t words,
               cudaStream_t stream);
void count_live(const uint32_t* bitmap, uint32_t* counts, uint32_t words, cudaStream_t stream);
void extract_live(const uint32_t* bitmap, const uint32_t* offsets, uint32_t words, uint32_t* live,
                  cudaStream_t stream);
void live_sizes(const char* arena, const uint32_t* live, uint32_t count, uint32_t* units, cudaStream_t stream);
// Copies databases i in [0, count): bytes[i] from src + src_off[i] to dst + dst_off[i] (16-byte aligned).
void copy_databases(char* dst, const uint64_t* dst_off, const char* src, const uint64_t* src_off,
                    const uint64_t* bytes, uint32_t count, cudaStream_t stream);
void remap_tasks(Task* stack, uint32_t stack_size, const uint32_t* live, const uint32_t* moved_to, uint32_t count,
                 cudaStream_t stream);

} // namespace ghuim::inline GHUIM_WIDTHS::dfs
