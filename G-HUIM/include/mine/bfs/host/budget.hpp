#pragma once
// Device buffers sized from the free memory.

#include "app/config.hpp"
#include "mine/bfs/placement.cuh"

namespace ghuim::bfs {

struct Budget
{
    uint64_t capacity;         // tasks per level
    uint64_t scratch_capacity; // bytes
    uint64_t region_bytes;     // each of the two arena regions
    uint64_t task_bytes;
    uint64_t per_task;
};

Budget plan_budget(const MiningOptions& options, const AutoTune& tune, uint64_t first_level_tasks);

} // namespace ghuim::bfs
