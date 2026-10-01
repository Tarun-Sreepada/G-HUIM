#pragma once
// Launchers of the depth-first kernels. Each processes one stack of tasks and pushes the children
// onto another.

#include "mine/dfs/types.cuh"

#include <cuda_runtime.h>

namespace ghuim::dfs {

struct MineArgs
{
    Control*     control;
    uint32_t     minimum_utility;
    Stack<Task>* current;
    Stack<Task>* next;
    // Adaptive method only: size histogram (8 buckets) and the stack for single-thread tasks.
    uint32_t*    distribution = nullptr;
    Stack<Task>* singles      = nullptr;
};

// One block of `threads` threads per task of `current`.
void launch_block_tasks(const MineArgs& args, uint32_t tasks, uint32_t threads);

// One thread per task of `current` (kSingleBlock threads per block); children go to `next`.
void launch_single_tasks(const MineArgs& args, uint32_t tasks);

} // namespace ghuim::dfs
