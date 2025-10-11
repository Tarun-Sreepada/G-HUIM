#pragma once

#include "args.hpp"

#include "patterns/patterns.cuh"

#include "task/task.cuh"
#include "task/stack.cuh"
// #include "file/build.hpp"
#include "file/build.cuh"
#include "memory/memory.cuh"
#include "timer.hpp"

#define scale 3/2
#define CAPACITY (128 * 1024 * 1024)


void static_parallel_mining(built_file &bf, patterns *p, ArgParser &parser, Timer &timer, size_t shared_memory, std::vector<std::vector<uint32_t>>& all_patterns);

__host__ __device__ uint32_t next_pow2(uint32_t n);