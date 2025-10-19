#pragma once

#include "args.hpp"
#include "patterns/patterns.cuh"
#include "task/stack.cuh"
#include "task/task.cuh"
#include "file/build.cuh"
#include "memory/memory.cuh"
#include "timer.hpp"

#define scale 3 / 2
#define CAPACITY (128 * 1024 * 1024)

// Launches the static-parallel mining pipeline on the prepared database.
void static_parallel_mining(built_file& bf, patterns* p, ArgParser& parser, Timer& timer, size_t shared_memory,
                            std::pair<uint32_t *, uint64_t>& all_patterns);

// Rounds an integer up to the next power of two.
__host__ __device__ uint32_t next_pow2(uint32_t n);