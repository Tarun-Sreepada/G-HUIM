#pragma once

#include "args.hpp"

#include "patterns/patterns.cuh"

#include "task/task.cuh"
#include "task/stack.cuh"
// #include "file/build.hpp"
#include "file/build.cuh"
#include "memory/memory.cuh"

#define scale 2
#define CAPACITY (1024 * 1024 * 1024)


void static_parallel_mining(built_file &bf, patterns *p, ArgParser &parser);

uint32_t next_pow2(uint32_t n);