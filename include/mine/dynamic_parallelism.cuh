#pragma once

#include "args.hpp"

#include "patterns/patterns.cuh"

#include "task/task.cuh"
// #include "file/build.hpp"
#include "file/build.cuh"
#include "task/deque.cuh"
#include "memory/memory.cuh"

#define scale 2
#define MAX_TASKS (1024 * 32)

void dynamic_parallel_mining(built_file &bf, patterns *p, ArgParser &parser);