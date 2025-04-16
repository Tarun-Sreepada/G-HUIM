#pragma once

#include "args.hpp"

#include "patterns/patterns.cuh"

#include "task/task.cuh"
#include "task/stack.cuh"
#include "file/build.hpp"
#include "memory/memory.cuh"

#define scale 2
#define CAPACITY (1024 * 1024 * 1024)


void static_parallel_mining(built_file &bf, patterns *p, ArgParser &parser);