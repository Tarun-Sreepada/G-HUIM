#pragma once

#include "args.hpp"

#include "patterns/patterns.cuh"

#include "task/task.cuh"
#include "task/stack.cuh"
#include "file/build.hpp"



void dynamic_parallel_mining(built_file &bf, patterns *p);