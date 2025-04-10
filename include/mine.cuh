#pragma once
#include "memory.cuh"
#include "stack.cuh"
#include "data_types.cuh"
#include "data_types.hpp"
#include "copy.cuh"
#include "build.hpp"
#include "file.hpp"
#include "timer.hpp"

#define scale 3

__global__ void mine(stack *work_stack,
    stack *new_work_stack,
    uint32_t *patterns,
    uint32_t min_util);