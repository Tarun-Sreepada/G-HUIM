#pragma once
#include "memory.cuh"
#include "stack.cuh"
#include "data_types.cuh"
#include "data_types.hpp"
#include "copy.cuh"
#include "build.hpp"
#include "file.hpp"
#include "timer.hpp"

#define scale 2

__global__ void mine(stack *work_stack, uint32_t *patterns, uint32_t min_util, uint32_t unique_item_count, uint32_t transaction_count);