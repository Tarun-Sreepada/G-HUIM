#pragma once

#include "../task/task.cuh"

#include <cstdint>

struct patterns
{
    uint32_t* data          = nullptr; ///< Pointer to array of patterns.
    uint64_t  offset        = 0;       ///< Offset in the patterns array.
    uint32_t  pattern_count = 0;       ///< Number of patterns.
    uint64_t  size          = 0;       ///< Size of the patterns array.

    // Initializes device storage for collecting patterns.
    __host__ void init(uint64_t s)
    {
        this->size = s;
        cudaMalloc(&data, size * sizeof(uint32_t));
        pattern_count = 0;
        offset        = 0;

    }

    // Releases device storage for patterns.
    __host__ void free()
    {
        if (data) {
            cudaFree(data);
            data = nullptr;
        }
    }

    // Adds a pattern using cooperative updates across a thread block.
    __device__ void add_pattern(Task& task, uint32_t min_util)
    {
        if (task.utility >= min_util) {
            __shared__ uint64_t pattern_offset;
            if (threadIdx.x == 0) {
                atomicAdd(&pattern_count, 1);
                pattern_offset = atomicAdd(
                        reinterpret_cast<unsigned long long int*>(&offset),
                        static_cast<unsigned long long int>(
                                task.pattern_length + 2));
                if (pattern_offset + task.pattern_length + 1 < size) {
                    data[pattern_offset + task.pattern_length] = task.utility;
                    for (uint32_t i = 0; i < task.pattern_length; i++) {
                        data[pattern_offset + i] = task.pattern[i];
                    }
                }
                data[pattern_offset + task.pattern_length + 1] = 0;
            }
            __syncthreads();
        }
    }

    // Adds a pattern using a single thread for serialization.
    __device__ void add_pattern_st(Task& task, uint32_t min_util)
    {
        if (task.utility >= min_util) {
            atomicAdd(&pattern_count, 1);
            uint64_t pattern_offset = atomicAdd(
                    reinterpret_cast<unsigned long long int*>(&offset),
                    static_cast<unsigned long long int>(task.pattern_length + 2));
            if (pattern_offset >= size)
                return;

            data[pattern_offset + task.pattern_length] = task.utility;
            for (int i = 0; i < task.pattern_length; i++) {
                data[pattern_offset + i] = task.pattern[i];
            }
            data[pattern_offset + task.pattern_length + 1] = 0;
        }
    }
};