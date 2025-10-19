#pragma once

#include "../task/task.cuh"

#include <cstdint>

struct patterns
{
    uint32_t* data          = nullptr; ///< Pointer to array of patterns.
    uint64_t  offset        = 0;       ///< Offset in the patterns array.
    uint32_t  pattern_count = 0;       ///< Number of patterns.
    uint64_t  size          = 0;       ///< Size of the patterns array.

    __host__ void init(uint64_t s)
    {
        // size = s;
        this->size = s;
        cudaMalloc(&data, size * sizeof(uint32_t));
        pattern_count = 0;
        offset        = 0;

    }

    // deconstructor
    __host__ void free()
    {
        if (data) {
            cudaFree(data);
            data = nullptr;
        }
    }

    __device__ void add_pattern(Task& task, uint32_t min_util)
    {
        if (task.utility >= min_util) {
            // Store the utility
            __shared__ uint64_t pattern_offset;
            if (threadIdx.x == 0) {
                atomicAdd(&pattern_count, 1); // Increment the pattern count
                pattern_offset = atomicAdd(
                        reinterpret_cast<unsigned long long int*>(&offset),
                        static_cast<unsigned long long int>(
                                task.pattern_length + 2)); // Get the current offset // 1 for Utility, 1 for spacer
                // if (pattern_offset  size);
                if (pattern_offset + task.pattern_length + 1 < size) {
                    data[pattern_offset + task.pattern_length] = task.utility; // Store the utility
                    for (uint32_t i = 0; i < task.pattern_length; i++) {
                        data[pattern_offset + i] = task.pattern[i]; // Copy the pattern
                    }
                }
                data[pattern_offset + task.pattern_length + 1] = 0; // Spacer
            }
            __syncthreads();
        }
    }

    __device__ void add_pattern_st(Task& task, uint32_t min_util)
    {
        if (task.utility >= min_util) {
            atomicAdd(&pattern_count, 1); // Increment the pattern count
            uint64_t pattern_offset = atomicAdd(
                    reinterpret_cast<unsigned long long int*>(&offset),
                    static_cast<unsigned long long int>(task.pattern_length +
                                                        2)); // Get the current offset // 1 for Utility, 1 for spacer
            if (pattern_offset >= size)
                return;

            data[pattern_offset + task.pattern_length] = task.utility; // Store the utility
            for (int i = 0; i < task.pattern_length; i++) {
                data[pattern_offset + i] = task.pattern[i]; // Copy the pattern
            }
            data[pattern_offset + task.pattern_length + 1] = 0; // Spacer
        }
    }
};