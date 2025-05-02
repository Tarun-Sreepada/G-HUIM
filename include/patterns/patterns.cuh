#pragma once

#include "../task/task.cuh"
#include <cstdint>

struct patterns
{
    uint32_t *data = nullptr;        ///< Pointer to array of patterns.
    uint32_t offset = 0; ///< Offset in the patterns array.
    uint32_t pattern_count = 0; ///< Number of patterns.

    __host__ void init(uint32_t size)
    {
        cudaMallocManaged(&data, size * sizeof(uint32_t));
        pattern_count = 0;
        offset = 0;
    }

    // deconstructor
    __host__ void free()
    {
        if (data)
        {
            cudaFree(data);
            data = nullptr;
        }
    }

    __device__ void add_pattern(Task &task, uint32_t min_util)
    {
        if (task.utility >= min_util)
        {
            // Store the utility
            __shared__ uint32_t pattern_offset;
            if (threadIdx.x == 0)
            {
                pattern_offset = atomicAdd(&offset, (task.pattern_length + 2)); // Get the current offset // 1 for Utility, 1 for spacer
                data[pattern_offset + task.pattern_length] = task.utility;      // Store the utility
                atomicAdd(&pattern_count, 1); // Increment the pattern count
            }
            __syncthreads();

            // Copy the pattern to the global memory
            for (int i = threadIdx.x; i < task.pattern_length; i += blockDim.x)
            {
                data[pattern_offset + i] = task.pattern[i]; // Copy the pattern
            }
            __syncthreads();
        }
    }
};