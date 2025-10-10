#pragma once

#include "../task/task.cuh"
#include <cstdint>

struct patterns
{
    uint32_t *data = nullptr;        ///< Pointer to array of patterns.
    uint32_t offset = 0; ///< Offset in the patterns array.
    uint32_t pattern_count = 0; ///< Number of patterns.
    uint32_t size = 0;          ///< Size of the patterns array.

    __host__ void init(uint32_t size)
    {
        // Using cudaMallocManaged for unified memory accessible by both host and device
        cudaMallocManaged(&data, size * sizeof(uint32_t));
        pattern_count = 0;
        offset = 0;
        this->size = size;
    }

    // Destructor
    __host__ void free()
    {
        if (data)
        {
            cudaFree(data);
            data = nullptr;
        }
    }

    /**
     * @brief Adds a pattern from a single thread within a block.
     * @note This version assumes only ONE thread (threadIdx.x == 0) per block will add a pattern.
     *       If multiple threads in a block can generate a valid pattern, only the first one's will be stored.
     */
    __device__ void add_pattern(Task &task, uint32_t min_util)
    {
        if (task.utility >= min_util)
        {
            // Only thread 0 in the block performs the addition to avoid race conditions within the block.
            if (threadIdx.x == 0)
            {
                uint32_t space_needed = task.pattern_length + 2; // +1 for utility, +1 for spacer
                uint32_t write_offset = atomicAdd(&offset, space_needed);

                // Check if there is enough space in the buffer
                if (write_offset + space_needed <= this->size)
                {
                    for (int i = 0; i < task.pattern_length; i++)
                    {
                        data[write_offset + i] = task.pattern[i]; // Copy the pattern
                    }
                    data[write_offset + task.pattern_length] = task.utility; // Store the utility
                    atomicAdd(&pattern_count, 1); // Increment the pattern count
                }
                else
                {
                    // Rollback the offset if there wasn't enough space
                    atomicSub(&offset, space_needed);
                }
            }
        }
    }

    /**
     * @brief Adds a pattern from a single, independent thread. (ST = Single Thread)
     * @note This version is the one you should use when ANY thread in a kernel launch can add a pattern.
     *       It is robust against out-of-bounds access.
     */
    __device__ void add_pattern_st(Task &task, uint32_t min_util)
    {
        if (task.utility >= min_util)
        {
            // Atomically get the current offset and increment it for the next thread.
            uint32_t space_needed = task.pattern_length + 2; // +1 for utility, +1 for spacer
            uint32_t write_offset = atomicAdd(&offset, space_needed);

            // --- KERNEL CRASH FIX ---
            // Check if the reserved space is within the bounds of the allocated buffer.
            if (write_offset + space_needed <= this->size)
            {
                // If it is, copy the pattern and utility.
                for (int i = 0; i < task.pattern_length; i++)
                {
                    data[write_offset + i] = task.pattern[i]; // Copy the pattern
                }
                data[write_offset + task.pattern_length] = task.utility; // Store the utility

                // Only now, increment the count of successfully stored patterns.
                atomicAdd(&pattern_count, 1);
            }
            else
            {
                // IMPORTANT: If we run out of space, roll back the offset we claimed.
                // This prevents `offset` from growing far beyond the buffer's actual size.
                atomicSub(&offset, space_needed);
            }
        }
    }
};