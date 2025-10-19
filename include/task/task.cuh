#pragma once
#include "../data_types/data_types.cuh"

#include <cstdint>

struct Task
{
    uint32_t* pattern        = nullptr;
    uint32_t  pattern_length = 0;

    uint32_t utility = 0;

    Database* db = nullptr;

    uint32_t unique_items = 0;
    uint32_t primary      = 0;

    // Clears the task to a default state.
    __host__ __device__ void clear()
    {
        pattern = nullptr;
        db = nullptr;

        pattern_length = 0;
        utility        = 0;
        unique_items   = 0;
        primary        = 0;
    }

    // Checks whether the task holds any work.
    __host__ __device__ bool is_empty() const
    {
        return pattern == 0 && db == 0 && pattern_length == 0;
    }

    // Returns a sentinel task with no associated work.
    __host__ __device__ static Task empty()
    {
        Task t;
        t.clear();
        return t;
    }
};
