#pragma once
#include <cstdint>
#include "data_types.cuh"

struct WorkItem
{
    uint32_t *pattern = nullptr;
    uint32_t pattern_length = 0;

    uint32_t utility = 0;

    Database *db = nullptr;

    uint32_t max_item = 0; // for util bucket 
    uint32_t primary = 0;

    uint32_t *work_done = nullptr;
    uint32_t work_count = 0;

    // Clear/reset the WorkItem
    __host__ __device__
    void clear()
    {
        pattern = nullptr;
        work_done = nullptr;
        db = nullptr;

        pattern_length = 0;
        utility = 0;
        max_item = 0;
        primary = 0;
        work_count = 0;
    }

    // Check if the WorkItem is "empty"
    __host__ __device__
    bool is_empty() const
    {
        return pattern == nullptr && db == nullptr && pattern_length == 0;
    }

    // Conversion to bool — allows: if (item) { ... }
    __host__ __device__
    explicit operator bool() const
    {
        return !is_empty();
    }

};

