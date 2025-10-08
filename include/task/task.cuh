#pragma once
#include <cstdint>
#include "../data_types/data_types.cuh"



struct Task
{
    uint64_t memory_usage = 0; // in bytes

    uint32_t *pattern = nullptr;
    uint32_t pattern_length = 0;

    uint32_t utility = 0;

    Database *db = nullptr;

    uint32_t unique_items = 0; // for util bucket 
    uint32_t primary = 0;

    // To keep track of the work done on this database so that it can be freed
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
        unique_items = 0;
        primary = 0;
        work_count = 0;
    }

    // Check if the WorkItem is "empty"
    __host__ __device__
    bool is_empty() const
    {
        return pattern == 0 && db == 0 && pattern_length == 0;
    }

    __host__ __device__
    static Task empty() {
        Task t;
        t.clear();
        return t;
    }

};

