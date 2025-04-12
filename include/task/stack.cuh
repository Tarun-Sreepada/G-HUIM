#pragma once
#include <cstdint>
#include "data_types/data_types.cuh"
#include "task.cuh"
#define CAPACITY (1024 * 1024)

struct Stack
{
    Task tasks[CAPACITY];

    volatile unsigned int top;                     // Points to the next free slot
    unsigned long long int total_transactions = 0; // Total transactions in the database

    __host__ void init()
    {
        top = 0;
        total_transactions = 0;
    }

    __device__ bool push(Task item)
    {
        uint32_t old_top = atomicAdd((unsigned int *)&top, 1);
        atomicAdd(&total_transactions, (unsigned long long int)item.db->num_transactions);
        tasks[old_top] = item;
        return true;
    }

    __device__ bool get(Task &item, int index)
    {
        if (index < top)
        {
            item = tasks[index];
            return true;
        }
        return false;
    }

    __device__ Task &operator[](int index)
    {
        return tasks[index];
    }
    __device__ const Task &operator[](int index) const
    {
        return tasks[index];
    }
};
