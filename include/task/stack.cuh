#pragma once
#include <cstdint>
#include "data_types/data_types.cuh"
#include "task.cuh"


template<typename T, int CAPACITY>
struct Stack
{
    T data[CAPACITY];

    volatile unsigned int top;                     // Points to the next free slot
    unsigned long long int total_transactions = 0; // Total transactions in the database

    __host__ void init()
    {
        top = 0;
        total_transactions = 0;
    }

    __device__ bool push(T item)
    {
        data[atomicAdd((unsigned int *)&top, 1)] = item;
        atomicAdd(&total_transactions, (unsigned long long int)item.db->num_transactions);
        return true;
    }

    __device__ bool get(T &item, int index)
    {
        if (index < top)
        {
            item = data[index];
            return true;
        }
        return false;
    }

    __device__ Task &operator[](int index)
    {
        return data[index];
    }
    __device__ const Task &operator[](int index) const
    {
        return data[index];
    }
};


