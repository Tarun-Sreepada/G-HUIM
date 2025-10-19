#pragma once
#include "data_types/data_types.cuh"
#include "task.cuh"

#include <cstdint>

template <typename T, int CAPACITY> struct Stack
{
    T* data;

    volatile unsigned int top;

    // Allocates unified memory for the stack data.
    __host__ void init_managed()
    {
        top = 0;
        printf("Managed Type size: %lu, Capacity: %d, Total size: %lu MB\n", sizeof(T), CAPACITY,
               (sizeof(T) * CAPACITY) / (1024 * 1024));
        cudaMallocManaged(&data, sizeof(T) * CAPACITY);
    }

    // Allocates device memory for the stack data.
    __host__ void init()
    {
        top = 0;
        printf("Type size: %lu, Capacity: %d, Total size: %lu MB\n", sizeof(T), CAPACITY,
               (sizeof(T) * CAPACITY) / (1024 * 1024));
        cudaMalloc(&data, sizeof(T) * CAPACITY);
    }

    // Clears the stack without freeing memory.
    __host__ void reset()
    {
        top = 0;
    }

    // Releases allocated stack memory.
    __host__ void de_init()
    {
        cudaFree(data);
    }

    // Pushes an item atomically and returns success.
    __device__ bool push(T item)
    {
        data[atomicAdd((unsigned int*)&top, 1)] = item;
        return true;
    }

    // Retrieves an item by index if it exists.
    __device__ bool get(T& item, int index)
    {
        if (index < top) {
            item = data[index];
            return true;
        }
        return false;
    }

    // Provides mutable access to an item by index.
    __device__ Task& operator[](int index)
    {
        return data[index];
    }
    // Provides read-only access to an item by index.
    __device__ const Task& operator[](int index) const
    {
        return data[index];
    }
};
