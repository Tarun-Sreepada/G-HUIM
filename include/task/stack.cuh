#pragma once
#include "data_types/data_types.cuh"
#include "task.cuh"

#include <cstdint>
#include <cstdio>

template <typename T> struct Stack
{
    T*                   data      = nullptr;
    volatile unsigned int top       = 0;
    uint32_t             capacity  = 0;

    // Allocates unified memory for the stack data.
    __host__ void init_managed(uint32_t requested_capacity)
    {
        capacity = requested_capacity;
        top      = 0;

        const unsigned long long total_bytes = static_cast<unsigned long long>(sizeof(T)) * capacity;
        printf("Managed Type size: %lu, Capacity: %u, Total size: %llu MB\n", sizeof(T), capacity,
               total_bytes / (1024ull * 1024ull));
        cudaMallocManaged(&data, total_bytes);
    }

    // Allocates device memory for the stack data.
    __host__ void init(uint32_t requested_capacity)
    {
        capacity = requested_capacity;
        top      = 0;

        const unsigned long long total_bytes = static_cast<unsigned long long>(sizeof(T)) * capacity;
        printf("Type size: %lu, Capacity: %u, Total size: %llu MB\n", sizeof(T), capacity,
               total_bytes / (1024ull * 1024ull));
        cudaMalloc(&data, total_bytes);
    }

    // Clears the stack without freeing memory.
    __host__ void reset()
    {
        top = 0;
    }

    // Releases allocated stack memory.
    __host__ void de_init()
    {
        if (data != nullptr)
            cudaFree(data);
        data     = nullptr;
        capacity = 0;
        top      = 0;
    }

    // Pushes an item atomically and returns success. Fails when the stack is full.
    __device__ bool push(T item)
    {
        const unsigned int index = atomicAdd((unsigned int*)&top, 1u);
        if (index >= capacity) {
            atomicSub((unsigned int*)&top, 1u);
            return false;
        }

        data[index] = item;
        return true;
    }

    // Retrieves an item by index if it exists.
    __device__ bool get(T& item, int index)
    {
        if (index < top && static_cast<uint32_t>(index) < capacity) {
            item = data[index];
            return true;
        }
        return false;
    }

    // Provides mutable access to an item by index.
    __device__ T& operator[](int index)
    {
        return data[index];
    }

    // Provides read-only access to an item by index.
    __device__ const T& operator[](int index) const
    {
        return data[index];
    }
};
