#pragma once
#include <cstdint>
#include "data_types.cuh"

#define CAPACITY (128 * 1024)

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

    // Default constructor
    __host__ __device__
    WorkItem() = default;

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



struct stack
{
    WorkItem items[CAPACITY];

    volatile unsigned int top;    // Points to the next free slot
    volatile unsigned int active; // Keeps track of active tasks / Maybe active tasks can push more items
    int lock;           // false = unlocked, true = locked

    __host__ void init()
    {
        top = 0;
        active = 0;
        lock = false;
    }

    // Device-side spinlock using atomicCAS
    __device__ void acquire_lock()
    {
        while (atomicCAS(&lock, 0, 1) != 0)
        {
            // Busy-wait
        }
    }

    __device__ void release_lock()
    {
        atomicExch(&lock, 0);
    }

    // Push a work item onto the stack (LIFO)
    // @brief Push a work item onto the stack
    // @param item The work item to push
    // @return 1 if the push was successful, 0 otherwise
    __device__ bool push(WorkItem item)
    {
        bool success = false;
        acquire_lock();
        if (top < CAPACITY)
        { // Check if stack is not full
            items[top] = item;
            top = top + 1;
            __threadfence(); // Ensure memory ordering
            atomicAdd((unsigned int *)&active, 1);
            success = true;
        }
        release_lock();
        return success;
    }
    
    __host__ bool host_push(WorkItem item)
    {
        bool success = false;
        if (top < CAPACITY)
        { // Check if stack is not full
            items[top] = item;
            top = top + 1;
            active = active + 1;
            success = true;
        }
        return success;
    }

    // Pop a work item from the stack (LIFO)
    __device__ bool pop(WorkItem &item)
    {
        bool success = false;
        acquire_lock();
        if (top > 0)
        { // Check if stack is not empty
            top = top - 1;
            item = items[top];
            __threadfence(); // Ensure memory ordering
            success = true;
        }
        release_lock();
        return success;
    }

    // @brief Check if the stack may pop
    // @return 1 if the stack may pop, 0 otherwise
    __host__ __device__ int may_pop()
    {
        return active;
    }

    // Mark a task as finished (decrement 'active')
    __device__ void finish_task()
    {
        atomicSub((unsigned int *)&active, 1);
    }

};
