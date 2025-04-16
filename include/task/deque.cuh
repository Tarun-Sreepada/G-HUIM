#pragma once
#include <cstdint>
#include "data_types/data_types.cuh"
#include "task.cuh"

#define MIN_TASKS 2 // for dynamic parallelism i dont want funny contention issues

// A struct for the head. We pack head.index and head.ctr into a 64-bit value.
struct alignas(16) Head
{
    int index; // Index where thieves steal.
    int ctr;   // A version counter (updated at critical points).
};

union AtomicHead
{
    Head h;
    unsigned long long value;
};

template <typename T, int CAPACITY>
struct Deque
{
    T data[CAPACITY];        // Array of tasks.
    int capacity = CAPACITY; // Capacity.

    // The head is stored in a union so that we can update it atomically.
    volatile AtomicHead a_head;
    volatile int tail;   // Index where owner pushes/pops.
    volatile int active; // Cumulative active count (increases on push, decreases on finish).

    __device__ __host__ Deque()
    {
        a_head.value = packHead(0, 0);
        tail = 0;
        active = 0;
    }

    __device__ void init()
    {
        a_head.value = packHead(0, 0);
        tail = 0;
        active = 0;
    }

    // Pack head values into a single 64-bit integer.
    __device__ __forceinline__ static unsigned long long packHead(int index, int ctr)
    {
        return (((unsigned long long)ctr) << 32) | ((unsigned int)index);
    }
    // Unpack a 64-bit value into a Head.
    __device__ __forceinline__ static Head unpackHead(unsigned long long value)
    {
        Head h;
        h.index = (int)(value & 0xffffffff);
        h.ctr = (int)(value >> 32);
        return h;
    }

    // --------------------- PUSH ---------------------
    // Push is called only by the owner.
    // Increments active to indicate that new work is submitted.
    __device__ bool push(T item)
    {
        // (It is assumed that your system guarantees tail - head <= capacity.)
        int idx = atomicAdd((int *)&tail, 1);
        data[idx] = item;
        atomicAdd((int *)&active, 1); // Increase active to signal a new task.
        __threadfence();              // Ensure the data is visible.
        return true;
    }

    // --------------------- POP ---------------------
    // This pop is owner-only.
    // It removes the task from the deque but does NOT subtract from active
    // because the task is now in processing.
    __device__ bool pop(T &item)
    {
        if (tail <= unpackHead(a_head.value).index)
        {
            return false; // Deque is empty.
        }
        // Decrement tail (owner only).
        int t = tail - 1;
        tail = t;
        __threadfence(); // Ensure ordering.
        item = data[t];
        // Do NOT subtract active here; the task remains "active" until processing is complete.
        return true;
    }

    // --------------------- STEAL ---------------------
    // A thief attempts to steal from the head.
    // On success, the task is removed and active is not decremented here (for the same reason).
    __device__ bool steal(T &item) {
        // Read the old head (atomicAdd(...,0) is effectively an atomic read).
        unsigned long long old_val = atomicAdd((unsigned long long *)&a_head.value, 0);
        Head old_head = unpackHead(old_val);
    
        // If the deque is empty, bail out.
        // if (tail <= old_head.index) {
        //     return false;
        // }
        if (tail - old_head.index < MIN_TASKS) {
            // Not enough items available to steal.
            return false;
        }
    
        // Prepare the new head: increment index *and* the version counter.
        Head new_head = old_head;
        new_head.index++; 
        new_head.ctr++;   // <--- important!
    
        unsigned long long new_val = packHead(new_head.index, new_head.ctr);
    
        // It is safest to read 'item' *after* CAS to avoid races,
        // or at least re-check that CAS succeeded before finalizing 'item'.
        // Typically we read the item first, but only "commit" it if CAS succeeds.
        T temp = data[old_head.index];
    
        // Attempt to claim that slot via CAS.
        unsigned long long res = atomicCAS((unsigned long long *)&a_head.value, old_val, new_val);
        if (res == old_val) {
            // Success: we have stolen 'temp'.
            item = temp;
            // Optionally do __threadfence() if you need memory ordering.
            __threadfence(); // Ensure the data is visible.
            return true;
        }
    
        // CAS failed -> revert the stolen item.
        // Another thief or the owner changed the head in the meantime.
        item = T::empty();
        return false;
    }
    

    // --------------------- FINISH TASK ---------------------
    // Only call finish_task() when a task is fully processed.
    // This subtracts from active to indicate one less work item is active.
    __device__ void finish_task()
    {
        atomicSub((int *)&active, 1);
    }

    // Helper: compute pending work (tasks waiting in the deque).
    __device__ int pending()
    {
        Head h = unpackHead(a_head.value);
        return tail - h.index;
    }

    __device__ void add_active(int count = 1)
    {
        atomicAdd((int *)&active, count);
    }

    // Optional: You can add a helper to get the active count.
    __device__ int get_active()
    {
        return atomicAdd((int *)&active, 0);
    }
};