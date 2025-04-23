#pragma once
#include <cstdint>
#include "data_types/data_types.cuh"
#include "task.cuh"

#define MIN_STEAL 1 // leave at least this many for the owner

template <typename T>
__device__ __forceinline__ T atomic_load(const volatile T *ptr)
{
    return atomicAdd(const_cast<T *>(ptr), static_cast<T>(0));
}

template <>
__device__ __forceinline__ unsigned long long
atomic_load<unsigned long long>(const volatile unsigned long long *ptr)
{
    return atomicAdd(const_cast<unsigned long long *>(ptr), 0ull);
}

struct alignas(16) Head
{
    int index;
    int ctr;
};

union AtomicHead
{
    Head h;
    unsigned long long value;
};

template <typename T, int CAPACITY>
struct Deque
{
    T data[CAPACITY];
    volatile AtomicHead a_head;
    volatile int tail;
    volatile int active;
    int capacity;
    int processed;

    __device__ __host__ Deque() { init(); }

    __device__ void init()
    {
        a_head.value = packHead(0, 0);
        tail = 0;
        active = 0;
        capacity = CAPACITY;
        processed = 0;
    }

    __device__ static __forceinline__ unsigned long long packHead(int idx, int ctr)
    {
        return (static_cast<unsigned long long>(ctr) << 32) |
               static_cast<unsigned int>(idx);
    }

    __device__ static __forceinline__
        Head
        unpackHead(unsigned long long v)
    {
        Head h;
        h.index = static_cast<int>(v & 0xffffffffu);
        h.ctr = static_cast<int>(v >> 32);
        return h;
    }

    __device__ bool push(const T &item)
    {
        int idx = atomicAdd((int *)&tail, 1); // atomic inc, returns old value
        data[idx] = item;
        __threadfence();              // make item visible
        atomicAdd((int *)&active, 1); // track in‑flight work
        return true;
    }

    __device__ bool pop(T &out)
    {
        // Snapshot head & tail *atomically*
        Head h = unpackHead(atomic_load(&a_head.value));
        int old = atomicAdd((int *)&tail, -1); // pre‑decrement

        int t = old - 1; // new tail value
        if (t < h.index)
        { // queue empty ⇒ roll back
            atomicAdd((int *)&tail, 1);
            return false;
        }
        __threadfence(); // make sure data is ready
        out = data[t];
        processed++;
        return true;
    }

    __device__ int pending()
    {
        Head h = unpackHead(atomic_load(&a_head.value));
        int t = atomic_load(&tail);
        return t - h.index;
    }

    __device__ int get_active() { return atomic_load(&active); }
    __device__ void add_active(int c = 1) { atomicAdd((int *)&active, c); }
    __device__ void finish_task() { atomicAdd((int *)&active, -1); }

    __device__ bool steal(T &out)
    {
        unsigned long long old_val = atomic_load(&a_head.value);
        Head old_h = unpackHead(old_val);

        int t = atomic_load(&tail);
        int sz = t - old_h.index;
        if (sz <= MIN_STEAL)
            return false; // nothing (fair) to steal

        // Grab prospective victim task *before* CAS so we only read once
        T tmp = data[old_h.index];

        Head new_h{old_h.index + 1, old_h.ctr + 1};
        unsigned long long new_val = packHead(new_h.index, new_h.ctr);

        if (atomicCAS((unsigned long long *)&a_head.value,
                      old_val, new_val) == old_val)
        {
            __threadfence(); // ensure prior writes visible
            out = tmp;
            return true;
        }
        return false; // raced – try again later
    }
};
