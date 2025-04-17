#pragma once
#include <cstdint>
#include "data_types/data_types.cuh"
#include "task.cuh"

#define MIN_STEAL 1  // leave at least this many for the owner

struct alignas(16) Head {
    int index;
    int ctr;
};

union AtomicHead {
    Head               h;
    unsigned long long value;
};

template <typename T, int CAPACITY>
struct Deque {
    T                   data[CAPACITY];
    volatile AtomicHead a_head;
    volatile int        tail;
    volatile int        active;

    __device__ __host__
    Deque()  { init(); }

    __device__ void init() {
        a_head.value = packHead(0,0);
        tail         = 0;
        active       = 0;
    }

    // pack / unpack
    __device__ static unsigned long long packHead(int idx,int ctr) {
        return ( (unsigned long long)ctr << 32 ) | (unsigned int)idx;
    }
    __device__ static Head unpackHead(unsigned long long v) {
        Head h; h.index=(int)(v&0xffffffff); h.ctr=(int)(v>>32); return h;
    }

    // owner-only push / pop
    __device__ bool push(const T &item) {
        int idx = atomicAdd((int*)&tail,1);
        data[idx] = item;
        atomicAdd((int*)&active,1);
        __threadfence();
        return true;
    }

    __device__ bool pop(T &out) {
        Head h = unpackHead(a_head.value);
        if (tail <= h.index) return false;
        int t = tail-1;
        tail = t;
        __threadfence();
        out = data[t];
        // keep `active` until finish_task()
        return true;
    }

    // how many are waiting (owner‑visible)
    __device__ int pending() {
        Head h = unpackHead(a_head.value);
        return tail - h.index;
    }

    // read the active count (pending + in‑flight)
    __device__ int get_active() {
        return atomicAdd((int*)&active, 0);
    }

    __device__ void add_active(int c=1) {
        atomicAdd((int*)&active, c);
    }
    __device__ void finish_task() {
        atomicSub((int*)&active, 1);
    }

    // thief
    __device__ bool steal(T &out) {
        unsigned long long old_val = atomicAdd((unsigned long long*)&a_head.value, 0ull);
        Head old_h = unpackHead(old_val);
        int sz = tail - old_h.index;
        if (sz <= MIN_STEAL) return false;

        // read before bumping
        T tmp = data[old_h.index];
        Head new_h{ old_h.index+1, old_h.ctr+1 };
        unsigned long long new_val = packHead(new_h.index, new_h.ctr);

        unsigned long long prev = atomicCAS((unsigned long long*)&a_head.value,
                                            old_val, new_val);
        if (prev == old_val) {
            __threadfence();
            out = tmp;
            return true;
        }
        return false;
    }
};
