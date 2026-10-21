#pragma once
// A Chase-Lev work-stealing deque. The owner (thread 0 of its block) pushes and pops at the bottom (LIFO: depth
// first, warm data); thieves take from the top (FIFO: the shallowest, largest subtrees). Orderings follow Le et al.,
// "Correct and Efficient Work-Stealing for Weak Memory Models" (PPoPP 2013), at the scope of the thieves:
// Scope::kCluster for a deque in shared memory (thieves are the cluster's blocks, through DSMEM), Scope::kGpu for
// one in global memory. The ring has a fixed capacity: push fails when it is full and the caller decides.

#include "mine/persistent/sync.cuh"

namespace ghuim::persistent {

struct DequeHeader
{
    uint64_t top;
    uint64_t bottom;
};

// Global deques get a line each: thieves CAS `top` while the owner writes `bottom`.
struct alignas(128) PaddedDequeHeader : DequeHeader
{
};

template <typename Task, Scope S> struct Deque
{
    DequeHeader* header;
    Task*        ring;
    uint32_t     capacity;

    __device__ uint64_t size() const
    {
        const uint64_t b = load_relaxed<S>(&header->bottom), t = load_relaxed<S>(&header->top);
        return b > t ? b - t : 0;
    }

    // Owner only.
    __device__ bool push(const Task& task) const
    {
        const uint64_t b = load_relaxed<S>(&header->bottom);
        const uint64_t t = load_acquire<S>(&header->top);
        if (b - t >= capacity)
            return false;
        ring[b % capacity] = task;
        store_release<S>(&header->bottom, b + 1); // the task is visible before the slot is claimable
        return true;
    }

    // Owner only.
    __device__ bool pop(Task& task) const
    {
        const uint64_t b = load_relaxed<S>(&header->bottom);
        if (b == load_relaxed<S>(&header->top))
            return false; // empty: skip the fence
        store_relaxed<S>(&header->bottom, b - 1);
        fence_sc<S>(); // the bottom store is ordered before the top load, against a thief's steal
        const uint64_t t = load_relaxed<S>(&header->top);
        if (t > b - 1) {
            store_relaxed<S>(&header->bottom, b);
            return false;
        }
        task = ring[(b - 1) % capacity];
        if (t == b - 1) { // last task: race the thieves for it
            const bool won = cas_acq_rel<S>(&header->top, t, t + 1) == t;
            store_relaxed<S>(&header->bottom, b);
            return won;
        }
        return true;
    }

    // Any block in scope, the owner included. False when empty or another thief (or the owner) won the race.
    __device__ bool steal(Task& task) const
    {
        const uint64_t t = load_acquire<S>(&header->top);
        fence_sc<S>();
        const uint64_t b = load_acquire<S>(&header->bottom);
        if (t >= b)
            return false;
        task = ring[t % capacity];
        return cas_acq_rel<S>(&header->top, t, t + 1) == t;
    }
};

} // namespace ghuim::persistent
