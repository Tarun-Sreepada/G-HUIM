#pragma once
// A device heap for databases shared between blocks. Sizes are rounded up to a power-of-two class of 256-byte
// units; a freed block goes on its class's lock-free free list (a Treiber stack whose head carries a 24-bit tag
// against ABA) and is reused by the next allocation of that class. A class with an empty list carves new memory
// from a bump pointer. Blocks are never split or merged, so up to half of an allocation can be slack.

#include "mine/persistent/sync.cuh"

#include <cstdint>

namespace ghuim::persistent {

struct Heap
{
    static constexpr uint32_t kUnitBits  = 8;
    static constexpr uint32_t kClasses   = 40;
    static constexpr uint64_t kNull      = ~0ull;
    static constexpr uint64_t kIndexMask = (uint64_t{1} << 40) - 1; // head: tag << 40 | (unit index + 1)

    char*     base;
    uint64_t  capacity;
    uint64_t* bump;  // bytes carved so far
    uint64_t* heads; // kClasses free-list heads
    uint64_t* live;  // bytes allocated and not yet released (whole classes)

    // Whether `bytes` more would take the live heap past 3/4: optional uses (splits, offers) back off then, so
    // what remains is kept for databases a block cannot do without.
    __device__ bool tight(uint64_t bytes) const
    {
        return load_relaxed(live) + bytes > capacity / 4 * 3;
    }

    __host__ __device__ static uint32_t class_of(uint64_t bytes)
    {
        const uint64_t units = bytes == 0 ? 1 : (bytes + (uint64_t{1} << kUnitBits) - 1) >> kUnitBits;
        uint32_t       c     = 0;
        while ((uint64_t{1} << c) < units)
            ++c;
        return c;
    }

    __host__ __device__ static uint64_t class_bytes(uint32_t c)
    {
        return uint64_t{1} << (c + kUnitBits);
    }

    // Offset from base of a block of at least `bytes` (256-byte aligned), or kNull when the heap is exhausted.
    __device__ uint64_t allocate(uint64_t bytes, uint32_t& cls) const
    {
        cls = class_of(bytes);
        if (cls >= kClasses)
            return kNull;
        uint64_t* head = heads + cls;
        uint64_t  h    = load_acquire(head);
        while ((h & kIndexMask) != 0) {
            const uint64_t offset = ((h & kIndexMask) - 1) << kUnitBits;
            // May read a block another thread just took; the tag makes the CAS below fail in that case.
            const uint64_t next    = load_relaxed(reinterpret_cast<const uint64_t*>(base + offset));
            const uint64_t desired = (((h >> 40) + 1) << 40) | (next & kIndexMask);
            const uint64_t seen    = cas_acq_rel(head, h, desired);
            if (seen == h) {
                atomicAdd(reinterpret_cast<unsigned long long*>(live), class_bytes(cls));
                return offset;
            }
            h = seen;
        }
        const uint64_t size   = class_bytes(cls);
        const uint64_t offset = atomicAdd(reinterpret_cast<unsigned long long*>(bump), size);
        if (offset + size > capacity)
            return kNull;
        atomicAdd(reinterpret_cast<unsigned long long*>(live), size);
        return offset;
    }

    __device__ void release(uint64_t offset, uint32_t cls) const
    {
        atomicAdd(reinterpret_cast<unsigned long long*>(live), ~class_bytes(cls) + 1); // -= class size
        uint64_t* head = heads + cls;
        uint64_t  h    = load_acquire(head);
        for (;;) {
            store_relaxed(reinterpret_cast<uint64_t*>(base + offset), h & kIndexMask);
            const uint64_t desired = (((h >> 40) + 1) << 40) | ((offset >> kUnitBits) + 1);
            const uint64_t seen    = cas_acq_rel(head, h, desired); // release: the link is written first
            if (seen == h)
                return;
            h = seen;
        }
    }
};

} // namespace ghuim::persistent
