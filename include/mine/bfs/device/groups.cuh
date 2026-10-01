#pragma once
// Thread groups: the lanes (a sub-warp group, a warp, or a block) that process one task together.

#include "mine/primitives.cuh"

namespace ghuim::bfs {

// ---------------------------------------------------------------------------------------------
// Thread groups. Every group exposes the lanes of its warp that work together (`lanes()`, a power
// of two <= 32, and their `mask()`); collectives are over those lanes only, so several groups can
// share a warp and finish at different times.
// ---------------------------------------------------------------------------------------------

struct WarpGroup
{
    __device__ uint32_t rank() const
    {
        return threadIdx.x & 31;
    }
    __device__ uint32_t size() const
    {
        return 32;
    }
    __device__ uint32_t lanes() const
    {
        return 32;
    }
    __device__ unsigned mask() const
    {
        return kFullMask;
    }
    __device__ void sync() const
    {
        __syncwarp();
    }
    __device__ uint32_t scan(uint32_t value) const
    {
        return warp_inclusive_scan(value);
    }
    __device__ uint32_t count(bool flag) const
    {
        return __popc(__ballot_sync(kFullMask, flag));
    }
};

// `lanes` consecutive lanes of a warp (a power of two) form a group.
struct SubGroup
{
    uint32_t n;
    __device__ explicit SubGroup(uint32_t lanes) : n(lanes) {}
    __device__ uint32_t rank() const
    {
        return threadIdx.x & (n - 1);
    }
    __device__ uint32_t size() const
    {
        return n;
    }
    __device__ uint32_t lanes() const
    {
        return n;
    }
    __device__ unsigned mask() const
    {
        return n == 32 ? kFullMask : ((1u << n) - 1) << (threadIdx.x & 31 & ~(n - 1));
    }
    __device__ void sync() const
    {
        __syncwarp(mask());
    }
    __device__ uint32_t scan(uint32_t value) const
    {
        for (uint32_t offset = 1; offset < n; offset <<= 1) {
            const uint32_t other = __shfl_up_sync(mask(), value, offset, n);
            if (rank() >= offset)
                value += other;
        }
        return value;
    }
    __device__ uint32_t count(bool flag) const
    {
        return __popc(__ballot_sync(mask(), flag));
    }
};

struct BlockGroup
{
    __device__ uint32_t rank() const
    {
        return threadIdx.x;
    }
    __device__ uint32_t size() const
    {
        return blockDim.x;
    }
    __device__ uint32_t lanes() const
    {
        return 32;
    }
    __device__ unsigned mask() const
    {
        return kFullMask;
    }
    __device__ void sync() const
    {
        __syncthreads();
    }
    __device__ uint32_t scan(uint32_t value) const
    {
        __shared__ uint32_t carry[32];
        return block_inclusive_scan<uint32_t>(value, carry);
    }
    __device__ uint32_t count(bool flag) const
    {
        return static_cast<uint32_t>(__syncthreads_count(flag));
    }
};

// The warp lanes of a group: lane `rank` handles transactions rank, rank + step, ... Loops bounded
// by `max` run in lockstep, and every lane of the group must call a reduction.
struct Lanes
{
    uint32_t count, rank;
    unsigned mask;

    template <typename Group>
    __device__ explicit Lanes(const Group& g)
        : count(g.lanes()), rank((threadIdx.x & 31) & (g.lanes() - 1)), mask(g.mask())
    {
    }
    __device__ uint32_t max(uint32_t value) const
    {
        return count == 1 ? value : __reduce_max_sync(mask, value);
    }
    template <typename T> __device__ T sum(T value) const
    {
        for (uint32_t offset = count >> 1; offset > 0; offset >>= 1)
            value += __shfl_xor_sync(mask, value, offset, count);
        return value;
    }
    __device__ bool leader() const
    {
        return rank == 0;
    }
};

} // namespace ghuim::bfs
