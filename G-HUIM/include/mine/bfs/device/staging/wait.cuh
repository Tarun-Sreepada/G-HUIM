#pragma once
// Bounded waits on the ring counters, with optional statistics.

#include "mine/bfs/device/staging/ring.cuh"

namespace ghuim::bfs {

__device__ __noinline__ inline void stage_fail(const StageCfg& c, uint32_t error)
{
    atomicOr(&c.control->error, error);
    __trap();
}

__device__ __forceinline__ bool counter_reached(uint32_t* counter, uint32_t value)
{
    return Flag(*counter).load(cuda::std::memory_order_acquire) >= value;
}

// Spins until `*counter >= value`, within the bound.
__device__ __forceinline__ void stage_wait(const StageCfg& c, uint32_t* counter, uint32_t value,
                                           unsigned long long* wait_counter)
{
#ifdef GHUIM_STAGE_STATS
    const long long t0 = clock64();
#endif
    uint32_t spins = 0;
    while (!counter_reached(counter, value)) {
        if (++spins > kStageSpinLimit)
            stage_fail(c, kStageTimeout);
        if (spins > 32)
            __nanosleep(32);
    }
#ifdef GHUIM_STAGE_STATS
    if ((threadIdx.x & 31) == 0)
        add(*wait_counter, static_cast<unsigned long long>(clock64() - t0));
#else
    (void)wait_counter;
#endif
}

} // namespace ghuim::bfs
