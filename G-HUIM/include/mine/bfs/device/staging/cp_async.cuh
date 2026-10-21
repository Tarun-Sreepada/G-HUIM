#pragma once
// cp.async (sm_80+) wrappers.

#include "mine/widths.hpp"

#include <cstdint>


namespace ghuim::inline GHUIM_WIDTHS::bfs {

__device__ __forceinline__ uint32_t smem_address(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async_4(void* dst, const void* src)
{
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem_address(dst)), "l"(src) : "memory");
}

__device__ __forceinline__ void cp_async_8(void* dst, const void* src)
{
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(smem_address(dst)), "l"(src) : "memory");
}

// Copies one value: cp.async for 4 and 8 bytes; a narrower value is copied by a plain load and store,
// which the tile's publish (wait, __syncwarp, release) orders like the asynchronous copies.
template <typename T> __device__ __forceinline__ void stage_copy(T* dst, const T* src)
{
    if constexpr (sizeof(T) == 8)
        cp_async_8(dst, src);
    else if constexpr (sizeof(T) == 4)
        cp_async_4(dst, src);
    else
        *dst = *src;
}

__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;" ::: "memory");
}

// Waits until at most N of this thread's most recent groups are still in flight.
template <int N> __device__ __forceinline__ void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
