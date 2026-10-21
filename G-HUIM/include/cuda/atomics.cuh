#pragma once
// Device atomics on plain integers, built on libcu++'s cuda::atomic_ref.

#include <cuda/atomic>

#include <cstdint>

namespace ghuim {

template <cuda::thread_scope Scope = cuda::thread_scope_device, typename T, typename V>
__device__ inline T fetch_add(T& target, V value)
{
    return cuda::atomic_ref<T, Scope>(target).fetch_add(static_cast<T>(value), cuda::std::memory_order_relaxed);
}

template <cuda::thread_scope Scope = cuda::thread_scope_device, typename T, typename V>
__device__ inline T fetch_sub(T& target, V value)
{
    return cuda::atomic_ref<T, Scope>(target).fetch_sub(static_cast<T>(value), cuda::std::memory_order_relaxed);
}

template <cuda::thread_scope Scope = cuda::thread_scope_device, typename T, typename V>
__device__ inline T fetch_max(T& target, V value)
{
    return cuda::atomic_ref<T, Scope>(target).fetch_max(static_cast<T>(value), cuda::std::memory_order_relaxed);
}

template <cuda::thread_scope Scope = cuda::thread_scope_device, typename T, typename V>
__device__ inline T fetch_min(T& target, V value)
{
    return cuda::atomic_ref<T, Scope>(target).fetch_min(static_cast<T>(value), cuda::std::memory_order_relaxed);
}

template <cuda::thread_scope Scope = cuda::thread_scope_device, typename T, typename V>
__device__ inline T fetch_or(T& target, V value)
{
    return cuda::atomic_ref<T, Scope>(target).fetch_or(static_cast<T>(value), cuda::std::memory_order_relaxed);
}

// Fire-and-forget add: nvcc turns an unused atomicAdd into a RED instruction.
// There is no 16-bit atomic add, so a 16-bit target adds into its half of the aligned 32-bit word. Only
// utilities are 16 bits wide and their sums always fit (mine/widths.hpp), so no carry leaves the half.
template <typename T, typename V> __device__ inline void add(T& target, V value)
{
    if constexpr (sizeof(T) == 8)
        atomicAdd(reinterpret_cast<unsigned long long*>(&target), static_cast<unsigned long long>(value));
    else if constexpr (sizeof(T) == 2) {
        const auto address = reinterpret_cast<uintptr_t>(&target);
        atomicAdd(reinterpret_cast<unsigned*>(address & ~uintptr_t{3}),
                  static_cast<unsigned>(static_cast<T>(value)) << (address & 2) * 8);
    }
    else
        atomicAdd(&target, static_cast<T>(value));
}

// Stores `desired` if `target` holds `expected`; returns the value seen (like atomicCAS).
template <cuda::thread_scope Scope = cuda::thread_scope_device, typename T>
__device__ inline T compare_swap(T& target, T expected, T desired)
{
    cuda::atomic_ref<T, Scope>(target).compare_exchange_strong(expected, desired, cuda::std::memory_order_relaxed);
    return expected;
}

} // namespace ghuim
