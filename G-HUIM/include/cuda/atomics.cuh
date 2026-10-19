#pragma once
// Device atomics on plain integers, built on libcu++'s cuda::atomic_ref.

#include <cuda/atomic>

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
template <typename T, typename V> __device__ inline void add(T& target, V value)
{
    if constexpr (sizeof(T) == 8)
        atomicAdd(reinterpret_cast<unsigned long long*>(&target), static_cast<unsigned long long>(value));
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
