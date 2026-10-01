#pragma once
// Device helpers shared by the mining kernels.

#include "mine/kernels.cuh"
#include "mine/layout.cuh"

#include <cuda/atomic>
#include <cuda/std/limits>

namespace ghuim {

constexpr uint32_t kNone = cuda::std::numeric_limits<uint32_t>::max();

// Relaxed atomic read-modify-writes through libcu++ (same ordering as the CUDA intrinsics).
// Scope defaults to the device; block-local counters pass cuda::thread_scope_block.
template <cuda::thread_scope Scope = cuda::thread_scope_device, typename T, typename V>
__device__ inline T fetch_add(T& target, V value)
{
    return cuda::atomic_ref<T, Scope>(target).fetch_add(static_cast<T>(value), cuda::std::memory_order_relaxed);
}

template <cuda::thread_scope Scope = cuda::thread_scope_device, typename T, typename V>
__device__ inline T fetch_max(T& target, V value)
{
    return cuda::atomic_ref<T, Scope>(target).fetch_max(static_cast<T>(value), cuda::std::memory_order_relaxed);
}

template <cuda::thread_scope Scope = cuda::thread_scope_device, typename T, typename V>
__device__ inline T fetch_or(T& target, V value)
{
    return cuda::atomic_ref<T, Scope>(target).fetch_or(static_cast<T>(value), cuda::std::memory_order_relaxed);
}

// Fire-and-forget add, for counters whose old value is not needed. Uses the intrinsic because
// nvcc turns an unused atomicAdd into a RED instruction (no round trip) and may aggregate it
// across the warp; libcu++ atomics are inline PTX and always pay for a full ATOM.
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

// Position of `item` in a sorted transaction, or -1.
__device__ inline int find_item(const Entry* entries, uint32_t length, uint32_t item)
{
    int low = 0, high = static_cast<int>(length) - 1;
    while (low <= high) {
        const int      mid = (low + high) >> 1;
        const uint32_t key = entries[mid].item;
        if (key == item)
            return mid;
        if (key < item)
            low = mid + 1;
        else
            high = mid - 1;
    }
    return -1;
}

// Whether two parent transactions project to the same kept items after `item`.
__device__ inline bool same_kept_items(Tx a, const Entry* entries, Tx b, uint32_t item, const uint32_t* rename)
{
    const Entry* ea = entries + a.begin;
    const Entry* eb = entries + b.begin;
    uint32_t     i  = find_item(ea, a.length, item) + 1;
    uint32_t     j  = find_item(eb, b.length, item) + 1;
    for (;;) {
        while (i < a.length && rename[ea[i].item] == kNone)
            ++i;
        while (j < b.length && rename[eb[j].item] == kNone)
            ++j;
        if (i == a.length || j == b.length)
            return i == a.length && j == b.length;
        if (ea[i].item != eb[j].item)
            return false;
        ++i;
        ++j;
    }
}

// Records a finished (not deferred) task and whether its itemset is high-utility.
__device__ inline void finish_task(const BatchView& b, const Task& task, bool high_utility)
{
    const uint32_t depth = task.depth < kMaxDepth ? task.depth : kMaxDepth - 1;
    add(b.control->depth_tasks[depth], 1ull);
    if (high_utility) {
        add(b.control->patterns, 1ull);
        add(b.control->depth_patterns[depth], 1ull);
    }
}

__device__ inline void push_task(const BatchView& b, const Task& task)
{
    const uint32_t slot = fetch_add(b.control->stack_top, 1u);
    if (slot < b.stack_capacity)
        b.stack[slot] = task;
    else
        fetch_or(b.control->error, kStackExhausted);
}

} // namespace ghuim
