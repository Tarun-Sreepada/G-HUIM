#pragma once
// Memory-ordered loads, stores, CAS and fences for the persistent scheduler, at cluster scope (deques in shared
// memory, reached by the cluster's blocks through DSMEM) or GPU scope (deques and counters in global memory).
// PTX on generic addresses: libcu++ in CUDA 12.8 has no cluster scope. Microbenchmark 15 measures both scopes.

#include <cstdint>

namespace ghuim::persistent {

enum class Scope
{
    kCluster,
    kGpu
};

// Thread-block clusters are sm_90+. Below that a cluster is one block, so its deque is reached only by its own
// block and CTA scope suffices.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#define GHUIM_CLUSTER_SCOPE "cta"
#else
#define GHUIM_CLUSTER_SCOPE "cluster"
#endif

#define GHUIM_SCOPED(body_cluster, body_gpu)                                                                           \
    if constexpr (S == Scope::kCluster)                                                                                \
        body_cluster;                                                                                                  \
    else                                                                                                               \
        body_gpu

template <Scope S = Scope::kGpu> __device__ inline uint64_t load_relaxed(const uint64_t* p)
{
    uint64_t v;
    GHUIM_SCOPED(asm volatile("ld.relaxed." GHUIM_CLUSTER_SCOPE ".u64 %0, [%1];"
                              : "=l"(v)
                              : "l"(p)
                              : "memory"),
                 asm volatile("ld.relaxed.gpu.u64 %0, [%1];"
                              : "=l"(v)
                              : "l"(p)
                              : "memory"));
    return v;
}

template <Scope S = Scope::kGpu> __device__ inline uint64_t load_acquire(const uint64_t* p)
{
    uint64_t v;
    GHUIM_SCOPED(asm volatile("ld.acquire." GHUIM_CLUSTER_SCOPE ".u64 %0, [%1];"
                              : "=l"(v)
                              : "l"(p)
                              : "memory"),
                 asm volatile("ld.acquire.gpu.u64 %0, [%1];"
                              : "=l"(v)
                              : "l"(p)
                              : "memory"));
    return v;
}

__device__ inline uint32_t load_relaxed(const uint32_t* p)
{
    uint32_t v;
    asm volatile("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ inline uint32_t load_acquire(const uint32_t* p)
{
    uint32_t v;
    asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

template <Scope S = Scope::kGpu> __device__ inline void store_relaxed(uint64_t* p, uint64_t v)
{
    GHUIM_SCOPED(asm volatile("st.relaxed." GHUIM_CLUSTER_SCOPE ".u64 [%0], %1;" ::"l"(p), "l"(v)
                              : "memory"),
                 asm volatile("st.relaxed.gpu.u64 [%0], %1;" ::"l"(p), "l"(v)
                              : "memory"));
}

template <Scope S = Scope::kGpu> __device__ inline void store_release(uint64_t* p, uint64_t v)
{
    GHUIM_SCOPED(asm volatile("st.release." GHUIM_CLUSTER_SCOPE ".u64 [%0], %1;" ::"l"(p), "l"(v)
                              : "memory"),
                 asm volatile("st.release.gpu.u64 [%0], %1;" ::"l"(p), "l"(v)
                              : "memory"));
}

// Returns the value seen; the swap happened iff it equals `expected`.
template <Scope S = Scope::kGpu>
__device__ inline uint64_t cas_acq_rel(uint64_t* p, uint64_t expected, uint64_t desired)
{
    uint64_t old;
    GHUIM_SCOPED(asm volatile("atom.acq_rel." GHUIM_CLUSTER_SCOPE ".cas.b64 %0, [%1], %2, %3;"
                              : "=l"(old)
                              : "l"(p), "l"(expected), "l"(desired)
                              : "memory"),
                 asm volatile("atom.acq_rel.gpu.cas.b64 %0, [%1], %2, %3;"
                              : "=l"(old)
                              : "l"(p), "l"(expected), "l"(desired)
                              : "memory"));
    return old;
}

// Acquire-release fence at GPU scope: loads after it see every write made visible to this thread's SM before
// (it also drops stale L1 lines, e.g. of a heap block reused since this SM last read it).
__device__ inline void fence_acq_rel_gpu()
{
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
}

__device__ inline uint32_t fetch_add_acq_rel(uint32_t* p, uint32_t v)
{
    uint32_t old;
    asm volatile("atom.acq_rel.gpu.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

template <Scope S = Scope::kGpu> __device__ inline void fence_sc()
{
    GHUIM_SCOPED(asm volatile("fence.sc." GHUIM_CLUSTER_SCOPE ";" ::: "memory"), asm volatile("fence.sc.gpu;" ::: "memory"));
}

#undef GHUIM_SCOPED
#undef GHUIM_CLUSTER_SCOPE

// Timing counters (cycle counts in BlockStats), compiled in with -DGHUIM_TIMERS=ON.
#ifdef GHUIM_TIMERS
constexpr bool kTimers = true;
#else
constexpr bool kTimers = false;
#endif

// clock64() with timers, else 0.
__device__ inline uint64_t timer()
{
    if constexpr (kTimers)
        return clock64();
    else
        return 0;
}

// Nanoseconds since an arbitrary epoch, the same on every SM.
__device__ inline uint64_t global_ns()
{
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

} // namespace ghuim::persistent
