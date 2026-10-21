// 15: cost of a work-stealing operation inside a thread-block cluster (DSMEM) versus through global memory (L2).
// Each block owns a deque header {top, bottom} and a ring of 16-byte tasks, either in its shared memory or in
// global memory. Thread 0 of every block runs `ops` operations back to back; the other threads stay idle.
//   cas_chain : dependent CAS on the victim's `top` (pure atomic latency)
//   steal     : a Chase-Lev steal: acquire-load bottom, load top, load the 16-byte task, CAS top
// Ordering: `acq_rel` (acquire loads, acq_rel CAS, as a Chase-Lev steal needs) or `relaxed` (no fences).
// Victims: `own` = the block's own header (local smem or its own global line), `peer` = the next rank of the
// cluster (one thief per victim, no contention), `hot` = every rank steals from rank 0 (cluster-size thieves
// contend on one header). Global victims are laid out like the shared ones, so only the memory path differs.
// Blocks reserve enough dynamic shared memory to be alone on their SM, as in the persistent miner. `load` = one
// (a single cluster: latency) or all (every SM runs a cluster at once: latency under device-wide traffic).
// Prints ns per successful operation of one thread, from the kernel time.

#include "bench.cuh"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

constexpr int      kBlock = 128;
constexpr uint32_t kRing  = 256;        // tasks per deque
constexpr int      kSmem  = 160 * 1024; // dynamic shared memory per block: one block per SM

struct alignas(16) Task
{
    u64 a, b;
};

struct alignas(128) Header
{ // one line per header, so global victims do not share lines
    u64 top, bottom;
};

enum Op
{
    kCasChain,
    kSteal
};
enum Victim
{
    kOwn,
    kPeer,
    kHot
};
enum Space
{
    kShared,
    kGlobal
};

// Loads and CAS at cluster or GPU scope, with acquire / acq_rel ordering or relaxed (no fences). libcu++ in CUDA 12.8
// has no cluster scope, so this is PTX on generic addresses (which reach the shared memory of other blocks in the
// cluster as well as global memory).
enum class Scope
{
    kCluster,
    kGpu
};

#define STEAL_LOAD(sem, scope) asm volatile("ld." sem "." scope ".u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory")
#define STEAL_CAS(sem, scope)                                                                                          \
    asm volatile("atom." sem "." scope ".cas.b64 %0, [%1], %2, %3;"                                                    \
                 : "=l"(old)                                                                                           \
                 : "l"(p), "l"(expected), "l"(desired)                                                                 \
                 : "memory")

template <Scope S, bool Relaxed> __device__ inline u64 load(const u64* p)
{
    u64 v;
    if constexpr (S == Scope::kCluster && Relaxed)
        STEAL_LOAD("relaxed", "cluster");
    else if constexpr (S == Scope::kCluster)
        STEAL_LOAD("acquire", "cluster");
    else if constexpr (Relaxed)
        STEAL_LOAD("relaxed", "gpu");
    else
        STEAL_LOAD("acquire", "gpu");
    return v;
}

template <Scope S, bool Relaxed> __device__ inline u64 cas(u64* p, u64 expected, u64 desired)
{
    u64 old;
    if constexpr (S == Scope::kCluster && Relaxed)
        STEAL_CAS("relaxed", "cluster");
    else if constexpr (S == Scope::kCluster)
        STEAL_CAS("acq_rel", "cluster");
    else if constexpr (Relaxed)
        STEAL_CAS("relaxed", "gpu");
    else
        STEAL_CAS("acq_rel", "gpu");
    return old;
}

#undef STEAL_LOAD
#undef STEAL_CAS

// One steal attempt: returns true and advances `top` on success. A failed CAS (contention) is retried by the caller.
template <Scope S, bool Relaxed> __device__ inline bool steal_once(Header* h, const Task* ring, u64& sink)
{
    const u64 b = load<S, Relaxed>(&h->bottom);
    const u64 t = load<S, Relaxed>(&h->top);
    if (t >= b)
        return false; // never happens: bottom is set far beyond the ops count
    const Task task = ring[t % kRing];
    if (cas<S, Relaxed>(&h->top, t, t + 1) != t)
        return false;
    sink += task.a ^ task.b;
    return true;
}

template <Scope S, bool Relaxed> __device__ inline bool cas_once(Header* h)
{
    const u64 t = load<S, Relaxed>(&h->top);
    return cas<S, Relaxed>(&h->top, t, t + 1) == t;
}

// Runs `ops` successful operations on (h, ring); returns the number of failed attempts.
template <Scope S, bool Relaxed>
__device__ uint32_t run_ops(Op op, Header* h, const Task* ring, uint32_t ops, u64& sink)
{
    uint32_t failed = 0;
    for (uint32_t done = 0; done < ops;) {
        const bool ok = op == kSteal ? steal_once<S, Relaxed>(h, ring, sink) : cas_once<S, Relaxed>(h);
        if (ok)
            ++done;
        else
            ++failed;
    }
    return failed;
}

__global__ void steal_kernel(Op op, bool relaxed, Victim victim, Space space, uint32_t ops, Header* g_headers,
                             Task* g_rings, u64* sink_out, unsigned long long* failed_out)
{
    __shared__ Header s_header;
    __shared__ Task   s_ring[kRing];
    cg::cluster_group cluster = cg::this_cluster();
    const uint32_t    rank    = cluster.block_rank();
    const uint32_t    size    = cluster.num_blocks();
    const uint32_t    bid     = blockIdx.x;
    const uint32_t    base    = bid - rank; // first block of this cluster

    for (uint32_t i = threadIdx.x; i < kRing; i += blockDim.x)
        s_ring[i] = {hash64(bid * kRing + i), i};
    if (threadIdx.x == 0)
        s_header = {0, ~0ull >> 1};
    cluster.sync(); // every header is initialised before any thief touches it

    const uint32_t target = victim == kOwn ? rank : victim == kPeer ? (rank + 1) % size : 0;
    u64            sink   = 0;
    uint32_t       failed = 0;
    if (threadIdx.x == 0) {
        if (space == kShared) {
            Header*     h    = cluster.map_shared_rank(&s_header, target);
            const Task* ring = cluster.map_shared_rank(s_ring, target);
            failed           = relaxed ? run_ops<Scope::kCluster, true>(op, h, ring, ops, sink)
                                       : run_ops<Scope::kCluster, false>(op, h, ring, ops, sink);
        }
        else {
            Header*     h    = g_headers + base + target;
            const Task* ring = g_rings + (size_t)(base + target) * kRing;
            failed           = relaxed ? run_ops<Scope::kGpu, true>(op, h, ring, ops, sink)
                                       : run_ops<Scope::kGpu, false>(op, h, ring, ops, sink);
        }
        sink_out[bid] = sink;
        atomicAdd(failed_out, (unsigned long long)failed);
    }
    cluster.sync(); // no block leaves while a peer may still read its shared memory
}

__global__ void init_global(Header* headers, Task* rings, uint32_t blocks)
{
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < blocks * kRing; i += gridDim.x * blockDim.x) {
        rings[i] = {hash64(i), i % kRing};
        if (i < blocks)
            headers[i] = {0, ~0ull >> 1};
    }
}

int main()
{
    print_device();
    CUDA_CHECK(cudaFuncSetAttribute(steal_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    CUDA_CHECK(cudaFuncSetAttribute(steal_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    std::printf("cluster,clusters,load,op,order,victim,space,ops_per_block,ms,ns_per_op,failed_per_op\n");

    const char* op_name[]     = {"cas_chain", "steal"};
    const char* victim_name[] = {"own", "peer", "hot"};
    const char* space_name[]  = {"dsmem", "global"};

    for (int cs : {1, 2, 4, 8, 16}) {
        cudaLaunchConfig_t  cfg{};
        cudaLaunchAttribute attr{};
        attr.id               = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim.x = cs;
        attr.val.clusterDim.y = 1;
        attr.val.clusterDim.z = 1;
        cfg.blockDim          = dim3(kBlock);
        cfg.dynamicSmemBytes  = kSmem;
        cfg.attrs             = &attr;
        cfg.numAttrs          = 1;
        cfg.gridDim           = dim3(cs);
        int clusters          = 0;
        if (cudaOccupancyMaxActiveClusters(&clusters, steal_kernel, &cfg) != cudaSuccess || clusters == 0) {
            cudaGetLastError();
            std::fprintf(stderr, "# cluster size %d: not launchable, skipped\n", cs);
            continue;
        }
        std::fprintf(stderr, "# cluster size %d: %d clusters resident\n", cs, clusters);
        const int      max_clusters = clusters;
        const uint32_t max_blocks   = max_clusters * cs;

        DeviceBuffer<Header>             headers(max_blocks);
        DeviceBuffer<Task>               rings((size_t)max_blocks * kRing);
        DeviceBuffer<u64>                sink(max_blocks);
        DeviceBuffer<unsigned long long> failed(1);

        for (int load : {0, 1})
            for (Op op : {kCasChain, kSteal})
                for (bool relaxed : {false, true})
                    for (Victim victim : {kOwn, kPeer, kHot})
                        for (Space space : {kShared, kGlobal}) {
                            if (cs == 1 && victim != kOwn)
                                continue;
                            const int      clusters = load ? max_clusters : 1;
                            const uint32_t blocks   = clusters * cs;
                            cfg.gridDim             = dim3(blocks);
                            const uint32_t ops      = victim == kHot ? 2000 : 20000;
                            auto           launch   = [&] {
                                init_global<<<256, 256>>>(headers.get(), rings.get(), blocks);
                                CUDA_CHECK(cudaLaunchKernelEx(&cfg, steal_kernel, op, relaxed, victim, space, ops,
                                                                          headers.get(), rings.get(), sink.get(),
                                                                          (unsigned long long*)failed.get()));
                            };
                            // Time only the steal kernel: subtract a run of the init kernel alone.
                            const float init_ms =
                                    time_ms([&] { init_global<<<256, 256>>>(headers.get(), rings.get(), blocks); });
                            const float ms = time_ms(launch) - init_ms;
                            failed.zero();
                            launch();
                            CUDA_CHECK(cudaDeviceSynchronize());
                            const double fails = (double)read_scalar(failed.get()) / ((double)ops * blocks);
                            // Per-thread latency: every block's thread 0 runs `ops` operations concurrently.
                            std::printf("%d,%d,%s,%s,%s,%s,%s,%u,%.4f,%.1f,%.3f\n", cs, clusters, load ? "all" : "one",
                                        op_name[op], relaxed ? "relaxed" : "acq_rel", victim_name[victim],
                                        space_name[space], ops, ms, ms * 1e6 / ops, fails);
                        }
    }
    return 0;
}
