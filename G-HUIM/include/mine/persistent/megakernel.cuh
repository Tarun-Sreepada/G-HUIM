#pragma once
// The persistent megakernel: one launch, one worker block per resident slot, run until no task is left.
//
// Phase 1 (wide): while the frontier has too few tasks for the grid, or a task too large for one block, the
// whole grid runs each frontier level together (Work::wide) and grid-syncs. Phase 2 (steal): the frontier is
// dealt round-robin to the blocks, which run their tasks depth first and steal when they run dry. A global
// `pending` count of live tasks ends the run; a %globaltimer deadline aborts it.
//
// Blocks are grouped into thread-block clusters (REFIM's NUMA nodes). Each block has two deques:
//   local  : in its shared memory; the owner pushes here first, and the cluster's blocks steal through DSMEM.
//   global : in global memory; takes the local deque's overflow, and the tasks exported for idle blocks
//            elsewhere: whenever some block is idle and the own global deque is empty, the owner moves the oldest
//            (shallowest) half of its local deque there. Any block can steal from it.
// A block looking for work tries its own deques, then every cluster peer's local deque, then random global
// deques. local_capacity = 0 turns the local deques off (all tasks go through global memory).
//
// Termination: a task's spawned children are added to `pending` before they become stealable, and a finished
// task's -1 is held back by its block (`retired`) until the block next spawns or runs dry, so `pending` never
// drops below the number of live tasks and reaches 0 only when there are none.
//
// A Work type provides:
//   using Task = ...;  // trivially copyable, small
//   static constexpr int kMaxThreads;  // block size the kernel is compiled for (launch bounds)
//   __device__ void wide(cg::grid_group&, const WideLevel<Task>&, BlockStats&) const;
//       Grid-cooperative, one level: expands frontier tasks (all of them when need_tasks, else at least those
//       larger than grain), appends their children and the tasks it did not expand to next, and atomicMax-es
//       the largest size of the next frontier. It must not append more than next_capacity tasks.
//   __device__ void init_block() const;  // every thread of a block, once, before its first task
//   __device__ static uint64_t describe(const Task&);  // a note on a task, logged for the longest one
//   __device__ void run(const Task&, const Spawn<Task>&, BlockStats&) const;
//       Block-cooperative: runs the task's subtree, handing parts to other blocks with spawn.push (thread 0).

#include "mine/persistent/control.cuh"
#include "mine/persistent/deque.cuh"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace ghuim::persistent {

// The block's thread-block cluster. Clusters are sm_90+; below that every block is a cluster of one (no DSMEM).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
struct Cluster
{
    __device__ uint32_t num_blocks() const { return 1; }
    __device__ uint32_t block_rank() const { return 0; }
    __device__ void     sync() const { __syncthreads(); }
    template <typename T> __device__ T* map_shared_rank(T* p, uint32_t) const { return p; }
};
__device__ inline Cluster this_cluster() { return {}; }
#else
using Cluster = cooperative_groups::cluster_group;
__device__ inline Cluster this_cluster() { return cooperative_groups::this_cluster(); }
#endif

// Host: whether the current device runs thread-block clusters (sm_90+); without them a cluster is one block.
inline bool clusters_supported()
{
    int device = 0, major = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess)
        return false;
    return major >= 9;
}

// Host: the cluster size to launch with on the current device: `want`, or 1 where clusters are unsupported.
inline uint32_t supported_cluster_size(uint32_t want)
{
    return clusters_supported() ? want : 1;
}

namespace cg = cooperative_groups;

constexpr uint32_t kStealTries   = 4; // random global victims tried per round before backing off
constexpr uint32_t kMinBackoffNs = 128;
constexpr uint32_t kMaxBackoffNs = 16 * 1024;

template <typename Work> struct Params
{
    using Task = typename Work::Task;
    Work               work;
    Control*           control;
    PaddedDequeHeader* headers; // global deques, one per block
    Task*              rings;   // global_capacity per block
    uint32_t           global_capacity;
    uint32_t           local_capacity; // tasks in each block's shared-memory deque (0 = none)
    Task*              frontier[2];
    uint32_t           frontier_capacity;
    uint32_t           target_tasks;    // the wide phase ends once the frontier has this many tasks ...
    uint64_t           grain;           // ... and none is larger than this,
    uint32_t           max_wide_levels; // or after this many levels, or when a level changes nothing
    BlockStats*        stats;
};

// One level of the wide phase, as Work::wide sees it.
template <typename Task> struct WideLevel
{
    const Task* frontier;
    uint32_t    n;
    Task*       next;
    uint32_t*   next_n;
    uint32_t    next_capacity;
    uint64_t*   largest;
    bool        need_tasks; // fewer than target_tasks: expand every task
    uint64_t    grain;
};

// Dynamic shared memory of the megakernel: the local deque's header, then its ring.
template <typename Task> __host__ __device__ constexpr size_t local_deque_bytes(uint32_t capacity)
{
    return capacity == 0
                   ? 0
                   : (sizeof(DequeHeader) + sizeof(Task) - 1) / sizeof(Task) * sizeof(Task) + capacity * sizeof(Task);
}

__device__ inline bool aborted(const Control* control)
{
    return load_acquire(&control->abort) != 0;
}

__device__ inline void raise_abort(Control* control, Abort reason)
{
    atomicCAS(&control->abort, 0u, static_cast<uint32_t>(reason));
}

__device__ inline uint32_t next_random(uint32_t& rng)
{
    rng ^= rng << 13, rng ^= rng >> 17, rng ^= rng << 5;
    return rng;
}

// Thread 0's view of the deques: its own two, and how to reach everyone else's.
template <typename Task> struct Deques
{
    Cluster                      cluster;
    Deque<Task, Scope::kCluster> local; // capacity 0: no local deque
    Deque<Task, Scope::kGpu>     global;
    PaddedDequeHeader*           headers;
    Task*                        rings;
    uint32_t                     global_capacity;

    __device__ Deque<Task, Scope::kGpu> global_of(uint32_t block) const
    {
        return {headers + block, rings + static_cast<size_t>(block) * global_capacity, global_capacity};
    }

    __device__ Deque<Task, Scope::kCluster> local_of(uint32_t rank) const
    {
        return {cluster.map_shared_rank(local.header, rank), cluster.map_shared_rank(local.ring, rank), local.capacity};
    }

    __device__ bool push(Control* control, const Task& task, BlockStats& stats) const
    {
        if (local.capacity != 0 && local.push(task)) {
            stats.max_local = max(stats.max_local, local.size());
            return true;
        }
        if (global.push(task)) {
            stats.max_global = max(stats.max_global, global.size());
            return true;
        }
        raise_abort(control, Abort::kOverflow);
        return false;
    }

    __device__ bool pop(Task& task) const
    {
        return (local.capacity != 0 && local.pop(task)) || global.pop(task);
    }

    // Some block is idle and nothing of ours is visible outside the cluster: publish the oldest half of the
    // local deque. The owner steals from its own top, racing the cluster's thieves like any of them.
    __device__ void export_for_idle(const Control* control, BlockStats& stats) const
    {
        if (local.capacity == 0 || load_acquire(&control->idle) == 0 || global.size() != 0)
            return;
        Task task;
        for (uint64_t k = local.size() / 2; k > 0 && local.steal(task); --k) {
            if (!global.push(task)) { // cannot happen: the global deque was empty and k <= local capacity
                local.push(task);
                return;
            }
            ++stats.exports;
        }
    }
};

__device__ inline void add_pending(Control* control, int64_t delta)
{
    if (delta != 0)
        atomicAdd(reinterpret_cast<unsigned long long*>(&control->pending), static_cast<unsigned long long>(delta));
}

// What Work::run uses to hand tasks to other blocks. Thread 0 only.
template <typename Task> struct Spawn
{
    Control*            control;
    const Deques<Task>* deques;
    BlockStats*         stats;
    uint32_t*           retired; // tasks this block finished and has not yet subtracted from pending

    // Pushes tasks[n-1] first, so tasks[0] is the next one this block pops.
    __device__ void push(const Task* tasks, uint32_t n) const
    {
        if (n == 0)
            return;
        add_pending(control, static_cast<int64_t>(n) - *retired); // before any of them can be stolen
        *retired = 0;
        for (uint32_t i = n; i-- > 0;)
            deques->push(control, tasks[i], *stats);
    }

    // Tasks that can be pushed now without overflowing this block's deques.
    __device__ uint32_t room() const
    {
        const uint64_t lc = deques->local.capacity, gc = deques->global.capacity;
        const uint64_t ls = deques->local.size(), gs = deques->global.size();
        const uint64_t free = (lc > ls ? lc - ls : 0) + (gc > gs ? gc - gs : 0);
        return static_cast<uint32_t>(free < 0xffffffffull ? free : 0xffffffffull);
    }

    // Blocks looking for work (relaxed: a hint, polled often).
    __device__ uint32_t idle() const
    {
        return load_relaxed(&control->idle);
    }

    // Some block is looking for work (REFIM's "hungry"). Relaxed: a hint, polled often.
    __device__ bool hungry() const
    {
        return load_relaxed(&control->idle) != 0;
    }
};

// Thread 0: the next task for this block, from its own deques or a victim's. False: the run is over.
template <typename Task>
__device__ bool find_task(Control* control, const Deques<Task>& d, uint32_t& rng, uint32_t& retired, Task& task,
                          BlockStats& stats)
{
    if (aborted(control))
        return false;
    if (d.pop(task))
        return true;
    add_pending(control, -static_cast<int64_t>(retired)); // before anyone could see pending == 0 because of us
    retired                = 0;
    const uint32_t blocks  = gridDim.x;
    const uint32_t size    = d.cluster.num_blocks();
    const uint32_t rank    = d.cluster.block_rank();
    const uint32_t first   = blockIdx.x - rank; // of this cluster
    const uint64_t start   = global_ns();
    bool           idle    = false;
    uint32_t       backoff = kMinBackoffNs;
    auto           found   = [&](uint64_t& counter) {
        ++counter;
        if (idle)
            atomicSub(&control->idle, 1u);
        stats.idle_ns += global_ns() - start;
        return true;
    };
    for (;;) {
        // Cluster peers through DSMEM, every one, starting at a random rank.
        if (d.local.capacity != 0 && size > 1) {
            const uint32_t offset = next_random(rng) % (size - 1);
            for (uint32_t k = 0; k < size - 1; ++k) {
                const uint32_t victim = (rank + 1 + (offset + k) % (size - 1)) % size;
                if (d.local_of(victim).steal(task))
                    return found(stats.steals_dsmem);
                ++stats.steal_misses;
            }
        }
        // Global deques of random blocks anywhere.
        if (blocks > 1)
            for (uint32_t k = 0; k < kStealTries; ++k) {
                const uint32_t victim = (blockIdx.x + 1 + next_random(rng) % (blocks - 1)) % blocks;
                if (d.global_of(victim).steal(task))
                    return found(victim - first < size ? stats.steals_near : stats.steals_far);
                ++stats.steal_misses;
            }
        if (load_acquire(&control->pending) == 0 || aborted(control))
            break;
        const uint64_t deadline = control->deadline_ns;
        if (deadline != 0 && global_ns() > deadline) {
            raise_abort(control, Abort::kTimeout);
            break;
        }
        if (!idle) {
            atomicAdd(&control->idle, 1u);
            idle = true;
        }
        __nanosleep(backoff);
        backoff = min(2 * backoff, kMaxBackoffNs);
    }
    stats.idle_ns += global_ns() - start;
    return false;
}

// One block per SM: let the compiler use registers up to what Work::kMaxThreads threads leave.
template <typename Work> __global__ void __launch_bounds__(Work::kMaxThreads, 1) megakernel(Params<Work> p)
{
    using Task                = typename Work::Task;
    cg::grid_group    grid    = cg::this_grid();
    Control*          control = p.control;
    extern __shared__ __align__(16) unsigned char local_memory[];
    __shared__ BlockStats                         stats;
    __shared__ Task                               task;
    __shared__ bool                               running;
    __shared__ uint32_t                           retired;

    const uint64_t     c_start = timer();
    const auto         add_cycles = [](uint64_t& counter, uint64_t start) { // lane 0 of each warp
        if constexpr (kTimers)
            if ((threadIdx.x & 31) == 0)
                atomicAdd(reinterpret_cast<unsigned long long*>(&counter),
                          static_cast<unsigned long long>(clock64() - start));
    };
    const Deques<Task> d{
            this_cluster(),
            {reinterpret_cast<DequeHeader*>(local_memory),
             reinterpret_cast<Task*>(local_memory + local_deque_bytes<Task>(p.local_capacity) -
                                     p.local_capacity * sizeof(Task)),
             p.local_capacity},
            {p.headers + blockIdx.x, p.rings + static_cast<size_t>(blockIdx.x) * p.global_capacity, p.global_capacity},
            p.headers,
            p.rings,
            p.global_capacity};
    if (threadIdx.x == 0) {
        stats   = {};
        retired = 0;
        if (p.local_capacity != 0)
            *d.local.header = {0, 0};
    }
    p.work.init_block(); // every thread: the Work's per-block state (e.g. its shared memory)
    d.cluster.sync();    // every peer has started and initialised its deque before any DSMEM access

    // Phase 1: the whole grid runs each level until the frontier decomposes into enough grain-sized tasks.
    // Every block reads the same counts after a grid sync, so all take the same branch.
    uint32_t cur = 0, prev_n = ~0u;
    uint64_t prev_largest = ~0ull;
    for (uint32_t level = 0;; ++level) {
        const uint32_t n       = load_acquire(&control->frontier_size[cur]);
        const uint64_t largest = load_acquire(&control->largest[cur]);
        const bool     enough  = n >= p.target_tasks && largest <= p.grain;
        const bool     stuck   = n == prev_n && largest == prev_largest;
        if (n == 0 || enough || stuck || level >= p.max_wide_levels)
            break;
        prev_n       = n;
        prev_largest = largest;
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            control->frontier_size[cur ^ 1] = 0;
            control->largest[cur ^ 1]       = 0;
            ++control->wide_levels;
        }
        grid.sync();
        const WideLevel<Task> wide{p.frontier[cur],     n,
                                   p.frontier[cur ^ 1], &control->frontier_size[cur ^ 1],
                                   p.frontier_capacity, &control->largest[cur ^ 1],
                                   n < p.target_tasks,  p.grain};
        p.work.wide(grid, wide, stats);
        grid.sync();
        cur ^= 1;
    }

    // Deal the frontier round-robin.
    const uint32_t n = load_acquire(&control->frontier_size[cur]);
    if (threadIdx.x == 0) {
        for (uint32_t i = blockIdx.x; i < n; i += gridDim.x)
            d.push(control, p.frontier[cur][i], stats);
        if (blockIdx.x == 0)
            atomicAdd(reinterpret_cast<unsigned long long*>(&control->pending), static_cast<unsigned long long>(n));
    }
    grid.sync(); // pending, every seed and every local deque header are visible before anyone looks for work
    add_cycles(stats.wide_cycles, c_start);

    // Phase 2: work stealing. Thread 0 schedules; the block runs each task together.
    uint32_t          rng = 0x9E3779B9u ^ (blockIdx.x * 0x85EBCA6Bu + 1);
    const Spawn<Task> spawn{control, &d, &stats, &retired};
    for (;;) {
        const uint64_t c_find = timer();
        if (threadIdx.x == 0)
            running = find_task(control, d, rng, retired, task, stats);
        __syncthreads();
        add_cycles(stats.find_cycles, c_find);
        if (!running)
            break;
        fence_acq_rel_gpu(); // every thread: the task's data was written on another SM
        const uint64_t start = global_ns();
        p.work.run(task, spawn, stats);
        __syncthreads();
        if (threadIdx.x == 0) {
            ++retired;
            d.export_for_idle(control, stats);
            ++stats.tasks;
            const uint64_t took = global_ns() - start;
            stats.busy_ns += took;
            if (took > stats.max_task_ns) {
                stats.max_task_ns   = took;
                stats.max_task_info = p.work.describe(task);
            }
            const uint64_t deadline = control->deadline_ns;
            if (deadline != 0 && global_ns() > deadline)
                raise_abort(control, Abort::kTimeout);
        }
        __syncthreads(); // `task` is reused
    }
    add_cycles(stats.total_cycles, c_start);
    __syncthreads();
    if (threadIdx.x == 0)
        p.stats[blockIdx.x] = stats;
    d.cluster.sync(); // no block leaves while a cluster peer may still be stealing from its shared memory
}

// Host: the kernel accepts `smem` bytes of dynamic shared memory and clusters of any size up to 16.
template <typename Work> cudaError_t prepare_megakernel(size_t smem)
{
    if (clusters_supported())
        if (const cudaError_t e =
                    cudaFuncSetAttribute(megakernel<Work>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
            e != cudaSuccess)
            return e;
    return cudaFuncSetAttribute(megakernel<Work>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
}

// Host: the largest block size (a multiple of 32) the kernel can run with: one block then owns an SM's threads.
template <typename Work> int max_block_size()
{
    cudaFuncAttributes attr{};
    if (cudaFuncGetAttributes(&attr, megakernel<Work>) != cudaSuccess)
        return 0;
    return attr.maxThreadsPerBlock / 32 * 32;
}

struct LaunchShape
{
    uint32_t cluster_size;
    uint32_t block;
    size_t   smem; // dynamic shared memory per block
};

namespace detail {

inline cudaLaunchConfig_t launch_config(const LaunchShape& shape, uint32_t clusters, cudaLaunchAttribute (&attrs)[2],
                                        bool cooperative, cudaStream_t stream)
{
    // Cooperative first, so a device without clusters launches with it alone.
    attrs[0].id               = cudaLaunchAttributeCooperative;
    attrs[0].val.cooperative  = 1;
    attrs[1].id               = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim.x = shape.cluster_size;
    attrs[1].val.clusterDim.y = 1;
    attrs[1].val.clusterDim.z = 1;
    const bool         cluster = clusters_supported();
    cudaLaunchConfig_t cfg{};
    cfg.gridDim          = dim3(clusters * shape.cluster_size);
    cfg.blockDim         = dim3(shape.block);
    cfg.dynamicSmemBytes = shape.smem;
    cfg.stream           = stream;
    cfg.attrs            = cooperative ? attrs : attrs + 1;
    cfg.numAttrs         = (cooperative ? 1 : 0) + (cluster ? 1 : 0);
    return cfg;
}

} // namespace detail

// Host: how many clusters of this shape the GPU keeps resident at once (0: the shape does not fit).
template <typename Work> uint32_t max_active_clusters(const LaunchShape& shape)
{
    if (!clusters_supported()) { // a cluster is one block
        int device = 0, sms = 0, per_sm = 0;
        if (shape.cluster_size != 1 || cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, megakernel<Work>, static_cast<int>(shape.block),
                                                          shape.smem) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        return static_cast<uint32_t>(sms * per_sm);
    }
    cudaLaunchAttribute      attrs[2];
    const cudaLaunchConfig_t cfg      = detail::launch_config(shape, 1, attrs, false, nullptr);
    int                      clusters = 0;
    if (cudaOccupancyMaxActiveClusters(&clusters, megakernel<Work>, &cfg) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return static_cast<uint32_t>(clusters);
}

// Host: cooperative cluster launch, so the driver refuses (instead of hanging) a grid that cannot be fully resident.
template <typename Work>
cudaError_t launch_megakernel(const Params<Work>& p, const LaunchShape& shape, uint32_t clusters,
                              cudaStream_t stream = nullptr)
{
    cudaLaunchAttribute      attrs[2];
    const cudaLaunchConfig_t cfg = detail::launch_config(shape, clusters, attrs, true, stream);
    return cudaLaunchKernelEx(&cfg, megakernel<Work>, p);
}

} // namespace ghuim::persistent
