// Milestones 1 and 3 of the persistent miner: the megakernel scheduler (with cluster deques) on a synthetic task tree,
// checked against a host walk of the same tree. Nothing is mined; a task burns `units` of hashing work and spawns up to
// four children.
//
// The tree is a pure function of the seeds: task (seed, depth) has child c with probability branch / 4 (until
// max_depth), and a size of `units << k` with k uniform in [0, skew], so sizes are heavy tailed. Every task is
// counted, summed and hashed exactly once in either phase, which the host verifies.
//
//   persistent_skeleton [--roots N] [--depth D] [--branch B] [--units U] [--skew K] [--block T] [--cluster C]
//                       [--clusters N] [--local-deque L] [--deque G] [--target-per-block F] [--grain S]
//                       [--timeout-ms M]
// --cluster: blocks per thread-block cluster (1..16); --clusters: at most this many (default: all that fit);
// --local-deque: tasks per shared-memory deque (0: global deques only, the milestone-1 scheduler).
// Exit status: 0 on a verified run, 1 on a mismatch, abort or CUDA error.

#include "mine/persistent/megakernel.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace ghuim::persistent;

#define CHECK(call)                                                                                                    \
    do {                                                                                                               \
        const cudaError_t err_ = (call);                                                                               \
        if (err_ != cudaSuccess) {                                                                                     \
            std::fprintf(stderr, "CUDA error: %s at %s:%d: %s\n", #call, __FILE__, __LINE__,                           \
                         cudaGetErrorString(err_));                                                                    \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

__host__ __device__ inline uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

struct Node
{
    uint64_t seed;
    uint32_t depth;
    uint32_t units;
};

struct TreeShape
{
    uint32_t max_depth;
    uint32_t keep;  // child c exists iff byte c of the hash is below this (branch / 4 * 256)
    uint32_t units; // smallest task size
    uint32_t skew;  // sizes are units << [0, skew]
};

constexpr uint32_t kMaxChildren = 4;

// Writes the children of `node`; returns how many.
__host__ __device__ inline uint32_t children_of(const TreeShape& shape, const Node& node, Node* out)
{
    if (node.depth >= shape.max_depth)
        return 0;
    const uint64_t h = mix64(node.seed);
    uint32_t       n = 0;
    for (uint32_t c = 0; c < kMaxChildren; ++c) {
        if (((h >> (8 * c)) & 0xff) >= shape.keep)
            continue;
        const uint64_t seed = mix64(node.seed * 31 + c + 1);
        out[n++]            = {seed, node.depth + 1, shape.units << (mix64(seed ^ 0x5bd1e995) % (shape.skew + 1))};
    }
    return n;
}

__device__ inline uint32_t burn(uint64_t seed, uint32_t i)
{
    uint32_t x = i ^ static_cast<uint32_t>(seed);
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        x ^= x >> 15;
        x *= 0x2c1b3c6du;
        x ^= x >> 12;
    }
    return x;
}

// Work counters: [0] checksum, [1] units, [2] nodes (tasks of the tree, whether queued or run inline).
struct SyntheticTree
{
    using Task                       = Node;
    static constexpr uint32_t kStack = 64; // private (unstealable) nodes per block
    static constexpr int      kMaxThreads = 1024;
    TreeShape                 shape;
    uint32_t                  inline_below; // children smaller than this run inside their parent's task
    uint32_t*                 sink;         // per block, keeps the hashing from being optimized away

    __device__ void init_block() const {}
    __device__ static uint64_t describe(const Task&) { return 0; }

    __device__ void credit(const Task& t, BlockStats& stats) const
    {
        stats.work[0] += mix64(t.seed ^ t.depth);
        stats.work[1] += t.units;
        ++stats.work[2];
    }

    // Expands every frontier task when tasks are needed, else only those larger than the grain (the rest move to
    // the next frontier). Each expanded task's hashing is spread over the whole grid; block i % grid emits task
    // i's children.
    __device__ void wide(cg::grid_group& grid, const WideLevel<Task>& w, BlockStats& stats) const
    {
        // Without room for a full level of children nothing is expanded: the level changes nothing and phase 1 ends.
        const bool room   = static_cast<uint64_t>(w.n) * kMaxChildren <= w.next_capacity;
        const auto expand = [&](const Task& t) { return room && (w.need_tasks || t.units > w.grain); };
        uint32_t   acc    = 0;
        for (uint32_t i = 0; i < w.n; ++i)
            if (expand(w.frontier[i]))
                for (uint64_t u = grid.thread_rank(); u < w.frontier[i].units; u += grid.size())
                    acc += burn(w.frontier[i].seed, static_cast<uint32_t>(u));
        if (acc == 0x12345678u)
            atomicAdd(sink + blockIdx.x, 1u);
        if (threadIdx.x != 0)
            return;
        for (uint32_t i = blockIdx.x; i < w.n; i += gridDim.x) {
            Task     kids[kMaxChildren];
            uint32_t k = 1;
            if (expand(w.frontier[i])) {
                k = children_of(shape, w.frontier[i], kids);
                credit(w.frontier[i], stats);
                ++stats.wide_tasks;
            }
            else
                kids[0] = w.frontier[i];
            const uint32_t slot = atomicAdd(w.next_n, k);
            for (uint32_t c = 0; c < k; ++c) {
                w.next[slot + c] = kids[c];
                atomicMax(reinterpret_cast<unsigned long long*>(w.largest),
                          static_cast<unsigned long long>(kids[c].units));
            }
        }
    }

    // Runs the task and, depth first, every descendant smaller than inline_below; larger ones (or all, while
    // some block is idle) are spawned for any block to take.
    __device__ void run(const Task& task, const Spawn<Task>& spawn, BlockStats& stats) const
    {
        __shared__ Task     stack[kStack];
        __shared__ uint32_t top;
        __shared__ Task     node;
        if (threadIdx.x == 0) {
            stack[0] = task;
            top      = 1;
        }
        for (;;) {
            __syncthreads();
            if (top == 0)
                break;
            __syncthreads(); // every thread has read top
            if (threadIdx.x == 0)
                node = stack[--top];
            __syncthreads();
            uint32_t acc = 0;
            for (uint32_t u = threadIdx.x; u < node.units; u += blockDim.x)
                acc += burn(node.seed, u);
            if (acc == 0x12345678u)
                atomicAdd(sink + blockIdx.x, 1u);
            if (threadIdx.x == 0) {
                credit(node, stats);
                Task           kids[kMaxChildren], out[kMaxChildren];
                const uint32_t k      = children_of(shape, node, kids);
                uint32_t       n      = 0;
                const bool     hungry = k != 0 && spawn.hungry();
                for (uint32_t c = k; c-- > 0;) // reversed: kids[0] ends on top of the stack
                    if (kids[c].units >= inline_below || hungry || top == kStack)
                        out[n++] = kids[c];
                    else
                        stack[top++] = kids[c];
                spawn.push(out, n);
            }
        }
    }
};

__global__ void read_clock(uint64_t* out)
{
    *out = global_ns();
}

struct Options
{
    uint32_t roots = 4, depth = 22, units = 256, skew = 6;
    double   branch  = 1.6;
    int      block   = 0;                                  // 0: the largest the kernel allows
    bool     cluster_set = false; // --cluster given: on a device without clusters, skip instead of using 1
    uint32_t cluster = 2, clusters = 0, local_deque = 256; // 2: the largest cluster that still covers every SM
    uint32_t target_per_block = 4;
    uint64_t grain            = 0; // 0: units << skew (any task fits a block)
    uint32_t deque            = 4096;
    uint32_t timeout_ms       = 10000;
    uint32_t inline_below     = 0; // children with fewer units run inside their parent's task (0: none)
    uint32_t max_wide_levels  = 16;
};

Options parse(int argc, char** argv)
{
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a    = argv[i];
        auto              next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", a.c_str());
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--roots")
            o.roots = std::strtoul(next(), nullptr, 10);
        else if (a == "--depth")
            o.depth = std::strtoul(next(), nullptr, 10);
        else if (a == "--branch")
            o.branch = std::strtod(next(), nullptr);
        else if (a == "--units")
            o.units = std::strtoul(next(), nullptr, 10);
        else if (a == "--skew")
            o.skew = std::strtoul(next(), nullptr, 10);
        else if (a == "--block")
            o.block = std::atoi(next());
        else if (a == "--cluster")
            o.cluster = std::strtoul(next(), nullptr, 10), o.cluster_set = true;
        else if (a == "--clusters")
            o.clusters = std::strtoul(next(), nullptr, 10);
        else if (a == "--local-deque")
            o.local_deque = std::strtoul(next(), nullptr, 10);
        else if (a == "--target-per-block")
            o.target_per_block = std::strtoul(next(), nullptr, 10);
        else if (a == "--grain")
            o.grain = std::strtoull(next(), nullptr, 10);
        else if (a == "--deque")
            o.deque = std::strtoul(next(), nullptr, 10);
        else if (a == "--inline-below")
            o.inline_below = std::strtoul(next(), nullptr, 10);
        else if (a == "--max-wide-levels")
            o.max_wide_levels = std::strtoul(next(), nullptr, 10);
        else if (a == "--timeout-ms")
            o.timeout_ms = std::strtoul(next(), nullptr, 10);
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            std::exit(1);
        }
    }
    return o;
}

struct Expected
{
    uint64_t tasks = 0, checksum = 0, units = 0;
    uint32_t max_depth = 0;
};

Expected walk(const TreeShape& shape, const std::vector<Node>& roots)
{
    Expected          e;
    std::vector<Node> stack(roots.begin(), roots.end());
    while (!stack.empty()) {
        const Node n = stack.back();
        stack.pop_back();
        ++e.tasks;
        e.checksum += mix64(n.seed ^ n.depth);
        e.units += n.units;
        e.max_depth = std::max(e.max_depth, n.depth);
        Node           kids[kMaxChildren];
        const uint32_t k = children_of(shape, n, kids);
        stack.insert(stack.end(), kids, kids + k);
    }
    return e;
}

} // namespace

int main(int argc, char** argv)
{
    const Options     o = parse(argc, argv);
    const TreeShape   shape{o.depth, static_cast<uint32_t>(o.branch / kMaxChildren * 256), o.units, o.skew};
    std::vector<Node> roots(o.roots);
    for (uint32_t r = 0; r < o.roots; ++r)
        roots[r] = {mix64(r + 1000), 0, o.units << o.skew}; // roots are the largest tasks
    const Expected expected = walk(shape, roots);

    int dev = 0, sms = 0;
    CHECK(cudaGetDevice(&dev));
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    const size_t smem = local_deque_bytes<Node>(o.local_deque);
    CHECK(prepare_megakernel<SyntheticTree>(smem));
    const uint32_t cluster = supported_cluster_size(o.cluster);
    if (cluster != o.cluster && o.cluster_set) {
        std::printf("skipped: --cluster %u needs thread-block clusters (sm_90+)\n", o.cluster);
        return 77; // CTest SKIP_RETURN_CODE
    }
    const LaunchShape launch{cluster, static_cast<uint32_t>(o.block ? o.block : max_block_size<SyntheticTree>()),
                             smem};
    const uint32_t    fit = max_active_clusters<SyntheticTree>(launch);
    if (fit == 0) {
        std::fprintf(stderr, "clusters of %u blocks x %u threads (%zu B shared) do not fit\n", launch.cluster_size,
                     launch.block, smem);
        return 1;
    }
    const uint32_t clusters = o.clusters ? std::min(o.clusters, fit) : fit;
    const uint32_t blocks   = clusters * launch.cluster_size;

    uint64_t grain             = o.grain ? o.grain : static_cast<uint64_t>(o.units) << o.skew;
    uint32_t frontier_capacity = std::max<uint32_t>(o.roots, blocks * o.target_per_block) * kMaxChildren * 8;
    std::printf("tree: %llu tasks, %llu units, depth %u | grid %u clusters x %u blocks x %u threads (%u SMs, %u "
                "clusters fit) | target %u tasks, grain %llu, deques %u local + %u global\n",
                (unsigned long long)expected.tasks, (unsigned long long)expected.units, expected.max_depth, clusters,
                launch.cluster_size, launch.block, sms, fit, blocks * o.target_per_block, (unsigned long long)grain,
                o.local_deque, o.deque);

    Control*           control  = nullptr;
    PaddedDequeHeader* headers  = nullptr;
    Node*              rings    = nullptr;
    Node*              frontier = nullptr;
    BlockStats*        stats    = nullptr;
    uint32_t*          sink     = nullptr;
    CHECK(cudaMalloc(&control, sizeof(Control)));
    CHECK(cudaMalloc(&headers, sizeof(PaddedDequeHeader) * blocks));
    CHECK(cudaMalloc(&rings, sizeof(Node) * blocks * o.deque));
    CHECK(cudaMalloc(&frontier, sizeof(Node) * 2 * frontier_capacity));
    CHECK(cudaMalloc(&stats, sizeof(BlockStats) * blocks));
    CHECK(cudaMalloc(&sink, sizeof(uint32_t) * blocks));
    CHECK(cudaMemset(headers, 0, sizeof(PaddedDequeHeader) * blocks));
    CHECK(cudaMemset(sink, 0, sizeof(uint32_t) * blocks));

    // The watchdog deadline is in %globaltimer time: read it on the device just before the launch.
    uint64_t* now = nullptr;
    CHECK(cudaMalloc(&now, sizeof(uint64_t)));
    read_clock<<<1, 1>>>(now);
    CHECK(cudaGetLastError());
    uint64_t now_ns = 0;
    CHECK(cudaMemcpy(&now_ns, now, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    Control c{};
    c.deadline_ns      = now_ns + static_cast<uint64_t>(o.timeout_ms) * 1000000;
    c.frontier_size[0] = o.roots;
    c.largest[0]       = static_cast<uint64_t>(o.units) << o.skew;
    CHECK(cudaMemcpy(control, &c, sizeof(Control), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(frontier, roots.data(), sizeof(Node) * o.roots, cudaMemcpyHostToDevice));

    Params<SyntheticTree> p{};
    p.work              = {shape, o.inline_below, sink};
    p.max_wide_levels   = o.max_wide_levels;
    p.control           = control;
    p.headers           = headers;
    p.rings             = rings;
    p.global_capacity   = o.deque;
    p.local_capacity    = o.local_deque;
    p.frontier[0]       = frontier;
    p.frontier[1]       = frontier + frontier_capacity;
    p.frontier_capacity = frontier_capacity;
    p.target_tasks      = blocks * o.target_per_block;
    p.grain             = grain;
    p.stats             = stats;

    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a));
    CHECK(launch_megakernel(p, launch, clusters));
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    CHECK(cudaGetLastError());
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, a, b));

    CHECK(cudaMemcpy(&c, control, sizeof(Control), cudaMemcpyDeviceToHost));
    std::vector<BlockStats> per_block(blocks);
    CHECK(cudaMemcpy(per_block.data(), stats, sizeof(BlockStats) * blocks, cudaMemcpyDeviceToHost));
    BlockStats          total{};
    std::vector<double> busy; // per block (each one an SM), percent of the kernel time
    for (const BlockStats& s : per_block) {
        total.tasks += s.tasks;
        total.wide_tasks += s.wide_tasks;
        total.steals_dsmem += s.steals_dsmem;
        total.steals_near += s.steals_near;
        total.steals_far += s.steals_far;
        total.steal_misses += s.steal_misses;
        total.exports += s.exports;
        total.busy_ns += s.busy_ns;
        total.idle_ns += s.idle_ns;
        total.max_local  = std::max(total.max_local, s.max_local);
        total.max_global = std::max(total.max_global, s.max_global);
        for (int w = 0; w < 4; ++w)
            total.work[w] += s.work[w];
        busy.push_back(100 * s.busy_ns / (ms * 1e6));
    }
    std::sort(busy.begin(), busy.end());

    const double wall_ns = ms * 1e6 * blocks;
    std::printf("run: %.3f ms | wide levels %u (%llu nodes), stealing phase %llu tasks, %llu nodes\n", ms,
                c.wide_levels, (unsigned long long)total.wide_tasks, (unsigned long long)total.tasks,
                (unsigned long long)(total.work[2] - total.wide_tasks));
    std::printf("steals: %llu dsmem, %llu same cluster (global), %llu other cluster | %llu misses | %llu exported\n",
                (unsigned long long)total.steals_dsmem, (unsigned long long)total.steals_near,
                (unsigned long long)total.steals_far, (unsigned long long)total.steal_misses,
                (unsigned long long)total.exports);
    std::printf("SMs: busy %.1f%% (min %.1f, median %.1f, max %.1f), idle %.1f%% | deepest deque %llu local, %llu "
                "global\n",
                100 * total.busy_ns / wall_ns, busy.front(), busy[busy.size() / 2], busy.back(),
                100 * total.idle_ns / wall_ns, (unsigned long long)total.max_local,
                (unsigned long long)total.max_global);

    bool ok = true;
    if (c.abort != 0) {
        std::printf("FAIL: aborted (%s)\n", abort_name(static_cast<Abort>(c.abort)));
        ok = false;
    }
    if (c.pending != 0 && c.abort == 0) {
        std::printf("FAIL: %llu tasks still pending\n", (unsigned long long)c.pending);
        ok = false;
    }
    const uint64_t ran = total.work[2];
    if (ran != expected.tasks || total.work[0] != expected.checksum || total.work[1] != expected.units) {
        std::printf("FAIL: ran %llu tasks (expected %llu), checksum %s, units %llu (expected %llu)\n",
                    (unsigned long long)ran, (unsigned long long)expected.tasks,
                    total.work[0] == expected.checksum ? "ok" : "wrong", (unsigned long long)total.work[1],
                    (unsigned long long)expected.units);
        ok = false;
    }
    if (ok)
        std::printf("OK\n");
    return ok ? 0 : 1;
}
