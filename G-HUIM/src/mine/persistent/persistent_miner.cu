// Host driver for the persistent search (REFIM on the GPU): one megakernel launch mines the whole dataset.
//
// Device memory: the root database (static), per-group scratch (one per block plus the grid's), per-block database
// stacks and frames, the deques and wide-phase frontiers, and the heap for databases shared between blocks. The
// stacks and the heap split what is left. Count-only: itemsets are counted, not written.

#include "app/config.hpp"
#include "cuda/buffer.cuh"
#include "data/dataset.hpp"
#include "mine/persistent/refim/work.cuh"
#include "mine/result.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vector>

namespace ghuim::inline GHUIM_WIDTHS {

namespace {

using namespace persistent;
using refim::kMaxWarps;
using refim::Work;

constexpr uint64_t kReservedBytes   = 512ull << 20; // runtime, kernels, and free-memory slack
constexpr uint32_t kClusterSize     = 2;            // the largest cluster that still covers every SM (milestone 3)
constexpr uint32_t kLocalDeque      = 256;
constexpr uint32_t kGlobalDeque     = 1u << 14;
constexpr uint32_t kTasksPerBlock   = 4;    // the wide phase stops at this many tasks per block ...
constexpr uint32_t kWarpItems       = 8192; // databases with at most this many entries run in warp mode
constexpr uint32_t kMaxWideLevels   = 4;
constexpr uint64_t kRefineMinItems  = 0;    // refine every database (REFIM's default): BMS expands 6x the nodes without
constexpr uint64_t kMergeMinItems   = 1024; // smaller databases skip merging
constexpr uint64_t kWatchdogSeconds = 3600; // abort a run longer than this (a hang); GHUIM_WATCHDOG_SECONDS overrides

// The dataset as one database: every transaction, every item secondary, the first-level extensions primary.
std::vector<char> root_database(const Dataset& data)
{
    const uint64_t        n_tx = data.transaction_count(), n_items = data.items.size(), n_prim = data.primary.size();
    std::vector<int32_t>  slot(data.original_item.size(), -1);
    std::vector<uint32_t> count(n_prim + 1, 0);
    for (uint32_t j = 0; j < n_prim; ++j)
        slot[data.primary[j]] = static_cast<int32_t>(j);
    for (const uint32_t x : data.items)
        if (slot[x] >= 0)
            ++count[slot[x] + 1];
    for (uint32_t j = 0; j < n_prim; ++j)
        count[j + 1] += count[j];
    const uint64_t        n_occ = count[n_prim];
    const refim::DbLayout L     = refim::db_layout(n_tx, n_items, n_prim, n_occ);
    // 32-bit occurrences (transaction above the position bits) and entry offsets, as REFIM packs them.
    if (n_items > UINT32_MAX || n_occ > UINT32_MAX || !occurrences_fit(n_tx, data.max_length))
        throw std::runtime_error("the dataset is too large for 32-bit occurrences and entry offsets");
    const uint32_t shift = occurrence_pos_bits(data.max_length);

    std::vector<char> image(L.bytes, 0);
    char*             base = image.data();
    auto*             h    = reinterpret_cast<refim::DbHeader*>(base);
    h->n_tx                = static_cast<uint32_t>(n_tx);
    h->n_primary           = static_cast<uint32_t>(n_prim);
    h->n_occ               = static_cast<uint32_t>(n_occ);
    h->top                 = static_cast<uint32_t>(data.original_item.size() - 1);
    h->n_items             = n_items;
    h->bytes               = L.bytes;
    h->refs                = 1;
    h->cls                 = refim::kStaticDb;
    h->occ_shift           = shift;
    const refim::Db d      = refim::Db::view(base);
    std::copy(count.begin(), count.end(), d.occ_begin);
    std::copy(data.primary.begin(), data.primary.end(), d.primary);
    for (uint64_t t = 0; t < n_tx; ++t) {
        const uint32_t b = data.tx_begin[t], e = data.tx_begin[t + 1];
        d.tx_off[t] = static_cast<uint32_t>(b);
        d.tx_len[t] = e - b;
        d.prefix[t] = 0;
        for (uint32_t i = b; i < e; ++i) {
            d.items[i] = static_cast<Key>(data.items[i]);
            d.utils[i] = static_cast<Util>(data.utilities[i]);
            if (const int32_t j = slot[data.items[i]]; j >= 0)
                d.occ[count[j]++] = refim::make_occ(static_cast<uint32_t>(t), i - b, shift);
        }
    }
    return image;
}

// An environment variable as a number (tuning knobs), or `fallback`.
uint64_t env_or(const char* name, uint64_t fallback)
{
    const char* v = std::getenv(name);
    return v ? std::strtoull(v, nullptr, 10) : fallback;
}

__global__ void read_clock(uint64_t* out)
{
    *out = global_ns();
}

} // namespace

MiningResult mine_persistent(const Dataset& data, const MiningOptions& options)
{
    MiningResult result;
    if (data.primary.empty())
        return result;
    const auto setup_start = std::chrono::steady_clock::now();

    const std::vector<char> root              = root_database(data);
    const uint64_t          items             = data.original_item.size();
    const uint32_t          frames            = data.max_length + 2; // a frame per itemset length, plus the task's
    const uint32_t          frontier_capacity = static_cast<uint32_t>(std::max<uint64_t>(1u << 20, 4 * items));

    // Launch shape: the largest block the kernel allows, two-block clusters, every cluster that fits. Dynamic shared
    // memory holds the local deque, then the scratch of the block and of each warp when they fit (else only the
    // block's, else none: they go to global memory).
    const uint32_t block = static_cast<uint32_t>(max_block_size<Work>());
    const uint32_t warps = block / 32;
    int            optin = 0, device = 0;
    check_cuda(cudaGetDevice(&device), "query device");
    check_cuda(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device), "query shared memory");
    cudaFuncAttributes attr{};
    check_cuda(cudaFuncGetAttributes(&attr, megakernel<Work>), "query megakernel");
    const uint64_t deque_bytes  = refim::align16(local_deque_bytes<Work::Task>(kLocalDeque));
    const uint64_t region_bytes = refim::scratch_bytes(items);
    const uint64_t pool_bytes   = refim::align16(sizeof(refim::LanePool));
    const uint64_t avail = optin - std::min<uint64_t>(optin, attr.sharedSizeBytes + deque_bytes + pool_bytes + 1024);
    const uint32_t smem_mode    = (1 + warps) * region_bytes <= avail ? 2 : region_bytes <= avail ? 1 : 0;
    const size_t   smem = deque_bytes + (smem_mode == 2 ? 1 + warps : smem_mode) * region_bytes + pool_bytes;
    check_cuda(prepare_megakernel<Work>(smem), "configure megakernel");
    const LaunchShape shape{kClusterSize, block, smem};
    const uint32_t    clusters = max_active_clusters<Work>(shape);
    if (clusters == 0)
        throw std::runtime_error("the persistent megakernel does not fit on the device");
    const uint32_t blocks     = clusters * shape.cluster_size;
    const char*    warp_env   = std::getenv("GHUIM_WARP_ITEMS"); // tuning: databases this small run in warp mode
    const uint32_t warp_items = warp_env ? std::strtoul(warp_env, nullptr, 10) : kWarpItems;

    // Scratch regions in global memory: the grid's, plus the blocks' and warps' that shared memory cannot hold.
    const uint64_t block_sets = smem_mode == 0 ? blocks : 0;
    const uint64_t warp_sets  = smem_mode < 2 && warp_items != 0 ? uint64_t{blocks} * warps : 0;
    const uint64_t sets       = 1 + block_sets + warp_sets;
    const bool     lane       = env_or("GHUIM_LANE", 1) != 0; // lane mode: a thread mines a tiny subtree alone
    const uint64_t lane_bytes = lane ? uint64_t{blocks} * block * refim::kLaneArena : 0;
    const uint64_t fixed      = root.size() + sets * region_bytes + lane_bytes +
                           uint64_t{blocks} * (1 + kMaxWarps) * frames * sizeof(refim::Frame) +
                           uint64_t{blocks} * kGlobalDeque * sizeof(Work::Task) +
                           2ull * frontier_capacity * sizeof(Work::Task) + (1u << 20);
    size_t free_bytes = 0, total_bytes = 0;
    // Each thread's local memory (the lane miner's arrays, spills) is reserved by the driver for every thread the GPU
    // can hold. Reserve it now, so the free memory measured next is what is really left.
    check_cuda(cudaDeviceSetLimit(cudaLimitStackSize, std::max<size_t>(attr.localSizeBytes, 1024)), "reserve stacks");
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "query device memory");
    const uint64_t budget =
            options.memory_bytes ? options.memory_bytes : free_bytes - std::min<uint64_t>(free_bytes, kReservedBytes);
    if (budget < fixed + (64ull << 20) * blocks)
        throw std::runtime_error("not enough device memory for the persistent search");
    // A block's stack holds one root-to-leaf path of databases. Children shrink quickly, so a few root sizes cover
    // most paths (a deeper one spills into the heap); the stacks take at most 60% and the heap the rest.
    const uint64_t rest        = budget - fixed;
    const uint64_t stack_bytes = std::clamp<uint64_t>(4 * root.size(), 64ull << 20, rest / 10 * 6 / blocks) / 256 * 256;
    const uint64_t heap_bytes  = (rest - stack_bytes * blocks) / 256 * 256;

    DeviceBuffer<char>              root_d(root.size());
    DeviceBuffer<char>              heap_d(heap_bytes);
    DeviceBuffer<uint64_t>          heap_state(2 + Heap::kClasses); // bump, free lists, live bytes
    DeviceBuffer<char>              stacks_d(stack_bytes * blocks);
    DeviceBuffer<refim::Frame>      frames_d(uint64_t{blocks} * frames);
    DeviceBuffer<refim::Frame>      warp_frames_d(uint64_t{blocks} * kMaxWarps * frames);
    DeviceBuffer<unsigned char>     sets_d(sets * region_bytes);
    DeviceBuffer<refim::Scratch>    scratch_d(blocks + 1);
    DeviceBuffer<refim::Scratch>    warp_scratch_d(uint64_t{blocks} * kMaxWarps);
    DeviceBuffer<refim::Counters>   grid_counters(1);
    DeviceBuffer<unsigned char>     lane_arena(std::max<uint64_t>(lane_bytes, 1));
    DeviceBuffer<PaddedDequeHeader> headers(blocks);
    DeviceBuffer<Work::Task>        rings(uint64_t{blocks} * kGlobalDeque);
    DeviceBuffer<Work::Task>        frontier(2ull * frontier_capacity);
    DeviceBuffer<Control>           control_d(1);
    DeviceBuffer<BlockStats>        stats_d(blocks);
    DeviceBuffer<uint64_t>          now_d(1);

    check_cuda(cudaMemcpy(root_d.get(), root.data(), root.size(), cudaMemcpyHostToDevice), "upload root");
    check_cuda(cudaMemset(heap_state.get(), 0, heap_state.size() * sizeof(uint64_t)), "clear heap");
    check_cuda(cudaMemset(headers.get(), 0, headers.size() * sizeof(PaddedDequeHeader)), "clear deques");
    check_cuda(cudaMemset(sets_d.get(), 0, sets_d.size()), "clear scratch");
    // Region r of the global scratch: Bound[items], then slot (-1), mark, touched, count, chosen.
    uint64_t   next_set = 0;
    const auto take_set = [&] {
        unsigned char* base = sets_d.get() + next_set++ * region_bytes;
        auto*          w    = reinterpret_cast<uint32_t*>(base + items * sizeof(refim::Bound));
        check_cuda(cudaMemset(w, 0xff, items * sizeof(uint32_t)), "clear scratch");
        return refim::Scratch{reinterpret_cast<refim::Bound*>(base),
                              reinterpret_cast<int32_t*>(w),
                              w + items,
                              w + 2 * items,
                              w + 3 * items,
                              w + 4 * items};
    };
    std::vector<refim::Scratch> scratch(blocks + 1), warp_scratch(uint64_t{blocks} * kMaxWarps);
    scratch[blocks] = take_set(); // the grid's
    for (uint64_t b = 0; b < block_sets; ++b)
        scratch[b] = take_set();
    for (uint64_t b = 0; warp_sets != 0 && b < blocks; ++b)
        for (uint32_t w = 0; w < warps; ++w)
            warp_scratch[b * kMaxWarps + w] = take_set();
    check_cuda(cudaMemcpy(scratch_d.get(), scratch.data(), scratch.size() * sizeof(refim::Scratch),
                          cudaMemcpyHostToDevice),
               "upload scratch");
    check_cuda(cudaMemcpy(warp_scratch_d.get(), warp_scratch.data(), warp_scratch.size() * sizeof(refim::Scratch),
                          cudaMemcpyHostToDevice),
               "upload scratch");

    // The first frontier: every first-level extension, one task each.
    const refim::Db         host_root = refim::Db::view(const_cast<char*>(root.data()));
    std::vector<Work::Task> first(data.primary.size());
    uint64_t                largest = 0, total_occ = 0;
    for (uint32_t k = 0; k < first.size(); ++k) {
        first[k]            = {reinterpret_cast<uint64_t>(root_d.get()), k, k + 1, 0, 0};
        const uint64_t size = host_root.occ_begin[k + 1] - host_root.occ_begin[k];
        largest             = std::max(largest, size);
        total_occ += size;
    }
    check_cuda(cudaMemcpy(frontier.get(), first.data(), first.size() * sizeof(Work::Task), cudaMemcpyHostToDevice),
               "upload frontier");

    read_clock<<<1, 1>>>(now_d.get());
    check_launch("read clock");
    uint64_t now_ns = 0;
    check_cuda(cudaMemcpy(&now_ns, now_d.get(), sizeof(now_ns), cudaMemcpyDeviceToHost), "read clock");
    Control        control{};
    const char*    watchdog  = std::getenv("GHUIM_WATCHDOG_SECONDS");
    const uint64_t seconds   = watchdog ? std::strtoull(watchdog, nullptr, 10) : kWatchdogSeconds;
    control.deadline_ns      = now_ns + seconds * 1000000000;
    control.frontier_size[0] = static_cast<uint32_t>(first.size());
    control.largest[0]       = largest;
    check_cuda(cudaMemcpy(control_d.get(), &control, sizeof(control), cudaMemcpyHostToDevice), "upload control");

    Params<Work> p{};
    const char*  refine_env = std::getenv("GHUIM_REFINE"); // REFIM's --refine: -1 (default) until no item drops
    p.work.set              = {options.minimum_utility,
                               static_cast<uint32_t>(items),
                               control_d.get(),
                               static_cast<uint32_t>(env_or("GHUIM_SPLIT_DEPTH", 2)),  // REFIM: kMaxSplitDepth
                               static_cast<uint32_t>(env_or("GHUIM_HUNGRY_DEPTH", 8)), // REFIM: kMaxHungryDepth
                               env_or("GHUIM_SPLIT_WORK", 1ull << 20),                 // REFIM: kSplitWork
                               env_or("GHUIM_HUNGRY_WORK", 1ull << 14),                // REFIM: kHungryWork
                               static_cast<uint32_t>(env_or("GHUIM_SPLIT_CHUNKS", 16)),
                  refine_env ? static_cast<int32_t>(std::strtol(refine_env, nullptr, 10)) : -1,
                               env_or("GHUIM_REFINE_MIN", kRefineMinItems),
                               env_or("GHUIM_MERGE_MIN", kMergeMinItems)};
    p.work.root             = root_d.get();
    p.work.root_bytes       = root.size();
    p.work.heap             = {heap_d.get(), heap_bytes, heap_state.get(), heap_state.get() + 1,
                               heap_state.get() + 1 + Heap::kClasses};
    p.work.scratch          = scratch_d.get();
    p.work.grid_counters    = grid_counters.get();
    p.work.stacks           = stacks_d.get();
    p.work.stack_bytes      = stack_bytes;
    p.work.frames           = frames_d.get();
    p.work.max_frames       = frames;
    p.work.warp_items       = warp_items;
    p.work.borrow_items     = static_cast<uint32_t>(env_or("GHUIM_BORROW_ITEMS", 8ull * warp_items));
    p.work.warp_scratch     = warp_scratch_d.get();
    p.work.warp_frames      = warp_frames_d.get();
    p.work.smem_scratch     = smem_mode;
    p.work.smem_offset      = static_cast<uint32_t>(deque_bytes);
    p.work.pool_offset      = static_cast<uint32_t>(smem - pool_bytes);
    p.work.lane_arena       = lane ? lane_arena.get() : nullptr;
    // Vertical mode is off by default: on dense data it is several times slower than the lane and warp paths.
    p.work.vert_on          = env_or("GHUIM_VERT", 0) != 0;
    p.work.control          = control_d.get();
    p.control               = control_d.get();
    p.headers               = headers.get();
    p.rings                 = rings.get();
    p.global_capacity       = kGlobalDeque;
    p.local_capacity        = kLocalDeque;
    p.frontier[0]           = frontier.get();
    p.frontier[1]           = frontier.get() + frontier_capacity;
    p.frontier_capacity     = frontier_capacity;
    p.target_tasks          = blocks * static_cast<uint32_t>(env_or("GHUIM_TASKS_PER_BLOCK", kTasksPerBlock));
    p.grain                 = std::max<uint64_t>(1024, total_occ / blocks); // ... and no task has more occurrences
    const char* wide_levels = std::getenv("GHUIM_WIDE_LEVELS");             // debugging: 0 skips the wide phase
    p.max_wide_levels       = wide_levels ? std::strtoul(wide_levels, nullptr, 10) : kMaxWideLevels;
    p.stats                 = stats_d.get();
    result.setup_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - setup_start).count();
    spdlog::info("Persistent: {} clusters x {} blocks x {} threads; root {} MiB, {} first-level tasks; heap {} MiB, "
                 "stacks {} MiB per block",
                 clusters, shape.cluster_size, shape.block, root.size() >> 20, first.size(), heap_bytes >> 20,
                 stack_bytes >> 20);
    spdlog::info("Persistent: scratch {} KiB per group, {}; warp mode up to {} entries", region_bytes >> 10,
                 smem_mode == 2   ? "block and warps in shared memory"
                 : smem_mode == 1 ? "block in shared memory, warps in global"
                                  : "in global memory",
                 warp_items);

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "create event");
    check_cuda(cudaEventCreate(&stop), "create event");
    // Keep the root database in L2 (REFIM keeps a copy per NUMA node): a persisting access-policy window on the
    // megakernel's stream, as large as the device allows.
    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream), "create stream");
    int max_window = 0, max_persist = 0;
    check_cuda(cudaDeviceGetAttribute(&max_window, cudaDevAttrMaxAccessPolicyWindowSize, device), "query L2");
    check_cuda(cudaDeviceGetAttribute(&max_persist, cudaDevAttrMaxPersistingL2CacheSize, device), "query L2");
    if (max_window > 0 && max_persist > 0 && env_or("GHUIM_L2_ROOT", 1) != 0) {
        const size_t window = std::min<size_t>(root.size(), static_cast<size_t>(max_window));
        check_cuda(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, std::min<size_t>(window, max_persist)),
                   "reserve persisting L2");
        cudaStreamAttrValue policy{};
        policy.accessPolicyWindow.base_ptr  = root_d.get();
        policy.accessPolicyWindow.num_bytes = window;
        policy.accessPolicyWindow.hitRatio  = std::min(1.0f, static_cast<float>(max_persist) / window);
        policy.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
        policy.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
        check_cuda(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &policy), "pin root in L2");
    }
    check_cuda(cudaEventRecord(start, stream), "record event");
    check_cuda(launch_megakernel(p, shape, clusters, stream), "launch megakernel");
    check_cuda(cudaEventRecord(stop, stream), "record event");
    spdlog::debug("Persistent: launched, watchdog {} s", seconds);
    check_cuda(cudaEventSynchronize(stop), "run megakernel");
    check_launch("megakernel");
    float ms = 0;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "time megakernel");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaCtxResetPersistingL2Cache(); // give the persisting lines back

    check_cuda(cudaMemcpy(&control, control_d.get(), sizeof(control), cudaMemcpyDeviceToHost), "read control");
    std::vector<BlockStats> stats(blocks);
    check_cuda(cudaMemcpy(stats.data(), stats_d.get(), blocks * sizeof(BlockStats), cudaMemcpyDeviceToHost),
               "read stats");
    if (control.abort != 0) {
        uint64_t tasks = 0, nodes = 0;
        for (const BlockStats& s : stats)
            tasks += s.tasks, nodes += s.work[1];
        spdlog::error(
                "Persistent: pending {}, idle {}, wide levels {} (frontiers {} / {}), {} tasks and {} nodes done; "
                "failed check at line {}",
                control.pending, control.idle, control.wide_levels, control.frontier_size[0], control.frontier_size[1],
                tasks, nodes, control.failed_line);
        spdlog::error("Persistent: debug n_tx {} n_primary {} n_occ {} top {} n_items {} bytes {} refs {} cls {}",
                      control.debug[1] >> 32, control.debug[1] & 0xffffffff, control.debug[2] >> 32,
                      control.debug[2] & 0xffffffff, control.debug[3], control.debug[4], control.debug[5] >> 32,
                      control.debug[5] & 0xffffffff);
    }
    if (control.abort != 0)
        throw std::runtime_error(std::string("persistent search aborted: ") +
                                 abort_name(static_cast<Abort>(control.abort)));

    BlockStats total{};
    for (const BlockStats& s : stats) {
        total.tasks += s.tasks;
        total.wide_tasks += s.wide_tasks;
        total.steals_dsmem += s.steals_dsmem;
        total.steals_near += s.steals_near;
        total.steals_far += s.steals_far;
        total.busy_ns += s.busy_ns;
        total.idle_ns += s.idle_ns;
        for (int w = 0; w < 24; ++w)
            total.work[w] += s.work[w];
    }
    const double wall_ns = ms * 1e6 * blocks;
    spdlog::info("Persistent: {:.3f} ms | wide levels {} ({} nodes) | {} tasks, {} nodes, {} counted directly, {} "
                 "splits/offers | steals {} dsmem, {} near, {} far | blocks busy {:.1f}%, idle {:.1f}%",
                 ms, control.wide_levels, total.wide_tasks, total.tasks, total.work[1], total.work[2], total.work[3],
                 total.steals_dsmem, total.steals_near, total.steals_far, 100 * total.busy_ns / wall_ns,
                 100 * total.idle_ns / wall_ns);
    const auto per = [](uint64_t a, uint64_t b) { return static_cast<double>(a) / std::max<uint64_t>(1, b); };
    spdlog::info("Persistent: nodes by block {}, warp {}, lane {} | lane pool: {} batches, {} jobs | {} siblings "
                 "borrowed from the block's loops",
                 total.work[1] - total.work[4] - total.work[5] - total.wide_tasks, total.work[4], total.work[5],
                 total.work[21], total.work[20], total.work[22]);
#ifdef GHUIM_TIMERS
    // Percentages of the warps' time in warp mode.
    const uint64_t warp_cycles = total.work[16];
    spdlog::info("Persistent: blocks in warp mode {:.1f}% | warp time: lane mode {:.1f}% (lanes busy {:.1f}% of it), "
                 "expanding {:.1f}%, checking lane eligibility {:.1f}%, deciding {:.1f}% (eligibility included), "
                 "taking work {:.1f}%, finding none {:.1f}% (sleeping {:.1f}%), at the end {:.1f}%",
                 100 * total.work[6] / wall_ns, 100 * per(total.work[7], warp_cycles),
                 100 * per(total.work[8], 32 * total.work[7]), 100 * per(total.work[9], warp_cycles),
                 100 * per(total.work[10], warp_cycles), 100 * per(total.work[18], warp_cycles),
                 100 * per(total.work[17], warp_cycles), 100 * per(total.work[19], warp_cycles),
                 100 * per(total.work[12], warp_cycles), 100 * per(total.work[15], warp_cycles));
    spdlog::info("Persistent: {:.0f} cycles per lane node, {:.0f} per warp node | lane converting {:.1f}% of lane "
                 "mining | {} lane subtrees in {} batches ({:.1f} per batch, {:.1f} nodes each)",
                 per(total.work[8], total.work[5]), per(total.work[9], total.work[4]),
                 100 * per(total.work[11], total.work[8]), total.work[13], total.work[14],
                 per(total.work[13], total.work[14]), per(total.work[5], total.work[13]));
#endif

    result.patterns  = total.work[0];
    result.tasks     = total.work[1];
    result.kernel_ms = ms;
    result.iterations.push_back({total.tasks, total.work[0], 0, ms});
    return result;
}

} // namespace ghuim::inline GHUIM_WIDTHS
