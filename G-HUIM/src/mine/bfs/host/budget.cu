#include "mine/bfs/host/budget.hpp"

#include "cuda/buffer.cuh"
#include "mine/bfs/kernels.cuh"
#include "mine/primitives.cuh"

#include <algorithm>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace ghuim::inline GHUIM_WIDTHS::bfs {

// Task indices and scan lengths are 32-bit: the most tasks a level can number.
constexpr uint64_t kIndexLimit = 0xfffffffeull;
constexpr uint64_t kReservedBytes = 512ull << 20; // runtime, kernels and slack kept free

Budget plan_budget(const MiningOptions& options, const AutoTune& tune, uint64_t first_level_tasks)
{
    size_t free_bytes = 0, total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "query device memory");
    if (free_bytes <= kReservedBytes + (64ull << 20))
        throw std::runtime_error("not enough free device memory");
    const uint64_t usable = free_bytes - kReservedBytes;
    const uint64_t vertical = uint64_t{tune.vert_warps} * vertical_warp_words(vertical_slots(tune)) * sizeof(uint32_t);
    const uint64_t budget =
            (options.memory_bytes == 0 ? usable : std::min<uint64_t>(options.memory_bytes, usable)) - std::min(vertical, usable / 2);

    Budget plan{};
    plan.per_task = 2 * sizeof(Task) + sizeof(Node) + sizeof(uint64_t) + sizeof(uint32_t);
    // The task arrays may take a quarter of the budget, unless --stack-capacity asks for fewer tasks
    // (0 = no limit of its own).
    const uint64_t fit       = std::min(budget / 4 / plan.per_task, kIndexLimit);
    const uint64_t requested = options.stack_capacity == 0 ? kIndexLimit : options.stack_capacity;
    plan.capacity            = std::min(requested, fit);
    if (plan.capacity < first_level_tasks)
        throw std::runtime_error("not enough device memory for the first level's tasks");
    if (plan.capacity < requested)
        spdlog::warn("Level capacity limited to {} tasks by device memory", plan.capacity);

    plan.task_bytes       = plan.capacity * plan.per_task + scan_block_count(plan.capacity + 1) * sizeof(uint64_t);
    plan.scratch_capacity = std::clamp<uint64_t>(budget / 8, 256ull << 20, 4ull << 30);
    if (budget < plan.task_bytes + plan.scratch_capacity + (64ull << 20))
        throw std::runtime_error("not enough device memory for the task arrays; lower --stack-capacity");
    plan.region_bytes =
            std::min<uint64_t>((budget - plan.task_bytes - plan.scratch_capacity) / 2 / kArenaUnit * kArenaUnit,
                               0xffffffffull * kArenaUnit);
    spdlog::info("BFS memory: {} MiB per arena region (x2), scratch {} MiB, {} tasks per level ({} MiB)",
                 plan.region_bytes >> 20, plan.scratch_capacity >> 20, plan.capacity, plan.task_bytes >> 20);
    return plan;
}

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
