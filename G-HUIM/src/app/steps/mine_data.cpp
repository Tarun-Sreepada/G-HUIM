#include "app/steps.hpp"

#include "cuda/nvtx.hpp"
#include "mine/miner.hpp"

#include <spdlog/spdlog.h>

namespace ghuim {
namespace {

void log_iterations(const MiningResult& result)
{
    spdlog::debug("Iteration: tasks / itemsets");
    for (size_t i = 0; i < result.iterations.size(); ++i)
        spdlog::debug("  {:>4} {:>12} / {:>12}", i + 1, result.iterations[i].tasks, result.iterations[i].patterns);
}

void log_depths(const MiningResult& result)
{
    spdlog::debug("Itemset length: tasks evaluated / high-utility itemsets");
    for (size_t d = 1; d < result.depth_tasks.size(); ++d)
        if (result.depth_tasks[d] != 0)
            spdlog::debug("  {:>3}{} {:>12} / {:>12}", d, d + 1 == result.depth_tasks.size() ? "+" : " ",
                          result.depth_tasks[d], result.depth_patterns[d]);
}

} // namespace

MiningResult mine_data(const Dataset& data, const Config& config)
{
    const auto          mine_start = Clock::now();
    const MiningResult result     = [&] {
        GHUIM_RANGE("mine");
        return mine(data, config.mining);
    }();
    spdlog::info("Mined with {} search, {} partitioning: {} tasks in {} iterations, kernels {:.3f} s, peak live arena "
                 "{} MiB ({:.3f} s)",
                 search_name(config.mining.search), method_name(config.mining.method), result.tasks,
                 result.iterations.size(), result.kernel_ms / 1000, result.peak_arena >> 20, seconds_since(mine_start));
    if (result.setup_ms > 0)
        spdlog::info("  {:<28} {:9.1f} ms", "build + upload GPU database", result.setup_ms);
    if (result.compactions != 0 || result.deferrals != 0)
        spdlog::info("  {} arena compactions, {} deferred tasks", result.compactions, result.deferrals);
    log_iterations(result);
    log_depths(result);
    return result;
}

} // namespace ghuim
