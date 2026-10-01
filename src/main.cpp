#include "app/cli.hpp"
#include "app/logging.hpp"
#include "data/dataset.hpp"
#include "mine/miner.hpp"

#include <chrono>
#include <spdlog/spdlog.h>

namespace {

double seconds_since(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void log_iterations(const ghuim::MiningResult& result)
{
    spdlog::debug("Iteration: tasks / itemsets");
    for (size_t i = 0; i < result.iterations.size(); ++i)
        spdlog::debug("  {:>4} {:>12} / {:>12}", i + 1, result.iterations[i].tasks, result.iterations[i].patterns);
}

} // namespace

int main(int argc, char* argv[])
{
    ghuim::Config config;
    CLI::App      app{"GPU-accelerated high-utility itemset mining (EFIM)", "G-HUIM"};
    ghuim::configure_cli(app, config);
    try {
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError& error) {
        return app.exit(error);
    }

    try {
        ghuim::initialize_logging(config);
        const auto start = std::chrono::steady_clock::now();

        const ghuim::Dataset data = ghuim::load_dataset(config.input, config.separator, config.mining.minimum_utility);
        const double         load_s = seconds_since(start);
        for (const auto& [stage, ms] : data.timings)
            spdlog::debug("  {:<28} {:9.1f} ms", stage, ms);
        spdlog::info("Loaded {}: {} transactions, {} promising items, {} first-level tasks ({:.3f} s)", config.input,
                     data.input_transactions, data.original_item.size(), data.primary.size(), load_s);

        const auto                mine_start = std::chrono::steady_clock::now();
        const ghuim::MiningResult result     = ghuim::mine(data, config.mining);
        spdlog::info("Mined with {}: {} tasks in {} iterations, kernels {:.3f} s, peak live arena {} MiB ({:.3f} s)",
                     ghuim::method_name(config.mining.method), result.tasks, result.iterations.size(),
                     result.kernel_ms / 1000, result.peak_arena >> 20, seconds_since(mine_start));
        log_iterations(result);

        spdlog::info("Number of patterns: {}", result.patterns);
        spdlog::info("Total time: {:.3f} s", seconds_since(start));
        return 0;
    }
    catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return 1;
    }
}
