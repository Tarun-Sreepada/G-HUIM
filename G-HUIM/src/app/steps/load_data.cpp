#include "app/steps.hpp"

#include "cuda/nvtx.hpp"

#include <spdlog/spdlog.h>

namespace ghuim {

Dataset load_data(const Config& config, Clock::time_point start)
{
    Dataset data = [&] {
        GHUIM_RANGE("load dataset");
        return load_dataset(config.input, config.separator, config.mining.minimum_utility);
    }();
    const double load_s = seconds_since(start);
    for (const auto& [stage, ms] : data.timings)
        spdlog::debug("  {:<28} {:9.1f} ms", stage, ms);
    spdlog::info("Loaded {}: {} transactions, {} promising items, {} first-level tasks ({:.3f} s)", config.input,
                 data.input_transactions, data.original_item.size(), data.primary.size(), load_s);
    return data;
}

} // namespace ghuim
