#include "app/steps.hpp"

#include <spdlog/spdlog.h>

namespace ghuim {

void report_totals(const MiningResult& result, Clock::time_point start, double context_wait_s)
{
    spdlog::info("Number of patterns: {}", result.patterns);
    spdlog::info("Total time: {:.3f} s", seconds_since(start) - context_wait_s);
    spdlog::info("Wall time: {:.3f} s (including CUDA context creation)", seconds_since(start));
}

} // namespace ghuim
