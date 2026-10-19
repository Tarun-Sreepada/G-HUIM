#include "app/steps.hpp"

#include <spdlog/spdlog.h>

namespace ghuim {

double wait_for_context(std::future<double>& context)
{
    const auto   wait_start = Clock::now();
    const double context_s  = context.get();
    const double wait_s     = seconds_since(wait_start);
    spdlog::info("CUDA context created ({:.3f} s, overlapped with loading; {:.3f} s waited after it)", context_s,
                 wait_s);
    return wait_s;
}

} // namespace ghuim
