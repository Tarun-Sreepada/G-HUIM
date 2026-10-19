#include "app/steps.hpp"

namespace ghuim {

double seconds_since(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

} // namespace ghuim
