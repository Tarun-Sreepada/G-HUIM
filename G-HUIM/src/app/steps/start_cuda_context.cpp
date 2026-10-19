#include "app/steps.hpp"

#include "cuda/nvtx.hpp"
#include "mine/miner.hpp"

namespace ghuim {

// The context is a fixed cost of the process, not of mining: "Total time" leaves out any wait for it,
// "Wall time" includes everything.
std::future<double> start_cuda_context()
{
    return std::async(std::launch::async, [] {
        GHUIM_RANGE("create CUDA context");
        const auto context_start = Clock::now();
        initialize_device();
        return seconds_since(context_start);
    });
}

} // namespace ghuim
