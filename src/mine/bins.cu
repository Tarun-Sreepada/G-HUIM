#include "mine/bins.hpp"

#include "cuda/buffer.cuh"
#include "mine/kernels.cuh"

#include <spdlog/spdlog.h>

namespace ghuim {
namespace {

// Measures the device: resident warps and the block sizes that keep full occupancy.
AutoTune make_auto_tune()
{
    int device = 0;
    check_cuda(cudaGetDevice(&device), "get device");
    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device), "query device");

    AutoTune tune{};
    tune.resident_warps = static_cast<uint32_t>(props.multiProcessorCount) * (props.maxThreadsPerMultiProcessor / 32);
    measure_optimized_occupancy(tune, static_cast<uint32_t>(props.multiProcessorCount));
    spdlog::info("Auto-tune: {} ({} SMs), {} resident warps, max block {} threads, packed-warp block {} threads, "
                 "lane-group block {} threads, huge tasks up to {} blocks",
                 props.name, props.multiProcessorCount, tune.resident_warps, tune.max_block_warps * 32,
                 tune.packed_block_warps * 32, tune.subwarp_block_warps * 32, tune.max_slices);
    return tune;
}

} // namespace

Bins bins_for(Method method)
{
    if (method == Method::Static)
        return {{false, 1, {}}, {Bins::Block}};
    if (method == Method::Adaptive) // up to 32 transactions on a thread, up to 1024 on a warp, else a block
        return {{false, 3, {32, 1024}}, {Bins::Thread, Bins::Warp, Bins::Block}};
    Bins bins{{true, kOptSubBin + kOptSubBins, {}, make_auto_tune()}, {}};
    for (uint32_t b = 0; b < kOptBins; ++b)
        bins.kind[b] = Bins::Optimized;
    bins.kind[kOptHugeBin]   = Bins::Huge;
    for (uint32_t i = 0; i < kOptSubBins; ++i)
        bins.kind[kOptSubBin + i] = Bins::Subwarp;
    return bins;
}

} // namespace ghuim
