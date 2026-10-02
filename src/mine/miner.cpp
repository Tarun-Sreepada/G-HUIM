#include "mine/miner.hpp"

#include "mine/bfs/bfs_miner.hpp"
#include "mine/dfs/dfs_miner.hpp"

#include <cuda_runtime.h>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ghuim {

void initialize_device()
{
    setenv("CUDA_VISIBLE_DEVICES", "0", /*overwrite=*/0); // read by the driver at its first initialization
    const cudaError_t error = cudaFree(nullptr); // the usual way to force context creation
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("create CUDA context: ") + cudaGetErrorString(error));
}

MiningResult mine(const Dataset& data, const MiningOptions& options)
{
    return options.search == Search::Bfs ? mine_bfs(data, options) : mine_dfs(data, options);
}

} // namespace ghuim
