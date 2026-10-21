#include "mine/bfs/host/level_miner.hpp"

#include "cuda/nvtx.hpp"

namespace ghuim::inline GHUIM_WIDTHS::bfs {

void LevelMiner::reset_control()
{
    GHUIM_RANGE("bfs: reset control");
    host_.arena_top  = 0;
    host_.next_count = 0;
    host_.error      = 0;
    check_cuda(cudaMemcpyAsync(control_.get(), &host_, sizeof(Control), cudaMemcpyHostToDevice, streams_[0]),
               "reset control");
}

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
