#include "mine/bfs/host/level_miner.hpp"

#include "cuda/nvtx.hpp"

#include <utility>

namespace ghuim::inline GHUIM_WIDTHS::bfs {

// Bins the chunk [begin, end) and runs every bin on its own stream.
void LevelMiner::mine_chunk(uint32_t level, uint32_t begin, uint32_t end, uint64_t scratch_base)
{
    const char* src = level == 1 ? root_.get() : regions_[(level - 1) & 1].get();
    char*       dst = regions_[level & 1].get();

    {
        GHUIM_RANGE("bfs: bin chunk + readback");
        launch_bin_chunk(control_.get(), current_, begin, end, order_.get(), tune_, streams_[0]);
        check_cuda(cudaMemcpyAsync(&host_, control_.get(), sizeof(Control), cudaMemcpyDeviceToHost, streams_[0]),
                   "read bins");
        check_cuda(cudaStreamSynchronize(streams_[0]), "bin chunk");
    }
    GHUIM_RANGE("bfs: launch bins + sync");

    const BatchView view{src,
                         dst,
                         budget_.region_bytes,
                         control_.get(),
                         current_,
                         prefix_.get(),
                         scratch_base,
                         scratch_.get(),
                         next_,
                         static_cast<uint32_t>(budget_.capacity),
                         nodes_.get(),
                         options_.minimum_utility,
                         tune_.staged ? tune_.io_warps : 0,
                         tune_.ring_bytes,
                         tune_.tiled,
                         options_.index_min_trans,
                         options_.index_sparsity,
                         options_.local_tail_items,
                         options_.local_tail_trans,
                         options_.refine_rounds,
                         !collect_,
                         vertical_.get(),
                         vertical_slots(tune_),
                         vertical_warp_words(vertical_slots(tune_))};
    cudaEvent_t     start = nullptr, stop = nullptr;
    std::vector<std::pair<uint32_t, std::pair<cudaEvent_t, cudaEvent_t>>> timed;
    for (uint32_t cell = 0; cell < kCells; ++cell) {
        const uint32_t  count  = host_.cell_count[cell];
        const uint32_t* tasks  = order_.get() + host_.cell_start[cell];
        const uint32_t  bin = cell / kCellKinds, tail = cell % kCellKinds;
        cudaStream_t    stream = options_.bin_times ? streams_[0] : streams_[bin + tail];
        if (count == 0)
            continue;
        if (options_.bin_times) {
            check_cuda(cudaEventCreate(&start), "create event");
            check_cuda(cudaEventCreate(&stop), "create event");
            check_cuda(cudaEventRecord(start, stream), "record event");
            bin_tasks_[cell] += count;
        }
        if (tail == kVertKind)
            launch_vertical_tasks(view, tasks, count, tune_, stream);
        else if (bin == 0)
            launch_warp_tasks(view, tasks, count, tune_, stream);
        else if (bin < kOptHugeBin)
            launch_block_tasks(view, tasks, count, bin, tune_, stream);
        else if (bin == kOptHugeBin)
            launch_huge_tasks(view, tasks, count, host_.cell_max_slices[cell], tune_, stream);
        else
            launch_subwarp_tasks(view, tasks, count, 1u << (bin - kOptSubBin), tail == kTailKind, tune_, stream);
        if (options_.bin_times) {
            check_cuda(cudaEventRecord(stop, stream), "record event");
            timed.push_back({cell, {start, stop}});
        }
    }
    synchronize("mine level");
    for (const auto& [bin, events] : timed) {
        float ms = 0;
        check_cuda(cudaEventElapsedTime(&ms, events.first, events.second), "event time");
        bin_ms_[bin] += ms;
        cudaEventDestroy(events.first);
        cudaEventDestroy(events.second);
    }
}

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
