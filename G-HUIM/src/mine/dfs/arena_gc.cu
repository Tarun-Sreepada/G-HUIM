#include "mine/dfs/arena_gc.hpp"

#include "mine/host/transfer.hpp"
#include "mine/primitives.cuh"
#include "mine/dfs/schedule.cuh"

#include <stdexcept>
#include <vector>

namespace ghuim::dfs {

Compaction compact_arena(const ArenaBuffers& buffers, uint32_t stack_size, cudaStream_t stream)
{
    auto&          arena_      = buffers.arena;
    auto&          scratch_    = buffers.scratch;
    auto&          stack_      = buffers.stack;
    auto&          control_    = buffers.control;
    auto&          bitmap_     = buffers.bitmap;
    auto&          counts_     = buffers.counts;
    auto&          block_sums_ = buffers.block_sums;
    auto&          scan_total_ = buffers.scan_total;
    const uint32_t words_      = buffers.words;
        mark_live(control_.get(), stack_.get(), stack_size, bitmap_.get(), words_, stream);
        count_live(bitmap_.get(), counts_.get(), words_, stream);
        exclusive_scan(counts_.get(), words_, block_sums_.get(), scan_total_.get(), stream);
        check_cuda(cudaStreamSynchronize(stream), "count live databases");
        uint32_t live_count = 0;
        download(&live_count, scan_total_.get(), 1);

        DeviceBuffer<uint32_t> live(live_count), moved_to(live_count);
        extract_live(bitmap_.get(), counts_.get(), words_, live.get(), stream);
        live_sizes(arena_.get(), live.get(), live_count, moved_to.get(), stream);
        exclusive_scan(moved_to.get(), live_count, block_sums_.get(), scan_total_.get(), stream);
        check_cuda(cudaStreamSynchronize(stream), "size live databases");
        uint32_t total_units = 0;
        download(&total_units, scan_total_.get(), 1);

        // Move databases in address-ordered groups that fit in scratch: gather a group into
        // scratch, then scatter it to its new place. A destination always lies below every
        // later database's source, so groups never clobber data still waiting to move.
        std::vector<uint32_t> from(live_count), to(live_count);
        download(from.data(), live.get(), live_count);
        download(to.data(), moved_to.get(), live_count);
        std::vector<uint64_t> src, staged, dst, bytes;
        auto flush = [&] {
            const auto n = static_cast<uint32_t>(src.size());
            DeviceBuffer<uint64_t> d_src(n), d_staged(n), d_dst(n), d_bytes(n);
            upload(d_src.get(), src.data(), n);
            upload(d_staged.get(), staged.data(), n);
            upload(d_dst.get(), dst.data(), n);
            upload(d_bytes.get(), bytes.data(), n);
            copy_databases(scratch_.get(), d_staged.get(), arena_.get(), d_src.get(), d_bytes.get(), n, stream);
            copy_databases(arena_.get(), d_dst.get(), scratch_.get(), d_staged.get(), d_bytes.get(), n, stream);
            check_cuda(cudaStreamSynchronize(stream), "move databases");
            src.clear(), staged.clear(), dst.clear(), bytes.clear();
        };
        uint64_t used = 0;
        for (uint32_t i = 0; i < live_count; ++i) {
            const uint64_t size = uint64_t{(i + 1 < live_count ? to[i + 1] : total_units) - to[i]} * kArenaUnit;
            if (from[i] == to[i] && src.empty())
                continue; // already in place, and nothing before it is still moving
            if (size > scratch_.size())
                throw std::runtime_error("a projected database is larger than the scratch buffer");
            if (used + size > scratch_.size()) {
                flush();
                used = 0;
            }
            src.push_back(uint64_t{from[i]} * kArenaUnit);
            staged.push_back(used);
            dst.push_back(uint64_t{to[i]} * kArenaUnit);
            bytes.push_back(size);
            used += size;
        }
        if (!src.empty())
            flush();
        remap_tasks(stack_.get(), stack_size, live.get(), moved_to.get(), live_count, stream);
    const auto arena_top = static_cast<decltype(Control::arena_top)>(total_units) * kArenaUnit;
    upload(&control_.get()->arena_top, &arena_top, 1);
    check_cuda(cudaStreamSynchronize(stream), "compact arena");
    return {live_count, arena_top};
}

} // namespace ghuim::dfs
