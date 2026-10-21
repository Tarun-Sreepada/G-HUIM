#pragma once

#include "cuda/buffer.cuh"
#include "data/dataset.hpp"
#include "mine/bfs/layout.cuh"

#include <vector>

namespace ghuim::inline GHUIM_WIDTHS::bfs {

// The whole input as one projected database in arena format (item names are the identity), on the device.
struct RootDatabase
{
    DeviceBuffer<char>    buffer;
    DbHeader              header{};
    std::vector<uint32_t> occ_counts; // occurrences of each item when the root is indexed, else empty
    uint32_t              occ_shift = 0; // position bits of a packed occurrence, for every database of the run
};

RootDatabase upload_root(const Dataset& data, bool tiled = false, uint32_t index_min_trans = 0, uint32_t index_sparsity = 0);

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
