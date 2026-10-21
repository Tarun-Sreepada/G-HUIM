#pragma once
// Searching and comparing transactions stored in the tile layout.

#include "mine/bfs/device/child.cuh"
#include "mine/bfs/device/modes.cuh"
#include "mine/bfs/placement.cuh"

namespace ghuim::inline GHUIM_WIDTHS::bfs {

// Position of `item` in a sorted tiled transaction (keys at col[k * kTile]), or -1.
__device__ inline int find_item_tiled(const Key* col, uint32_t length, uint32_t item, uint32_t stride)
{
    int low = 0, high = static_cast<int>(length) - 1;
    while (low <= high) {
        const int      mid = (low + high) >> 1;
        const uint32_t key = col[static_cast<uint32_t>(mid) * stride];
        if (key == item)
            return mid;
        if (key < item)
            low = mid + 1;
        else
            high = mid - 1;
    }
    return -1;
}

// same_kept_items over tiled key arrays.
__device__ inline bool same_kept_items_tiled(const Key* ea, uint32_t a_length, uint32_t i, const Key* eb,
                                             uint32_t b_length, uint32_t j, const uint32_t* rename, uint32_t stride)
{
    for (;;) {
        while (i < a_length && rename[ea[i * stride]] == kNone)
            ++i;
        while (j < b_length && rename[eb[j * stride]] == kNone)
            ++j;
        if (i == a_length || j == b_length)
            return i == a_length && j == b_length;
        if (ea[i * stride] != eb[j * stride])
            return false;
        ++i;
        ++j;
    }
}

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
