#include "mine/bfs/layout.cuh"
#include "mine/bfs/root_image.hpp"

#include <algorithm>

namespace ghuim::bfs {

namespace {

// The root database in the tile layout: transactions sorted by length, 32 to a tile, entries column-major.
std::vector<char> tiled_root_image(const Dataset& data)
{
    const uint64_t n_trans = data.transaction_count();
    const uint64_t n_items = data.original_item.size();
    auto           length  = [&](size_t t) { return data.tx_begin[t + 1] - data.tx_begin[t]; };

    std::vector<uint32_t> order(n_trans);
    for (size_t t = 0; t < n_trans; ++t)
        order[t] = static_cast<uint32_t>(t);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return length(a) < length(b); });

    const uint64_t stride = tile_stride(n_trans);
    uint64_t       padded = 0;
    uint32_t       max_len = 0;
    for (uint64_t first = 0; first < n_trans; first += stride) {
        const uint32_t tile_len = length(order[std::min(first + stride, n_trans) - 1]); // sorted: the last is longest
        padded += uint64_t{tile_len} * stride;
        max_len = std::max(max_len, tile_len);
    }
    const DbLayout    layout = DbLayout::of(n_trans, padded, n_items, true);
    std::vector<char> image(layout.bytes, 0);
    *reinterpret_cast<DbHeader*>(image.data()) =
            DbHeader{static_cast<uint32_t>(n_trans), static_cast<uint32_t>(padded), static_cast<uint32_t>(n_items), max_len,
                     layout.bytes};
    auto* globals = reinterpret_cast<uint32_t*>(image.data() + layout.items);
    for (uint32_t i = 0; i < n_items; ++i)
        globals[i] = i;
    auto* txs   = reinterpret_cast<Tx*>(image.data() + layout.tx);
    auto* eitem = reinterpret_cast<uint32_t*>(image.data() + layout.eitem);
    auto* eutil = reinterpret_cast<uint32_t*>(image.data() + layout.eutil);

    uint64_t tile_base = 0;
    for (uint64_t first = 0; first < n_trans; first += stride) {
        const uint64_t end      = std::min(first + stride, n_trans);
        const uint32_t tile_len = length(order[end - 1]);
        for (uint64_t slot = first; slot < end; ++slot) {
            const uint32_t t     = order[slot];
            const uint32_t begin = static_cast<uint32_t>(tile_base + slot - first);
            txs[slot]            = Tx{begin, length(t), 0};
            for (uint32_t k = 0; k < length(t); ++k) {
                eitem[begin + uint64_t{k} * stride] = data.items[data.tx_begin[t] + k];
                eutil[begin + uint64_t{k} * stride] = data.utilities[data.tx_begin[t] + k];
            }
        }
        tile_base += uint64_t{tile_len} * stride;
    }
    return image;
}

} // namespace

std::vector<char> root_image(const Dataset& data, bool tiled, uint32_t index_min_trans)
{
    if (tiled)
        return tiled_root_image(data);
    const uint64_t    n_trans = data.transaction_count(), n_entries = data.items.size();
    const uint64_t    n_items = data.original_item.size();
    const bool indexed = index_min_trans != 0 && n_trans >= index_min_trans;
    const DbLayout    layout  = DbLayout::of(n_trans, n_entries, n_items, false, indexed);
    std::vector<char> image(layout.bytes, 0);

    uint32_t max_len = 0;
    for (size_t t = 0; t < n_trans; ++t)
        max_len = std::max(max_len, data.tx_begin[t + 1] - data.tx_begin[t]);
    *reinterpret_cast<DbHeader*>(image.data()) =
            DbHeader{static_cast<uint32_t>(n_trans), static_cast<uint32_t>(n_entries), static_cast<uint32_t>(n_items),
                     max_len, layout.bytes, indexed};

    auto* globals = reinterpret_cast<uint32_t*>(image.data() + layout.items);
    for (uint32_t i = 0; i < n_items; ++i)
        globals[i] = i;
    auto* txs     = reinterpret_cast<Tx*>(image.data() + layout.tx);
    auto* eitem   = reinterpret_cast<uint32_t*>(image.data() + layout.eitem);
    auto* eutil   = reinterpret_cast<uint32_t*>(image.data() + layout.eutil);
    for (size_t t = 0; t < n_trans; ++t)
        txs[t] = Tx{data.tx_begin[t], data.tx_begin[t + 1] - data.tx_begin[t], 0};
    for (size_t i = 0; i < n_entries; ++i) {
        eitem[i] = data.items[i];
        eutil[i] = data.utilities[i];
    }
    if (indexed) {
        auto* begin = reinterpret_cast<uint32_t*>(image.data() + layout.occ_begin);
        auto* occ = reinterpret_cast<Occurrence*>(image.data() + layout.occ);
        for (const uint32_t item : data.items)
            ++begin[item + 1];
        for (uint32_t i = 0; i < n_items; ++i)
            begin[i + 1] += begin[i];
        std::vector<uint32_t> cursor(begin, begin + n_items);
        for (uint32_t t = 0; t < n_trans; ++t)
            for (uint32_t pos = 0; pos < txs[t].length; ++pos)
                occ[cursor[eitem[txs[t].begin + pos]]++] = Occurrence{t, pos};
    }
    return image;
}

} // namespace ghuim::bfs
