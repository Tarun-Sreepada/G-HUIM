#include "mine/dfs/root_image.hpp"

#include "mine/dfs/layout.cuh"

#include <algorithm>

namespace ghuim::dfs {

std::vector<char> root_image(const Dataset& data)
{
    const uint64_t    n_trans = data.transaction_count(), n_entries = data.items.size();
    const uint64_t    n_items = data.original_item.size();
    const DbLayout    layout  = DbLayout::of(n_trans, n_entries, n_items);
    std::vector<char> image(layout.bytes, 0);
    uint32_t          max_len = 0;
    uint32_t          entries_by_length[kLengthBins] = {};
    for (size_t t = 0; t < n_trans; ++t) {
        const uint32_t length = data.tx_begin[t + 1] - data.tx_begin[t];
        max_len               = std::max(max_len, length);
        entries_by_length[length_bin(length)] += length;
    }
    *reinterpret_cast<DbHeader*>(image.data()) =
            DbHeader{static_cast<uint32_t>(n_trans), static_cast<uint32_t>(n_entries), static_cast<uint32_t>(n_items),
                     max_len,                         layout.bytes,                    median_length_bin(entries_by_length),
                     0};
    auto* globals = reinterpret_cast<uint32_t*>(image.data() + layout.items);
    for (uint32_t i = 0; i < n_items; ++i)
        globals[i] = i;
    auto* txs     = reinterpret_cast<Tx*>(image.data() + layout.tx);
    auto* entries = reinterpret_cast<Entry*>(image.data() + layout.entries);
    for (size_t t = 0; t < n_trans; ++t)
        txs[t] = Tx{data.tx_begin[t], data.tx_begin[t + 1] - data.tx_begin[t], 0};
    for (size_t i = 0; i < n_entries; ++i)
        entries[i] = Entry{data.items[i], data.utilities[i]};
    return image;
}

} // namespace ghuim::dfs
