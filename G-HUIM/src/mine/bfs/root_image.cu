#include "mine/bfs/layout.cuh"
#include "mine/bfs/root_image.hpp"
#include "mine/host/transfer.hpp"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <spdlog/spdlog.h>
#include <bit>
#include <stdexcept>
#include <string>

namespace ghuim::inline GHUIM_WIDTHS::bfs {

namespace {

// The root database in the tile layout: transactions sorted by length, 32 to a tile, entries column-major.
// With `indexed`, the occurrence index follows (built here: each item's occurrences in slot order, padding
// left out), and `occ_counts` receives every item's occurrence count.
std::vector<char> tiled_root_image(const Dataset& data, bool indexed, uint32_t occ_shift, std::vector<uint32_t>& occ_counts)
{
    if (data.original_item.size() > kMaxDbItems)
        throw std::runtime_error("too many items for " + std::to_string(8 * sizeof(Key)) + "-bit keys (raise --minutil)");
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
    spdlog::info("Tiles: root holds {} entries in {} slots, padding {:.1f}%", data.items.size(), padded,
                 100.0 * static_cast<double>(padded - data.items.size()) / static_cast<double>(std::max<uint64_t>(padded, 1)));
    const DbLayout    layout = DbLayout::of(n_trans, padded, n_items, true, indexed);
    std::vector<char> image(layout.bytes, 0);
    *reinterpret_cast<DbHeader*>(image.data()) =
            DbHeader{static_cast<uint32_t>(n_trans), static_cast<uint32_t>(padded), static_cast<uint32_t>(n_items), max_len,
                     layout.bytes, indexed};
    auto* globals = reinterpret_cast<uint32_t*>(image.data() + layout.items);
    for (uint32_t i = 0; i < n_items; ++i)
        globals[i] = i;
    auto* txs   = reinterpret_cast<Tx*>(image.data() + layout.tx);
    auto* eitem = reinterpret_cast<Key*>(image.data() + layout.eitem);
    auto* eutil = reinterpret_cast<Util*>(image.data() + layout.eutil);

    uint64_t tile_base = 0;
    for (uint64_t first = 0; first < n_trans; first += stride) {
        const uint64_t end      = std::min(first + stride, n_trans);
        const uint32_t tile_len = length(order[end - 1]);
        for (uint64_t slot = first; slot < end; ++slot) {
            const uint32_t t     = order[slot];
            const uint32_t begin = static_cast<uint32_t>(tile_base + slot - first);
            txs[slot]            = Tx{begin, length(t), 0};
            for (uint32_t k = 0; k < length(t); ++k) {
                eitem[begin + uint64_t{k} * stride] = static_cast<Key>(data.items[data.tx_begin[t] + k]);
                eutil[begin + uint64_t{k} * stride] = data.utilities[data.tx_begin[t] + k];
            }
        }
        tile_base += uint64_t{tile_len} * stride;
    }
    if (indexed) {
        auto* begin = reinterpret_cast<uint32_t*>(image.data() + layout.occ_begin);
        auto* occ   = reinterpret_cast<Occurrence*>(image.data() + layout.occ);
        occ_counts.assign(n_items, 0);
        for (uint64_t e = 0; e < data.items.size(); ++e)
            ++occ_counts[data.items[e]];
        std::vector<uint32_t> cursor(n_items);
        uint32_t              run = 0;
        for (uint32_t z = 0; z < n_items; ++z) {
            begin[z] = cursor[z] = run;
            run += occ_counts[z];
        }
        begin[n_items] = run;
        for (uint32_t slot = 0; slot < n_trans; ++slot)
            for (uint32_t k = 0; k < txs[slot].length; ++k)
                occ[cursor[eitem[txs[slot].begin + uint64_t{k} * stride]]++] = make_occurrence(slot, k, occ_shift);
    }
    return image;
}

// Occurrence of entry e of transaction t at position p, packed (layout.cuh).
__global__ void entry_occurrences(const Tx* txs, uint32_t n_trans, uint32_t occ_shift, Occurrence* occ)
{
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n_trans; t += gridDim.x * blockDim.x) {
        const Tx tx = txs[t];
        for (uint32_t pos = 0; pos < tx.length; ++pos)
            occ[tx.begin + pos] = make_occurrence(t, pos, occ_shift);
    }
}

__global__ void count_items(const Key* eitem, uint64_t n_entries, uint32_t* count)
{
    for (uint64_t e = blockIdx.x * uint64_t{blockDim.x} + threadIdx.x; e < n_entries; e += uint64_t{gridDim.x} * blockDim.x)
        atomicAdd(count + eitem[e], 1u);
}

// The root's occurrence index, on the device: every entry's (transaction, position), sorted stably by item
// so each item's occurrences stay in transaction order, and the exclusive prefix of the item counts.
void build_root_index(char* root, const DbLayout& layout, uint32_t n_trans, uint32_t n_entries, uint32_t n_items,
                      uint32_t occ_shift)
{
    auto*          eitem = reinterpret_cast<const Key*>(root + layout.eitem);
    auto*          occ   = reinterpret_cast<Occurrence*>(root + layout.occ);
    auto*          begin = reinterpret_cast<uint32_t*>(root + layout.occ_begin);
    const uint32_t grid  = std::min<uint64_t>((n_entries + 255) / 256 + 1, 65535);
    DeviceBuffer<Occurrence>         unsorted(n_entries);
    DeviceBuffer<Key>                keys(n_entries);
    DeviceBuffer<uint32_t>           count(n_items + 1);
    entry_occurrences<<<grid, 256>>>(reinterpret_cast<const Tx*>(root + layout.tx), n_trans, occ_shift, unsorted.get());
    check_cuda(cudaMemset(count.get(), 0, (n_items + 1) * sizeof(uint32_t)), "clear item counts");
    count_items<<<grid, 256>>>(eitem, n_entries, count.get());
    check_launch("launch root index kernels");

    const int end_bit = std::max(1, static_cast<int>(std::bit_width(n_items - 1))); // keys are below n_items
    size_t    sort_bytes = 0, scan_bytes = 0;
    check_cuda(cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, eitem, keys.get(), unsorted.get(), occ, n_entries, 0, end_bit),
               "size root index sort");
    check_cuda(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, count.get(), begin, n_items + 1), "size root index scan");
    DeviceBuffer<char> temp(std::max(sort_bytes, scan_bytes));
    check_cuda(cub::DeviceRadixSort::SortPairs(temp.get(), sort_bytes, eitem, keys.get(), unsorted.get(), occ, n_entries, 0, end_bit),
               "sort root index");
    check_cuda(cub::DeviceScan::ExclusiveSum(temp.get(), scan_bytes, count.get(), begin, n_items + 1), "scan root index");
    synchronize("build root index");
}

// Position bits of the run's packed occurrences; with an index, the root's transaction slots must fit above
// them (no projected database has more transactions or longer ones).
uint32_t occurrence_shift(uint64_t n_trans, uint32_t max_len, bool indexed)
{
    if (indexed && !occurrences_fit(n_trans, max_len))
        throw std::runtime_error(std::to_string(n_trans) + " transactions of up to " + std::to_string(max_len) +
                                 " items do not fit 32-bit occurrences (turn indexes off with --index-min-trans 0)");
    return occurrence_pos_bits(max_len);
}

} // namespace

RootDatabase upload_root(const Dataset& data, bool tiled, uint32_t index_min_trans, uint32_t index_sparsity)
{
    RootDatabase root;
    if (tiled) {
        const bool indexed = wants_index(data.transaction_count(), data.items.size(), data.original_item.size(),
                                         index_min_trans, index_sparsity);
        root.occ_shift                = occurrence_shift(data.transaction_count(), data.max_length, indexed);
        const std::vector<char> image = tiled_root_image(data, indexed, root.occ_shift, root.occ_counts);
        root.header                   = *reinterpret_cast<const DbHeader*>(image.data());
        root.buffer                   = DeviceBuffer<char>(image.size());
        upload(root.buffer.get(), image.data(), image.size());
        return root;
    }
    // Only the header, item names and transaction table are built here; the entry arrays are the dataset's
    // own and go straight to their place in the device image.
    const uint64_t n_trans = data.transaction_count(), n_entries = data.items.size();
    const uint64_t n_items = data.original_item.size();
    if (n_items > kMaxDbItems)
        throw std::runtime_error(std::to_string(n_items) + " items survive TWU pruning; at most " +
                                 std::to_string(kMaxDbItems) + " fit " + std::to_string(8 * sizeof(Key)) +
                                 "-bit keys (raise --minutil)");
    const bool     indexed = wants_index(n_trans, n_entries, n_items, index_min_trans, index_sparsity);
    const DbLayout layout  = DbLayout::of(n_trans, n_entries, n_items, false, indexed);
    std::vector<char> head(layout.eitem, 0);

    uint32_t max_len = 0;
    auto*    txs     = reinterpret_cast<Tx*>(head.data() + layout.tx);
    for (size_t t = 0; t < n_trans; ++t) {
        txs[t]  = Tx{data.tx_begin[t], data.tx_begin[t + 1] - data.tx_begin[t], 0};
        max_len = std::max(max_len, txs[t].length);
    }
    root.header = DbHeader{static_cast<uint32_t>(n_trans), static_cast<uint32_t>(n_entries), static_cast<uint32_t>(n_items),
                           max_len, layout.bytes, indexed};
    *reinterpret_cast<DbHeader*>(head.data()) = root.header;
    root.occ_shift                            = occurrence_shift(n_trans, max_len, indexed);
    auto* globals = reinterpret_cast<uint32_t*>(head.data() + layout.items);
    for (uint32_t i = 0; i < n_items; ++i)
        globals[i] = i;

    root.buffer = DeviceBuffer<char>(layout.bytes);
    char* base  = root.buffer.get();
    upload(base, head.data(), head.size());
    upload_as(reinterpret_cast<Key*>(base + layout.eitem), data.items.data(), n_entries);
    upload_as(reinterpret_cast<Util*>(base + layout.eutil), data.utilities.data(), n_entries);
    if (indexed) {
        build_root_index(base, layout, static_cast<uint32_t>(n_trans), static_cast<uint32_t>(n_entries),
                         static_cast<uint32_t>(n_items), root.occ_shift);
        std::vector<uint32_t> begin(n_items + 1);
        download(begin.data(), reinterpret_cast<const uint32_t*>(base + layout.occ_begin), n_items + 1);
        root.occ_counts.resize(n_items);
        for (uint32_t z = 0; z < n_items; ++z)
            root.occ_counts[z] = begin[z + 1] - begin[z];
    }
    return root;
}

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
