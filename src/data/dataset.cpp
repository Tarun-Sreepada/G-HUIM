#include "data/dataset.hpp"

#include "data/mapped_file.hpp"
#include "data/parallel.hpp"
#include "data/parse.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace ghuim {
namespace {

constexpr uint32_t kRemoved = std::numeric_limits<uint32_t>::max();

// Merged transactions sum utilities in 32 bits; the database total bounds every such sum.
void check_total_utility(const RawInput& in)
{
    uint64_t total = 0;
    for (const auto& t : in.transactions)
        total += t.utility;
    if (total > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("total database utility exceeds 32 bits");
}

// Transaction-weighted utility of each item: per-thread histograms, summed. Each extra
// histogram costs `bins` words, so fan out only while that stays small next to the input.
std::vector<uint64_t> transaction_weighted_utility(const RawInput& in)
{
    const uint32_t max_item    = *std::max_element(in.items.begin(), in.items.end());
    const size_t   bins        = static_cast<size_t>(max_item) + 1;
    const size_t   twu_workers = worker_count(in.items.size() / bins, 1);
    std::vector<std::vector<uint64_t>> partial(twu_workers, std::vector<uint64_t>(bins, 0));
    parallel_for(twu_workers, in.transactions.size(), [&](size_t w, size_t begin, size_t end) {
        auto& hist = partial[w];
        for (size_t t = begin; t < end; ++t)
            for (size_t i = in.transactions[t].begin; i < in.transactions[t].end; ++i)
                hist[in.items[i]] += in.transactions[t].utility;
    });
    for (size_t w = 1; w < twu_workers; ++w)
        for (size_t b = 0; b < bins; ++b)
            partial[0][b] += partial[w][b];
    return std::move(partial[0]);
}

// Promising items in ascending TWU order (EFIM's processing order); fills data.original_item and
// returns the map from input item to new name (kRemoved for pruned items).
std::vector<uint32_t> rename_items(const std::vector<uint64_t>& twu, uint64_t minimum_utility, Dataset& data)
{
    for (uint32_t item = 0; item < twu.size(); ++item)
        if (twu[item] >= minimum_utility)
            data.original_item.push_back(item);
    std::stable_sort(data.original_item.begin(), data.original_item.end(),
                     [&](uint32_t a, uint32_t b) { return twu[a] < twu[b]; });
    std::vector<uint32_t> rename(twu.size(), kRemoved);
    for (uint32_t id = 0; id < data.original_item.size(); ++id)
        rename[data.original_item[id]] = id;
    return rename;
}

// Rebuilds transactions with renamed, sorted items. Counts the surviving items per transaction,
// prefix-sums into output offsets (dropping empty transactions), then fills concurrently.
void rebuild_transactions(const RawInput& in, const std::vector<uint32_t>& rename, Dataset& data)
{
    const auto&           raw             = in.transactions;
    const size_t          rebuild_workers = worker_count(in.items.size(), size_t{1} << 16);
    std::vector<uint32_t> kept(raw.size());
    parallel_for(rebuild_workers, raw.size(), [&](size_t, size_t begin, size_t end) {
        for (size_t t = begin; t < end; ++t) {
            uint32_t n = 0;
            for (size_t i = raw[t].begin; i < raw[t].end; ++i)
                n += rename[in.items[i]] != kRemoved;
            kept[t] = n;
        }
    });
    std::vector<uint32_t> out_tx(raw.size()); // raw transaction -> output index, or kRemoved
    data.tx_begin.push_back(0);
    uint64_t total_items = 0;
    for (size_t t = 0; t < raw.size(); ++t) {
        if (kept[t] == 0) {
            out_tx[t] = kRemoved;
            continue;
        }
        out_tx[t] = static_cast<uint32_t>(data.tx_begin.size() - 1);
        total_items += kept[t];
        if (total_items >= std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("too many item occurrences for 32-bit offsets");
        data.tx_begin.push_back(static_cast<uint32_t>(total_items));
    }
    data.items.resize(total_items);
    data.utilities.resize(total_items);
    parallel_for(rebuild_workers, raw.size(), [&](size_t, size_t begin, size_t end) {
        std::vector<std::pair<uint32_t, uint32_t>> scratch;
        for (size_t t = begin; t < end; ++t) {
            if (out_tx[t] == kRemoved)
                continue;
            scratch.clear();
            for (size_t i = raw[t].begin; i < raw[t].end; ++i)
                if (rename[in.items[i]] != kRemoved)
                    scratch.emplace_back(rename[in.items[i]], in.utilities[i]);
            std::sort(scratch.begin(), scratch.end());
            size_t o = data.tx_begin[out_tx[t]];
            for (const auto& [item, utility] : scratch) {
                data.items[o]     = item;
                data.utilities[o] = utility;
                ++o;
            }
        }
    });
}

// Subtree utility of each item in the revised database selects the first-level extensions.
void select_primary_items(Dataset& data, uint64_t minimum_utility)
{
    const size_t items_n         = data.original_item.size();
    const size_t subtree_workers = worker_count(data.items.size() / std::max<size_t>(items_n, 1), 1);
    std::vector<std::vector<uint64_t>> partial(subtree_workers, std::vector<uint64_t>(items_n, 0));
    parallel_for(subtree_workers, data.transaction_count(), [&](size_t w, size_t begin, size_t end) {
        auto& hist = partial[w];
        for (size_t t = begin; t < end; ++t) {
            uint64_t suffix = 0;
            for (uint32_t i = data.tx_begin[t + 1]; i-- > data.tx_begin[t];) {
                suffix += data.utilities[i];
                hist[data.items[i]] += suffix;
            }
        }
    });
    for (size_t w = 1; w < subtree_workers; ++w)
        for (size_t b = 0; b < items_n; ++b)
            partial[0][b] += partial[w][b];
    for (uint32_t id = 0; id < items_n; ++id)
        if (partial[0][id] >= minimum_utility)
            data.primary.push_back(id);
}

} // namespace

Dataset load_dataset(const std::string& path, char separator, uint64_t minimum_utility)
{
    std::vector<std::pair<const char*, double>> timings;
    auto lap = std::chrono::steady_clock::now();
    auto stage = [&](const char* name) {
        const auto now = std::chrono::steady_clock::now();
        timings.emplace_back(name, std::chrono::duration<double, std::milli>(now - lap).count());
        lap = now;
    };

    const MappedFile file(path);
    stage("map file");

    const RawInput input = parse_input(file.view(), separator);
    stage("parse");
    check_total_utility(input);
    stage("merge chunks");

    const std::vector<uint64_t> twu = transaction_weighted_utility(input);
    stage("TWU");

    Dataset data;
    data.input_transactions = input.transactions.size();
    const std::vector<uint32_t> rename = rename_items(twu, minimum_utility, data);
    stage("prune + rename items");

    rebuild_transactions(input, rename, data);
    stage("rebuild + sort transactions");

    select_primary_items(data, minimum_utility);
    stage("subtree utilities");
    data.timings = std::move(timings);
    return data;
}

} // namespace ghuim
