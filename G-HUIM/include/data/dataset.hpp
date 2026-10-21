#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ghuim {

// The input after EFIM preprocessing: items whose TWU is below the threshold are removed,
// the rest are renamed 0..n-1 in ascending TWU order, and every transaction is sorted by
// the new names. Transactions left empty are dropped.
struct Dataset
{
    std::vector<uint32_t> original_item; // new name -> item as written in the input
    std::vector<uint32_t> tx_begin;      // CSR offsets into items/utilities, size = transactions + 1
    std::vector<uint32_t> items;
    std::vector<uint64_t> utilities;
    std::vector<uint32_t> primary;       // items whose subtree utility meets the threshold
    uint64_t              input_transactions = 0;
    uint64_t              total_utility      = 0; // sum over the input
    // Largest TWU over the kept items, from the rebuilt transactions: bounds every utility the miners store or
    // sum, so it sets the utility width (mine/widths.hpp).
    uint64_t              max_twu            = 0;
    // Longest kept transaction: with the transaction count, sets the BFS occurrence width (mine/widths.hpp).
    uint32_t              max_length         = 0;
    std::vector<std::pair<const char*, double>> timings; // preprocessing stage -> milliseconds

    size_t transaction_count() const
    {
        return tx_begin.size() - 1;
    }
};

Dataset load_dataset(const std::string& path, char separator, uint64_t minimum_utility);

} // namespace ghuim
