#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <omp.h>
#include "file/parse.hpp"
#include <algorithm>
#include <numeric>
#include "data_types/data_types.hpp"

struct built_file
{
    std::map<uint32_t, uint32_t> ordered_twu;                      ///< Items ordered by transaction-weighted utility.
    std::vector<std::pair<uint32_t, uint32_t>> ordered_twu_vector; ///< Vector version of ordered_twu for fast access.

    std::unordered_map<uint32_t, uint32_t> item_to_itemID; ///< Maps original item IDs to internal IDs.
    std::unordered_map<uint32_t, uint32_t> itemID_to_item; ///< Maps internal item IDs back to original item IDs.

    size_t transaction_count;    ///< Number of transactions.
    size_t total_items;          ///< Total number of items after pruning.
    size_t max_transaction_size; ///< Size of the largest transaction.

    std::vector<std::vector<Item>> transactions; ///< List of compacted transactions.

    std::vector<uint32_t> primary; ///< Items considered primary (for mining root).

    built_file()
    {
        transaction_count = 0;
        max_transaction_size = 0;
        total_items = 0;
    }

    ~built_file()
    {
        ordered_twu.clear();
        ordered_twu_vector.clear();
        item_to_itemID.clear();
        itemID_to_item.clear();
    }
};

built_file build_file(parsed_file &pf, int threads, uint64_t min_utility);