#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <utility>
#include <charconv>
#include <stdexcept>
#include <omp.h>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include "data_types.hpp"
#include "timer.hpp"
#include "file.hpp"

struct built_file
{
    std::map<uint32_t, uint32_t> ordered_twu;
    std::vector<std::pair<uint32_t, uint32_t>> ordered_twu_vector;

    std::unordered_map<uint32_t, uint32_t> item_to_itemID;
    std::unordered_map<uint32_t, uint32_t> itemID_to_item;

    size_t transaction_count;
    size_t total_items;
    size_t max_transaction_size;

    std::vector<uint32_t> primary;

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

built_file build_file(parsed_file &pf, int threads, Timer &timer, uint64_t min_utility);