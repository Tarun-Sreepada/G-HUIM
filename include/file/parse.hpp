#pragma once
#include <cctype>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <omp.h>
#include "file/load.hpp"
#include <algorithm>
#include <charconv>
#include "data_types/data_types.hpp"

enum ParseState
{
    PARSE_ITEMS,         /**< Parsing item keys. */
    PARSE_TOTAL_UTILITY, /**< Parsing total utility field. */
    PARSE_UTILITIES      /**< Parsing per-item utilities. */
};

struct parsed_file
{
    std::vector<std::vector<Item>> database;

    std::unordered_map<uint32_t, uint32_t> twu;
};

parsed_file parse(raw_file &file, int threads, const char separator);