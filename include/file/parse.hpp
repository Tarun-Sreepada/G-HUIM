#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <omp.h>
#include "file/load.hpp"
#include <algorithm>
#include <charconv>


enum ParseState { 
    PARSE_ITEMS,      /**< Parsing item keys. */
    PARSE_TOTAL_UTILITY, /**< Parsing total utility field. */
    PARSE_UTILITIES   /**< Parsing per-item utilities. */
};


struct parsed_file
{
    std::vector<
        std::vector<
            std::pair<std::vector<uint32_t>, std::vector<uint32_t>>>> key_value_pairs;
            // each thread's key-value pairs
    
    std::unordered_map<uint32_t, uint32_t> twu;
};


parsed_file parse_file(raw_file &file, int threads, const char separator);