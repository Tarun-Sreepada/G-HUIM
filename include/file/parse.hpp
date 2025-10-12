#pragma once
#include "data_types/data_types.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <omp.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

struct parsed_file
{
    std::vector<std::vector<Item>> database;

    std::unordered_map<uint32_t, uint32_t> twu;
    std::vector<uint32_t>                  twu_v;

    std::vector<uint32_t> total_util;
};

parsed_file parse(const std::string& filename, int threads, char sep);

bool keysAreContiguousFromOne(const std::unordered_map<uint32_t, uint32_t>& twu);