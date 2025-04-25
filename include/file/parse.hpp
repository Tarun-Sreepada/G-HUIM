#pragma once
#include <cctype>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <omp.h>
#include <algorithm>
#include <charconv>
#include "data_types/data_types.hpp"
#include <iostream>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>


struct parsed_file
{
    std::vector<std::vector<Item>> database;

    std::unordered_map<uint32_t, uint32_t> twu;
    std::vector<uint32_t> twu_v;

    std::vector<uint32_t> total_util;

};

parsed_file parse(const std::string &filename, int threads, char sep);

bool keysAreContiguousFromOne(
    const std::unordered_map<uint32_t, uint32_t> &twu);