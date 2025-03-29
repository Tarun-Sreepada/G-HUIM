#pragma once

#include <iostream>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include <memory>
#include "timer.hpp"
#include <vector>
#include <string>
#include <omp.h>
#include <liburing.h>
#include <unordered_map>
#include <optional>
#include <stdexcept>
#include <algorithm>
#include <utility>
#include <charconv>


struct raw_file
{
    char *data;
    size_t size_bytes;
    size_t submitted_bytes; // bytes submitted to io_uring
    size_t retrieved_bytes; // bytes retrieved from io_uring
    size_t processed_bytes; // bytes processed from data
    std::vector<std::pair<size_t, size_t>> retrieved_indices;

    int fd;

    raw_file() : data(nullptr), size_bytes(0), submitted_bytes(0), retrieved_bytes(0), processed_bytes(0), fd(-1) {}

    void close()
    {
        if (data != nullptr)
            free(data);
    }

};

raw_file load_file(std::string filename, int threads, Timer &timer);

struct parsed_file
{
    std::vector<std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>>> key_value_pairs;
    std::unordered_map<uint32_t, uint32_t> twu;
};

parsed_file parse_file(raw_file &file, int threads, const char separator, Timer &timer);

enum ParseState { PARSE_ITEMS, PARSE_TOTAL_UTILITY, PARSE_UTILITIES };
