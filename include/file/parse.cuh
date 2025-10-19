#pragma once

// file info
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

// for sorting line indices
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

// for database
#include "data_types/data_types.cuh"

struct Parsed
{
    Database* db;
    uint32_t  max_item;
};

// Parses the dataset directly on the GPU and returns device-resident buffers.
Parsed gpuParse(const std::string& filename, const char separator);
