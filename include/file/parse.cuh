#pragma once


// file info
#include <iostream>
#include <string>
#include <algorithm>
#include <cctype>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <cstdio>

// for sorting line indices
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

// for database
#include "data_types/data_types.cuh"


struct Parsed
{
    Database *db;
    uint32_t max_item;
};

Parsed gpuParse(const std::string &filename, const char separator);
