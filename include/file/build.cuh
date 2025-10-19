#pragma once

#include "args.hpp"
#include "data_types/data_types.cuh"
#include "file/parse.cuh"
#include "file/parse.hpp"

struct Item_mapping
{
    uint32_t item;
    uint32_t id;
    uint32_t local_utility;
    uint32_t subtree_utility;
};

// Forward declarations for kernels
__global__ void buildingKernel(Item* data, uint32_t num_items, const Item_mapping* item_map);
__global__ void sortTransactionsKernel(Transaction* transactions, uint32_t num_transactions, Utilities* utils);

struct built_file
{
    Database*     db;
    Item_mapping* item_map;
    Utilities*    utils;
    uint32_t      max_id;

    std::vector<uint32_t> primary;
};

// Builds the GPU-friendly database layout from parsed input and pruning data.
built_file build(Parsed& pf, uint32_t minutil, ArgParser& parser, Item_mapping* item_map);
