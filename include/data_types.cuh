/**
 * @file data_types.cuh
 * @brief GPU data structures for transactions and database.
 */

#pragma once
#include <cstdint>
#include "data_types.hpp"

/**
 * @brief Represents a single transaction on GPU.
 */
struct Transaction
{
    Item *data = nullptr;    ///< Pointer to array of items.
    unsigned int length = 0; ///< Number of items.
    int utility = 0;         ///< Transaction utility.

    __device__ __host__ unsigned int size() const { return length; }

    __device__ __host__ Item &operator[](unsigned int index) { return data[index]; }
    __device__ __host__ const Item &operator[](unsigned int index) const { return data[index]; }
};

/**
 * @brief Represents a GPU-resident transaction database.
 */
struct Database
{
    Item *data;                ///< Flat item data.
    Transaction *transactions; ///< Array of transactions.
    int num_transactions;      ///< Number of transactions.
    int num_items;             ///< Total number of items.

    __device__ __host__ bool compare_transactions(int idx1, int idx2) const;
};
