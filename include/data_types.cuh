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
    Item *data = nullptr; ///< Pointer to array of items.
    // uint32_t *cumulative_sum = nullptr; ///< Cumulative sum of items.
    uint32_t length = 0;  ///< Number of items.
    uint32_t utility = 0; ///< Transaction utility.

    __device__ __host__ uint32_t size() const { return length; }

    __device__ __host__ Item &operator[](unsigned int index) { return data[index]; }
    __device__ __host__ const Item &operator[](unsigned int index) const { return data[index]; }
    __device__ __host__ Item *begin() { return data; }
    __device__ __host__ Item *end() { return data + length; }
    __device__ __host__ const Item *begin() const { return data; }
    __device__ __host__ const Item *end() const { return data + length; }
    __device__ __host__ bool is_empty() const { return length == 0; }
    __device__ __host__ void clear()
    {
        length = 0;
        utility = 0;
        data = nullptr;
    }

    __device__ __host__ int linear_search(uint32_t item)
    {
        if (is_empty() || length == 0)
        {
            return -1; // Not found
        }
        for (int i = 0; i < length; ++i)
        {
            if (data[i].key == item)
            {
                return i;
            }
        }
        return -1; // Not found
    }

    __device__ __host__ int binary_search(uint32_t item)
    {
        int left = 0;
        int right = length - 1;
        while (left <= right)
        {
            int mid = left + (right - left) / 2;
            if (data[mid].key == item)
            {
                return mid;
            }
            else if (data[mid].key < item)
            {
                left = mid + 1;
            }
            else
            {
                right = mid - 1;
            }
        }
        return -1; // Not found
    }

    __device__ __host__ uint32_t util_sum(uint32_t start = 0)
    {
        uint32_t total = 0;
        for (int i = start; i < length; ++i)
        {
            total += data[i].util;
        }
        return total;
    }

    // struct CumsumProxy {
    //     Transaction *trans;
    //     __device__ __host__ uint32_t &operator[](unsigned int index) {
    //         return trans->cumulative_sum[index];
    //     }
    //     __device__ __host__ const uint32_t &operator[](unsigned int index) const {
    //         return trans->cumulative_sum[index];
    //     }
    // };

    // // Provide an accessor that returns the proxy.
    // __device__ __host__ CumsumProxy cumsum() { return {this}; }
    // __device__ __host__ const CumsumProxy cumsum() const { return {const_cast<Transaction*>(this)}; }
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
    // uint32_t *cumulative_sum = nullptr; // for util bucket

    __device__ __host__ bool compare_transactions(int idx1, int idx2) const
    {
        const Transaction &trans1 = transactions[idx1];
        const Transaction &trans2 = transactions[idx2];

        if (trans1.length != trans2.length)
            return false;

        for (int i = 0; i < trans1.length; i++)
        {
            if (trans1.data[i].key != trans2.data[i].key)
                return false;
        }
        return true;
    }
};

struct Utilities
{
    unsigned int key;
    unsigned int local_utility;
    unsigned int subtree_utility;

    static const int EMPTY_KEY = 0;
    static const int EMPTY_LOCAL_UTILITY = 0;
    static const int EMPTY_SUBTREE_UTILITY = 0;

    // enum for local and subtree utility
    enum UtilityType
    {
        LOCAL_UTILITY,
        SUBTREE_UTILITY
    };

    __device__ void clear()
    {
        local_utility = EMPTY_LOCAL_UTILITY;
        subtree_utility = EMPTY_SUBTREE_UTILITY;
    }
};
