#pragma once
#include <cstdint>

// /**
//  * @brief Represents a single transaction on GPU.
//  */
struct Transaction
{
    uint32_t *keys = nullptr; ///< Pointer to array of keys.
    uint32_t *utilities = nullptr; ///< Pointer to array of utilities.
    uint32_t length = 0;  ///< Number of items.
    uint32_t utility = 0; ///< Transaction utility.

    __device__ __host__ uint32_t size() const { return length; }

    __device__ __host__ void clear()
    {
        keys = nullptr;
        utilities = nullptr;
        length = 0;
        utility = 0;
    }

    __device__ __host__ int linear_search(uint32_t item)
    {
        if (length == 0)
        {
            return -1; // Not found
        }
        for (int i = 0; i < length; ++i)
        {
            if (keys[i] == item)
            {
                return i;
            }
        }
        return -1; // Not found
    }

    __device__ __host__ int binary_search(uint32_t item)
    {
        int lr[2] = {0, (int)length - 1};
        int arr[2] = {1, -1};
        int &left = lr[0];
        int &right = lr[1];

        while (left <= right)
        {
            int mid = left + (right - left) / 2;
            if (keys[mid] == item)
            {
                return mid;
            }
            int idx = static_cast<int>(keys[mid] > item);
            lr[idx] = mid + arr[idx];
        }
        return -1;
    }

    __device__ __host__ int binary_search_old(uint32_t item)
    {
        int left = 0;
        int right = length - 1;

        if (length == 0)
        {
            return -1; // Not found
        }
        if (item < keys[left] || item > keys[right])
        {
            return -1; // Not found
        }

        if (item < keys[0] || item > keys[length - 1])
        {
            return -1; // Not found
        }

        if (keys[left] == item)
        {
            return left;
        }
        if (keys[right] == item)
        {
            return right;
        }

        while (left < right)
        {
            int mid = left + (right - left) / 2;
            if (keys[mid] == item)
            {
                return mid;
            }
            else if (keys[mid] < item)
            {
                left = mid + 1;
            }
            else
            {
                right = mid;
            }
        }

        return -1; // Not found
    }

    __device__ __host__ uint32_t util_sum(uint32_t start = 0)
    {
        uint32_t total = 0;
        for (int i = start; i < length; ++i)
        {
            total += utilities[i];
        }
        return total;
    }
};

struct Database
{
    uint32_t *keys;           ///< Array of keys.
    uint32_t *utilities;      ///< Array of utilities.
    Transaction *transactions; ///< Array of transactions.
    int num_transactions;      ///< Number of transactions.
    int num_items;             ///< Total number of items.

    __device__ __host__ bool compare_transactions(int idx1, int idx2) const
    {
        const Transaction &trans1 = transactions[idx1];
        const Transaction &trans2 = transactions[idx2];

        if (trans1.length != trans2.length)
            return false;

        for (int i = 0; i < trans1.length; i++)
        {
            if (trans1.keys[i] != trans2.keys[i])
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

    __device__ void clear()
    {
        key = EMPTY_KEY;
        local_utility = EMPTY_LOCAL_UTILITY;
        subtree_utility = EMPTY_SUBTREE_UTILITY;
    }
};
