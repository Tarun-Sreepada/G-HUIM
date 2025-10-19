#pragma once
#include "data_types.hpp"

#define KILO 1024ULL
#define MEGA (1024ULL * KILO)
#define GIGA (1024ULL * MEGA)

#include <cstdint>

struct Transaction
{
    Item*    data    = nullptr; ///< Pointer to array of items.
    uint32_t length  = 0;       ///< Number of items.
    uint32_t utility = 0;       ///< Transaction utility.

    // Returns the number of items in the transaction.
    __device__ __host__ uint32_t size() const
    {
        return length;
    }

    // Provides mutable access to an item by index.
    __device__ __host__ Item& operator[](unsigned int index)
    {
        return data[index];
    }
    // Provides read-only access to an item by index.
    __device__ __host__ const Item& operator[](unsigned int index) const
    {
        return data[index];
    }

    // Returns an iterator to the first item.
    __device__ __host__ Item* begin()
    {
        return data;
    }
    // Returns an iterator past the last item.
    __device__ __host__ Item* end()
    {
        return data + length;
    }
    // Returns a const iterator to the first item.
    __device__ __host__ const Item* begin() const
    {
        return data;
    }
    // Returns a const iterator past the last item.
    __device__ __host__ const Item* end() const
    {
        return data + length;
    }

    // Checks whether the transaction contains no items.
    __device__ __host__ bool is_empty() const
    {
        return length == 0;
    }
    // Resets the transaction to an empty state.
    __device__ __host__ void clear()
    {
        data    = nullptr;
        length  = 0;
        utility = 0;
    }

    // Performs a binary search for the given item key.
    __device__ __host__ int binary_search(uint32_t item) const
    {
        int  lr[2]  = {0, (int)length - 1};
        int  arr[2] = {1, -1};
        int& left   = lr[0];
        int& right  = lr[1];

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (data[mid].key == item) {
                return mid;
            }
            int idx = static_cast<int>(data[mid].key > item);
            lr[idx] = mid + arr[idx];
        }
        return -1;
    }

    // Legacy binary search variant retained for reference.
    __device__ __host__ int binary_search_old(uint32_t item) const
    {
        int left  = 0;
        int right = (int)length - 1;

        while (left <= right) {
            int      mid = left + ((right - left) >> 1); // Use bit-shift for division by 2
            uint32_t key = data[mid].key;                // Cache the key locally

            if (key == item) {
                return mid;
            }
            else if (key > item) {
                right = mid - 1;
            }
            else {
                left = mid + 1;
            }
        }
        return -1;
    }

    // Computes the sum of utilities from the given starting position.
    __device__ __host__ uint32_t util_sum(uint32_t start = 0) const
    {
        uint32_t total = 0;
        for (uint32_t i = start; i < length; ++i) {
            total += data[i].util;
        }
        return total;
    }

    // Checks for equality with another transaction by comparing keys.
    __device__ __host__ bool operator==(const Transaction& other) const
    {
        if (length != other.length)
            return false;
        for (uint32_t i = 0; i < length; ++i)
            if (data[i].key != other.data[i].key)
                return false;
        return true;
    }
};

/**
 * @brief Represents a GPU-resident transaction database.
 */
struct Database
{
    Item*        data;             ///< Flat item data.
    Transaction* transactions;     ///< Array of transactions.
    int          num_transactions; ///< Number of transactions.
    int          num_items;        ///< Total number of items.

    // Compares two transactions identified by their indices for equality.
    __device__ __host__ bool compare_transactions(int idx1, int idx2) const
    {
        const Transaction& trans1 = transactions[idx1];
        const Transaction& trans2 = transactions[idx2];

        if (trans1.length != trans2.length)
            return false;

        for (uint32_t i = 0; i < trans1.length; i++) {
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

    static const int EMPTY_KEY             = 0;
    static const int EMPTY_LOCAL_UTILITY   = 0;
    static const int EMPTY_SUBTREE_UTILITY = 0;

    enum UtilityType
    {
        LOCAL_UTILITY,
        SUBTREE_UTILITY
    };

    // Resets the utility entry to denote an unused slot.
    __device__ void clear()
    {
        key             = EMPTY_KEY;
        local_utility   = EMPTY_LOCAL_UTILITY;
        subtree_utility = EMPTY_SUBTREE_UTILITY;
    }
};
