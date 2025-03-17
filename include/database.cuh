#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include "functions.cuh"

// Represents an item with key and utility.
struct Item
{
    unsigned int key;
    unsigned int util;

    // define empty key as -1 -1
    static const unsigned int EMPTY_KEY = 0;
    static const unsigned int EMPTY_UTIL = 0;

    static const Item EMPTY_ITEM;
};

const Item Item::EMPTY_ITEM = {Item::EMPTY_KEY, Item::EMPTY_UTIL};

struct Utilities
{
    unsigned int key;
    unsigned int local_utility;
    unsigned int subtree_utility;

    static const int EMPTY_KEY = 0;
    static const int EMPTY_LOCAL_UTILITY = 0;
    static const int EMPTY_SUBTREE_UTILITY = 0;

    static const Utilities EMPTY_UTILITIES;

    // enum for local and subtree utility
    enum UtilityType
    {
        LOCAL_UTILITY,
        SUBTREE_UTILITY
    };
};

const Utilities Utilities::EMPTY_UTILITIES = {Utilities::EMPTY_KEY, Utilities::EMPTY_LOCAL_UTILITY, Utilities::EMPTY_SUBTREE_UTILITY};


template <typename T>
struct HashTable;

template<>
struct HashTable<Utilities>
{
    // Device pointer to an array of `Utilities`
    Utilities* table;
    int tableSize;

    // Insert or update the key with the given utility
    __device__ bool insert(int key, int util, Utilities::UtilityType type)
    {
        // Use MurmurHash3 on the key, then mod by tableSize
        uint32_t hashVal = murmur_hash3(&key, sizeof(int)) % tableSize;

        for (int i = 0; i < tableSize; i++)
        {
            int probe = (hashVal + i) % tableSize;

            // Attempt to claim an empty slot
            int oldKey = atomicCAS(&table[probe].key, 
                                   Utilities::EMPTY_KEY, 
                                   key);

            if (oldKey == Utilities::EMPTY_KEY)
            {
                // We own this slot now; set the chosen utility
                if (type == Utilities::LOCAL_UTILITY)
                    // table[probe].local_utility += util;
                    atomicAdd(&table[probe].local_utility, util);
                else
                    // table[probe].subtree_utility += util;
                    atomicAdd(&table[probe].subtree_utility, util);

                return true;
            }

            // If the key is already in the slot, just update
            if (table[probe].key == key)
            {
                if (type == Utilities::LOCAL_UTILITY)
                    // table[probe].local_utility += util;
                    atomicAdd(&table[probe].local_utility, util);
                else
                    // table[probe].subtree_utility += util;
                    atomicAdd(&table[probe].subtree_utility, util);

                return true;
            }
        }
        // Table is full
        return false;
    }

    // Retrieve an index for the given key
    __device__ int retrieve(int key)
    {
        uint32_t hashVal = murmur_hash3(&key, sizeof(int)) % tableSize;

        for (int i = 0; i < tableSize; i++)
        {
            int probe = (hashVal + i) % tableSize;

            if (table[probe].key == key)
            {
                return probe;
            }
            // If we hit an empty slot, no need to probe further
            if (table[probe].key == Utilities::EMPTY_KEY)
            {
                return -1;
            }
        }

        return -1; // Not found
    }

    __host__ __device__ void print()
    {
        for (int i = 0; i < tableSize; i++)
        {
            if (table[i].key != Utilities::EMPTY_KEY || table[i].local_utility != Utilities::EMPTY_LOCAL_UTILITY || table[i].subtree_utility != Utilities::EMPTY_SUBTREE_UTILITY)
                printf("Key: %d | Local Utility: %d | Subtree Utility: %d\n", table[i].key, table[i].local_utility, table[i].subtree_utility);
        }
    }
};

// Represents a transaction containing multiple items.
struct Transaction
{
    Item *data;  // Pointer to items array
    int length;  // Number of items
    int utility; // Transaction utility value

    __device__ __host__ inline int size() const { return length; }

    __device__ __host__ inline Item &operator[](int i) { return data[i]; }
    __device__ __host__ inline Item *operator->() { return data; }
    __device__ __host__ inline Item *get() { return data; }

    // Optimized binary search for a given key.
    __device__ Item find(int search_id) const
    {
        int left = 0, right = length - 1;
        while (left <= right)
        {
            int mid = left + (right - left) / 2;
            if (data[mid].key == search_id)
                return data[mid];
            data[mid].key < search_id ? left = mid + 1 : right = mid - 1;
        }
        // return -1;
        return {Item::EMPTY_KEY, Item::EMPTY_UTIL};
    }
};

// Represents the database storing transactions.
struct Database
{
    Item *d_data;                // Pointer to item array
    Transaction *d_transactions; // Pointer to transactions
    int numTransactions;
    int transaction_tracker;
    int numItems;

    // Checks if two transactions have the same keys
    __device__ bool sameKey(int t1, int t2) const
    {
        const Transaction &trans1 = d_transactions[t1];
        const Transaction &trans2 = d_transactions[t2];

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

// Prints the database (for debugging)
__host__ __device__ void printDatabase(const Database *db)
{
    if (!db)
    {
        printf("Database pointer is NULL! (Device)\n");
        return;
    }

    for (int t = 0; t < db->numTransactions; t++)
    {
        const Transaction &tran = db->d_transactions[t];

        printf("T%d | Utility: %d | ", t, tran.utility);

        for (int i = 0; i < tran.length; i++)
        {
            printf("[%d:%d] ", tran.data[i].key, tran.data[i].util);
        }
        printf("\n");
    }
}
