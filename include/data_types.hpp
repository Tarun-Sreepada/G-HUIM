#pragma once

#include <cstdio>

// Represents an item with key and utility.
struct Item
{
    unsigned int key;
    unsigned int util;

    static const unsigned int EMPTY_KEY = 0;
    static const unsigned int EMPTY_UTIL = 0;
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
};
