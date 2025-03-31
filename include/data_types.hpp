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

