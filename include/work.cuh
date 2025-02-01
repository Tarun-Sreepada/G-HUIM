#pragma once
#include <cstdint>

#define CAPACITY 16384

struct Item {
    int key;
    int util;
} __attribute__((packed)); // Ensures no padding


struct WorkItem {
    int *pattern;

    Item *items;
    int num_items;

    int *start;
    int *end;
    int *utility;
    Item *tran_hash;
    int num_transactions;

    int *primary;
    int num_primary;
    int max_item;

    Item *local_util;
    Item *subtree_util;

    int *work_done;
    int work_count;

    int bytes;
    void *base_ptr;

};

typedef struct {
    int item_counter;
    int transaction_counter;
    int pattern_utility;
} ProjectionResult;


typedef struct {
    int new_item_counter;
    int new_transaction_counter;
} MergeResult;

