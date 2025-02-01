
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include "args.h"     // Include the args parser
#include "parser.h"   // Include the file reader
#include "work.cuh"   // Include the work queue
#include "memory.cuh" // Include the memory manager

#define KILO 1024ULL
#define MEGA KILO *KILO
#define GIGA KILO *MEGA

#define scale 2
#define page_size 512
#define total_memory 24 * GIGA

#define blocks 32

// Binary Search Utility
__device__ int binarySearchItems(const Item *items, int n, uint32_t search_id, int offset, int length)
{
    if (offset < 0 || offset >= n || length <= 0 || (offset + length) > n)
        return -1;
    int l = offset, r = offset + length - 1;
    while (l <= r)
    {
        int mid = l + (r - l) / 2;
        if (items[mid].key == search_id)
            return mid;
        items[mid].key < search_id ? l = mid + 1 : r = mid - 1;
    }
    return -1;
}

__device__ uint32_t pcg_hash(uint32_t input)
{
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ uint32_t hashFunction(uint32_t key, uint32_t tableSize)
{
    return pcg_hash(key) % tableSize;
}

__device__ uint32_t items_hasher(const Item *items, int n, int tableSize)
{
    uint32_t hash = 0;
    for (int i = 0; i < n; i++)
        hash ^= pcg_hash(items[i].key);
    return hash % tableSize;
}

__device__ WorkItem *allocateProjectionMemory(CudaMemoryManager *memory_manager,
                                              const WorkItem *work_item)
{
    int bytesNeeded = sizeof(WorkItem) + (work_item->pattern[0] + 2) * sizeof(int) // pattern
                      + work_item->num_items * sizeof(Item)                        // items
                      + work_item->num_transactions * sizeof(int)                  // start
                      + work_item->num_transactions * sizeof(int)                  // end
                      + work_item->num_transactions * sizeof(int)                  // utility
                      + work_item->num_transactions * sizeof(Item) * scale         // tran_hash
                      + work_item->max_item * sizeof(int)                          // primary
                      + work_item->max_item * sizeof(Item) * scale                 // local_util
                      + work_item->max_item * sizeof(Item) * scale                 // subtree_util
                      + 1 * sizeof(int);                                           // work_done

    void *base_ptr = deviceMemMalloc(memory_manager, bytesNeeded);
    if (!base_ptr)
    {
        // Allocation failed
        return nullptr;
    }

    // Zero out memory
    memset(base_ptr, 0, bytesNeeded);

    // Construct the new WorkItem in the allocated buffer
    WorkItem *newWorkItem = (WorkItem *)base_ptr;

    // Layout (pointer arithmetic) for sub-structures
    newWorkItem->pattern = (int *)(newWorkItem + 1);
    newWorkItem->items = (Item *)(newWorkItem->pattern + work_item->pattern[0] + 2);
    newWorkItem->start = (int *)(newWorkItem->items + work_item->num_items);
    newWorkItem->end = (int *)(newWorkItem->start + work_item->num_transactions);
    newWorkItem->utility = (int *)(newWorkItem->end + work_item->num_transactions);

    newWorkItem->tran_hash = (Item *)(newWorkItem->utility + work_item->num_transactions);
    newWorkItem->primary = (int *)(newWorkItem->tran_hash + work_item->num_transactions * scale);
    newWorkItem->local_util = (Item *)(newWorkItem->primary + work_item->max_item);
    newWorkItem->subtree_util = (Item *)(newWorkItem->local_util + work_item->max_item * scale);
    newWorkItem->work_done = (int *)(newWorkItem->subtree_util + work_item->max_item * scale);

    // Fill in metadata
    newWorkItem->bytes = bytesNeeded;
    newWorkItem->base_ptr = base_ptr;

    return newWorkItem;
}

__device__ void copyAndExtendPattern(const WorkItem *work_item, WorkItem *n_wi, const int item)
{
    // Copy the current pattern (first n integers) and then extend it
    // memcpy(pm->n_pattern, work_item->pattern, (work_item->pattern[0] + 1) * sizeof(int));
    memcpy(n_wi->pattern, work_item->pattern, (work_item->pattern[0] + 1) * sizeof(int));
    n_wi->pattern[0] += 1;
    n_wi->pattern[n_wi->pattern[0]] = item;
}

__device__ ProjectionResult performProjection(const WorkItem *work_item, WorkItem *pm, const int item)
{
    ProjectionResult res = {0, 0, 0};
    int item_counter = 0;
    int transaction_counter = 0;
    int pattern_utility = 0;

    for (int i = 0; i < work_item->num_transactions; i++)
    {
        // Find the index for the primary item in this transaction.
        int idx = binarySearchItems(work_item->items, work_item->num_items,
                                    item, work_item->start[i],
                                    work_item->end[i] - work_item->start[i]);
        if (idx == -1)
            continue;

        // Record the starting index for the new transaction
        pm->start[transaction_counter] = item_counter;
        // Combine the current transaction utility with the utility at idx.
        pm->utility[transaction_counter] = work_item->utility[i] + work_item->items[idx].util;
        pattern_utility += pm->utility[transaction_counter];
        int temp_util = pm->utility[transaction_counter];

        // Copy the remaining items of the transaction
        for (int j = idx + 1; j < work_item->end[i]; j++)
        {
            pm->items[item_counter] = work_item->items[j];
            temp_util += work_item->items[j].util;
            item_counter++;
        }
        pm->end[transaction_counter] = item_counter;
        transaction_counter++;

        // Update local utility table for all items in this transaction.
        for (int j = idx + 1; j < work_item->end[i]; j++)
        {
            uint32_t hash = hashFunction(work_item->items[j].key, work_item->max_item);
            while (true)
            {
                if (pm->local_util[hash].key == 0 ||
                    pm->local_util[hash].key == work_item->items[j].key)
                {
                    pm->local_util[hash].key = work_item->items[j].key;
                    pm->local_util[hash].util += temp_util;
                    break;
                }
                hash = (hash + 1) % (work_item->max_item * scale);
            }
        }
    }
    res.item_counter = item_counter;
    res.transaction_counter = transaction_counter;
    res.pattern_utility = pattern_utility;
    return res;
}

__device__ void storeHighUtilityPattern(const WorkItem *wi, int pattern_utility,
                                        int *high_utility_patterns, int min_util)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (pattern_utility >= min_util)
    {
        atomicAdd(&high_utility_patterns[0], 1);
        printf("%d|High Utility Patterns: %d\n", tid, high_utility_patterns[0]);
        int offset = atomicAdd(&high_utility_patterns[1], wi->pattern[0] + 2);
        for (int i = 0; i < wi->pattern[0]; i++)
        {
            high_utility_patterns[offset + i] = wi->pattern[i + 1];
        }
        high_utility_patterns[offset + wi->pattern[0]] = pattern_utility;
    }
}

__device__ MergeResult trimMergeAndComputeSubtree(const WorkItem *src, WorkItem *dst,
                                                  int proj_transaction_count, int min_util)
{
    MergeResult mres{0, 0};
    int &new_item_counter = mres.new_item_counter;
    int &new_transaction_counter = mres.new_transaction_counter;

    int mod_val = src->max_item * scale;

    // For each projected transaction
    for (int i = 0; i < proj_transaction_count; i++)
    {
        // Start index of new, trimmed transaction
        int start_idx = new_item_counter;
        for (int j = dst->start[i]; j < dst->end[i]; j++)
        {
            uint32_t hash = hashFunction(dst->items[j].key, mod_val);
            while (true)
            {
                if (dst->local_util[hash].key == dst->items[j].key)
                {
                    // Keep only if local utility >= min_util
                    if (dst->local_util[hash].util >= min_util)
                    {
                        dst->items[new_item_counter] = dst->items[j];
                        new_item_counter++;
                    }
                    break;
                }
                hash = (hash + 1) % mod_val;
            }
        }

        // If trimming removed all items, skip
        if (start_idx == new_item_counter)
            continue;

        // Update transaction boundaries
        dst->start[new_transaction_counter] = start_idx;
        dst->end[new_transaction_counter] = new_item_counter;
        dst->utility[new_transaction_counter] = dst->utility[i];

        // Compute partial utility for subtree
        int temp_util = dst->utility[new_transaction_counter];
        for (int j = dst->start[new_transaction_counter]; j < dst->end[new_transaction_counter]; j++)
        {
            temp_util += dst->items[j].util;
        }

        // Update subtree utility
        int consumed_util = 0;
        for (int j = dst->start[new_transaction_counter]; j < dst->end[new_transaction_counter]; j++)
        {
            uint32_t hash = hashFunction(dst->items[j].key, mod_val);
            while (true)
            {
                if (dst->subtree_util[hash].key == 0 ||
                    dst->subtree_util[hash].key == dst->items[j].key)
                {
                    dst->subtree_util[hash].key = dst->items[j].key;
                    dst->subtree_util[hash].util += (temp_util - consumed_util);
                    consumed_util = dst->items[j].util;
                    break;
                }
                hash = (hash + 1) % mod_val;
            }
        }

        // Attempt to merge duplicate transactions
        int cur_tran_idx = new_transaction_counter;
        int tran_length = dst->end[cur_tran_idx] - dst->start[cur_tran_idx];
        int t_hash = items_hasher(dst->items + start_idx, tran_length,
                                  src->num_transactions * scale);

        bool merged = false;
        while (true)
        {
            if (dst->tran_hash[t_hash].key == 0)
            {
                // This transaction shape hasn't been seen
                dst->tran_hash[t_hash].key = t_hash;
                dst->tran_hash[t_hash].util = cur_tran_idx;
                break;
            }

            // If we find an identical transaction signature
            if (dst->tran_hash[t_hash].key == t_hash)
            {
                int existing_idx = dst->tran_hash[t_hash].util;
                int existing_length = dst->end[existing_idx] - dst->start[existing_idx];
                if (existing_length == tran_length)
                {
                    // Check if items match
                    bool same = true;
                    for (int j = 0; j < tran_length; j++)
                    {
                        if (dst->items[dst->start[existing_idx] + j].key != dst->items[dst->start[cur_tran_idx] + j].key)
                        {
                            same = false;
                            break;
                        }
                    }
                    // Merge
                    if (same)
                    {
                        dst->utility[existing_idx] += dst->utility[cur_tran_idx];
                        for (int j = 0; j < tran_length; j++)
                        {
                            dst->items[dst->start[existing_idx] + j].util +=
                                dst->items[dst->start[cur_tran_idx] + j].util;
                        }
                        new_item_counter -= tran_length;
                        merged = true;
                        break;
                    }
                }
            }
            t_hash = (t_hash + 1) % (src->num_transactions * scale);
        }

        // If not merged, it's a new unique transaction
        if (!merged)
            new_transaction_counter++;
    }

    return mres;
}

__device__ int countSubtreeUtility(const WorkItem *work_item, const WorkItem *pm, int min_util)
{
    int counter = 0;
    for (int i = 0; i < work_item->max_item * scale; i++)
    {
        if (pm->subtree_util[i].util >= min_util)
        {
            pm->primary[counter] = pm->subtree_util[i].key;

            counter++;
        }
    }
    return counter;
}

__global__ void mine(CudaMemoryManager *memory_manager,
                     WorkItem *work_item,
                     int min_util,
                     int *high_utility_patterns)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // If there's no valid item in primary[tid], skip
    if (work_item->primary[tid] == 0)
    {
        return;
    }

    // 1. Allocate memory for the new, projected WorkItem
    WorkItem *new_work_item = allocateProjectionMemory(memory_manager, work_item);
    if (!new_work_item)
    {
        printf("Memory allocation failed.\n");
        return;
    }

    // 2. Copy and extend the pattern
    copyAndExtendPattern(work_item, new_work_item, work_item->primary[tid]);

    // 3. Perform projection
    ProjectionResult proj = performProjection(work_item, new_work_item,
                                              new_work_item->pattern[new_work_item->pattern[0]]);

    // 4. Check if the newly formed pattern is high-utility
    storeHighUtilityPattern(new_work_item, proj.pattern_utility, high_utility_patterns, min_util);

    // 5. If no transactions remain after projection, free memory & mark done
    if (proj.transaction_counter == 0)
    {
        deviceMemFree(memory_manager, new_work_item->base_ptr, new_work_item->bytes);
        atomicAdd(&work_item->work_done[0], 1);
        return;
    }

    // 6. Trim, merge, and compute subtree utilities
    MergeResult mergeRes = trimMergeAndComputeSubtree(work_item, new_work_item,
                                                      proj.transaction_counter, min_util);

    // 7. Identify next-level primary items
    int subtree_util_count = countSubtreeUtility(work_item, new_work_item, min_util);
    new_work_item->num_primary = subtree_util_count;
    new_work_item->work_done[0] = 0;
    new_work_item->work_count = subtree_util_count;
    new_work_item->num_items = mergeRes.new_item_counter;
    new_work_item->num_transactions = mergeRes.new_transaction_counter;
    new_work_item->max_item = work_item->max_item;

    // 8. Recurse if there are any items in the subtree
    if (subtree_util_count > 0)
    {
        mine<<<subtree_util_count, 1>>>(memory_manager,
                                        new_work_item,
                                        min_util,
                                        high_utility_patterns);
    }

    // 9. Wait until all child calls complete (polling on work_done)
    while (atomicAdd(new_work_item->work_done, 0) != new_work_item->work_count)
    {
    }

    // 10. Free memory for the new WorkItem
    deviceMemFree(memory_manager, new_work_item->base_ptr, new_work_item->bytes);

    // Mark that this thread's work is done
    atomicAdd(&work_item->work_done[0], 1);
}

__global__ void copy(CudaMemoryManager *memory_manager,
                     WorkItem *work_item,
                     const Item *items,
                     const int *start,
                     const int *end,
                     const int *primary,
                     int num_primary,
                     int num_transactions,
                     int max_item,
                     int *d_high_utility_patterns,
                     int min_util)
{
    int item_count = end[num_transactions - 1] - start[0];

    // Calculate memory needed for the initial WorkItem
    int bytesNeeded = 2 * sizeof(int)                                              // pattern (2 ints: length, next item)
                      + item_count * sizeof(Item)                                  // items
                      + num_transactions * sizeof(int)                             // start
                      + num_transactions * sizeof(int)                             // end
                      + num_transactions * sizeof(int)                             // utility
                      + num_primary * sizeof(int)                                  // primary
                      + 1 * sizeof(int);                                           // work_done

    // Allocate on the device
    void *base_ptr = deviceMemMalloc(memory_manager, bytesNeeded);
    memset(base_ptr, 0, bytesNeeded);

    // Layout
    void *ptr = base_ptr;
    int *pattern = reinterpret_cast<int *>(ptr);
    ptr = reinterpret_cast<char *>(ptr) + 2 * sizeof(int);

    Item *n_items = reinterpret_cast<Item *>(ptr);
    ptr = reinterpret_cast<char *>(ptr) + item_count * sizeof(Item);

    int *n_start = reinterpret_cast<int *>(ptr);
    ptr = reinterpret_cast<char *>(ptr) + num_transactions * sizeof(int);

    int *n_end = reinterpret_cast<int *>(ptr);
    ptr = reinterpret_cast<char *>(ptr) + num_transactions * sizeof(int);

    int *n_utility = reinterpret_cast<int *>(ptr);
    ptr = reinterpret_cast<char *>(ptr) + num_transactions * sizeof(int);

    int *n_primary = reinterpret_cast<int *>(ptr);
    ptr = reinterpret_cast<char *>(ptr) + num_primary * sizeof(int);

    int *work_done = reinterpret_cast<int *>(ptr);
    ptr = reinterpret_cast<char *>(ptr) + sizeof(int);

    // Initialize
    pattern[0] = 0; // initial pattern length = 0
    work_done[0] = 0;
    memcpy(n_items, items, item_count * sizeof(Item));
    memcpy(n_start, start, num_transactions * sizeof(int));
    memcpy(n_end, end, num_transactions * sizeof(int));
    memcpy(n_primary, primary, num_primary * sizeof(int));

    // Fill out the WorkItem
    work_item->pattern = pattern;
    work_item->items = n_items;
    work_item->num_items = item_count;
    work_item->start = n_start;
    work_item->end = n_end;
    work_item->utility = n_utility;
    work_item->num_transactions = num_transactions;
    work_item->primary = n_primary;
    work_item->num_primary = num_primary;
    work_item->max_item = max_item;
    work_item->work_done = work_done;
    work_item->work_count = num_primary;
    work_item->bytes = bytesNeeded;
    work_item->base_ptr = base_ptr;

    // Launch the recursive mining kernel
    mine<<<num_primary, 1>>>(memory_manager, work_item, min_util, d_high_utility_patterns);
}

int main(int argc, char *argv[])
{

    // set stack limit to 64kb
    cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024);


    // Parse command-line arguments using args_parser
    ParsedArgs args;
    if (!parseArguments(argc, argv, args))
    {
        // Parsing failed; exit the program
        return EXIT_FAILURE;
    }

    ReadFileResult fileResult = read_file(args.filename, args.separator, args.utility);

    // Access the parsed data
    auto &filteredTransactions = fileResult.filteredTransactions;
    auto &primary = fileResult.primary;
    auto &rename = fileResult.rename;
    int max_item = fileResult.max_item;

    // Flatten filteredTransactions
    std::vector<Item> items;
    std::vector<int> start;
    std::vector<int> end;

    for (const auto &[key, val] : filteredTransactions)
    {
        start.push_back(items.size());
        for (int i = 0; i < key.size(); i++)
        {
            items.push_back({key[i], val[i]});
        }
        end.push_back(items.size());
    }

    // copy items to device
    Item *d_items;
    int *d_start;
    int *d_end;
    int *d_primary;

    size_t num_items = items.size();
    size_t num_transactions = start.size();

    cudaMalloc(&d_items, num_items * sizeof(Item));
    cudaMalloc(&d_start, num_transactions * sizeof(int));
    cudaMalloc(&d_end, num_transactions * sizeof(int));
    cudaMalloc(&d_primary, primary.size() * sizeof(int));

    cudaMemcpy(d_items, items.data(), num_items * sizeof(Item), cudaMemcpyHostToDevice);
    cudaMemcpy(d_start, start.data(), num_transactions * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_end, end.data(), num_transactions * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_primary, primary.data(), primary.size() * sizeof(int), cudaMemcpyHostToDevice);

    int32_t *d_high_utility_patterns;
    cudaMallocManaged(&d_high_utility_patterns, 128 * MEGA); // 1GB
    d_high_utility_patterns[1] = 2;

    // memory
    CudaMemoryManager *memory_manager = createMemoryManager(total_memory, page_size);

    // Work Item
    WorkItem *work_item;
    cudaMalloc(&work_item, sizeof(WorkItem));

    copy<<<1, 1>>>(memory_manager, work_item, d_items, d_start, d_end, d_primary, primary.size(), num_transactions, max_item, d_high_utility_patterns, args.utility);
    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();

    // free the memory
    cudaFree(d_items);
    cudaFree(d_start);
    cudaFree(d_end);
    cudaFree(d_primary);

    std::cout << "High Utility Patterns: " << d_high_utility_patterns[0] << "\n";

    std::map<std::string, int> Patterns;

    // convert high utility patterns to string
    std::string high_utility_patterns_str = "";
    std::vector<int> high_utility_patten;
    for (int i = 0; i < d_high_utility_patterns[1]; i++)
    {
        while (d_high_utility_patterns[i + 2] != 0)
        {
            high_utility_patten.push_back(d_high_utility_patterns[i + 2]);
            i++;
        }
        // if empty, skip
        if (high_utility_patten.size() == 0)
        {
            continue;
        }

        for (int j = 0; j < high_utility_patten.size() - 1; j++)
        {
            high_utility_patterns_str += rename[high_utility_patten[j]] + " ";
        }
        Patterns[high_utility_patterns_str] = high_utility_patten[high_utility_patten.size() - 1];

        high_utility_patterns_str = "";
        high_utility_patten.clear();
    }
    for (const auto &p : Patterns)
    {
        std::cout << p.first << "UTIL: " << p.second << std::endl;
    }

    std::cout << "High Utility Patterns: " << d_high_utility_patterns[0] << "\n";

    return 0;
}