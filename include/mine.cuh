#pragma once

#include "patterns.cuh"
#include "memory.cuh"

#define scale 2

// -------------------------------------------------------------------------
// Cleaned-up kernel to copy work data and push work items onto the queue.
__global__ void copy_work(AtomicWorkStack *curr_work_queue,
                          WorkItem *work_item,
                          int *primary,
                          int primary_size)
{
    const int tid = threadIdx.x;
    __shared__ WorkItem local_item;

    // === Initialization (by thread 0) ===
    if (tid == 0)
    {
        local_item.pattern = (int *)global_malloc(sizeof(int)); // Allocate pattern space.
        local_item.pattern_length = 0;
        local_item.utility = 0;
        local_item.db = (Database *)global_malloc(sizeof(Database));
        local_item.db->numItems = work_item->db->numItems;
        local_item.db->numTransactions = work_item->db->numTransactions;
        local_item.db->d_data = (Item *)global_malloc(sizeof(Item) * work_item->db->numItems);
        local_item.db->d_transactions = (Transaction *)global_malloc(sizeof(Transaction) * work_item->db->numTransactions);
        local_item.max_item = work_item->max_item;

        local_item.work_done = (int *)global_malloc(sizeof(int));
        local_item.work_done[0] = 0;
        local_item.work_count = primary_size;
    }
    __syncthreads();

    // === Copy Database Data ===
    // Copy d_data array in parallel.
    for (int i = tid; i < work_item->db->numItems; i += blockDim.x)
    {
        local_item.db->d_data[i] = work_item->db->d_data[i];
    }
    __syncthreads();

    // Copy each transaction's metadata.
    for (int i = tid; i < work_item->db->numTransactions; i += blockDim.x)
    {
        // Calculate offset into d_data.
        int offset = work_item->db->d_transactions[i].data - work_item->db->d_data;
        local_item.db->d_transactions[i].data = local_item.db->d_data + offset;
        local_item.db->d_transactions[i].length = work_item->db->d_transactions[i].length;
        local_item.db->d_transactions[i].utility = 0;
    }
    __syncthreads();

    // === Push Work Items ===
    // Only one thread pushes items; for each primary value, update the local work item and push it.
    if (tid == 0)
    {
        for (int i = 0; i < primary_size; i++)
        {
            local_item.primary = primary[i];
            curr_work_queue->push(local_item);
        }
    }
}

__device__ void add_bucket_util(Item *local_util, int table_size, int key, int total_util)
{
    // hash the key
    int idx = hashFunction(key, table_size);

    // find the key
    while (true)
    {
        // we are adding in atomic so do compare and swap for the key
        int old = atomicCAS(&local_util[idx].key, 0, key);
        if (old == key)
        {
            // if the key is already present, add the utility
            atomicAdd(&local_util[idx].util, total_util);
            return;
        }
        else if (old == 0)
        {
            // if the key is not present, add the utility since we cas'd it
            atomicAdd(&local_util[idx].util, total_util);
            return;
        }
        // if the key is not present, find the next slot
        idx = (idx + 1) % (table_size);
    }
}

// -------------------------------------------------------------------------
__global__ void mine(AtomicWorkStack *stack,
    HighUtilityPatterns *patterns, int min_util)
{

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    // Use shared memory only for values that one block will process together.
    __shared__ WorkItem work_item; // the work-item popped from the queue
    __shared__ bool s_popped;      // did we successfully pop a work-item?

    // Shared copies for data that one block uses to process the work-item.
    __shared__ WorkItem new_work_item;
    __shared__ Transaction *temp_transaction;
    __shared__ Item *local_util;
    __shared__ int num_items;
    __shared__ int num_transactions;

    __shared__ int *hashes;
    __shared__ Item *subtree_util;
    __shared__ int max_item;
    __shared__ int primary_count;
    
    printf("Tid: %d\tBid: %d\n", tid, bid);
    printf("Active: %d\n", stack->get_active());


    while(stack->get_active() > 0)
    {

        // (1) Only thread 0 pops from the work queue.
        if (tid == 0)
        {
            s_popped = stack->pop(&work_item);
        }
        __syncthreads();

         // If no work-item was popped, then skip this iteration.
         if (!s_popped)
         {
             __syncthreads(); // all threads must sync before next iteration
             continue;
         }

         // (2) Thread 0 initializes the new work item.
        if (tid == 0)
        {
            // printf("Starting work item\n");
            memset(&new_work_item, 0, sizeof(WorkItem));
            new_work_item.utility = 0;
            // Allocate and copy the pattern
            new_work_item.pattern = reinterpret_cast<int *>(
                global_malloc((work_item.pattern_length + 1) * sizeof(int)));
            memcpy(new_work_item.pattern,
                   work_item.pattern,
                   work_item.pattern_length * sizeof(int));
            new_work_item.pattern[work_item.pattern_length] = work_item.primary;
            new_work_item.pattern_length = work_item.pattern_length + 1;

            num_items = 0;
            num_transactions = 0;
            local_util = reinterpret_cast<Item *>(
                global_malloc(work_item.max_item * scale * sizeof(Item)));
            memset(local_util, 0, work_item.max_item * scale * sizeof(Item));
            temp_transaction = reinterpret_cast<Transaction *>(
                global_malloc(work_item.db->numTransactions * sizeof(Transaction)));
            memset(temp_transaction, 0, work_item.db->numTransactions * sizeof(Transaction));
        }
        __syncthreads();


        // (4) Process each transaction in parallel.
        for (int i = tid; i < work_item.db->numTransactions; i += blockDim.x)
        {
            Transaction &oldTrans = work_item.db->d_transactions[i];
            int idx = oldTrans.findItem(work_item.primary);
            if (idx == -1)
                continue;

            // Update new_work_item.utility atomically.
            atomicAdd(&new_work_item.utility, oldTrans.utility + oldTrans.data[idx].util);

            int suffix_count = oldTrans.length - (idx + 1);
            if (suffix_count <= 0)
                continue;

            // Reserve space for suffix items.
            atomicAdd(&num_items, suffix_count);
            int t_idx = atomicAdd(&num_transactions, 1);

            // Create a temporary transaction with the suffix.
            Transaction &newTrans = temp_transaction[t_idx];
            newTrans.data = oldTrans.data + idx + 1;
            newTrans.length = suffix_count;
            newTrans.utility = oldTrans.utility + oldTrans.data[idx].util;

            // Compute the total local utility for the transaction.
            int total_local_util = newTrans.utility;
            for (int j = 0; j < suffix_count; j++)
            {
                total_local_util += newTrans.data[j].util;
            }

            // Update the local utility buckets.
            for (int j = 0; j < suffix_count; j++)
            {
                add_bucket_util(local_util,
                                work_item.max_item * scale,
                                newTrans.data[j].key,
                                total_local_util);
            }
        }
        // After processing all transactions in parallel:
        __syncthreads();


        if (tid == 0)
        {
            // printf("Utility: %d\n", new_work_item.utility);
            if (new_work_item.utility >= min_util)
            {
                patterns->add_pattern(&new_work_item);
            }
        }
        __syncthreads();


        // Now handle the case when there are no surviving transactions:
        if (num_transactions == 0)
        {
            if (tid == 0)
            {
                stack->finish_task();

                // Free allocated memory here if necessary.
                global_free(local_util);
                global_free(temp_transaction);

                // Free the pattern.
                global_free(new_work_item.pattern);

                int ret = atomicAdd(&work_item.work_done[0], 1);
                if (ret == work_item.work_count - 1)
                {
                    global_free(work_item.work_done);
                    global_free(work_item.pattern);
                    global_free(work_item.db->d_data);
                    global_free(work_item.db->d_transactions);
                    global_free(work_item.db);
                }
            }
            __syncthreads();
            continue;
        }

         // (6) Allocate memory for the new database in new_work_item.
         if (tid == 0)
         {
             // printf("Allocating memory\n");
             new_work_item.db = reinterpret_cast<Database *>(global_malloc(sizeof(Database)));
             new_work_item.db->d_data = reinterpret_cast<Item *>(global_malloc(num_items * sizeof(Item)));
             new_work_item.db->d_transactions = reinterpret_cast<Transaction *>(
                 global_malloc(num_transactions * sizeof(Transaction)));
             new_work_item.db->numItems = 0;
             new_work_item.db->numTransactions = 0;
 
             max_item = 0;
             primary_count = 0;
             hashes = reinterpret_cast<int *>(
                 global_malloc(num_transactions * scale * sizeof(int)));
             memset(hashes, -1, num_transactions * scale * sizeof(int));
             subtree_util = reinterpret_cast<Item *>(
                 global_malloc(work_item.max_item * scale * sizeof(Item)));
             memset(subtree_util, 0, work_item.max_item * scale * sizeof(Item));
         }
         __syncthreads();


         // (8) For each temporary transaction, filter and copy it into the new DB.
        for (int i = tid; i < num_transactions; i += blockDim.x)
        {
            Transaction tempTrans = temp_transaction[i];
            int total_subtree_util = tempTrans.utility;
            int count = 0;
            // Count how many items survive filtering.
            for (int j = 0; j < tempTrans.length; j++)
            {
                int idx = find_item(local_util, work_item.max_item * scale, tempTrans.data[j].key);
                if (idx == -1)
                    continue;
                if (local_util[idx].util >= min_util)
                {
                    count++;
                    total_subtree_util += tempTrans.data[j].util;
                }
            }
            if (count == 0)
                continue;

            int new_trans_idx = atomicAdd(&new_work_item.db->numTransactions, 1);
            Transaction &newTrans = new_work_item.db->d_transactions[new_trans_idx];

            int start = atomicAdd(&new_work_item.db->numItems, count);
            newTrans.data = new_work_item.db->d_data + start;
            newTrans.length = count;
            newTrans.utility = tempTrans.utility;

            int trans_idx = 0;
            int temp_util = 0;
            for (int j = 0; j < tempTrans.length; j++)
            {
                int idx = find_item(local_util, work_item.max_item * scale, tempTrans.data[j].key);
                if (idx == -1)
                    continue;
                if (local_util[idx].util >= min_util)
                {
                    newTrans.data[trans_idx++] = tempTrans.data[j];
                    add_bucket_util(subtree_util,
                                    work_item.max_item * scale,
                                    tempTrans.data[j].key,
                                    total_subtree_util - temp_util);
                    temp_util += tempTrans.data[j].util;
                }
            }

            // (9) Use open addressing to try to merge transactions.
            int hash_idx = items_hasher(newTrans.data, newTrans.length, num_transactions * scale);
            while (true)
            {
                int old = atomicCAS(&hashes[hash_idx], -1, new_trans_idx);
                if (old == -1)
                {
                    // Inserted successfully.
                    // cumulative_indices[new_trans_idx + 1] = 1;
                    break;
                }
                // If the transactions have the same key pattern, merge them.
                if (new_work_item.db->sameKey(old, new_trans_idx))
                {
                    Transaction &oldTran = new_work_item.db->d_transactions[old];
                    atomicAdd(&oldTran.utility, newTrans.utility);
                    for (int j = 0; j < oldTran.length; j++)
                    {
                        atomicAdd(&oldTran.data[j].util, newTrans.data[j].util);
                    }
                    // Mark this transaction as merged.
                    newTrans.data = nullptr;
                    newTrans.length = 0;
                    newTrans.utility = 0;
                    break;
                }
                hash_idx = (hash_idx + 1) % (num_transactions * scale);
            }
        }
        __syncthreads();


        // If no transactions survived filtering, then skip this iteration.
        if (new_work_item.db->numTransactions == 0)
        {
            if (tid == 0)
            {
                stack->finish_task();

                // Free allocated memory here if necessary.
                global_free(local_util);
                global_free(temp_transaction);
                global_free(hashes);
                global_free(subtree_util);

                // Free the pattern.
                global_free(new_work_item.pattern);
            }
            __syncthreads();
            continue;
        }


        // (10) Update max_item and primary_count in parallel.
        for (int i = tid; i < work_item.max_item * scale; i += blockDim.x)
        {
            if (local_util[i].util >= min_util)
                atomicAdd(&max_item, 1);
            if (subtree_util[i].util >= min_util)
                atomicAdd(&primary_count, 1);
        }
        __syncthreads();


        if (tid == 0)
        {
            int compact_index = 0;
            // int temp_old = new_work_item.db->numTransactions;
            for (int i = 0; i < new_work_item.db->numTransactions; i++)
            {
                if (new_work_item.db->d_transactions[i].data != nullptr)
                {
                    new_work_item.db->d_transactions[compact_index++] =
                        new_work_item.db->d_transactions[i];
                }
            }
            new_work_item.db->numTransactions = compact_index;

            new_work_item.max_item = max_item;
            new_work_item.work_count = primary_count;
            new_work_item.work_done = reinterpret_cast<int *>(
                global_malloc(sizeof(int)));
            new_work_item.work_done[0] = 0;
            // (12) For every surviving primary in subtree_util, push a new work-item.
            for (int i = 0; i < work_item.max_item * scale; i++)
            {
                if (subtree_util[i].util >= min_util)
                {
                    new_work_item.primary = subtree_util[i].key;
                    stack->push(new_work_item);
                }
            }
            stack->finish_task();

            // Free allocated memory here if necessary.
            global_free(local_util);
            global_free(temp_transaction);
            global_free(hashes);
            global_free(subtree_util);

            int ret = atomicAdd(&work_item.work_done[0], 1);

            if (ret == (work_item.work_count - 1))
            {

                global_free(work_item.work_done);
                global_free(work_item.pattern);
                global_free(work_item.db->d_data);
                global_free(work_item.db->d_transactions);
                global_free(work_item.db);
            }
        }
        __syncthreads();

















    }

}