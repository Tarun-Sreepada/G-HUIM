#include "mine.cuh"

__device__ __forceinline__ bool fetchWorkItem(stack *work_stack, WorkItem &outItem)
{
    // Compute thread ID inside the function
    if (threadIdx.x == 0)
    {
        outItem.clear();
        // Keep popping until we get a non-empty item or stack is empty
        while (outItem.is_empty() && work_stack->may_pop())
        {
            work_stack->pop(outItem);
        }

    }
    __syncthreads();
    // If outItem is empty here, there's no more work
    return !outItem.is_empty();
}

__device__ __forceinline__ void finishTask(stack *work_stack)

{
    if (threadIdx.x == 0)
    {
        work_stack->finish_task();
    }
    __syncthreads();
}

__device__ __forceinline__ void reset_structs(
    WorkItem *new_work_item,
    const WorkItem &curr_work_item,
    Transaction *temporary_transactions,
    int32_t *transaction_hashes,
    Utilities *local_subtree_utilities,
    uint32_t *p_temp_transaction_count, // pointer
    uint32_t *p_temp_item_count,        // pointer
    int max_transactions,
    int max_utilities)
{
    // Set the shared variable to zero from thread 0
    if (threadIdx.x == 0)
    {
        *p_temp_transaction_count = 0;
        *p_temp_item_count = 0;
        new_work_item->clear();
        new_work_item->pattern = (uint32_t *)gmalloc(sizeof(uint32_t) * (curr_work_item.pattern_length + 1));
        new_work_item->pattern_length = curr_work_item.pattern_length + 1;
        new_work_item->pattern[curr_work_item.pattern_length] = curr_work_item.primary;
    }
    __syncthreads();

    // Copy the pattern
    for (int i = threadIdx.x; i < curr_work_item.pattern_length; i += blockDim.x)
    {
        new_work_item->pattern[i] = curr_work_item.pattern[i];
    }
    __syncthreads();

    // Clear out any necessary structures
    for (int i = threadIdx.x; i < max_transactions; i += blockDim.x)
    {
        temporary_transactions[i].clear();
        transaction_hashes[2 * i] = -1;
        transaction_hashes[2 * i + 1] = -1;
    }

    for (int i = threadIdx.x; i < max_utilities; i += blockDim.x)
    {
        local_subtree_utilities[i].clear();
    }
    __syncthreads();
}

__device__ __forceinline__ void initalize_structs(Transaction **pTemporaryPtr,
                                                  Utilities *pUtilitiesPtr,
                                                  int32_t **pTransactionHashes,
                                                  int max_transactions,
                                                  int max_utilities)
{
    if (threadIdx.x == 0)
    {
        // Now we write into the caller's pointer
        *pTemporaryPtr = (Transaction *)gmalloc(sizeof(Transaction) * max_transactions);
        *pTransactionHashes = (int32_t *)gmalloc(sizeof(int32_t) * max_transactions * scale);
    }
    __syncthreads();

    // Then do your initialization loops using (*pTemporaryPtr) etc.

    for (int i = threadIdx.x; i < max_utilities; i += blockDim.x)
    {
        pUtilitiesPtr[i].key = i;
    }
    __syncthreads();
}

__device__ __forceinline__ void add_pattern(WorkItem &new_work_item,
                                            uint32_t *patterns,
                                            uint32_t min_util)
{
    // Get thread ID
    __shared__ uint32_t pattern_offset;

    if (new_work_item.utility >= min_util)
    {
        // add_pattern(&new_work_item, patterns, min_util);
        if (threadIdx.x == 0)
        {
            pattern_offset = atomicAdd(&patterns[1], (new_work_item.pattern_length + 2));    // Get the current offset // 1 for Utility, 1 for spacer
            patterns[pattern_offset + new_work_item.pattern_length] = new_work_item.utility; // Store the utility
            // Copy the pattern to the global memory
        }
        __syncthreads();
        for (int i = threadIdx.x; i < new_work_item.pattern_length; i += blockDim.x)
        {
            patterns[pattern_offset + i] = new_work_item.pattern[i]; // Copy the pattern
        }
    }

    __syncthreads();

    return;
}

__device__ __forceinline__ void scan(const WorkItem &curr_work_item,
                                     WorkItem &new_work_item,
                                     Utilities *local_subtree_utilities,
                                     Transaction *temporary_transactions,
                                     uint32_t *temp_transaction_count,
                                     uint32_t *temp_item_count)
{
    // scan
    for (int i = threadIdx.x; i < curr_work_item.db->num_transactions; i += blockDim.x)
    {
        Transaction &transaction = curr_work_item.db->transactions[i];
        int idx = transaction.binary_search(curr_work_item.primary);

        if (idx != -1)
        {
            uint32_t transaction_pattern_utility = transaction.utility + transaction.data[idx].util;
            atomicAdd(&new_work_item.utility, transaction_pattern_utility);
            uint32_t remaining_items = transaction.length - idx - 1;

            if (remaining_items != 0)
            {
                int temp_transaction_idx = atomicAdd(temp_transaction_count, 1);
                atomicAdd(temp_item_count, remaining_items);
                // Copy the transaction data to the temporary transactions
                temporary_transactions[temp_transaction_idx].data = transaction.data + idx + 1;
                temporary_transactions[temp_transaction_idx].utility = transaction_pattern_utility;
                temporary_transactions[temp_transaction_idx].length = remaining_items;
            }
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < *temp_transaction_count; i += blockDim.x)
    {
        Transaction &transaction = temporary_transactions[i];
        uint32_t total_util = transaction.utility + transaction.util_sum();
        for (int j = 0; j < transaction.length; j++)
        {
            atomicAdd(&local_subtree_utilities[transaction.data[j].key].local_utility, total_util);
        }
    }

}

__device__ void allocate(WorkItem &wi,
                         uint32_t temp_item_count,
                         uint32_t temp_transaction_count)
{

    if (threadIdx.x == 0)
    {
        wi.db = (Database *)gmalloc(sizeof(Database));
        wi.db->data = (Item *)gmalloc(sizeof(Item) * temp_item_count);
        wi.db->transactions = (Transaction *)gmalloc(sizeof(Transaction) * temp_transaction_count);
    }
    __syncthreads();
}

__device__ uint64_t fasthash(uint64_t i)
{
    i += 1ULL;
    i ^= i >> 33ULL;
    i *= 0xff51afd7ed558ccdULL;
    i ^= i >> 33ULL;
    i *= 0xc4ceb9fe1a85ec53ULL;
    i ^= i >> 33ULL;
    return i;
}

__device__ __forceinline__ void write_merge(
    Transaction *temporary_transactions,
    uint32_t temp_transaction_count,
    int32_t *transaction_hashes,
    Utilities *local_subtree_utilities,
    WorkItem &new_work_item,
    uint32_t min_util, uint32_t transaction_count)
{

    for (int i = threadIdx.x; i < temp_transaction_count; i += blockDim.x)
    {

        Transaction &transaction = temporary_transactions[i];

        // acquire a new transaction
        int offset = atomicAdd(&new_work_item.db->num_transactions, 1);
        int item_offset = atomicAdd(&new_work_item.db->num_items, transaction.length);

        // copy the transaction context
        new_work_item.db->transactions[offset].data = new_work_item.db->data + item_offset;
        new_work_item.db->transactions[offset].utility = transaction.utility;

        int cumsum = 0;
        int idx = 0;

        uint64_t hash = 0;

        // write
        for (int j = 0; j < transaction.length; ++j)
        {
            Item &item = transaction.data[j];
            // if item's local utility is greater than min util copy it
            if (local_subtree_utilities[item.key].local_utility >= min_util)
            {
                hash ^= fasthash(item.key + idx);

                new_work_item.db->transactions[offset].data[idx] = item;
                cumsum += item.util;
                ++idx;
            }
        }

        // if idx == 0 skip
        if (idx == 0)
            continue;

        // update the length
        new_work_item.db->transactions[offset].length = idx;

        // update the subtree utility
        int temp = 0;
        for (int j = 0; j < idx; ++j)
        {
            atomicAdd(&local_subtree_utilities
                           [new_work_item.db->transactions[offset].data[j].key]
                               .subtree_utility,
                      transaction.utility + cumsum - temp);
            temp += new_work_item.db->transactions[offset].data[j].util;
        }

        // merge identical transactions
        uint64_t hash_index = hash % (transaction_count * scale);

        while (true)
        {
            int old = atomicCAS(&transaction_hashes[hash_index], -1, offset);
            if (old == -1)
                break; // Successfully inserted
            if (new_work_item.db->compare_transactions(old, offset))
            {
                // Found a duplicate transaction
                Transaction &old_transaction = new_work_item.db->transactions[old];

                atomicAdd(&old_transaction.utility, transaction.utility);
                for (int j = 0; j < idx; j++)
                {
                    atomicAdd(&old_transaction.data[j].util, new_work_item.db->transactions[offset].data[j].util);
                }

                // Mark the new transaction as a duplicate
                new_work_item.db->transactions[offset].data = nullptr;
                break;
            }
            hash_index = (hash_index + 1) % (transaction_count * scale);
        }
    }

    __syncthreads();
}

__device__ __forceinline__ void trim(WorkItem &new_work_item)
{
    // Shared memory
    __shared__ uint32_t temp_transaction_count;

    // Initialize the transaction count

    if (threadIdx.x == 0)
    {
        temp_transaction_count = 0;
    }
    __syncthreads();

    // push up the valid transactions
    if (threadIdx.x < 32)
    {
        for (int i = threadIdx.x; i < new_work_item.db->num_transactions; i += 32)
        {
            Transaction &transaction = new_work_item.db->transactions[i];
            if (transaction.data != nullptr)
            {
                int ret = atomicAdd(&temp_transaction_count, 1);
                new_work_item.db->transactions[ret].data = transaction.data;
                new_work_item.db->transactions[ret].utility = transaction.utility;
                new_work_item.db->transactions[ret].length = transaction.length;
            }
        }
    }

    __syncthreads();

    // update the transaction count
    if (threadIdx.x == 0)
    {
        new_work_item.db->num_transactions = temp_transaction_count;
    }

    __syncthreads();
}

__device__ __forceinline__ void push_work_item(Utilities *local_subtree_utilities,
                                               WorkItem &new_work_item, stack *work_stack,
                                               uint32_t min_util, uint32_t unique_item_count)
{
    // push the new work item to the stack
    for (int i = threadIdx.x; i < unique_item_count; i += blockDim.x)
    {
        // if the subtree utility is greater than min util push it
        WorkItem copy = new_work_item;
        if (local_subtree_utilities[i].subtree_utility >= min_util)
        {
            copy.primary = i;
            work_stack->push(copy);
        }
    }
}

__global__ void mine(stack *work_stack,
                     uint32_t *patterns,
                     uint32_t min_util,
                     uint32_t unique_item_count,
                     uint32_t transaction_count)
{
    // Shared memory
    __shared__ WorkItem curr_work_item;
    __shared__ Transaction *temporary_transactions;
    __shared__ uint32_t temp_transaction_count;
    __shared__ uint32_t temp_item_count;

    __shared__ int32_t *transaction_hashes;

    // __shared__ Utilities *local_subtree_utilities;
    extern __shared__ Utilities local_subtree_utilities[];

    __shared__ WorkItem new_work_item;

    // __shared__ uint32_t pattern_offset;

    initalize_structs(&temporary_transactions,
                      local_subtree_utilities,
                      &transaction_hashes,
                      transaction_count,
                      unique_item_count);
    __syncthreads();

    // While there is potentially another item to pop...
    while (work_stack->may_pop())
    {
        // Attempt to fetch an item
        bool gotItem = fetchWorkItem(work_stack, curr_work_item);

        // No more work available -> exit loop
        if (!gotItem)
            break;

        // Reset the shared variables
        reset_structs(&new_work_item,
                      curr_work_item,
                      temporary_transactions,
                      transaction_hashes,
                      local_subtree_utilities,
                      &temp_transaction_count,
                      &temp_item_count,
                      transaction_count,
                      unique_item_count);

        scan(curr_work_item,
             new_work_item,
             local_subtree_utilities,
             temporary_transactions,
             &temp_transaction_count,
             &temp_item_count);

        add_pattern(new_work_item, patterns, min_util);

        allocate(new_work_item, temp_item_count, temp_transaction_count);

        // merge write....
        write_merge(temporary_transactions, temp_transaction_count, transaction_hashes, local_subtree_utilities, new_work_item, min_util, transaction_count);

        trim(new_work_item);

        push_work_item(local_subtree_utilities, new_work_item, work_stack, min_util, unique_item_count);

        // Indicate we're done with this item
        finishTask(work_stack);
    }
}
