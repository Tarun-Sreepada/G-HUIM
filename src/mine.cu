#include "mine.cuh"

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


__device__ int find_item(const Utilities *buckets, int table_size, int key)
{
    uint32_t hash = fasthash(key) % table_size;
    while (true)
    {
        if (buckets[hash].key == key)
            return hash;
        if (buckets[hash].key == 0)
            return -1;
        hash = (hash + 1) % table_size;
    }
}

__device__ __forceinline__ bool fetchWorkItem(stack *work_stack, WorkItem &outItem)
{
    // Compute thread ID inside the function
    if (threadIdx.x == 0)
    {
        outItem = work_stack->items[blockIdx.x];
    }
    __syncthreads();
    // If outItem is empty here, there's no more work
    return true;
}

__device__ __forceinline__ void initalize_structs(Transaction **pTemporaryPtr,
                                                  Utilities **pUtilitiesPtr,
                                                  int32_t **pTransactionHashes,
                                                  int max_transactions,
                                                  int max_utilities)
{
    if (threadIdx.x == 0)
    {
        // Now we write into the caller's pointer
        *pTemporaryPtr = (Transaction *)gmalloc(sizeof(Transaction) * max_transactions);
        *pTransactionHashes = (int32_t *)gmalloc(sizeof(int32_t) * max_transactions * scale);
        *pUtilitiesPtr = (Utilities *)gmalloc(sizeof(Utilities) * max_utilities * scale);
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
    }

    for (int i = threadIdx.x; i < max_transactions * scale; i += blockDim.x)
    {
        transaction_hashes[i] = -1;
    }

    for (int i = threadIdx.x; i < max_utilities * scale; i += blockDim.x)
    {
        local_subtree_utilities[i].clear();
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

__device__ int binary_search_segment(const Transaction &transaction,
                                     uint32_t item,
                                     int seg_start,
                                     int seg_length)
{
    int left = seg_start;
    int right = seg_start + seg_length - 1;
    while (left <= right)
    {
        int mid = left + (right - left) / 2;
        if (transaction.data[mid].key == item)
        {
            return mid;
        }
        else if (transaction.data[mid].key < item)
        {
            left = mid + 1;
        }
        else
        {
            right = mid - 1;
        }
    }
    return -1; // Not found in this segment.
}

__device__ __forceinline__ void load_batch_data(
    const WorkItem &curr_work_item,
    int batchStart,
    int batchSize,
    Transaction *sharedTransactions,
    int *segmentCounts,
    int *foundIndices, uint16_t segment_size)
{
    int t = threadIdx.x;
    if (t < batchSize)
    {
        sharedTransactions[t] = curr_work_item.db->transactions[batchStart + t];
        segmentCounts[t] = (sharedTransactions[t].length + (segment_size - 1)) / segment_size;
        foundIndices[t] = -1; // initialize no match found
    }
}

__device__ __forceinline__ int parallel_prefix_sum(
    int *segmentCounts,
    int *prefixSegments,    // Output exclusive scan
    int *temp_scan_storage, // Temporary storage for scan
    int batchSize)
{
    int t = threadIdx.x;

    // (2a) Initialize temp array
    if (t < batchSize)
    {
        temp_scan_storage[t] = segmentCounts[t];
    }
    else
    {
        // Zero out
        int max_idx = blockDim.x;
        if (t < max_idx)
        {
            temp_scan_storage[t] = 0;
        }
    }
    if (t == 0)
    {
        prefixSegments[0] = 0;
    }
    __syncthreads(); // Ensure temp and prefixSegments[0] are initialized

    // (2b) Parallel Inclusive Scan
    for (int offset = 1; offset < batchSize; offset *= 2)
    {
        int val = 0;
        // Read phase
        if (t >= offset && t < batchSize)
        {
            val = temp_scan_storage[t] + temp_scan_storage[t - offset];
        }
        else if (t < batchSize)
        {
            val = temp_scan_storage[t];
        }
        __syncthreads();

        // Write phase
        if (t < batchSize)
        {
            temp_scan_storage[t] = val;
        }
        __syncthreads();
    }

    // (2c) Write Exclusive Scan result to prefixSegments
    if (t < batchSize)
    {
        prefixSegments[t + 1] = temp_scan_storage[t];
    }
    __syncthreads(); // Ensure prefixSegments is fully written before returning total

    // (2d) Return total segments
    return (batchSize > 0) ? prefixSegments[batchSize] : 0;
}

__device__ __forceinline__ void process_segments(
    const WorkItem &curr_work_item,
    int batchSize,
    int totalSegments,
    const Transaction *sharedTransactions,    // read-only access needed
    const int *prefixSegments,                // read-only access needed
    int *foundIndices, uint16_t segment_size) // Output results
{
    // Each thread processes one or more segments (round-robin)
    for (int seg = threadIdx.x; seg < totalSegments; seg += blockDim.x)
    {
        // Find the transaction index 'transIndex' such that:
        // prefixSegments[transIndex] <= seg < prefixSegments[transIndex + 1]
        int low = 0;
        int high = batchSize;
        int transIndex = -1; // Initialize to invalid index

        while (low < high)
        {
            int mid = low + ((high - low) >> 1);

            if (prefixSegments[mid] <= seg)
            {
                low = mid + 1;
            }
            else
            {
                high = mid;
            }
        }

        transIndex = low - 1;

        if (transIndex >= 0 && transIndex < batchSize)
        {
            // Within the found transaction, determine the segment details
            int segInTransaction = seg - prefixSegments[transIndex];
            const Transaction &transaction = sharedTransactions[transIndex];
            int segStart = segInTransaction * segment_size;

            if (segStart < transaction.length)
            {
                int segLength = min((int)segment_size, transaction.length - segStart); // Cast segment_size if necessary

                int localIdx = binary_search_segment(transaction, curr_work_item.primary, segStart, segLength);

                if (localIdx != -1)
                {
                    atomicCAS(&foundIndices[transIndex], -1, localIdx);
                }
            }
        }

    } // End for loop over segments
}
__device__ __forceinline__ void consolidate_batch_results(
    int batchSize,
    const Transaction *sharedTransactions, // read-only access needed
    const int *foundIndices,               // read-only access needed
    WorkItem &new_work_item,
    Transaction *temporary_transactions,
    uint32_t *temp_transaction_count,
    uint32_t *temp_item_count)
{
    int t = threadIdx.x;
    if (t < batchSize)
    {
        const Transaction &transaction = sharedTransactions[t];
        int foundIdx = foundIndices[t];

        if (foundIdx != -1)
        {
            // Item found in this transaction
            uint32_t transaction_pattern_utility = transaction.utility + transaction.data[foundIdx].util;

            // Add utility to the new work item
            atomicAdd(&new_work_item.utility, transaction_pattern_utility);

            // Check if there are remaining items to form a temporary transaction
            int remaining_items = transaction.length - foundIdx - 1;
            if (remaining_items > 0)
            {
                // Atomically get an index for the new temporary transaction
                int temp_transaction_idx = atomicAdd(temp_transaction_count, 1);
                // Atomically update the total count of items in temporary transactions
                atomicAdd(temp_item_count, remaining_items);

                // Populate the temporary transaction entry
                temporary_transactions[temp_transaction_idx].data = transaction.data + foundIdx + 1;
                temporary_transactions[temp_transaction_idx].utility = transaction_pattern_utility;
                temporary_transactions[temp_transaction_idx].length = remaining_items;
            }
        }
    }
}

__device__ __forceinline__ void process_temporary_transactions(
    Transaction *temporary_transactions,
    uint32_t *temp_transaction_count_ptr, // Pointer to the count
    Utilities *local_subtree_utilities)
{

    uint32_t temp_count = *temp_transaction_count_ptr;

    // Process temporary transactions in parallel
    for (int i = threadIdx.x; i < temp_count; i += blockDim.x)
    {
        // Get a reference to the temporary transaction
        Transaction &transaction = temporary_transactions[i];

        // Calculate the total utility for this projected transaction
        uint32_t total_util = transaction.utility + transaction.util_sum();

        // Add this utility to the local utility count for each item in it
        for (int j = 0; j < transaction.length; j++)
        {
            // Ensure key is within bounds of local_subtree_utilities if it's an array
            atomicAdd(&local_subtree_utilities[transaction.data[j].key].local_utility, total_util);
        }
    }
}

__device__ __forceinline__ void scan(const WorkItem &curr_work_item,
                                     WorkItem &new_work_item,
                                     Utilities *local_subtree_utilities,
                                     Transaction *temporary_transactions,
                                     uint32_t *temp_transaction_count,
                                     uint32_t *temp_item_count, uint16_t segment_size)
{
    // Define constants and shared memory
    const int TRANSACTIONS_PER_BATCH = 512;
    __shared__ Transaction sharedTransactions[TRANSACTIONS_PER_BATCH];
    __shared__ int segmentCounts[TRANSACTIONS_PER_BATCH];
    __shared__ int prefixSegments[TRANSACTIONS_PER_BATCH + 1];
    __shared__ int foundIndices[TRANSACTIONS_PER_BATCH];
    __shared__ int temp_scan_storage[TRANSACTIONS_PER_BATCH]; // For parallel scan

    // Loop over batches
    for (int batchStart = 0; batchStart < curr_work_item.db->num_transactions; batchStart += TRANSACTIONS_PER_BATCH)
    {
        int batchSize = min(TRANSACTIONS_PER_BATCH, curr_work_item.db->num_transactions - batchStart);

        // Stage 1: Load data for the current batch
        load_batch_data(curr_work_item, batchStart, batchSize,
                        sharedTransactions, segmentCounts, foundIndices, segment_size);
        __syncthreads(); // Ensure loading is complete

        // Stage 2: Compute prefix sum of segment counts
        int totalSegments = parallel_prefix_sum(segmentCounts, prefixSegments,
                                                temp_scan_storage, batchSize);
        __syncthreads(); // Ensure prefix sum calculation is complete

        // Stage 3: Process segments (perform searches)
        process_segments(curr_work_item, batchSize, totalSegments,
                         sharedTransactions, prefixSegments, foundIndices, segment_size);
        __syncthreads(); // Ensure segment processing is complete

        // Stage 4: Consolidate results from this batch
        consolidate_batch_results(batchSize, sharedTransactions, foundIndices,
                                  new_work_item, temporary_transactions,
                                  temp_transaction_count, temp_item_count);
        __syncthreads(); // Ensure consolidation is complete before next batch
    }

    __syncthreads(); // Ensure all batches are processed before moving to temp transactions

    // Stage 5: Process all accumulated temporary transactions
    process_temporary_transactions(temporary_transactions, temp_transaction_count,
                                   local_subtree_utilities);
}

__device__ void add_bucket_util(Utilities *buckets, uint32_t table_size, uint32_t key, uint32_t utility, bool local_utility)
{
    // hash the key
    int idx = fasthash(key) % table_size;

    // find the key
    while (true)
    {
        // we are adding in atomic so do compare and swap for the key
        int old = atomicCAS(&buckets[idx].key, 0, key);
        if (old == key || old == 0)
        {
            if (local_utility == true)
            {
                atomicAdd(&buckets[idx].local_utility, utility);
            }
            else
            {
                atomicAdd(&buckets[idx].subtree_utility, utility);
                // printf("Key: %d, Subtree Utility: %d\n", buckets[idx].key, atomicAdd(&buckets[idx].subtree_utility, utility));
            }
            return;
        }
        // if the key is not present, find the next slot
        idx = (idx + 1) % (table_size);
    }
}

// --- Helper Function Implementations ---

__device__ __forceinline__ void old_scan(const WorkItem &curr_work_item,
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

    // local utilities
    for (int i = threadIdx.x; i < *temp_transaction_count; i += blockDim.x)
    {
        Transaction &transaction = temporary_transactions[i];
        uint32_t total_util = transaction.utility + transaction.util_sum();
        for (int j = 0; j < transaction.length; j++)
        {
            // atomicAdd(&local_subtree_utilities[transaction.data[j].key].local_utility, total_util);

            add_bucket_util(local_subtree_utilities,
                            curr_work_item.max_item * scale,
                            transaction.data[j].key, total_util, true);
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

__device__ __forceinline__ void write_merge(
    Transaction *temporary_transactions,
    uint32_t temp_transaction_count,
    int32_t *transaction_hashes,
    Utilities *local_subtree_utilities,
    WorkItem &new_work_item,
    uint32_t min_util, uint32_t transaction_count, uint32_t max_item)
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

        // write
        for (int j = 0; j < transaction.length; ++j)
        {
            Item &item = transaction.data[j];
            // if item's local utility is greater than min util copy it
            // if (local_subtree_utilities[item.key].local_utility >= min_util)
            // {
            //     hash ^= fasthash(item.key + idx);

            //     new_work_item.db->transactions[offset].data[idx] = item;
            //     cumsum += item.util;
            //     ++idx;
            // }
            int found = find_item(local_subtree_utilities, max_item * scale, item.key);
            if (local_subtree_utilities[found].local_utility >= min_util)
            {
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
        uint64_t hash = 0;

        // update the subtree utility
        int temp = 0;
        for (int j = 0; j < idx; ++j)
        {
            add_bucket_util(local_subtree_utilities,
                            max_item * scale,
                            new_work_item.db->transactions[offset].data[j].key,
                            transaction.utility + cumsum - temp, false);

            hash ^= fasthash(new_work_item.db->transactions[offset].data[j].key + idx);
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

    __shared__ uint32_t new_unique_item_count;

    if (threadIdx.x == 0)
    {
        new_unique_item_count = 0;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < unique_item_count * scale; i += blockDim.x)
    {
        // get the new unique item count
        if (local_subtree_utilities[i].local_utility >= min_util)
        {
            atomicAdd(&new_unique_item_count, 1);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        new_work_item.max_item = new_unique_item_count;
    }
    __syncthreads();

    // push the new work item to the stack
    for (int i = threadIdx.x; i < unique_item_count * scale; i += blockDim.x)
    {
        // if the subtree utility is greater than min util push it
        WorkItem copy = new_work_item;
        if (local_subtree_utilities[i].subtree_utility >= min_util)
        {
            copy.primary = local_subtree_utilities[i].key;
            work_stack->push(copy);
        }
    }
}

__global__ void mine(stack *work_stack,
                     stack *new_work_stack,
                     uint32_t *patterns,
                     uint32_t min_util)
{
    // Shared memory
    __shared__ WorkItem curr_work_item;
    __shared__ Transaction *temporary_transactions;
    __shared__ uint32_t temp_transaction_count;
    __shared__ uint32_t temp_item_count;

    __shared__ int32_t *transaction_hashes;

    __shared__ Utilities *local_subtree_utilities;

    __shared__ WorkItem new_work_item;

    // bool gotItem =
    fetchWorkItem(work_stack, curr_work_item);

    initalize_structs(&temporary_transactions,
                      &local_subtree_utilities,
                      &transaction_hashes,
                      curr_work_item.db->num_transactions,
                      curr_work_item.max_item);
    __syncthreads();

    // uint16_t segment_size = 16;

    // While there is potentially another item to pop...
    // while (work_stack->may_pop())
    // {
    // Attempt to fetch an item

    // // No more work available -> exit loop
    // if (!gotItem)
    //     return;

    // Reset the shared variables
    reset_structs(&new_work_item,
                  curr_work_item,
                  temporary_transactions,
                  transaction_hashes,
                  local_subtree_utilities,
                  &temp_transaction_count,
                  &temp_item_count,
                  curr_work_item.db->num_transactions,
                  curr_work_item.max_item);

    // if (scan_bool == false)
    // {
    // scan(curr_work_item,
    //      new_work_item,
    //      local_subtree_utilities,
    //      temporary_transactions,
    //      &temp_transaction_count,
    //      &temp_item_count, segment_size);
    // scan_bool = true;
    // }
    // else
    // {
    old_scan(curr_work_item,
             new_work_item,
             local_subtree_utilities,
             temporary_transactions,
             &temp_transaction_count,
             &temp_item_count);
    // }

    add_pattern(new_work_item, patterns, min_util);

    allocate(new_work_item, temp_item_count, temp_transaction_count);

    write_merge(temporary_transactions, temp_transaction_count, transaction_hashes, local_subtree_utilities, new_work_item, min_util, curr_work_item.db->num_transactions, curr_work_item.max_item);

    trim(new_work_item);

    push_work_item(local_subtree_utilities, new_work_item, new_work_stack, min_util, curr_work_item.max_item);

    // // Indicate we're done with this item
    // finishTask(work_stack);
    // }
}