#include "mine/static_parallelism.cuh"

__global__ void copy_work(Stack<Task, CAPACITY> *work_stack, Database *db, uint32_t *primary, uint32_t primary_size, uint32_t unique)
{
    int tid = threadIdx.x;
    __shared__ Task task;
    // WorkItem item;
    if (tid == 0)
    {
        task.pattern = nullptr;
        task.pattern_length = 0;
        task.utility = 0;
        task.db = (Database *)gmalloc(sizeof(Database));
        task.db->num_items = db->num_items;
        task.db->num_transactions = db->num_transactions;

        task.db->data = (Item *)gmalloc(sizeof(Item) * db->num_items);

        task.db->transactions = (Transaction *)gmalloc(sizeof(Transaction) * db->num_transactions);

        task.unique_items = unique;

        task.work_done = (uint32_t *)gmalloc(sizeof(uint32_t));
        task.work_done[0] = 0;
        task.work_count = primary_size;
    }

    __syncthreads();

    // copy items
    for (int i = tid; i < db->num_items; i += blockDim.x)
    {
        task.db->data[i] = db->data[i];
    }

    // copy transaction structures
    for (int i = tid; i < db->num_transactions; i += blockDim.x)
    {
        task.db->transactions[i].utility = 0;
        task.db->transactions[i].length = db->transactions[i].length;
        task.db->transactions[i].data = task.db->data + (db->transactions[i].data - db->data); // data + offset (offset)
    }

    // push the work item to the stack

    Task copy = task;
    for (int i = tid; i < primary_size; i += blockDim.x)
    {
        copy.primary = primary[i];
        work_stack->push(copy);
    }

    __syncthreads();

    return;
}

uint32_t next_pow2(uint32_t n)
{
    uint32_t p = 1;
    while (p < n)
    {
        p <<= 1;
    }
    return p;
}

__device__ __forceinline__ void fetch_task(Stack<Task, CAPACITY> *work_stack, Task &item)
{
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    if (thread_id == 0)
    {
        work_stack->get(item, block_id);
    }

    __syncthreads();
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
    Task *new_work_item,
    const Task &curr_work_item,
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

__device__ __forceinline__ void add_bucket_util(Utilities *buckets, uint32_t table_size, uint32_t key, uint32_t utility, bool local_utility)
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
            if (local_utility)
            {
                atomicAdd(&buckets[idx].local_utility, utility);
            }
            else
            {
                atomicAdd(&buckets[idx].subtree_utility, utility);
            }
            return;
        }
        // if the key is not present, find the next slot
        idx = (idx + 1) % (table_size);
    }
}

__device__ void scan(const Task &curr_work_item,
                     Task &new_work_item,
                     Utilities *local_subtree_utilities,
                     Transaction *temporary_transactions,
                     uint32_t *temp_transaction_count,
                     uint32_t *temp_item_count)
{
    // scan
    for (int i = threadIdx.x; i < curr_work_item.db->num_transactions; i += blockDim.x)
    {
        Transaction transaction = curr_work_item.db->transactions[i];
        int idx = transaction.binary_search(curr_work_item.primary);

        if (idx != -1)
        {
            uint32_t transaction_pattern_utility = transaction.utility + transaction[idx].util;
            atomicAdd(&new_work_item.utility, transaction_pattern_utility);
            uint32_t remaining_items = transaction.size() - idx - 1;

            if (remaining_items > 0)
            {
                int temp_transaction_idx = atomicAdd(temp_transaction_count, 1);
                atomicAdd(temp_item_count, remaining_items);
                temporary_transactions[temp_transaction_idx].data = &transaction[idx + 1];
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
            add_bucket_util(local_subtree_utilities,
                            curr_work_item.unique_items * scale,
                            transaction[j].key, total_util, true);
        }
    }

    __syncthreads();
}

__device__ void allocate(Task &wi,
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

__device__ void write_merge(
    Transaction *temporary_transactions,
    uint32_t temp_transaction_count,
    int32_t *transaction_hashes,
    Utilities *local_subtree_utilities,
    Task &new_work_item,
    uint32_t min_util, uint32_t transaction_count, uint32_t max_item)
{

    for (int i = threadIdx.x; i < temp_transaction_count; i += blockDim.x)
    {

        Transaction &transaction = temporary_transactions[i];

        // acquire a new transaction
        int offset = atomicAdd(&new_work_item.db->num_transactions, 1);
        int item_offset = atomicAdd(&new_work_item.db->num_items, transaction.length);

        // copy the transaction context
        Transaction &new_transaction = new_work_item.db->transactions[offset];
        new_transaction.data = &new_work_item.db->data[item_offset];
        new_transaction.utility = transaction.utility;

        int cumsum = transaction.utility;
        int idx = 0;

        // write
        for (int j = 0; j < transaction.length; ++j)
        {
            Item item = transaction.data[j];
            int found = find_item(local_subtree_utilities, max_item * scale, item.key);
            if (local_subtree_utilities[found].local_utility >= min_util)
            {
                new_transaction[idx++] = item;
                cumsum += item.util;
            }
        }

        // if no items are found, skip the transaction
        if (idx == 0)
            continue;

        // update the length
        new_transaction.length = idx;
        uint64_t hash = 0;

        // update the subtree utility
        int temp = 0;
        for (int j = 0; j < idx; ++j)
        {
            Item item = new_work_item.db->transactions[offset].data[j];
            // add the utility to the transaction
            add_bucket_util(local_subtree_utilities,
                            max_item * scale,
                            item.key,
                            cumsum - temp, false);

            hash ^= fasthash(item.key + j);
            temp += item.util;
        }

        // merge identical transactions
        uint64_t hash_index = hash % (transaction_count * scale);

        while (true)
        {
            int old = atomicCAS(&transaction_hashes[hash_index], -1, offset);
            if (old == -1)
                break;                                               // Successfully inserted
            if (new_work_item.db->compare_transactions(old, offset)) // if same
            {
                // Found a duplicate transaction
                Transaction &old_transaction = new_work_item.db->transactions[old];

                atomicAdd(&old_transaction.utility, new_transaction.utility);
                for (int j = 0; j < idx; j++)
                {
                    atomicAdd(&old_transaction[j].util, new_transaction[j].util);
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

__device__ void trim(Task &new_work_item)
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

__device__ void push_work_item(Utilities *local_subtree_utilities,
                               Task &new_work_item,
                               Stack<Task, CAPACITY> *work_stack,
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
        new_work_item.unique_items = new_unique_item_count;
    }
    __syncthreads();

    // push the new work item to the stack
    for (int i = threadIdx.x; i < unique_item_count * scale; i += blockDim.x)
    {
        // if the subtree utility is greater than min util push it
        Task copy = new_work_item;
        if (local_subtree_utilities[i].subtree_utility >= min_util)
        {
            copy.primary = local_subtree_utilities[i].key;
            work_stack->push(copy);
        }
    }

    __syncthreads();
}

__global__ void mine(
    Stack<Task, CAPACITY> *curr_stack,
    Stack<Task, CAPACITY> *new_stack,
    patterns *p,
    uint32_t min_util)
{
    // Shared memory
    __shared__ Task curr_task;
    __shared__ Transaction *temporary_transactions;
    __shared__ uint32_t temp_transaction_count;
    __shared__ uint32_t temp_item_count;
    __shared__ int32_t *transaction_hashes;
    __shared__ Utilities *local_subtree_utilities;
    __shared__ Task new_task;

    fetch_task(curr_stack, curr_task);

    initalize_structs(&temporary_transactions,
                      &local_subtree_utilities,
                      &transaction_hashes,
                      curr_task.db->num_transactions,
                      curr_task.unique_items);

    reset_structs(&new_task,
                  curr_task,
                  temporary_transactions,
                  transaction_hashes,
                  local_subtree_utilities,
                  &temp_transaction_count,
                  &temp_item_count,
                  curr_task.db->num_transactions,
                  curr_task.unique_items);

    scan(curr_task,
         new_task,
         local_subtree_utilities,
         temporary_transactions,
         &temp_transaction_count,
         &temp_item_count);

    p->add_pattern(new_task, min_util);

    allocate(new_task, temp_item_count, temp_transaction_count);

    write_merge(temporary_transactions, temp_transaction_count, transaction_hashes, local_subtree_utilities, new_task, min_util, curr_task.db->num_transactions, curr_task.unique_items);

    trim(new_task);

    push_work_item(local_subtree_utilities, new_task, new_stack, min_util, curr_task.unique_items);
}

void static_parallel_mining(built_file &bf, patterns *p, ArgParser &parser)
{
    // Allocate memory for the stack
    Stack<Task, CAPACITY> *curr_stack;
    cudaMallocManaged(&curr_stack, sizeof(Stack<Task, CAPACITY>));
    curr_stack->init();

    // Allocate the stack
    Stack<Task, CAPACITY> *new_stack;
    cudaMallocManaged(&new_stack, sizeof(Stack<Task, CAPACITY>));
    new_stack->init();

    uint32_t *primary;
    cudaMallocManaged(&primary, sizeof(uint32_t) * bf.primary.size());
    memcpy(primary, bf.primary.data(), sizeof(uint32_t) * bf.primary.size());

    // Copy the database to the stack
    copy_work<<<1, 1024>>>(curr_stack, bf.db, primary, bf.primary.size(), bf.max_id + 1);
    cudaDeviceSynchronize();

    while (curr_stack->top != 0)
    {
// printf("Stack Size: %d\n", curr_stack->top);
#if DEBUG
        printf("Stack Size: %d\n", curr_stack->top);
#endif

        uint32_t thread_count = parser.get_uint32("--block_size");

        if (parser.get_bool("--adaptive"))
        {
            uint32_t avg = curr_stack->total_transactions / curr_stack->top;
            if (avg < thread_count)
            {
                thread_count = next_pow2(avg);
            }
            if (thread_count < 32)
            {
                thread_count = 32;
            }
        }

        mine<<<curr_stack->top, thread_count>>>(curr_stack, new_stack, p, (uint32_t)parser.get_uint32("--minutil"));

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        std::swap(curr_stack, new_stack);
        new_stack->init();
    }

    cudaFree(curr_stack);
    cudaFree(new_stack);

    cudaFree(bf.db->data);
    cudaFree(bf.db->transactions);
    cudaFree(bf.db);
    cudaFree(primary);
}