// static_parallelism.cu
#include "mine/static_parallelism.cuh"

#define TEMP_ITEM_BUFFER_SIZE 256
#define single_st 128

//------------------------------------------------------------------------------
// copy_work: seed per-primary tasks onto the stack
//------------------------------------------------------------------------------
__global__ void copy_work(Stack<Task, CAPACITY> *work_stack,
                          Database *db,
                          uint32_t *primary,
                          uint32_t primary_size,
                          uint32_t unique)
{
    const int tid = threadIdx.x;
    __shared__ Task task;

    if (tid == 0)
    {
        task.pattern = nullptr;
        task.pattern_length = 0;
        task.utility = 0;

        const uint32_t memory_needed =
            sizeof(Database) +
            sizeof(Item) * db->num_items +
            sizeof(Transaction) * db->num_transactions;

        task.db = (Database *)gmalloc(memory_needed);
        task.db->data = (Item *)((char *)task.db + sizeof(Database));
        task.db->transactions = (Transaction *)((char *)task.db->data + sizeof(Item) * db->num_items);
        task.work_done = (uint32_t *)((char *)task.db->transactions + sizeof(Transaction) * db->num_transactions);

        task.work_done[0] = 0;
        task.work_count = primary_size;

        task.db->num_items = db->num_items;
        task.db->num_transactions = db->num_transactions;

        task.memory_usage = memory_needed;
        task.unique_items = unique;
    }
    __syncthreads();

    // copy items
    for (int i = tid; i < db->num_items; i += blockDim.x)
        task.db->data[i] = db->data[i];

    // copy transaction structures (rebase by element offset)
    for (int i = tid; i < db->num_transactions; i += blockDim.x)
    {
        task.db->transactions[i].utility = 0;
        task.db->transactions[i].length = db->transactions[i].length;
        const size_t off_items = (size_t)(db->transactions[i].data - db->data);
        task.db->transactions[i].data = task.db->data + off_items;
    }

    // push each primary as a task
    Task copy = task;
    for (int i = tid; i < (int)primary_size; i += blockDim.x)
    {
        copy.primary = primary[i];
        work_stack->push(copy);
    }
    __syncthreads();
}

uint32_t next_pow2(uint32_t n)
{
    uint32_t p = 1;
    while (p < n)
        p <<= 1;
    return p;
}
static inline uint32_t round_up_to_warp_size(uint32_t n)
{
    const uint32_t warp = 32;
    return ((n + warp - 1) / warp) * warp;
}

//------------------------------------------------------------------------------
// Work fetch
//------------------------------------------------------------------------------
__device__ __forceinline__ void fetch_task(Stack<Task, CAPACITY> *work_stack, Task &item, uint32_t offset)
{
    if (threadIdx.x == 0)
        work_stack->get(item, blockIdx.x + offset);
    __syncthreads();
}

__device__ static inline uint64_t fasthash(uint64_t i)
{
    i += 1ULL;
    i ^= i >> 33ULL;
    i *= 0xff51afd7ed558ccdULL;
    i ^= i >> 33ULL;
    i *= 0xc4ceb9fe1a85ec53ULL;
    i ^= i >> 33ULL;
    return i;
}

__device__ static inline int find_item(const Utilities *buckets, int table_size, int key)
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

__device__ __forceinline__ void add_bucket_util(Utilities *buckets, uint32_t table_size,
                                                uint32_t key, uint32_t utility, bool local_utility)
{
    int idx = fasthash(key) % table_size;
    while (true)
    {
        int old = atomicCAS(&buckets[idx].key, 0, key);
        if (old == key || old == 0)
        {
            if (local_utility)
                atomicAdd(&buckets[idx].local_utility, utility);
            else
                atomicAdd(&buckets[idx].subtree_utility, utility);
            return;
        }
        idx = (idx + 1) % table_size;
    }
}

//------------------------------------------------------------------------------
// Global-memory temp structure init/reset
//------------------------------------------------------------------------------
// ---------- Initialization: allocate only what's needed up-front (utilities etc.) ----------
__device__ __forceinline__ void initalize_structs(Utilities **pUtilitiesPtr,
                                                  int32_t **pTransactionHashes,
                                                  int max_utilities)
{
    // Only one thread allocates the shared utilities area for the block
    if (threadIdx.x == 0)
    {
        const uint32_t utilities_bytes = sizeof(Utilities) * max_utilities * scale;
        *pUtilitiesPtr = (Utilities *)gmalloc(utilities_bytes);

        // transaction hashes allocation is optional; keep commented or allocate if needed
        // *pTransactionHashes = (int32_t *)gmalloc(sizeof(int32_t) * max_transactions * scale);
    }
    __syncthreads();
}

// ---------- reset: clear counters and prepare new_work_item (no big temporary array touched) ----------
__device__ __forceinline__ void reset_structs(
    Task *new_work_item,
    const Task &curr_work_item,
    Utilities *local_subtree_utilities,
    uint32_t *p_temp_transaction_count,
    uint32_t *p_temp_item_count)
{
    if (threadIdx.x == 0)
    {
        *p_temp_transaction_count = 0;
        *p_temp_item_count = 0;
        new_work_item->clear();

        if (new_work_item->pattern == nullptr)
            new_work_item->pattern = (uint32_t *)gmalloc(sizeof(uint32_t) * (curr_work_item.pattern_length + 1));
        new_work_item->pattern_length = curr_work_item.pattern_length + 1;
        new_work_item->pattern[curr_work_item.pattern_length] = curr_work_item.primary;
    }
    __syncthreads();

    // copy parent pattern in parallel
    for (int i = threadIdx.x; i < curr_work_item.pattern_length; i += blockDim.x)
        new_work_item->pattern[i] = curr_work_item.pattern[i];

    // clear utilities area in parallel
    for (int i = threadIdx.x; i < curr_work_item.unique_items * scale; i += blockDim.x)
        local_subtree_utilities[i].clear();

    __syncthreads();
}

//------------------------------------------------------------------------------
// scan: build projected transactions + local utilities
//------------------------------------------------------------------------------
__device__ void scan(const Task &curr_work_item,
                     Task &new_work_item,
                     Utilities *local_subtree_utilities,
                     Transaction **temporary_transactions_ptr, // will be allocated inside scan
                     uint32_t *temp_transaction_count,
                     uint32_t *temp_item_count)
{

    for (int i = threadIdx.x; i < curr_work_item.db->num_transactions; i += blockDim.x)
    {
        const Transaction &dbt = curr_work_item.db->transactions[i];
        int idx = dbt.binary_search(curr_work_item.primary);
        if (idx != -1)
        {
            if (dbt.size() - idx - 1 > 0)
                atomicAdd((unsigned int *)temp_transaction_count, 1u);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        // temporary_transactions_ptr
        *temporary_transactions_ptr = (Transaction *)gmalloc(sizeof(Transaction) * *temp_transaction_count);
        temp_transaction_count[0] = 0; // reset to use as allocation index
    }
    __syncthreads();

    Transaction *temporary_transactions = *temporary_transactions_ptr;

    // Phase 1: project transactions containing primary
    for (int i = threadIdx.x; i < curr_work_item.db->num_transactions; i += blockDim.x)
    {
        Transaction transaction = curr_work_item.db->transactions[i];
        int idx = transaction.binary_search(curr_work_item.primary);
        if (idx != -1)
        {
            uint32_t trans_pat_util = transaction.utility + transaction[idx].util;
            atomicAdd(&new_work_item.utility, trans_pat_util);

            uint32_t remain = transaction.size() - idx - 1;
            if (remain > 0)
            {
                int t_idx = atomicAdd(temp_transaction_count, 1);
                atomicAdd(temp_item_count, remain);

                temporary_transactions[t_idx].data = &transaction[idx + 1];
                temporary_transactions[t_idx].utility = trans_pat_util;
                temporary_transactions[t_idx].length = remain;
            }
        }
    }
    __syncthreads();

    // Phase 2: per-item local utilities
    for (int i = threadIdx.x; i < *temp_transaction_count; i += blockDim.x)
    {
        Transaction &t = temporary_transactions[i];
        uint32_t total_util = t.utility + t.util_sum();
        for (int j = 0; j < t.length; j++)
        {
            add_bucket_util(local_subtree_utilities,
                            curr_work_item.unique_items * scale,
                            t[j].key, total_util, true);
        }
    }
    __syncthreads();
}



//------------------------------------------------------------------------------
// allocate (global) projected DB
//------------------------------------------------------------------------------
__device__ void allocate(Task &wi, uint32_t temp_item_count, uint32_t temp_transaction_count)
{
    if (threadIdx.x == 0)
    {
        const uint32_t memory_needed =
            sizeof(Database) +
            sizeof(Item) * temp_item_count +
            sizeof(Transaction) * temp_transaction_count;

        wi.db = (Database *)gmalloc(memory_needed);
        wi.db->data = (Item *)((char *)wi.db + sizeof(Database));
        wi.db->transactions = (Transaction *)((char *)wi.db->data + sizeof(Item) * temp_item_count);
        wi.memory_usage = memory_needed;

        // atomicMax(&max_task_memory, memory_needed);
    }
    __syncthreads();
}

//------------------------------------------------------------------------------
// write_merge: compact, update subtree utils, de-dup projected transactions
//------------------------------------------------------------------------------
__device__ void write_merge(Transaction *temporary_transactions,
                            uint32_t temp_transaction_count,
                            int32_t *transaction_hashes,
                            Utilities *local_subtree_utilities,
                            Task &new_work_item,
                            uint32_t min_util,
                            uint32_t transaction_count,
                            uint32_t max_item)
{
    // const int TEMP_ITEM_BUFFER_SIZE = 256;
    Item temp_item_buffer[TEMP_ITEM_BUFFER_SIZE];

    for (int i = threadIdx.x; i < temp_transaction_count; i += blockDim.x)
    {
        Transaction &t = temporary_transactions[i];

        // 1) filter into private buffer
        int idx = 0;
        int cumsum = t.utility;
        for (int j = 0; j < t.length; ++j)
        {
            Item it = t.data[j];
            int found = find_item(local_subtree_utilities, max_item * scale, it.key);
            if (found != -1 && local_subtree_utilities[found].local_utility >= min_util)
            {
                if (idx < TEMP_ITEM_BUFFER_SIZE)
                {
                    temp_item_buffer[idx++] = it;
                    cumsum += it.util;
                }
            }
        }
        if (idx == 0)
            continue;

        // 2) allocate exact space, copy
        int t_off = atomicAdd(&new_work_item.db->num_transactions, 1);
        int i_off = atomicAdd(&new_work_item.db->num_items, idx);

        Transaction &nt = new_work_item.db->transactions[t_off];
        nt.data = &new_work_item.db->data[i_off];
        nt.utility = t.utility;
        nt.length = idx;

        for (int j = 0; j < idx; ++j)
            nt[j] = temp_item_buffer[j];

        // 3) subtree utils + merge duplicates
        uint64_t hash = 0;
        int temp = 0;
        for (int j = 0; j < idx; ++j)
        {
            Item it = nt[j];
            add_bucket_util(local_subtree_utilities,
                            max_item * scale,
                            it.key, cumsum - temp, false);

            hash ^= fasthash(it.key + j);
            temp += it.util;
        }

        uint64_t h = hash % (temp_transaction_count * scale);
        while (true)
        {
            int old = atomicCAS(&transaction_hashes[h], -1, t_off);
            if (old == -1)
                break;

            if (new_work_item.db->compare_transactions(old, t_off))
            {
                Transaction &ot = new_work_item.db->transactions[old];
                atomicAdd(&ot.utility, nt.utility);
                for (int j = 0; j < idx; j++)
                    atomicAdd(&ot[j].util, nt[j].util);

                new_work_item.db->transactions[t_off].data = nullptr;
                break;
            }
            h = (h + 1) % (temp_transaction_count * scale);
        }
    }
    __syncthreads();
}

//------------------------------------------------------------------------------
// trim: remove duplicates (null data) and compact transactions
//------------------------------------------------------------------------------
__device__ void trim(Task &new_work_item)
{
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    __shared__ Transaction tile[1024];
    __shared__ int scan_storage[1024];
    __shared__ uint32_t final_count;

    if (tid == 0)
        final_count = 0;
    __syncthreads();

    for (int i = 0; i < new_work_item.db->num_transactions; i += block_size)
    {
        const int read_idx = i + tid;
        bool is_valid = false;

        if (read_idx < new_work_item.db->num_transactions)
        {
            tile[tid] = new_work_item.db->transactions[read_idx];
            is_valid = (tile[tid].data != nullptr);
        }

        scan_storage[tid] = is_valid ? 1 : 0;
        __syncthreads();

        // inclusive scan
        for (int offset = 1; offset < block_size; offset <<= 1)
        {
            int temp = (tid >= offset) ? scan_storage[tid - offset] : 0;
            __syncthreads();
            scan_storage[tid] += temp;
            __syncthreads();
        }

        __shared__ int valid_in_tile;
        if (tid == block_size - 1)
            valid_in_tile = scan_storage[tid];
        __syncthreads();

        __shared__ int write_base;
        if (tid == 0 && valid_in_tile > 0)
            write_base = atomicAdd(&final_count, valid_in_tile);
        __syncthreads();

        if (is_valid)
        {
            const int local_off = scan_storage[tid] - 1;
            new_work_item.db->transactions[write_base + local_off] = tile[tid];
        }
        __syncthreads();
    }

    if (tid == 0)
        new_work_item.db->num_transactions = final_count;

    __syncthreads();
}

//------------------------------------------------------------------------------
// push_work_item: push new tasks, fill histogram
//------------------------------------------------------------------------------
__device__ void push_work_item(Utilities *local_subtree_utilities,
                               Task &new_work_item,
                               Stack<Task, CAPACITY> *work_stack,
                               uint32_t min_util, uint32_t unique_item_count,
                               uint32_t *transaction_distribution, bool adaptive = false, Stack<Task, CAPACITY> *one_stack = nullptr)
{
    __shared__ uint32_t new_unique_item_count;
    if (threadIdx.x == 0)
        new_unique_item_count = 0;
    __syncthreads();

    for (int i = threadIdx.x; i < unique_item_count * scale; i += blockDim.x)
        if (local_subtree_utilities[i].local_utility >= min_util)
            atomicAdd(&new_unique_item_count, 1);
    __syncthreads();

    if (threadIdx.x == 0)
        new_work_item.unique_items = new_unique_item_count;
    __syncthreads();

    for (int i = threadIdx.x; i < unique_item_count * scale; i += blockDim.x)
    {
        Task copy = new_work_item;
        if (local_subtree_utilities[i].subtree_utility >= min_util)
        {
            copy.primary = local_subtree_utilities[i].key;

            if (!adaptive)
            {
                work_stack->push(copy);
                continue;
            }
            if (copy.db->num_transactions <= 16)
            {
                // uint32_t index = 0;
                // atomicAdd(&transaction_distribution[index], 1);
                one_stack->push(copy);
            }
            else
            {
                // uint32_t index = (copy.db->num_transactions + 31) / 32;
                // if (index > 33)
                //     index = 33;
                // atomicAdd(&transaction_distribution[index], 1);
                if (copy.db->num_transactions > 1024)
                    atomicAdd(&transaction_distribution[7], 1); // Overflow bucket
                else
                {
                    int idx = 0;
                    int val = 32;
                    while (val < copy.db->num_transactions)
                    {
                        val <<= 1;
                        idx++;
                    }
                    atomicAdd(&transaction_distribution[idx], 1);
                }

                work_stack->push(copy);
            }
        }
    }
    __syncthreads();
}

//------------------------------------------------------------------------------
// Global-memory mining kernel
//------------------------------------------------------------------------------
__global__ void mine(Stack<Task, CAPACITY> *curr_stack,
                     Stack<Task, CAPACITY> *new_stack,
                     patterns *p,
                     uint32_t min_util,
                     uint32_t *transaction_distribution, uint32_t offset = 0, bool adaptive = false, Stack<Task, CAPACITY> *one_stack = nullptr)
{
    __shared__ Task curr_task;
    __shared__ Transaction *temporary_transactions;
    __shared__ uint32_t temp_transaction_count;
    __shared__ uint32_t temp_item_count;
    __shared__ int32_t *transaction_hashes;
    __shared__ Utilities *local_subtree_utilities;
    __shared__ Task new_task;

    fetch_task(curr_stack, curr_task, offset);

    initalize_structs(&local_subtree_utilities,
                      &transaction_hashes,
                      curr_task.unique_items);

    reset_structs(&new_task,
                  curr_task,
                  local_subtree_utilities,
                  &temp_transaction_count,
                  &temp_item_count);

    scan(curr_task,
         new_task,
         local_subtree_utilities,
         &temporary_transactions,
         &temp_transaction_count,
         &temp_item_count);

    // initalize_structs(&temporary_transactions,
    //                   &local_subtree_utilities,
    //                   &transaction_hashes,
    //                   curr_task.db->num_transactions,
    //                   curr_task.unique_items);

    // reset_structs(&new_task,
    //               curr_task,
    //               temporary_transactions,
    //               transaction_hashes,
    //               local_subtree_utilities,
    //               &temp_transaction_count,
    //               &temp_item_count,
    //               curr_task.db->num_transactions,
    //               curr_task.unique_items);

    // scan(curr_task,
    //      new_task,
    //      local_subtree_utilities,
    //      temporary_transactions,
    //      &temp_transaction_count,
    //      &temp_item_count);

    p->add_pattern(new_task, min_util);
    __syncthreads();

    if (temp_transaction_count == 0)
    {
        return;
    }

    if (threadIdx.x == 0)
    {
        // allocate hashes
        transaction_hashes = (int32_t *)gmalloc(sizeof(int32_t) * temp_transaction_count * scale);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < temp_transaction_count * scale; i += blockDim.x)
        transaction_hashes[i] = -1;
    __syncthreads();

    allocate(new_task, temp_item_count, temp_transaction_count);

    write_merge(temporary_transactions, temp_transaction_count,
                transaction_hashes, local_subtree_utilities,
                new_task, min_util,
                curr_task.db->num_transactions,
                curr_task.unique_items);

    trim(new_task);

    push_work_item(local_subtree_utilities, new_task, new_stack,
                   min_util, curr_task.unique_items, transaction_distribution, adaptive, one_stack);
}

// Simplified version for single-thread access, no atomics needed.
__device__ __forceinline__ void add_bucket_util_st(Utilities *buckets, uint32_t table_size,
                                                   uint32_t key, uint32_t utility, bool local_utility)
{
    int idx = fasthash(key) % table_size;
    while (true)
    {
        // In a single thread, atomicCAS is not needed. We check if the bucket is empty or matches the key.
        if (buckets[idx].key == 0)
            buckets[idx].key = key;

        if (buckets[idx].key == key)
        {
            if (local_utility)
                buckets[idx].local_utility += utility;
            else
                buckets[idx].subtree_utility += utility;
            return;
        }
        idx = (idx + 1) % table_size;
    }
}

//------------------------------------------------------------------------------
// New Mining Kernel (1 Thread per Task)
//------------------------------------------------------------------------------

__device__ void printDatabase(const Task &task)
{
    printf("Pattern: ");
    for (int i = 0; i < task.pattern_length; ++i)
        printf("%d ", task.pattern[i]);
    printf(" Primary: %d Utility: %d\n", task.primary, task.utility);

    for (int i = 0; i < task.db->num_transactions; ++i)
    {
        Transaction t = task.db->transactions[i];
        printf(" %d:%d| ", i, t.utility);
        for (int j = 0; j < t.length; ++j)
            printf("(%d,%d) ", t[j].key, t[j].util);
        printf("\n");
    }
    printf("\n");
}

__device__ void mine_recursive(Task &curr_task,
                               patterns *p,
                               uint32_t min_util,
                               Stack<Task, CAPACITY> *new_stack)
{
    // Cache reused values locally to reduce register pressure and global memory reads
    const Database *curr_db = curr_task.db;
    const uint32_t curr_unique = curr_task.unique_items;
    const uint32_t ui_scale = curr_unique * scale;
    const uint32_t curr_num_trans = curr_db->num_transactions;
    const uint32_t primary_key = curr_task.primary;

    // Small fixed buffer on stack (guaranteed <=16)
    // Transaction temporary_transactions[16];

    __shared__ Transaction all_shared[16 * single_st]; // 16 transactions per thread, single_st threads
    Transaction *temporary_transactions = &all_shared[threadIdx.x * 16];

// Fast initialization: set lengths/utilities to 0
#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        temporary_transactions[i].length = 0;
        temporary_transactions[i].utility = 0;
        temporary_transactions[i].data = nullptr;
    }

    Utilities local_subtree_utilities[TEMP_ITEM_BUFFER_SIZE * scale]; // assuming max 1024*scale items

// Initialize to zero without memset
#pragma unroll
    for (uint32_t i = 0; i < ui_scale; ++i)
    {
        local_subtree_utilities[i].key = 0;
        local_subtree_utilities[i].local_utility = 0;
        local_subtree_utilities[i].subtree_utility = 0;
    }

    // Prepare new task
    Task new_task;
    new_task.clear();
    new_task.pattern_length = curr_task.pattern_length + 1;
    new_task.pattern = (uint32_t *)gmalloc(sizeof(uint32_t) * new_task.pattern_length);

// Copy pattern
#pragma unroll
    for (uint32_t i = 0; i < curr_task.pattern_length; ++i)
        new_task.pattern[i] = curr_task.pattern[i];
    new_task.pattern[curr_task.pattern_length] = primary_key;

    new_task.utility = 0;
    new_task.unique_items = 0;
    new_task.memory_usage = 0;

    uint32_t temp_transaction_count = 0;
    uint32_t temp_item_count = 0;

    const Transaction *in_trans = curr_db->transactions;

    // 1. Project transactions containing the primary item
    for (uint32_t i = 0; i < curr_num_trans; ++i)
    {
        const Transaction &t = in_trans[i];
        int idx = t.binary_search(primary_key);
        if (idx == -1)
            continue;

        const uint32_t trans_pat_util = t.utility + t[idx].util;
        new_task.utility += trans_pat_util;

        const int remain = t.size() - idx - 1;
        if (remain > 0 && temp_transaction_count < 16)
        {
            Transaction &dst = temporary_transactions[temp_transaction_count++];
            dst.data = &t.data[idx + 1];
            dst.utility = trans_pat_util;
            dst.length = remain;
            temp_item_count += remain;
        }
    }

    // Add the pattern before deeper exploration
    p->add_pattern_st(new_task, min_util);

    if (temp_transaction_count == 0)
        return;

    // 2. First pass: compute local subtree utilities
    for (uint32_t i = 0; i < temp_transaction_count; ++i)
    {
        Transaction &t = temporary_transactions[i];
        const uint32_t total_util = t.utility + t.util_sum();
        for (uint32_t j = 0; j < t.length; ++j)
        {
            const Item &it = t[j];
            add_bucket_util_st(local_subtree_utilities, ui_scale, it.key, total_util, true);
        }
    }

    // 3. Allocate projected database
    const uint32_t memory_needed_db =
        sizeof(Database) +
        sizeof(Item) * temp_item_count +
        sizeof(Transaction) * temp_transaction_count;

    new_task.db = (Database *)gmalloc(memory_needed_db);
    new_task.db->data = (Item *)((char *)new_task.db + sizeof(Database));
    new_task.db->transactions =
        (Transaction *)((char *)new_task.db->data + sizeof(Item) * temp_item_count);
    new_task.db->num_transactions = 0;
    new_task.db->num_items = 0;
    new_task.memory_usage = memory_needed_db;

    // Local small buffer
    // const int TEMP_ITEM_BUFFER_SIZE = 256;
    Item temp_item_buffer[TEMP_ITEM_BUFFER_SIZE];

    // 4. Build filtered projected database
    for (uint32_t i = 0; i < temp_transaction_count; ++i)
    {
        Transaction &t = temporary_transactions[i];

        int idx_buf = 0;
        uint32_t cumsum = t.utility;

        for (uint32_t j = 0; j < t.length; ++j)
        {
            const Item &it = t.data[j];
            int found = find_item(local_subtree_utilities, ui_scale, it.key);
            if (found != -1 && local_subtree_utilities[found].local_utility >= min_util)
            {
                if (idx_buf < TEMP_ITEM_BUFFER_SIZE)
                {
                    temp_item_buffer[idx_buf++] = it;
                    cumsum += it.util;
                }
            }
        }

        if (idx_buf == 0)
            continue;

        int t_off = new_task.db->num_transactions;
        int i_off = new_task.db->num_items;
        new_task.db->num_transactions += 1;
        new_task.db->num_items += idx_buf;

        Transaction &nt = new_task.db->transactions[t_off];
        nt.data = &new_task.db->data[i_off];
        nt.utility = t.utility;
        nt.length = idx_buf;

        uint32_t temp_util = 0;
        for (int j = 0; j < idx_buf; ++j)
        {
            nt[j] = temp_item_buffer[j];
            const Item &it = nt[j];
            add_bucket_util_st(local_subtree_utilities, ui_scale, it.key, cumsum - temp_util, false);
            temp_util += it.util;
        }
    }

    // 5. Count items meeting local utility threshold
    for (uint32_t i = 0; i < ui_scale; ++i)
    {
        if (local_subtree_utilities[i].local_utility >= min_util)
            new_task.unique_items++;
    }

    // 6. Push new recursive tasks
    for (uint32_t i = 0; i < ui_scale; ++i)
    {
        if (local_subtree_utilities[i].subtree_utility >= min_util)
        {
            new_task.primary = local_subtree_utilities[i].key;
            new_stack->push(new_task);
        }
    }
}

__global__ void mine_st(Stack<Task, CAPACITY> *curr_stack,
                        Stack<Task, CAPACITY> *new_stack,
                        patterns *p,
                        uint32_t min_util, uint32_t offset = 0)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid + offset >= curr_stack->top)
        return;

    Task curr_task;

    // Fetch the task for this thread
    curr_stack->get(curr_task, tid + offset);

    mine_recursive(curr_task, p, min_util, new_stack);
}

void run_static_mining(built_file &bf,
                       uint32_t *primary,
                       patterns *p,
                       ArgParser &parser,
                       Timer &timer)
{
    // --- Setup for Static Mining ---
    Stack<Task, CAPACITY> *curr_stack, *new_stack;
    gpuErrchk(cudaMallocManaged(&curr_stack, sizeof(*curr_stack)));
    gpuErrchk(cudaMallocManaged(&new_stack, sizeof(*new_stack)));
    curr_stack->init();
    new_stack->init();

    // Seed initial work from primary patterns
    copy_work<<<1, 1024>>>(curr_stack, bf.db, primary, bf.primary.size(), bf.max_id + 1);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // --- Main Mining Loop ---
    int iter = 1;
    const uint32_t thread_count = parser.get_uint32("--block_size"); // Fixed block size

    uint64_t initial_offset = allocators::atomic::allocator_offset;
    uint64_t total_boundary = allocators::atomic::allocator_total;
    uint64_t safety_boundary = 0;
    uint64_t current_boundary = 0;

    while (curr_stack->top != 0)
    {

        // const unsigned long long before_offset = allocators::atomic::allocator_offset;
        safety_boundary = allocators::atomic::allocator_offset;

        // Launch the kernel directly using one block per task
        mine<<<curr_stack->top, thread_count>>>(
            curr_stack, new_stack, p,
            (uint32_t)parser.get_uint32("--minutil"),
            nullptr, // transaction_distribution is not used in static mode
            0);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        current_boundary = allocators::atomic::allocator_offset;
        timer.recordPoint("Iteration " + std::to_string(iter), safety_boundary, current_boundary);

        // Prepare for the next iteration
        std::swap(curr_stack, new_stack);
        new_stack->init();

        uint64_t head = total_boundary - current_boundary;
        uint64_t tail = safety_boundary - initial_offset;
        std::cout << "H: " << head / (1024 * 1024) << " T: " << tail / (1024 * 1024) << std::endl;
        if (head < tail)
        {
            std::cout << "Reset Mem: " << initial_offset / (1024 * 1024) << " to " << safety_boundary / (1024 * 1024) << std::endl;
            allocators::atomic::allocator_offset = initial_offset;
            cudaMemset(allocators::atomic::allocator_base + initial_offset, 0, tail);
        }

        iter++;
    }

    // --- Cleanup for Static Mining ---
    cudaFree(curr_stack);
    cudaFree(new_stack);
}



void run_adaptive_mining(built_file &bf,
                         uint32_t *primary,
                         patterns *p,
                         ArgParser &parser,
                         Timer &timer)
{
    
    // --- Setup for Adaptive Mining ---
    Stack<Task, CAPACITY> *curr_stack, *new_stack, *one_stack, *one_stack_next;
    gpuErrchk(cudaMallocManaged(&curr_stack, sizeof(*curr_stack)));
    gpuErrchk(cudaMallocManaged(&new_stack, sizeof(*new_stack)));
    gpuErrchk(cudaMallocManaged(&one_stack, sizeof(*one_stack)));
    gpuErrchk(cudaMallocManaged(&one_stack_next, sizeof(*one_stack_next)));
    curr_stack->init();
    new_stack->init();
    one_stack->init();
    one_stack_next->init();

    // Seed initial work from primary patterns
    copy_work<<<1, 1024>>>(curr_stack, bf.db, primary, bf.primary.size(), bf.max_id + 1);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Histogram for adaptive scheduling
    uint32_t *transaction_distribution;
    const int dist_size = 8; // 1, 32, 64, 128, 256, 512, 1024, >1024
    gpuErrchk(cudaMallocManaged(&transaction_distribution, sizeof(uint32_t) * dist_size));
    memset(transaction_distribution, 0, sizeof(uint32_t) * dist_size);

    transaction_distribution[7] = bf.primary.size(); // Seed with initial patterns to avoid zero-thread launch

    // --- Main Mining Loop for Multi-Threaded Tasks ---
    int iter = 1;
    uint32_t thread_count = parser.get_uint32("--block_size"); // Initial block size

    uint64_t initial_offset = allocators::atomic::allocator_offset;
    uint64_t safety_boundary = 0;
    uint64_t current_boundary = 0;

    std::vector<uint32_t> dist_copy(transaction_distribution, transaction_distribution + dist_size);

    while (curr_stack->top != 0 || one_stack->top != 0)
    {
        safety_boundary = allocators::atomic::allocator_offset;

        // copy distribution for logging
        memcpy(dist_copy.data(), transaction_distribution, sizeof(uint32_t) * dist_size);

        // mine blah blah blah
        if (curr_stack->top != 0)
        {

            // --- Adaptive Logic: Calculate optimal block size for this iteration ---
            unsigned long long weighted_sum = 0;
            uint32_t tasks_in_dist = 0;

            for (int i = 1; i < dist_size; ++i)
            {
                const uint32_t num_tasks = transaction_distribution[i];
                if (num_tasks > 0)
                {
                    const uint32_t rep = (1 << (i + 4)); // 32, 64, 128, ..., 2048
                    weighted_sum += (unsigned long long)rep * num_tasks;
                    tasks_in_dist += num_tasks;
                }
            }

            if (tasks_in_dist > 0)
            {
                const uint32_t weighted_avg = (uint32_t)(weighted_sum / tasks_in_dist);
                thread_count = round_up_to_warp_size(weighted_avg);
            }
            else if (curr_stack->top > 0) // Fallback for the first iteration
            {
                const uint32_t avg = curr_stack->total_transactions / curr_stack->top;
                if (avg > 0)
                    thread_count = next_pow2(avg);
            }

            // Clamp thread_count to valid CUDA limits
            thread_count = std::max(32u, std::min(1024u, thread_count));

            // Reset distribution for the next kernel launch to populate it
            memset(transaction_distribution, 0, sizeof(uint32_t) * dist_size);
            // The 'mine' kernel now has added logic to offload single-thread tasks to 'one_stack'
            mine<<<curr_stack->top, thread_count>>>(
                curr_stack, new_stack, p,
                (uint32_t)parser.get_uint32("--minutil"),
                transaction_distribution, 0, true, one_stack);

            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

            // Prepare for the next iteration
            std::swap(curr_stack, new_stack);
            new_stack->init();

            if (curr_stack->top == 0)
            {
                curr_stack->de_init();
                new_stack->de_init();
            }

        }

        transaction_distribution[0] = one_stack->top;

        if (one_stack->top != 0)
        {
            mine_st<<<(one_stack->top + single_st - 1) / single_st, single_st>>>(
                one_stack, one_stack_next, p,
                (uint32_t)parser.get_uint32("--minutil"), 0);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
        }

        std::swap(one_stack, one_stack_next);
        one_stack_next->init();

        current_boundary = allocators::atomic::allocator_offset;
        timer.recordPoint("Iteration " + std::to_string(iter), safety_boundary, current_boundary, dist_copy);

        // so from safety to current we will need it. but initial to safety we can clear it but we only set it if the head is less than tail
        // head being the current boundary to the total
        // tail being previous to safety
        uint64_t head = allocators::atomic::allocator_total - current_boundary;
        uint64_t tail = safety_boundary - initial_offset;
        // std::cout << "H: " << head / (1024 * 1024) << " T: " << tail / (1024 * 1024) << std::endl;
        if (head < tail)
        {
            // std::cout << "Reset Mem: " << initial_offset / (1024 * 1024) << " to " << safety_boundary / (1024 * 1024) << std::endl;
            allocators::atomic::allocator_offset = initial_offset;
            cudaMemset(allocators::atomic::allocator_base + initial_offset, 0, tail);
        }

        iter++;
    }

    // --- Cleanup for Adaptive Mining ---
    cudaFree(transaction_distribution);
    cudaFree(curr_stack);
    cudaFree(new_stack);
    cudaFree(one_stack); // Free the additional stack
    cudaFree(one_stack_next);
}

void static_parallel_mining(built_file &bf,
                            patterns *p,
                            ArgParser &parser,
                            Timer &timer,
                            size_t available_shared_memory)
{
    // Allocate and copy primary patterns to the device
    uint32_t *primary;
    gpuErrchk(cudaMallocManaged(&primary, sizeof(uint32_t) * bf.primary.size()));
    memcpy(primary, bf.primary.data(), sizeof(uint32_t) * bf.primary.size());

    // --- Dispatch to the correct mining strategy ---
    if (parser.get_bool("--adaptive"))
    {
        printf("Running Adaptive Mining...\n");
        run_adaptive_mining(bf, primary, p, parser, timer);
    }
    else
    {
        printf("Running Static Mining...\n");
        run_static_mining(bf, primary, p, parser, timer);
    }

    cudaFree(bf.db->data);
    cudaFree(bf.db->transactions);
    cudaFree(bf.db);
    cudaFree(primary);
}