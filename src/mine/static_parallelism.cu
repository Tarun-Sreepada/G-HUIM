// static_parallelism.cu 
#include "mine/static_parallelism.cuh"

__managed__ uint32_t max_task_memory;
const size_t SHARED_MEM_HEURISTIC_FACTOR = 3; // input DB + output DB + temp

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
            sizeof(Transaction) * db->num_transactions +
            sizeof(uint32_t); // work_done

        task.db = (Database *)gmalloc(memory_needed);
        task.db->data = (Item *)((char *)task.db + sizeof(Database));
        task.db->transactions = (Transaction *)((char *)task.db->data + sizeof(Item) * db->num_items);
        task.work_done = (uint32_t *)((char *)task.db->transactions + sizeof(Transaction) * db->num_transactions);

        task.work_done[0] = 0;
        task.work_count = primary_size;

        task.db->num_items = db->num_items;
        task.db->num_transactions = db->num_transactions;

        task.memory_usage = memory_needed;
        max_task_memory = task.memory_usage;
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

//------------------------------------------------------------------------------
// Work fetch
//------------------------------------------------------------------------------
__device__ __forceinline__ void fetch_task(Stack<Task, CAPACITY> *work_stack, Task &item)
{
    if (threadIdx.x == 0)
        work_stack->get(item, blockIdx.x);
    __syncthreads();
}

//------------------------------------------------------------------------------
// Global-memory temp structure init/reset
//------------------------------------------------------------------------------
__device__ __forceinline__ void initalize_structs(Transaction **pTemporaryPtr,
                                                  Utilities **pUtilitiesPtr,
                                                  int32_t **pTransactionHashes,
                                                  int max_transactions,
                                                  int max_utilities)
{
    if (threadIdx.x == 0)
    {
        const uint32_t memory_needed =
            sizeof(Transaction) * max_transactions +
            sizeof(Utilities) * max_utilities * scale +
            sizeof(int32_t) * max_transactions * scale;

        *pTemporaryPtr = (Transaction *)gmalloc(memory_needed);
        *pUtilitiesPtr = (Utilities *)((char *)(*pTemporaryPtr) + sizeof(Transaction) * max_transactions);
        *pTransactionHashes = (int32_t *)((char *)(*pUtilitiesPtr) + sizeof(Utilities) * max_utilities * scale);
    }
    __syncthreads();
}

__device__ __forceinline__ void reset_structs(
    Task *new_work_item,
    const Task &curr_work_item,
    Transaction *temporary_transactions,
    int32_t *transaction_hashes,
    Utilities *local_subtree_utilities,
    uint32_t *p_temp_transaction_count,
    uint32_t *p_temp_item_count,
    int max_transactions,
    int max_utilities)
{
    if (threadIdx.x == 0)
    {
        *p_temp_transaction_count = 0;
        *p_temp_item_count = 0;
        new_work_item->clear();
        new_work_item->utility = 0;

        if (new_work_item->pattern == nullptr)
            new_work_item->pattern = (uint32_t *)gmalloc(sizeof(uint32_t) * (curr_work_item.pattern_length + 1));

        new_work_item->pattern_length = curr_work_item.pattern_length + 1;
        new_work_item->pattern[curr_work_item.pattern_length] = curr_work_item.primary;
    }
    __syncthreads();

    // Copy parent pattern
    for (int i = threadIdx.x; i < curr_work_item.pattern_length; i += blockDim.x)
        new_work_item->pattern[i] = curr_work_item.pattern[i];

    // Clear temps
    for (int i = threadIdx.x; i < max_transactions; i += blockDim.x)
        temporary_transactions[i].clear();

    for (int i = threadIdx.x; i < max_transactions * scale; i += blockDim.x)
        transaction_hashes[i] = -1;

    for (int i = threadIdx.x; i < max_utilities * scale; i += blockDim.x)
        local_subtree_utilities[i].clear();

    __syncthreads();
}

//------------------------------------------------------------------------------
// Hash helpers
//------------------------------------------------------------------------------
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
// scan: build projected transactions + local utilities
//------------------------------------------------------------------------------
__device__ void scan(const Task &curr_work_item,
                     Task &new_work_item,
                     Utilities *local_subtree_utilities,
                     Transaction *temporary_transactions,
                     uint32_t *temp_transaction_count,
                     uint32_t *temp_item_count)
{
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

        atomicMax(&max_task_memory, memory_needed);
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
    const int TEMP_ITEM_BUFFER_SIZE = 256;
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

            // temp_item_buffer[idx] = Item{0, 0}; // sentinel

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

        uint64_t h = hash % (transaction_count * scale);
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
            h = (h + 1) % (transaction_count * scale);
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
                               uint32_t *transaction_distribution)
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

    if (threadIdx.x == 0 && new_unique_item_count > 0 && new_work_item.db->num_transactions > 0)
    {
        uint32_t index = (new_work_item.db->num_transactions + 31) / 32;
        if (index > 32)
            index = 32;
        atomicAdd(&transaction_distribution[index], new_unique_item_count);
    }

    for (int i = threadIdx.x; i < unique_item_count * scale; i += blockDim.x)
    {
        Task copy = new_work_item;
        if (local_subtree_utilities[i].subtree_utility >= min_util)
        {
            copy.primary = local_subtree_utilities[i].key;
            work_stack->push(copy);
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
                     uint32_t *transaction_distribution)
{
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

    write_merge(temporary_transactions, temp_transaction_count,
                transaction_hashes, local_subtree_utilities,
                new_task, min_util,
                curr_task.db->num_transactions,
                curr_task.unique_items);

    trim(new_task);

    push_work_item(local_subtree_utilities, new_task, new_stack,
                   min_util, curr_task.unique_items, transaction_distribution);
}

//------------------------------------------------------------------------------
// Shared-memory scaffolding
//------------------------------------------------------------------------------
struct SmemAllocator
{

    char *buf;
    size_t offset;

    __device__ __forceinline__ void init(char *smem_buffer)
    {
        if (threadIdx.x == 0)
        {
            // *smem_base_ptr   = smem_buffer;
            // *smem_offset_ptr = 0;
            buf = smem_buffer;
            offset = 0;
        }
        // __syncthreads();
    }

    __device__ __forceinline__ void *alloc(size_t size)
    {
        const size_t aligned = (size + 7) & ~size_t(7);
        void *ptr = nullptr;
        if (threadIdx.x == 0)
        {
            // ptr = (void*)(smem_base + *smem_offset_ptr);
            ptr = (void *)(buf + offset);
            // *smem_offset_ptr += aligned;
            offset += aligned;
        }
        return ptr;
    }
};

__device__ __forceinline__ void smem_copy_in_task_db(const Task &g_task,
                                                     Task &s_task,
                                                     SmemAllocator &allocator)
{

    if (threadIdx.x == 0)
    {
        s_task = g_task; // copy scalars
        s_task.db = (Database *)allocator.alloc(g_task.memory_usage);
    }
    __syncthreads();


    // Parallel memcpy
    {
        char *dst = (char *)s_task.db;
        char *src = (char *)g_task.db;
        size_t n = g_task.memory_usage;
        for (size_t i = threadIdx.x; i < n; i += blockDim.x)
            dst[i] = src[i];
    }
    __syncthreads();

    // Rebase internal pointers using ITEM index offsets
    if (threadIdx.x == 0)
    {
        s_task.db->data = (Item *)((char *)s_task.db + sizeof(Database));
        s_task.db->transactions = (Transaction *)((char *)s_task.db->data + sizeof(Item) * s_task.db->num_items);

        for (int i = 0; i < s_task.db->num_transactions; ++i)
        {
            size_t item_idx_off = (size_t)(g_task.db->transactions[i].data - g_task.db->data);
            s_task.db->transactions[i].data = s_task.db->data + item_idx_off;
        }
    }
    __syncthreads();
}


__device__ __forceinline__ void smem_init_temporaries(Transaction **ppTemp,
                                                      Utilities **ppUtils,
                                                      int32_t **ppHashes,
                                                      int max_transactions,
                                                      int max_utilities,
                                                      SmemAllocator &alloc)
{
    if (threadIdx.x == 0)
    {
        const uint32_t bytes =
            sizeof(Transaction) * max_transactions + sizeof(Utilities) * max_utilities * scale + sizeof(int32_t) * max_transactions * scale;

        *ppTemp = (Transaction *)alloc.alloc(bytes);
        *ppUtils = (Utilities *)((char *)(*ppTemp) + sizeof(Transaction) * max_transactions);
        *ppHashes = (int32_t *)((char *)(*ppUtils) + sizeof(Utilities) * max_utilities * scale);
    }
    __syncthreads();
}

__device__ __forceinline__ void smem_reset_for_candidate(Task *pNew,
                                                         const Task &curr,
                                                         Transaction *temporary_transactions,
                                                         int32_t *transaction_hashes,
                                                         Utilities *local_subtree_utilities,
                                                         uint32_t *p_temp_transaction_count,
                                                         uint32_t *p_temp_item_count,
                                                         int max_transactions,
                                                         int max_utilities)
{
    if (threadIdx.x == 0)
    {
        *p_temp_transaction_count = 0;
        *p_temp_item_count = 0;
        pNew->clear();
        pNew->utility = 0;

        // IMPORTANT: keep pattern in GLOBAL memory for safety
        if (pNew->pattern == nullptr)
            pNew->pattern = (uint32_t *)gmalloc(sizeof(uint32_t) * (curr.pattern_length + 1));

        pNew->pattern_length = curr.pattern_length + 1;
        pNew->pattern[curr.pattern_length] = curr.primary;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < curr.pattern_length; i += blockDim.x)
        pNew->pattern[i] = curr.pattern[i];

    for (int i = threadIdx.x; i < max_transactions; i += blockDim.x)
        temporary_transactions[i].clear();

    for (int i = threadIdx.x; i < max_transactions * scale; i += blockDim.x)
        transaction_hashes[i] = -1;

    for (int i = threadIdx.x; i < max_utilities * scale; i += blockDim.x)
        local_subtree_utilities[i].clear();
    __syncthreads();
}

// Shared memory version of allocate
__device__ void allocate_smem(Task &wi, uint32_t temp_item_count, uint32_t temp_transaction_count,
                              SmemAllocator &allocator)
{
    if (threadIdx.x == 0)
    {
        uint32_t memory_needed = sizeof(Database) + sizeof(Item) * temp_item_count + sizeof(Transaction) * temp_transaction_count;
        wi.db = (Database *)allocator.alloc(memory_needed);
        wi.db->data = (Item *)((char *)wi.db + sizeof(Database));
        wi.db->transactions = (Transaction *)((char *)wi.db->data + sizeof(Item) * temp_item_count);
        wi.memory_usage = memory_needed;
    }
    __syncthreads();
}

// `scan` logic, explicitly for shared memory inputs and outputs.
__device__ void scan_smem(const Task &curr_work_item, // .db is in smem
                          Task &new_work_item,         // struct is in smem
                          Utilities *local_subtree_utilities, // in smem
                          Transaction *temporary_transactions, // in smem
                          uint32_t *temp_transaction_count, // in smem
                          uint32_t *temp_item_count)       // in smem
{
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

// `write_merge` logic, explicitly for shared memory.
__device__ void write_merge_smem(Transaction *temporary_transactions, // in smem
                                 uint32_t temp_transaction_count,
                                 int32_t *transaction_hashes,      // in smem
                                 Utilities *local_subtree_utilities, // in smem
                                 Task &new_work_item,              // .db is in smem
                                 uint32_t min_util,
                                 uint32_t transaction_count,
                                 uint32_t max_item)
{
    const int TEMP_ITEM_BUFFER_SIZE = 256;
    Item temp_item_buffer[TEMP_ITEM_BUFFER_SIZE]; // thread-private

    for (int i = threadIdx.x; i < temp_transaction_count; i += blockDim.x)
    {
        Transaction &t = temporary_transactions[i];

        int idx = 0;
        int cumsum = t.utility;
        // for (int j = 0; j < t.length; ++j)
        // {
        //     Item it = t.data[j];
        //     int found = find_item(local_subtree_utilities, max_item * scale, it.key);
        //     if (found != -1 && local_subtree_utilities[found].local_utility >= min_util)
        //     {
        //         if (idx < TEMP_ITEM_BUFFER_SIZE) {
        //             temp_item_buffer[idx++] = it;
        //             cumsum += it.util;
        //         }
        //     }
        // }
        // if (idx == 0) continue;

        // int t_off = atomicAdd(&new_work_item.db->num_transactions, 1);
        // int i_off = atomicAdd(&new_work_item.db->num_items, idx);

        // Transaction &nt = new_work_item.db->transactions[t_off];
        // nt.data = &new_work_item.db->data[i_off];
        // nt.utility = t.utility;
        // nt.length = idx;

        // for (int j = 0; j < idx; ++j) nt[j] = temp_item_buffer[j];

        // uint64_t hash = 0;
        // int temp = 0;
        // for (int j = 0; j < idx; ++j)
        // {
        //     Item it = nt[j];
        //     add_bucket_util(local_subtree_utilities, max_item * scale, it.key, cumsum - temp, false);
        //     hash ^= fasthash(it.key + j);
        //     temp += it.util;
        // }

        // uint64_t h = hash % (transaction_count * scale);
        // while (true)
        // {
        //     int old = atomicCAS(&transaction_hashes[h], -1, t_off);
        //     if (old == -1) break;

        //     if (new_work_item.db->compare_transactions(old, t_off)) {
        //         Transaction &ot = new_work_item.db->transactions[old];
        //         atomicAdd(&ot.utility, nt.utility);
        //         for (int j = 0; j < idx; j++) atomicAdd(&ot[j].util, nt[j].util);
        //         new_work_item.db->transactions[t_off].data = nullptr;
        //         break;
        //     }
        //     h = (h + 1) % (transaction_count * scale);
        // }
    }
    __syncthreads();
}

// `trim` logic, explicitly for shared memory.
__device__ void trim_smem(Task &new_work_item) // .db is in smem
{
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    __shared__ Transaction tile[1024];
    __shared__ int scan_storage[1024];
    __shared__ uint32_t final_count;

    if (tid == 0) final_count = 0;
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

        for (int offset = 1; offset < block_size; offset <<= 1)
        {
            int temp = (tid >= offset) ? scan_storage[tid - offset] : 0;
            __syncthreads();
            scan_storage[tid] += temp;
            __syncthreads();
        }

        __shared__ int valid_in_tile;
        if (tid == block_size - 1) valid_in_tile = scan_storage[tid];
        __syncthreads();

        __shared__ int write_base;
        if (tid == 0 && valid_in_tile > 0) write_base = atomicAdd(&final_count, valid_in_tile);
        __syncthreads();

        if (is_valid)
        {
            const int local_off = scan_storage[tid] - 1;
            new_work_item.db->transactions[write_base + local_off] = tile[tid];
        }
        __syncthreads();
    }

    if (tid == 0) new_work_item.db->num_transactions = final_count;
    __syncthreads();
}

// --- NEW: Orchestrator for building projected DB in Shared Memory ---
__device__ __forceinline__ void smem_build_projected_db(Task &s_curr,
                                                        Task &s_new,
                                                        Transaction *temporary_transactions,
                                                        int32_t *transaction_hashes,
                                                        Utilities *local_subtree_utilities,
                                                        uint32_t *p_temp_transaction_count,
                                                        uint32_t *p_temp_item_count,
                                                        uint32_t min_util,
                                                        SmemAllocator &alloc,
                                                        patterns *p)
{
    // Step 1: Scan current DB, calculate local utilities, create temp projected transactions.
    scan_smem(s_curr, s_new, local_subtree_utilities, temporary_transactions,
              p_temp_transaction_count, p_temp_item_count);

    if (threadIdx.x == 0)
    {
        printf("Thread %d, scanning for item %u found %u transactions with %u items\n",
               threadIdx.x, s_curr.primary, *p_temp_transaction_count, *p_temp_item_count);

        printf("Original DB had %u transactions, %u unique items\n",
               s_curr.db->num_transactions, s_curr.unique_items);

        printf("Local utilities:\n");
        for (int i = 0; i < s_curr.unique_items * scale; ++i)
        {
            if (local_subtree_utilities[i].key != 0)
            {
                printf("  Item %u: local_util=%u, subtree_util=%u\n",
                       local_subtree_utilities[i].key,
                       local_subtree_utilities[i].local_utility,
                       local_subtree_utilities[i].subtree_utility);
            }
        }
    }

    // Step 2: Add the candidate pattern to the results list.
    p->add_pattern(s_new, min_util);

    // Step 3: If any transactions were projected, build the new database.
    if (*p_temp_transaction_count > 0)
    {
        allocate_smem(s_new, *p_temp_item_count, *p_temp_transaction_count, alloc);

        // write_merge_smem(temporary_transactions, *p_temp_transaction_count,
        //                  transaction_hashes, local_subtree_utilities, s_new, min_util,
        //                  s_curr.db->num_transactions, s_curr.unique_items);
        
    //     // If merging duplicates creates empty slots, compact the transaction array.
    //     // After this, s_new.db->num_transactions might be 0.
    //     trim_smem(s_new);
    }
    else
    {
        // No transactions were projected; mark the new DB as null.
        if (threadIdx.x == 0) s_new.db = nullptr;
    }
    __syncthreads();
}


__device__ __forceinline__ void smem_copy_out_and_push(Task &s_curr,
                                                       Task &s_new,
                                                       Utilities *local_subtree_utilities,
                                                       Stack<Task, CAPACITY> *new_stack,
                                                       uint32_t min_util,
                                                       uint32_t *transaction_distribution)
{
    __shared__ bool needs_push;

    // A valid, non-empty projected DB must exist to be copied out.
    if (threadIdx.x == 0)
        needs_push = (s_new.db != nullptr && s_new.db->num_transactions > 0);
    __syncthreads();

    if (!needs_push)
        return;

    __shared__ Task out_task;
    // Allocate global DB blob
    if (threadIdx.x == 0)
    {
        out_task = s_new; // copy scalars (+ global pattern ptr)
        out_task.db = (Database *)gmalloc(s_new.memory_usage);
        atomicMax(&max_task_memory, s_new.memory_usage);
    }
    __syncthreads();

    // Copy DB from shared to global
    char *dst = (char *)out_task.db;
    char *src = (char *)s_new.db;
    size_t n = s_new.memory_usage;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x)
        dst[i] = src[i];
    __syncthreads();

    // Rebase internal pointers in the new GLOBAL copy
    if (threadIdx.x == 0)
    {
        out_task.db->data = (Item *)((char *)out_task.db + sizeof(Database));
        out_task.db->transactions = (Transaction *)((char *)out_task.db->data + sizeof(Item) * out_task.db->num_items);

        for (int i = 0; i < out_task.db->num_transactions; ++i)
        {
            size_t off_items = (size_t)(s_new.db->transactions[i].data - s_new.db->data);
            out_task.db->transactions[i].data = out_task.db->data + off_items;
        }
    }
    __syncthreads();

    // Push children
    push_work_item(local_subtree_utilities, out_task, new_stack,
                   min_util, s_curr.unique_items, transaction_distribution);
}

// --- REFACTORED: `mine_shared` Kernel ---
__global__ void mine_shared(
    Stack<Task, CAPACITY> *curr_stack,
    Stack<Task, CAPACITY> *new_stack,
    patterns *p,
    uint32_t min_util,
    uint32_t *transaction_distribution)
{
    // --- Shared arena + allocator ---
    extern __shared__ char smem[];
    SmemAllocator allocator;
    allocator.init(smem);

    // --- Per-block shared state ---
    __shared__ Task g_task;
    __shared__ Task s_task;
    __shared__ Task s_new;

    __shared__ Transaction *temporary_transactions;
    __shared__ Utilities *local_subtree_utilities;
    __shared__ int32_t *transaction_hashes;
    __shared__ uint32_t temp_transaction_count;
    __shared__ uint32_t temp_item_count;

    // 1. Fetch task from global stack
    fetch_task(curr_stack, g_task);

    // 2. Copy task's DB from global to shared memory
    smem_copy_in_task_db(g_task, s_task, allocator);

    // 3. Allocate temporary structures in shared memory
    smem_init_temporaries(&temporary_transactions, &local_subtree_utilities, &transaction_hashes,
                          s_task.db->num_transactions, s_task.unique_items,
                          allocator);

    // 4. Reset state for the new candidate task
    smem_reset_for_candidate(&s_new, s_task,
                             temporary_transactions, transaction_hashes, local_subtree_utilities,
                             &temp_transaction_count, &temp_item_count,
                             s_task.db->num_transactions, s_task.unique_items);

    // 5. Build the projected database entirely within shared memory
    smem_build_projected_db(s_task, s_new,
                            temporary_transactions, transaction_hashes, local_subtree_utilities,
                            &temp_transaction_count, &temp_item_count,
                            min_util, allocator, p);

    // // 6. If a valid projected DB was created, copy it to global memory and push new child tasks
    // smem_copy_out_and_push(s_task, s_new,
    //                        local_subtree_utilities,
    //                        new_stack, min_util, transaction_distribution);
}


//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------
static inline uint32_t round_up_to_warp_size(uint32_t n)
{
    const uint32_t warp = 32;
    return ((n + warp - 1) / warp) * warp;
}

//------------------------------------------------------------------------------
// Host driver
//------------------------------------------------------------------------------
void static_parallel_mining(built_file &bf,
                            patterns *p,
                            ArgParser &parser,
                            Timer &timer,
                            size_t available_shared_memory)
{
    // Stacks
    Stack<Task, CAPACITY> *curr_stack;
    Stack<Task, CAPACITY> *new_stack;
    gpuErrchk(cudaMallocManaged(&curr_stack, sizeof(*curr_stack)));
    gpuErrchk(cudaMallocManaged(&new_stack, sizeof(*new_stack)));
    curr_stack->init();
    new_stack->init();

    max_task_memory = 0;

    // Histogram (33 buckets)
    uint32_t *transaction_distribution;
    const int dist_size = 33;
    gpuErrchk(cudaMallocManaged(&transaction_distribution, sizeof(uint32_t) * dist_size));
    memset(transaction_distribution, 0, sizeof(uint32_t) * dist_size);

    // Primaries
    uint32_t *primary;
    gpuErrchk(cudaMallocManaged(&primary, sizeof(uint32_t) * bf.primary.size()));
    memcpy(primary, bf.primary.data(), sizeof(uint32_t) * bf.primary.size());

    // Seed
    copy_work<<<1, 1024>>>(curr_stack, bf.db, primary, bf.primary.size(), bf.max_id + 1);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    int iter = 1;
    uint32_t cand_count = 0;

    while (curr_stack->top != 0)
    {
        // Heuristic (kept for logging), but we won't use the shared path.
        const bool shared_capable =
            (max_task_memory * SHARED_MEM_HEURISTIC_FACTOR < available_shared_memory);

        // We explicitly disable the shared path for stability.
        const bool use_shared_memory_path = shared_capable;
        const size_t dynamic_smem_size =
            use_shared_memory_path ? (max_task_memory * SHARED_MEM_HEURISTIC_FACTOR) : 0;

        printf("Iter %d: Estimated memory/task: %u KB. %s\n",
               iter,
               (unsigned)((max_task_memory * SHARED_MEM_HEURISTIC_FACTOR + 1023) / 1024),
               shared_capable
                   ? "SHARED-capable (disabled → using GLOBAL path)."
                   : "Using GLOBAL path.");

        max_task_memory = 0;
        cand_count += curr_stack->top;

        // Adaptive block size (same as before)
        uint32_t thread_count = parser.get_uint32("--block_size");
        if (parser.get_bool("--adaptive"))
        {
            unsigned long long weighted_sum = 0;
            uint32_t tasks_in_dist = 0;

            for (int i = 1; i < dist_size; ++i)
            {
                printf("  Dist[%d] = %6d\n", i*32, transaction_distribution[i]);
            }

            for (int i = 1; i < dist_size; ++i)
            {
                const uint32_t num_tasks = transaction_distribution[i];
                if (num_tasks > 0)
                {
                    const uint32_t rep = i * 32;
                    weighted_sum += (unsigned long long)rep * num_tasks;
                    tasks_in_dist += num_tasks;
                }
            }

            if (tasks_in_dist > 0)
            {
                const uint32_t weighted_avg = (uint32_t)(weighted_sum / tasks_in_dist);
                thread_count = round_up_to_warp_size(weighted_avg);
            }
            else if (curr_stack->top > 0)
            {
                const uint32_t avg = curr_stack->total_transactions / curr_stack->top;
                if (avg > 0)
                    thread_count = next_pow2(avg);
            }

            if (thread_count > 1024)
                thread_count = 1024;
            if (thread_count < 32)
                thread_count = 32;

            memset(transaction_distribution, 0, sizeof(uint32_t) * dist_size);
        }

        const unsigned long long before_offset = allocators::atomic::allocator_offset;
        reset_allocator_wrapped_flag();

        // // Always launch GLOBAL path (shared path temporarily disabled)
        // (void)dynamic_smem_size; // silence unused warning
        // mine<<<curr_stack->top, thread_count>>>(
        //     curr_stack, new_stack, p,
        //     (uint32_t)parser.get_uint32("--minutil"),
        //     transaction_distribution);
        // if (use_shared_memory_path)
        // {
        //     // printf("  Launching %u blocks of %u threads (shared mem: %zu KB)\n",
        //     //        curr_stack->top, thread_count, dynamic_smem_size / 1024);
        //     // mine_shared
        //         mine
        //         <<<
        //         curr_stack->top
        //         // 1
        //         , thread_count, dynamic_smem_size>>>(
        //         curr_stack, new_stack, p,
        //         (uint32_t)parser.get_uint32("--minutil"),
        //         transaction_distribution);
        // }
        // else
        // {
        mine<<<curr_stack->top, thread_count>>>(
            curr_stack, new_stack, p,
            (uint32_t)parser.get_uint32("--minutil"),
            transaction_distribution);
        // }

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        const unsigned long long after_offset = allocators::atomic::allocator_offset;
        timer.recordPoint("Iteration " + std::to_string(iter), before_offset, after_offset);
        iter++;

        reclaim_memory();

        std::swap(curr_stack, new_stack);
        new_stack->init();
    }

    // Cleanup
    cudaFree(curr_stack);
    cudaFree(new_stack);
    cudaFree(transaction_distribution);
    cudaFree(bf.db->data);
    cudaFree(bf.db->transactions);
    cudaFree(bf.db);
    cudaFree(primary);
}
