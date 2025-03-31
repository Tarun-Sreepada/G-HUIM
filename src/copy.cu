#include "copy.cuh"

__global__ void copy_work(stack *work_stack, Database *db, uint32_t *primary, uint32_t primary_size, uint32_t max_item)
{
    int tid = threadIdx.x;
    __shared__ WorkItem item;
    // WorkItem item;
    if (tid == 0)
    {
        item.pattern = (uint32_t *)gmalloc(sizeof(int));
        item.pattern_length = 0;
        item.utility = 0;
        item.db = (Database *)gmalloc(sizeof(Database));
        item.db->num_items = db->num_items;
        item.db->num_transactions = db->num_transactions;

        item.db->data = (Item *)gmalloc(sizeof(Item) * db->num_items);
        // item.db->cumulative_sum = (uint32_t *)gmalloc(sizeof(uint32_t) * (db->num_items));


        item.db->transactions = (Transaction *)gmalloc(sizeof(Transaction) * db->num_transactions);
        item.db->num_transactions = db->num_transactions;

        item.max_item = max_item;

        item.work_done = (uint32_t *)gmalloc(sizeof(uint32_t));
        item.work_done[0] = 0;
        item.work_count = primary_size;
    }

    __syncthreads();

    // copy items
    for (int i = tid; i < db->num_items; i += blockDim.x)
    {
        item.db->data[i] = db->data[i];
    }
    __syncthreads();

    // copy transaction structures
    for (int i = tid; i < db->num_transactions; i += blockDim.x)
    {
        item.db->transactions[i].data = item.db->data + (db->transactions[i].data - db->data); // data + offset (offset)
        // item.db->transactions[i].cumulative_sum = item.db->cumulative_sum + (db->transactions[i].data - db->data); // cumsum + offset
        item.db->transactions[i].utility = 0;
        item.db->transactions[i].length = db->transactions[i].length;
    }

    __syncthreads();

    // push the work item to the stack

    WorkItem copy = item;
    for (int i = tid; i < primary_size; i += blockDim.x)
    {
        copy.primary = primary[i];
        work_stack->push(copy);
    }

    __syncthreads();

    // cumulative sums

    // for (int i = tid; i < db->num_transactions ; i += blockDim.x)
    // {
    //     Transaction &tran = item.db->transactions[i];
    //     tran.cumsum()[0] = tran.data[0].util;
    //     for (int j = 0; j < tran.length; j++)
    //     {
    //         tran.cumsum()[j] = tran.cumsum()[j - 1] + tran.data[j].util;
    //     }
    // }

    return;
}

std::tuple<Database *, uint32_t *> // return type
initialize_copy(built_file &bf, stack *s, Timer &timer)
{
    Database *db;
    gpuAssert(cudaMallocManaged(&db, sizeof(Database)), __FILE__, __LINE__);
    db->num_transactions = bf.transaction_count;
    db->num_items = bf.total_items;
    Item *items;
    gpuAssert(cudaMallocManaged(&items, sizeof(Item) * bf.total_items), __FILE__, __LINE__);
    db->data = items;
    gpuAssert(cudaMallocManaged(&db->transactions, sizeof(Transaction) * bf.transaction_count), __FILE__, __LINE__);

    timer.recordPoint("GPU Memory Allocated");

    unsigned int item_offset = 0;
    for (size_t t = 0; t < bf.transactions.size(); ++t)
    {
        const std::vector<Item> &tranItems = bf.transactions[t];
        // Set the pointer for the transaction to point to the correct location in db->data.
        db->transactions[t].data = &db->data[item_offset];
        db->transactions[t].length = tranItems.size();

        // Copy each item from the vector into the unified memory block.
        memcpy(&db->data[item_offset], tranItems.data(), sizeof(Item) * tranItems.size());

        // Update the offset for the next transaction.
        item_offset += tranItems.size();
    }

    timer.recordPoint("Transactions Copied to GPU");

    uint32_t *primary;
    gpuErrchk(cudaMallocManaged(&primary, sizeof(uint32_t) * bf.primary.size()));
    gpuErrchk(cudaMemcpy(primary, bf.primary.data(), sizeof(uint32_t) * bf.primary.size(), cudaMemcpyHostToDevice));

    copy_work<<<1, 1024>>>(s, db, primary, bf.primary.size(), bf.itemID_to_item.size());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // return db, primary
    return std::make_tuple(std::move(db), std::move(primary));
}