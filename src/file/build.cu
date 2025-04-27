#include "file/build.cuh"

// building<<<block_count, block_size>>>(db->data, db->num_items, item_map);
__global__ void buildingKernel(Item *data, uint32_t num_items, const Item_mapping *item_map) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;
    Item it = data[idx];
    data[idx].key = item_map[it.key].id;
}

__device__ void insertion_sort_ignore_zeros(Transaction &tx) {
    Item *arr = tx.data;
    int len = tx.length;
    int count = 0;
    // Compact non-zeros
    for (int i = 0; i < len; ++i) {
        if (arr[i].key != 0) arr[count++] = arr[i];
    }
    // Insertion sort on prefix
    for (int i = 1; i < count; ++i) {
        Item key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j].key > key.key) {
            arr[j + 1] = arr[j]; --j;
        }
        arr[j + 1] = key;
    }
    tx.length = count;
}

__global__ void sortTransactionsKernel(Transaction *transactions,
                                       uint32_t num_transactions,
                                       Utilities *utils) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_transactions) return;

    Transaction &tx = transactions[tid];
    insertion_sort_ignore_zeros(tx);

    uint32_t sum = tx.util_sum();
    uint32_t prefix = 0;
    for (int i = 0; i < tx.length; ++i) {
        uint32_t id = tx.data[i].key;
        atomicAdd(&utils[id].subtree_utility, sum - prefix);
        prefix += tx.data[i].util;
    }
}

namespace builder
{

    // Initializes the item mapping array and returns the number of valid mapped IDs
    uint32_t initItemMapping(Item_mapping *item_map,
                             const std::unordered_map<uint32_t, uint32_t> &twu,
                             uint32_t minutil)
    {
        uint32_t max_id = 0;

        for (auto &entry : twu)
        {
            item_map[entry.first].item = entry.first;
            item_map[entry.first].subtree_utility = 0;
            item_map[entry.first].local_utility = entry.second;
            if (entry.second > minutil)
                ++max_id;
        }
        return max_id;
    }

    // Sort by local utility descending, assign new IDs for items above threshold, then restore order
    void assignItemIDs(Item_mapping *item_map,
                       size_t map_size,
                       uint32_t minutil,
                       uint32_t max_id)
    {
        // Sort by utility
        std::sort(item_map, item_map + map_size,
                  [](auto &a, auto &b)
                  { return a.local_utility > b.local_utility; });

        // Assign IDs
        int next_id = static_cast<int>(max_id);
        for (size_t i = 0; i < map_size; ++i)
        {
            if (item_map[i].local_utility < minutil)
                break;
            item_map[i].id = next_id--;
        }

        // Restore original ordering by item value
        std::sort(item_map, item_map + map_size,
                  [](auto &a, auto &b)
                  { return a.item < b.item; });
    }

    // Prepare managed memory for Database, Items, and Transactions
    void prepareDatabase(Database *&db,
                         const parsed_file &pf)
    {
        // Count total items
        uint32_t total_items = 0;
        for (const auto &tx : pf.database)
            total_items += static_cast<uint32_t>(tx.size());

        // Allocate managed memory
        cudaMallocManaged(&db, sizeof(Database));
        db->num_transactions = static_cast<uint32_t>(pf.database.size());
        db->num_items = total_items;
        cudaMallocManaged(&db->data, total_items * sizeof(Item));
        cudaMallocManaged(&db->transactions, pf.database.size() * sizeof(Transaction));

        // Copy each transaction into contiguous array
        uint32_t offset = 0;
        for (size_t i = 0; i < pf.database.size(); ++i)
        {
            const auto &src = pf.database[i];
            Transaction &dest = db->transactions[i];
            dest.length = static_cast<int>(src.size());
            dest.data = &db->data[offset];
            memcpy(dest.data, src.data(), src.size() * sizeof(Item));
            offset += static_cast<uint32_t>(src.size());
        }
    }

    // Launch build-key kernel
    void launchBuilding(Item *data,
                        uint32_t num_items,
                        const Item_mapping *item_map,
                        uint32_t block_size)
    {
        uint32_t grid = (num_items + block_size - 1) / block_size;
        buildingKernel<<<grid, block_size>>>(data, num_items, item_map);
        cudaDeviceSynchronize();
    }

    // Launch sort-and-aggregate kernel
    void launchSortTransactions(Transaction *transactions,
                                uint32_t num_transactions,
                                Utilities *utils,
                                uint32_t block_size)
    {
        uint32_t grid = (num_transactions + block_size - 1) / block_size;
        sortTransactionsKernel<<<grid, block_size>>>(transactions, num_transactions, utils);
        cudaDeviceSynchronize();
    }

} // namespace builder

built_file build(Parsed &pf, uint32_t minutil, ArgParser &parser, Item_mapping *item_map)
{
    // // Allocate and initialize item map
    // size_t map_size = pf.twu.size() + 1;
    // Item_mapping *item_map;
    // cudaMallocManaged(&item_map, (map_size) * sizeof(Item_mapping));


    // Init mapping and filter items
    // uint32_t max_id = builder::initItemMapping(item_map, pf.twu, minutil);
    // builder::assignItemIDs(item_map, map_size, minutil, max_id);
    builder::assignItemIDs(item_map, pf.max_item, minutil, pf.max_item);


    // // Prepare database memory and copy transactions
    // Database *db = nullptr;
    // builder::prepareDatabase(pf.db, pf);


    // Remap item keys
    uint32_t block_size = parser.get_uint32("--block_size");
    // builder::launchBuilding(db->data, db->num_items, item_map, block_size);
    builder::launchBuilding(pf.db->data, pf.db->num_items, item_map, block_size);

    // Allocate and init utilities
    Utilities *utils;
    cudaMallocManaged(&utils, sizeof(Utilities) * (pf.max_item));
    for (uint32_t i = 0; i <= pf.max_item; ++i)
        utils[i].key = i;

    // Sort transactions and compute subtree utilities
    // builder::launchSortTransactions(db->transactions, db->num_transactions, utils, block_size);
    builder::launchSortTransactions(pf.db->transactions, pf.db->num_transactions, utils, block_size);


    std::vector<uint32_t> primary;
    for (uint32_t i = 0; i <= pf.max_item; ++i)
    {
        if (utils[i].subtree_utility >= minutil)
        {
            primary.push_back(i);
        }
    }

    std::sort(primary.begin(), primary.end());

    built_file bf;
    bf.db = pf.db;
    bf.item_map = item_map;
    bf.utils = utils;
    bf.max_id = pf.max_item;
    bf.primary = primary;

    return bf;
}