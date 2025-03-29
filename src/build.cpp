#include "build.hpp"


void buildOrderedTWU(const parsed_file &pf, uint64_t min_utility, built_file &bf, Timer &timer)
{
    for (const auto &item : pf.twu)
    {
        if (item.second < min_utility)
        {
            continue;
        }
        bf.ordered_twu[item.first] = item.second;
    }

    bf.ordered_twu_vector = std::vector<std::pair<uint32_t, uint32_t>>(bf.ordered_twu.begin(), bf.ordered_twu.end());
    std::sort(bf.ordered_twu_vector.begin(), bf.ordered_twu_vector.end(),
              [](const std::pair<uint32_t, uint32_t> &a, const std::pair<uint32_t, uint32_t> &b)
              {
                  return a.second > b.second;
              });
}

void createItemMapping(built_file &bf, Timer &timer)
{
    uint32_t itemID = static_cast<uint32_t>(bf.ordered_twu.size());
    for (const auto &item : bf.ordered_twu_vector)
    {
        bf.itemID_to_item[itemID] = item.first;
        bf.item_to_itemID[item.first] = itemID--;
    }
}

void formatTransactions(const parsed_file &pf,
                        int threads,
                        const built_file &bf,
                        Timer &timer,
                        std::vector<std::vector<Item>> &transactions,
                        std::unordered_map<uint32_t, uint32_t> &global_twu)
{
    // Create thread-local storage for transactions and TWU.
    std::vector<std::vector<std::vector<Item>>> local_transactions(threads);
    std::vector<std::unordered_map<uint32_t, uint32_t>> local_twu(threads);

#pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        auto &localTrans = local_transactions[tid];
        auto &ltwu = local_twu[tid];

        // Assume pf.key_value_pairs is partitioned per thread.
        for (const auto &[items, utilities] : pf.key_value_pairs[tid])
        {
            std::vector<Item> transaction;
            // Filter items based on TWU and map them to their itemID.
            for (size_t i = 0; i < items.size(); ++i)
            {
                // if item is not in the ordered_twu, skip it
                if (bf.ordered_twu.find(items[i]) == bf.ordered_twu.end())
                {
                    continue;
                }
                // Add the item to the transaction.
                transaction.push_back({bf.item_to_itemID.at(items[i]), utilities[i]});
            }

            // Sort the transaction in descending order by item key.
            std::sort(transaction.begin(), transaction.end(),
                      [](const Item &a, const Item &b)
                      {
                          return a.key > b.key;
                      });

            // Calculate the total utility of the transaction.
            uint32_t total_utility = std::accumulate(transaction.begin(), transaction.end(), 0,
                                                     [](uint32_t sum, const Item &item)
                                                     {
                                                         return sum + item.util;
                                                     });

            // Update the subtree utility for each item in the transaction.
            uint32_t temp = 0;
            for (const auto &item : transaction)
            {
                ltwu[item.key] += total_utility - temp;
                temp += item.util;
            }

            localTrans.emplace_back(std::move(transaction));
        }
    }

    // Merge local transactions into global transactions.
    for (const auto &localTrans : local_transactions)
    {
        for (const auto &trans : localTrans)
        {
            transactions.push_back(trans);
        }
    }

    #ifdef DEBUG
    // print transactions
    // std::cout << "Transactions:" << std::endl;
    // for (const auto &trans : transactions)
    // {
    //     for (const auto &item : trans)
    //     {
    //         std::cout << item.key << ":" << item.util << " ";
    //     }
    //     std::cout << std::endl;
    // }

    #endif

    // Merge local TWU values into global TWU.
    for (const auto &ltwu : local_twu)
    {
        for (const auto &[item, utility] : ltwu)
        {
            global_twu[item] += utility;
        }
    }
}

void createPrimary(built_file &bf, uint64_t min_utility, std::unordered_map<uint32_t, uint32_t> &subtree_utility)
{
    // Create primary and secondary lists based on subtree utility.
    for (const auto &item : subtree_utility)
    {
        if (item.second >= min_utility)
        {
            bf.primary.push_back(item.first);
        }
    }
}

built_file build_file(parsed_file &pf, int threads, Timer &timer, uint64_t min_utility)
{
    built_file bf;

    // Step 1: Build the ordered TWU structure.
    buildOrderedTWU(pf, min_utility, bf, timer);

    // Step 2: Create the item to itemID mapping.
    createItemMapping(bf, timer);

    // Step 3: Format transactions in parallel and merge local results.
    std::vector<std::vector<Item>> transactions;
    std::unordered_map<uint32_t, uint32_t> subtree_utility;
    formatTransactions(pf, threads, bf, timer, transactions, subtree_utility);

    // Step 4: Create primary and secondary lists.
    createPrimary(bf, min_utility, subtree_utility);

    timer.recordPoint("Build");

    return bf;
}