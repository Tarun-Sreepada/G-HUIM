#include "file/parse.hpp"
#include "file/build.hpp"

void buildOrderedTWU(const parsed_file &pf, uint64_t min_utility, built_file &bf)
{
    for (const auto &item : pf.twu)
    {
        if (item.second < min_utility)
        {
            continue;
        }
        bf.ordered_twu[item.first] = item.second;
    }

    // Sort the ordered_twu map by utility in descending order.
    bf.ordered_twu_vector = std::vector<std::pair<uint32_t, uint32_t>>(bf.ordered_twu.begin(), bf.ordered_twu.end());
    std::sort(bf.ordered_twu_vector.begin(), bf.ordered_twu_vector.end(),
              [](const std::pair<uint32_t, uint32_t> &a, const std::pair<uint32_t, uint32_t> &b)
              {
                  return a.second > b.second;
              });
}

void createItemMapping(built_file &bf)
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
                        built_file &bf,
                        std::unordered_map<uint32_t, uint32_t> &global_twu)
{
    // We'll store each thread's transactions in two parallel vectors:
    // local_keys[tid] and local_utils[tid] hold multiple transactions.
    // local_keys[tid][t] and local_utils[tid][t] describe the t-th transaction in thread tid.
    std::vector<std::vector<std::vector<uint32_t>>> local_keys(threads);
    std::vector<std::vector<std::vector<uint32_t>>> local_utils(threads);

    // Also store each thread’s local TWU for merging.
    std::vector<std::unordered_map<uint32_t, uint32_t>> local_twu(threads);

#pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        auto &lk = local_keys[tid];
        auto &lu = local_utils[tid];
        auto &ltwu = local_twu[tid];

        // For each (items, utilities) chunk that belongs to this thread:
        for (const auto &[items, utilities] : pf.key_value_pairs[tid])
        {
            // Temporary storage of <internalKey, utility> so we can sort them together.
            std::vector<std::pair<uint32_t, uint32_t>> filtered;
            filtered.reserve(items.size());

            // 1) Filter out items not in the bf.ordered_twu map, and map to internal IDs.
            for (size_t i = 0; i < items.size(); ++i)
            {
                if (bf.ordered_twu.find(items[i]) == bf.ordered_twu.end())
                    continue; // skip pruned item

                // Convert original ID -> internal ID
                const uint32_t internalID = bf.item_to_itemID.at(items[i]);
                filtered.push_back({internalID, utilities[i]});
            }

            if (filtered.empty())
                continue; // skip empty transactions

            // 2) Sort by ascending internal ID. make sure that the utility order is sorted using the internal ID.
            // now descending
            std::sort(filtered.begin(), filtered.end(),
            [](const auto &a, const auto &b)
            {
                return a.first < b.first;
            });


            // Calculate the total utility of the transaction.
            uint32_t total_utility = std::accumulate(filtered.begin(), filtered.end(), 0,
                                                     [](uint32_t sum, const std::pair<uint32_t, uint32_t> &kv)
                                                     {
                                                         return sum + kv.second;
                                                     });

            // 4) Update local TWU.  Subtree utility for each item is totalUtility minus
            //    the sum of utilities of preceding items in sorted order.
            uint32_t prefixSum = 0;
            for (const auto &kv : filtered)
            {
                ltwu[kv.first] += (total_utility - prefixSum);
                prefixSum += kv.second;
            }

            // 5) Split into separate key & utility vectors for final storage.
            std::vector<uint32_t> transKeys, transUtils;
            transKeys.reserve(filtered.size());
            transUtils.reserve(filtered.size());

            for (auto &kv : filtered)
            {
                transKeys.push_back(kv.first);
                transUtils.push_back(kv.second);
            }

            lk.emplace_back(std::move(transKeys));
            lu.emplace_back(std::move(transUtils));
        }
    } // end parallel region

    // Merge local TWU into global TWU
    for (auto &ltwu : local_twu)
    {
        for (const auto &[itemID, twuVal] : ltwu)
        {
            global_twu[itemID] += twuVal;
        }
    }

    // At this point, local_keys and local_utils each hold all transactions, per thread.
    // We need to flatten them into a single list of transactions for bf.


    // Count total transactions
    size_t totalTrans = 0;
    for (int t = 0; t < threads; ++t)
    {
        totalTrans += local_keys[t].size();
    }
    bf.transaction_count = totalTrans;

    // Prepare to store sizes and offsets for each transaction.
    bf.transaction_sizes.resize(totalTrans);
    bf.transaction_offsets.resize(totalTrans + 1);
    bf.transaction_offsets[0] = 0; // First transaction starts at offset 0.

    // Gather all per-thread transactions into one vector of (keys, utils).
    std::vector<std::vector<uint32_t>> allKeys;
    allKeys.reserve(totalTrans);
    std::vector<std::vector<uint32_t>> allUtils;
    allUtils.reserve(totalTrans);

    // Flatten
    for (int t = 0; t < threads; ++t)
    {
        auto &lk = local_keys[t];
        auto &lu = local_utils[t];
        for (size_t i = 0; i < lk.size(); ++i)
        {
            allKeys.push_back(std::move(lk[i]));
            allUtils.push_back(std::move(lu[i]));
        }
    }

    // Track offsets and maximum transaction size
    size_t offset = 0;
    size_t maxSize = 0;

    // for (size_t i = 0; i < totalTrans; i++)
    // {
    //     const size_t sz = allKeys[i].size();
    //     bf.transaction_sizes[i] = static_cast<uint32_t>(sz);
    //     bf.transaction_offsets[i+1] = static_cast<uint32_t>(offset);

    //     if (sz > maxSize)
    //         maxSize = sz;

    //     offset += sz;
    // }

    // bf.max_transaction_size = maxSize;
    // bf.total_items = offset;

    // Finally, allocate the big arrays and copy data

    size_t pos = 0;
    for (size_t i = 0; i < totalTrans; i++)
    {
        const auto &kVec = allKeys[i];
        const auto &uVec = allUtils[i];
        for (size_t j = 0; j < kVec.size(); ++j)
        {
            bf.keys.push_back(kVec[j]);
            bf.values.push_back(uVec[j]);
            pos++;
        }
        bf.transaction_sizes[i] = static_cast<uint32_t>(kVec.size());
        bf.transaction_offsets[i + 1] = static_cast<uint32_t>(pos);

        if (bf.transaction_sizes[i] > maxSize)
            maxSize = bf.transaction_sizes[i];
    }

    bf.max_transaction_size = maxSize;
    bf.total_items = pos;

    #ifdef DEBUG
    std::cout << "Total items: " << bf.total_items << "\n";
    std::cout << "Transaction count: " << bf.transaction_count << "\n";
    #endif
}

built_file build_file(parsed_file &pf, int threads, uint64_t min_utility)
{
    built_file bf;

    // Step 1: Build the ordered TWU structure.
    buildOrderedTWU(pf, min_utility, bf);

    // Step 2: Create the item to itemID mapping.
    createItemMapping(bf);

    // Step 3: Initialize the transaction sizes and offsets.
    std::unordered_map<uint32_t, uint32_t> subtree_utility;
    formatTransactions(pf, threads, bf, subtree_utility);

    // bf.primary
    for (const auto &[itemID, twuVal] : subtree_utility)
    {
        if (twuVal >= min_utility)
        {
            bf.primary.push_back(itemID);
        }
    }
    std::sort(bf.primary.begin(), bf.primary.end());

    return bf;
}