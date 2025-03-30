/**
 * @file build_file.hpp
 * @brief Utilities for building transaction structures from parsed input.
 *
 * Contains the `built_file` struct and the `build_file` function to transform
 * a parsed file into a structured, utility-aware transaction format.
 */

 #pragma once

 #include <iostream>
 #include <vector>
 #include <string>
 #include <unordered_map>
 #include <map>
 #include <utility>
 #include <charconv>
 #include <stdexcept>
 #include <omp.h>
 #include <cstdint>
 #include <algorithm>
 #include <numeric>
 #include "data_types.hpp"
 #include "timer.hpp"
 #include "file.hpp"
 
 /**
  * @struct built_file
  * @brief Structure holding transformed transaction data for mining.
  *
  * Stores item mappings, transaction utility info, and compacted transactions.
  */
 struct built_file
 {
     std::map<uint32_t, uint32_t> ordered_twu;               ///< Items ordered by transaction-weighted utility.
     std::vector<std::pair<uint32_t, uint32_t>> ordered_twu_vector; ///< Vector version of ordered_twu for fast access.
 
     std::unordered_map<uint32_t, uint32_t> item_to_itemID;  ///< Maps original item IDs to internal IDs.
     std::unordered_map<uint32_t, uint32_t> itemID_to_item;  ///< Maps internal item IDs back to original item IDs.
 
     size_t transaction_count;    ///< Number of transactions.
     size_t total_items;          ///< Total number of items after pruning.
     size_t max_transaction_size;///< Size of the largest transaction.
 
     std::vector<std::vector<Item>> transactions; ///< List of compacted transactions.
     std::vector<uint32_t> primary;               ///< Items considered primary (for mining root).
 
     /**
      * @brief Default constructor.
      *
      * Initializes counts and sizes to zero.
      */
     built_file()
     {
         transaction_count = 0;
         max_transaction_size = 0;
         total_items = 0;
     }
 
     /**
      * @brief Destructor.
      *
      * Clears internal mappings and vectors.
      */
     ~built_file()
     {
         ordered_twu.clear();
         ordered_twu_vector.clear();
         item_to_itemID.clear();
         itemID_to_item.clear();
     }
 };
 
 /**
  * @brief Builds a structured and filtered transaction dataset from parsed input.
  *
  * Transforms a parsed_file into a built_file by filtering low-utility items,
  * remapping item IDs, and computing statistics like TWU and max transaction size.
  *
  * @param pf The parsed file input.
  * @param threads Number of threads to use (OpenMP enabled).
  * @param timer A Timer object for benchmarking.
  * @param min_utility The minimum utility threshold to keep an item.
  * @return A built_file structure containing the processed transaction data.
  */
 built_file build_file(parsed_file &pf, int threads, Timer &timer, uint64_t min_utility);
 