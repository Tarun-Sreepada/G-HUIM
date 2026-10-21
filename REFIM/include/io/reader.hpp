#pragma once

#include "io/reader/chunk.hpp"

#include <cstdint>
#include <string>
#include <vector>

// The input after renaming, before the root database is built (make_root, io/root.hpp).
struct Transactions {
	std::vector<Chunk> chunks;   // renamed transactions, items ascending
	std::size_t num_items = 0;   // items keep ids 1..num_items
	std::uint64_t max_twu = 0;   // largest TWU of a kept item over the renamed transactions; bounds every utility sum
};

// Reads the input, removes items with TWU < min_util and renames items so a larger id means a larger TWU.
// Parsing uses up to `threads` threads. rename[id] = original item name (index 0 unused).
Transactions read_transactions(const std::string& path, const std::string& sep, std::int64_t min_util,
                               std::vector<std::string>& rename, int threads = 1);
