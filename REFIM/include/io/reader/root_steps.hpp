#pragma once
// The stages of make_root(), in the order it runs them. Each is implemented in src/io/reader/.

#include "io/reader/chunk.hpp"
#include "mine/level.hpp"

#include <cstdint>
#include <memory>
#include <vector>

inline namespace REFIM_WIDTHS {

// The transactions of every chunk back to back; transaction t is keys/vals[begin[t]..begin[t + 1]).
struct Flat {
	std::unique_ptr<Item[]> keys;
	std::unique_ptr<Util[]> vals;
	std::unique_ptr<std::size_t[]> begin;
	std::size_t num_tx = 0;
};

// Concatenates the chunks into global arrays of this build's widths and frees them.
Flat concatenate(std::vector<Chunk>& chunks);
// Transaction order, reverse-lexicographic (compare from the last item).
std::vector<std::uint32_t> sort_transactions(const Flat& flat, int threads);
// Builds the root level from the sorted order, merging equal transactions.
void build_root(const Flat& flat, const std::vector<std::uint32_t>& order, int threads, Level& root);
// Primary items: those whose subtree utility is >= min_util.
void find_primary(Level& root, std::size_t num_items, std::uint64_t min_util, int threads);

}  // namespace REFIM_WIDTHS
