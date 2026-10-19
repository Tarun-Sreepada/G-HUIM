#pragma once
// The stages of read_database(), in the order it runs them. Each is implemented in src/io/reader/.

#include "io/reader/chunk.hpp"
#include "mine/level.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

// The transactions of every chunk back to back; transaction t is keys/vals[begin[t]..begin[t + 1]).
struct Flat {
	std::unique_ptr<Item[]> keys;
	std::unique_ptr<Util[]> vals;
	std::unique_ptr<std::size_t[]> begin;
	std::size_t num_tx = 0;
};

// Reads the whole file with parallel pread()s.
std::string read_all(const std::string& path, int threads);
// Splits the text at line boundaries so each thread parses whole lines.
std::vector<Chunk> split_into_chunks(std::string_view text, int threads);
// Parses every chunk in parallel; throws "line N: ..." for the first bad line.
void parse_chunks(std::string_view text, const std::string& sep, std::vector<Chunk>& chunks);
// Merges the per-chunk name tables in chunk order (= first appearance in the file) and sets chunk.remap.
NameTable merge_names(std::vector<Chunk>& chunks);
// Keeps items with TWU >= min_util, ids by TWU ascending (a larger id is a larger TWU).
// Returns the old-name index -> new id map (0 = pruned) and fills rename[id].
std::vector<Item> rank_items(const NameTable& names, Utility min_util, std::vector<std::string>& rename);
// Renamed transactions per chunk, items ascending.
void rename_transactions(std::vector<Chunk>& chunks, const std::vector<Item>& new_id, std::size_t num_items);
// Throws if a utility is negative or an item's total utility overflows the stored width.
void check_utilities(const std::vector<Chunk>& chunks, std::size_t num_items, const std::vector<std::string>& rename);
// Concatenates the chunks into global arrays and frees them.
Flat concatenate(std::vector<Chunk>& chunks);
// Transaction order, reverse-lexicographic (compare from the last item).
std::vector<std::uint32_t> sort_transactions(const Flat& flat, int threads);
// Builds the root level from the sorted order, merging equal transactions.
void build_root(const Flat& flat, const std::vector<std::uint32_t>& order, int threads, Level& root);
// Primary items: those whose subtree utility is >= min_util.
void find_primary(Level& root, std::size_t num_items, Utility min_util, int threads);
