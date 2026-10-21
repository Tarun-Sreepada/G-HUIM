#pragma once
// The stages of read_transactions(), in the order it runs them. Each is implemented in src/io/reader/.
// The root build that follows (make_root, io/root.hpp) has its own stages in io/reader/root_steps.hpp.

#include "io/reader/chunk.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// Items keep ids 1..num_items, and the widest item width is 32 bits.
constexpr std::size_t kMaxItems = 0xFFFFFFFEu;

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
std::vector<std::uint32_t> rank_items(const NameTable& names, std::int64_t min_util, std::vector<std::string>& rename);
// Renamed transactions per chunk, items ascending, and each chunk's TWU of every kept item.
void rename_transactions(std::vector<Chunk>& chunks, const std::vector<std::uint32_t>& new_id, std::size_t num_items);
// Throws if a utility is negative or a sum of utilities overflows 64 bits; returns the largest TWU of a kept
// item over the renamed transactions.
std::uint64_t check_utilities(const std::vector<Chunk>& chunks, std::size_t num_items);
