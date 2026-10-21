#pragma once

#include "io/reader/name_table.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// One thread's share of the input: a block of whole lines.
struct Chunk {
	std::size_t begin = 0, end = 0;  // byte range in the input
	NameTable table;
	std::vector<std::int32_t> raw_names;  // transactions back to back, as indices into table.names
	std::vector<std::int64_t> raw_utils;
	std::vector<std::size_t> raw_begin{0};
	std::vector<std::int32_t> remap;  // local name id -> global name id
	// Renamed transactions (items ascending), pruned items removed, in the widest widths; the root build
	// narrows them to the widths the search runs with.
	std::vector<std::uint32_t> keys;
	std::vector<std::uint64_t> vals;
	std::vector<std::uint64_t> twu;  // per renamed item: sum of its transactions' utilities over this chunk
	std::vector<std::size_t> tx_begin;  // tx_begin[t]..tx_begin[t + 1]; tx_begin[0] = 0
	bool failed = false;
	bool bad_util = false;  // a negative utility
	bool overflow = false;  // a sum of utilities above 64 bits
	std::size_t error_pos = 0;
	std::string error;
};
