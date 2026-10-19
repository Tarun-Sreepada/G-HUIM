#pragma once

#include "io/reader/name_table.hpp"
#include "mine/level.hpp"

#include <cstddef>
#include <string>
#include <vector>

// One thread's share of the input: a block of whole lines.
struct Chunk {
	std::size_t begin = 0, end = 0;  // byte range in the input
	NameTable table;
	std::vector<std::int32_t> raw_names;  // transactions back to back, as indices into table.names
	std::vector<Utility> raw_utils;
	std::vector<std::size_t> raw_begin{0};
	std::vector<std::int32_t> remap;  // local name id -> global name id
	// Renamed transactions (items ascending), pruned items removed.
	std::vector<Item> keys;
	std::vector<Util> vals;
	std::vector<Utility> item_sum;  // per renamed item: sum of |utility| over this chunk (overflow check)
	std::vector<std::size_t> tx_begin;  // tx_begin[t]..tx_begin[t + 1]; tx_begin[0] = 0
	bool failed = false;
	bool bad_util = false;
	std::size_t error_pos = 0;
	std::string error;
};
