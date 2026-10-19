#pragma once
// Interning table for item names (open addressing, string_views into the input buffer).

#include "mine/level.hpp"

#include <cstdint>
#include <string_view>
#include <vector>

struct NameTable {
	std::vector<std::int32_t> slots = std::vector<std::int32_t>(1024, -1);
	std::vector<std::string_view> names;  // first-appearance order
	std::vector<std::uint64_t> hashes;
	std::vector<Utility> twu;

	static std::uint64_t hash(std::string_view s);
	std::int32_t intern(std::string_view s);
	void grow();
};
