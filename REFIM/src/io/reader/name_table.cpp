#include "io/reader/name_table.hpp"

std::uint64_t NameTable::hash(std::string_view s) {
	std::uint64_t h = 1469598103934665603ull;
	for (const unsigned char c : s) h = (h ^ c) * 1099511628211ull;
	return h ^ (h >> 29);
}

std::int32_t NameTable::intern(std::string_view s) {
	const std::uint64_t h = hash(s);
	std::size_t i = h & (slots.size() - 1);
	while (slots[i] >= 0) {
		const auto id = static_cast<std::size_t>(slots[i]);
		if (hashes[id] == h && names[id] == s) return slots[i];
		i = (i + 1) & (slots.size() - 1);
	}
	const auto id = static_cast<std::int32_t>(names.size());
	slots[i] = id;
	names.push_back(s);
	hashes.push_back(h);
	twu.push_back(0);
	if (names.size() * 2 > slots.size()) grow();
	return id;
}

void NameTable::grow() {
	slots.assign(slots.size() * 2, -1);
	for (std::size_t id = 0; id < names.size(); ++id) {
		std::size_t i = hashes[id] & (slots.size() - 1);
		while (slots[i] >= 0) i = (i + 1) & (slots.size() - 1);
		slots[i] = static_cast<std::int32_t>(id);
	}
}
