#include "io/reader/steps.hpp"

NameTable merge_names(std::vector<Chunk>& chunks) {
	NameTable global;
	for (Chunk& c : chunks) {
		c.remap.resize(c.table.names.size());
		for (std::size_t l = 0; l < c.table.names.size(); ++l) {
			const std::int32_t g = global.intern(c.table.names[l]);
			global.twu[static_cast<std::size_t>(g)] += c.table.twu[l];
			c.remap[l] = g;
		}
	}
	return global;
}
