#include "io/reader/steps.hpp"

#include <algorithm>
#include <stdexcept>

std::uint64_t check_utilities(const std::vector<Chunk>& chunks, std::size_t num_items) {
	std::vector<std::uint64_t> twu(num_items + 1, 0);
	bool overflow = false;
	for (const Chunk& c : chunks) {
		if (c.bad_util) throw std::runtime_error("a utility is negative");
		overflow |= c.overflow;
		for (std::size_t x = 1; x <= num_items; ++x) overflow |= __builtin_add_overflow(twu[x], c.twu[x], &twu[x]);
	}
	if (overflow) throw std::runtime_error("a transaction-weighted utility overflows 64 bits");
	return *std::max_element(twu.begin(), twu.end());
}
