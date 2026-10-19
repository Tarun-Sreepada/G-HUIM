#include "io/reader/steps.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

void check_utilities(const std::vector<Chunk>& chunks, std::size_t num_items, const std::vector<std::string>& rename_) {
	std::vector<Utility> total(num_items + 1, 0);
	for (const Chunk& c : chunks) {
		if (c.bad_util) throw std::runtime_error("a utility is negative or does not fit in 64 bits");
		for (std::size_t x = 1; x <= num_items; ++x) total[x] += c.item_sum[x];
	}
	const auto worst = std::max_element(total.begin(), total.end());
	if (*worst > std::numeric_limits<Util>::max())
		throw std::runtime_error("item '" + rename_[worst - total.begin()] + "' has total utility " +
								 std::to_string(*worst) + " which overflows the 64-bit utility storage");
}
