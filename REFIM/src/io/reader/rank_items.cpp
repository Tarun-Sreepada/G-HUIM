#include "io/reader/steps.hpp"

#include <algorithm>
#include <spdlog/spdlog.h>
#include <stdexcept>

std::vector<std::uint32_t> rank_items(const NameTable& global, std::int64_t min_util_, std::vector<std::string>& rename_) {
	const auto& names = global.names;
	const auto& twu = global.twu;

	// Keep promising items, sorted by TWU descending (stable, like Python's sorted).
	std::vector<std::int32_t> order;
	for (std::int32_t i = 0; i < static_cast<std::int32_t>(names.size()); ++i)
		if (twu[i] >= min_util_) order.push_back(i);
	std::stable_sort(order.begin(), order.end(), [&](std::int32_t a, std::int32_t b) { return twu[a] > twu[b]; });

	if (order.size() > kMaxItems)
		throw std::runtime_error(std::to_string(order.size()) + " items survive TWU pruning at minutil " +
								 std::to_string(min_util_) + "; items are at most 32 bits (at most " +
								 std::to_string(kMaxItems) + "). Raise minutil.");
	const std::size_t num_items = order.size();
	std::vector<std::uint32_t> str_to_int(names.size(), 0);  // 0 = pruned
	spdlog::debug("reader: {} distinct items, {} kept", names.size(), num_items);
	rename_.assign(num_items + 1, "");
	std::size_t next_id = num_items;
	for (const std::int32_t i : order) {
		str_to_int[i] = static_cast<std::uint32_t>(next_id);
		rename_[next_id] = std::string(names[i]);
		--next_id;
	}
	return str_to_int;
}
