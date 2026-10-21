#include "io/reader/steps.hpp"

#include "sys/parallel.hpp"

#include <algorithm>

void rename_transactions(std::vector<Chunk>& chunks, const std::vector<std::uint32_t>& str_to_int, std::size_t num_items) {
	parallel_run(static_cast<int>(chunks.size()), [&](int ci) {
		Chunk& c = chunks[ci];
		std::vector<std::pair<std::uint32_t, std::uint64_t>> transaction;
		c.tx_begin.assign(1, 0);
		c.twu.assign(num_items + 1, 0);
		c.max_length = 0;
		for (std::size_t t = 0; t + 1 < c.raw_begin.size(); ++t) {
			transaction.clear();
			for (std::size_t i = c.raw_begin[t]; i < c.raw_begin[t + 1]; ++i)
				if (const std::uint32_t x = str_to_int[c.remap[c.raw_names[i]]]) {
					const std::int64_t u = c.raw_utils[i];
					if (u < 0) c.bad_util = true;
					transaction.emplace_back(x, static_cast<std::uint64_t>(u));
				}
			if (transaction.empty()) continue;
			std::sort(transaction.begin(), transaction.end(),
					  [](const auto& a, const auto& b) { return a.first < b.first; });
			// An item listed twice in one transaction (kosarak) becomes one entry with the summed utility.
			const std::size_t first = c.keys.size();
			std::uint64_t utility = 0;
			for (const auto& [x, u] : transaction) {
				c.overflow |= __builtin_add_overflow(utility, u, &utility);
				if (c.keys.size() > first && c.keys.back() == x) {
					c.vals.back() += u;  // at most `utility`, which did not overflow
					continue;
				}
				c.keys.push_back(x);
				c.vals.push_back(u);
			}
			for (std::size_t i = first; i < c.keys.size(); ++i)
				c.overflow |= __builtin_add_overflow(c.twu[c.keys[i]], utility, &c.twu[c.keys[i]]);
			c.max_length = std::max(c.max_length, static_cast<std::uint32_t>(c.keys.size() - first));
			c.tx_begin.push_back(c.keys.size());
		}
		std::vector<std::int32_t>().swap(c.raw_names);
		std::vector<std::int64_t>().swap(c.raw_utils);
	});
}
