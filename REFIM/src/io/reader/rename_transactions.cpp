#include "io/reader/steps.hpp"

#include "sys/parallel.hpp"

#include <algorithm>
#include <limits>

void rename_transactions(std::vector<Chunk>& chunks, const std::vector<Item>& str_to_int, std::size_t num_items) {
	parallel_run(static_cast<int>(chunks.size()), [&](int ci) {
		Chunk& c = chunks[ci];
		std::vector<std::pair<Item, Util>> transaction;
		c.tx_begin.assign(1, 0);
		c.item_sum.assign(num_items + 1, 0);
		for (std::size_t t = 0; t + 1 < c.raw_begin.size(); ++t) {
			transaction.clear();
			for (std::size_t i = c.raw_begin[t]; i < c.raw_begin[t + 1]; ++i)
				if (const Item x = str_to_int[c.remap[c.raw_names[i]]]) {
					const Utility u = c.raw_utils[i];
					if (u < 0 || u > std::numeric_limits<Util>::max()) c.bad_util = true;
					c.item_sum[x] += u;
					transaction.emplace_back(x, static_cast<Util>(u));
				}
			if (transaction.empty()) continue;
			std::sort(transaction.begin(), transaction.end(),
					  [](const auto& a, const auto& b) { return a.first < b.first; });
			// An item listed twice in one transaction (kosarak) becomes one entry with the summed utility.
			const std::size_t first = c.keys.size();
			for (const auto& [x, u] : transaction) {
				if (c.keys.size() > first && c.keys.back() == x) {
					const Utility sum = Utility{c.vals.back()} + u;
					if (sum > std::numeric_limits<Util>::max()) c.bad_util = true;
					c.vals.back() = static_cast<Util>(sum);
					continue;
				}
				c.keys.push_back(x);
				c.vals.push_back(u);
			}
			c.tx_begin.push_back(c.keys.size());
		}
		std::vector<std::int32_t>().swap(c.raw_names);
		std::vector<Utility>().swap(c.raw_utils);
	});
}
