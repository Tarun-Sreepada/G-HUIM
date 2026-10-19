#include "io/reader/steps.hpp"

#include "sys/parallel.hpp"

void find_primary(Level& root, std::size_t num_items, Utility min_util_, int threads) {
	std::vector<Utility> subtree(num_items + 1, 0);
	{
		const int parts = root.txs.size() < 4096 ? 1 : threads;
		std::vector<std::vector<Utility>> partial(static_cast<std::size_t>(parts), std::vector<Utility>(num_items + 1, 0));
		parallel_run(parts, [&](int p) {
			const std::size_t lo = root.txs.size() / parts * p, hi = p + 1 == parts ? root.txs.size() : root.txs.size() / parts * (p + 1);
			for (std::size_t ti = lo; ti < hi; ++ti) {
				const auto& t = root.txs[ti];
				Utility remaining = 0;
				for (std::uint32_t i = 0; i < t.len; ++i) remaining += t.utils[i];
				for (std::uint32_t i = 0; i < t.len; ++i) {
					partial[p][t.items[i]] += remaining;
					remaining -= t.utils[i];
				}
			}
		});
		for (const auto& v : partial)
			for (std::size_t x = 0; x < subtree.size(); ++x) subtree[x] += v[x];
	}
	for (std::size_t x = 1; x <= num_items; ++x)
		if (subtree[x] >= min_util_) root.primary.push_back(static_cast<Item>(x));
}
