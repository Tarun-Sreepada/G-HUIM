#include "mine/worker.hpp"

#include <algorithm>

void Worker::select_extensions(Level& child, int depth) {
	const int next = depth + 1, restore = depth > 0 ? depth : -1;
	// Candidates for child.secondary with their subtree bound. `dropped`: some item of the child's
	// transactions was ruled out, so their bounds still count utility no extension can use.
	chosen.clear();
	bool dropped = false;
	const std::size_t candidates = touched.size();
	for (const Item x : touched) {
		const Bound bx = bound[x];
		bound[x] = {};
		if (bx.local < min_util) {
			dropped = true;
			continue;
		}
		chosen.push_back({x, bx.subtree});
	}
	touched.clear();

	std::uint64_t first_primary = 0;
	for (const auto& [x, su] : chosen) first_primary += su >= min_util;
	Bound* __restrict b = bound.data();
	int rounds = 0;
	// A pass that dropped few items rarely makes a rescan pay for itself.
	std::size_t last_drop = candidates - chosen.size();
	const auto worth = [&] { return static_cast<double>(last_drop) >= refine_min_drop * static_cast<double>(chosen.size() + last_drop); };
	while (dropped && !chosen.empty() && (refine < 0 || rounds < refine) && worth()) {
		++rounds;
		for (const auto& [x, su] : chosen) mark[x] = next;
		for (const auto& t : child.txs) {
			Utility suffix = 0;
			for (std::uint32_t i = t.len; i-- > 0;) {
				const Item x = t.items[i];
				if (mark[x] != next) continue;
				suffix += t.utils[i];
				b[x].subtree += t.prefix + suffix;
			}
			const Utility base = t.prefix + suffix;
			for (std::uint32_t i = 0; i < t.len; ++i)
				if (mark[t.items[i]] == next) b[t.items[i]].local += base;
		}
		for (const auto& [x, su] : chosen) mark[x] = restore;
		dropped = false;
		std::size_t w = 0;
		for (const auto& [x, su] : chosen) {
			const Bound bx = b[x];
			b[x] = {};
			if (bx.local < min_util) {
				++stats.refine_secondary;
				dropped = true;
				continue;
			}
			chosen[w++] = {x, bx.subtree};
		}
		last_drop = chosen.size() - w;
		chosen.resize(w);
	}
	stats.refine_rounds += rounds;

	std::sort(chosen.begin(), chosen.end());
	if (compact && chosen.size() < candidates && !chosen.empty()) {
		for (const auto& [x, su] : chosen) mark[x] = next;
		compact_child(child, next);
		for (const auto& [x, su] : chosen) mark[x] = restore;
	}
	for (const auto& [x, su] : chosen) {
		child.secondary.push_back(x);
		if (su >= min_util) child.primary.push_back(x);
	}
	stats.refine_primary += first_primary - child.primary.size();
	stats.primary += child.primary.size();
	stats.secondary += child.secondary.size();
}
