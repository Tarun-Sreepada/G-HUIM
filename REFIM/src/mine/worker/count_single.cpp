#include "mine/worker.hpp"

#include <algorithm>
#include <functional>

inline namespace REFIM_WIDTHS {

namespace {

// Subsets of u[i, n) (sorted descending, rest[i] = their sum) whose sum reaches `need`.
std::uint64_t count_subsets(const Util* u, const std::uint64_t* rest, std::uint32_t n, std::uint32_t i, std::uint64_t need) {
	if (need == 0) return std::uint64_t{1} << (n - i);
	if (rest[i] < need) return 0;
	// rest[i] >= need > 0, so i < n.
	const std::uint64_t take = u[i] >= need ? std::uint64_t{1} << (n - i - 1) : count_subsets(u, rest, n, i + 1, need - u[i]);
	return take + count_subsets(u, rest, n, i + 1, need);
}

}  // namespace

std::uint64_t Worker::count_tail(Util prefix) {
	const std::uint32_t n = static_cast<std::uint32_t>(tail.size());
	std::sort(tail.begin(), tail.end(), std::greater<>());
	tail_rest.resize(n + 1);
	tail_rest[n] = 0;
	for (std::uint32_t i = n; i-- > 0;) tail_rest[i] = tail_rest[i + 1] + tail[i];
	const std::uint64_t need = prefix >= min_util ? 0 : min_util - prefix;
	// The empty subset is the prefix itself, already recorded.
	return count_subsets(tail.data(), tail_rest.data(), n, 0, need) - (need == 0);
}

bool Worker::count_full(const Level& child) {
	// Compaction leaves each transaction holding only secondary items, so one of full length holds all
	// of them and supports every extension.
	const std::size_t k = child.secondary.size();
	if (collect_patterns || !compact || k == 0 || k >= 64) return false;
	Util lower = 0;
	bool all = true;
	for (const auto& t : child.txs) {
		if (t.len == k)
			lower += t.prefix;
		else
			all = false;
	}
	std::uint64_t found;
	if (lower >= min_util) {
		// Each extension's utility is at least the prefix utility in the transactions holding all of them.
		found = (std::uint64_t{1} << k) - 1;
		++stats.full_lower;
	} else if (all) {
		// Every transaction holds every item, so utilities add up per item as in one merged transaction.
		tail.assign(k, 0);
		for (const auto& t : child.txs)
			for (std::uint32_t i = 0; i < k; ++i) tail[i] += t.utils[i];
		found = count_tail(lower);
		++stats.full_exact;
	} else {
		return false;
	}
	stats.full_patterns += found;
	pattern_count += found;
	return true;
}

bool Worker::count_single(Level& child) {
	if (collect_patterns || child.txs.size() != 1 || child.txs[0].len >= 64) return false;
	// Every extension is supported by this one transaction alone, so u(prefix + Y) = prefix + u(Y):
	// count the nonempty subsets Y reaching min_util instead of searching them.
	for (const Item x : touched) bound[x] = {};
	touched.clear();
	const Transaction& t = child.txs[0];
	const std::uint32_t n = t.len;
	tail.assign(t.utils, t.utils + n);
	const std::uint64_t found = count_tail(t.prefix);
	++stats.single;
	stats.single_patterns += found;
	pattern_count += found;
	return true;
}

}  // namespace REFIM_WIDTHS
