#include "mine/worker.hpp"

#include <algorithm>
#include <functional>

inline namespace REFIM_WIDTHS {

namespace {

// Subsets of u[0, n) (sorted descending, rest[i] = sum of u[i, n)) whose sum reaches `need`; n < 64.
// Iterative, so its speed does not hinge on how far the compiler inlines a recursion (which under LTO
// depends on the size of the whole binary). A frame (i, need) walks the subsets skipping u[i], u[i+1], ...
// and pushes a frame for each u[j] taken that leaves need > 0. Frames are pushed with increasing i on top
// of a smaller one, so the stack holds at most n frames.
std::uint64_t count_subsets(const Util* u, const std::uint64_t* rest, std::uint32_t n, std::uint64_t need) {
	if (need == 0) return std::uint64_t{1} << n;
	struct Frame {
		std::uint32_t i;
		std::uint64_t need;
	};
	Frame stack[64];
	int top = 0;
	stack[top++] = {0, need};
	std::uint64_t found = 0;
	while (top > 0) {
		auto [i, left] = stack[--top];
		// rest[i] >= left > 0, so i < n.
		for (; rest[i] >= left; ++i) {
			if (u[i] >= left)
				found += std::uint64_t{1} << (n - i - 1);
			else
				stack[top++] = {i + 1, left - u[i]};
		}
	}
	return found;
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
	return count_subsets(tail.data(), tail_rest.data(), n, need) - (need == 0);
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
