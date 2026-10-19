#include "io/reader/steps.hpp"

#include "sys/parallel.hpp"

#include <algorithm>
#include <iterator>
#include <numeric>

// Projection keeps this order, so identical projected transactions stay adjacent. Sorted in
// parallel slices, then merged pairwise.
std::vector<std::uint32_t> sort_transactions(const Flat& flat, int threads) {
	std::vector<std::uint32_t> perm(flat.num_tx);
	std::iota(perm.begin(), perm.end(), 0);
	const auto less = [&](std::uint32_t a, std::uint32_t b) {
		return std::lexicographical_compare(
			std::make_reverse_iterator(flat.keys.get() + flat.begin[a + 1]), std::make_reverse_iterator(flat.keys.get() + flat.begin[a]),
			std::make_reverse_iterator(flat.keys.get() + flat.begin[b + 1]), std::make_reverse_iterator(flat.keys.get() + flat.begin[b]));
	};
	const int parts = perm.size() < 4096 ? 1 : threads;
	std::vector<std::size_t> cut(static_cast<std::size_t>(parts) + 1);
	for (int i = 0; i <= parts; ++i) cut[i] = perm.size() / parts * i;
	cut[parts] = perm.size();
	const auto at = [&](int i) { return perm.begin() + static_cast<std::ptrdiff_t>(cut[i]); };
	parallel_run(parts, [&](int i) { std::sort(at(i), at(i + 1), less); });
	// Each round merges pairs of runs into `buf`. Every merge is split into slices of the
	// output by co-ranking, so all threads stay busy even in the last (single-pair) round.
	std::vector<std::uint32_t> buf(perm.size());
	std::uint32_t *src = perm.data(), *dst = buf.data();
	for (int width = 1; width < parts; width *= 2) {
		const int merges = (parts + 2 * width - 1) / (2 * width);
		const int slices = std::max(1, threads / merges);
		parallel_run(merges * slices, [&](int job) {
			const int m = job / slices, sl = job % slices;
			const int lo = m * 2 * width, mid = std::min(lo + width, parts), hi = std::min(lo + 2 * width, parts);
			const std::uint32_t *a = src + cut[lo], *b = src + cut[mid];
			const std::size_t na = cut[mid] - cut[lo], nb = cut[hi] - cut[mid], n = na + nb;
			// Smallest i with i + j = d such that the first i of `a` and first j of `b` are
			// the first d outputs of a stable merge (ties take `a` first).
			const auto corank = [&](std::size_t d) {
				std::size_t lo_i = d > nb ? d - nb : 0, hi_i = std::min(d, na);
				while (lo_i < hi_i) {
					const std::size_t i = (lo_i + hi_i) / 2, j = d - i - 1;
					if (less(b[j], a[i])) hi_i = i;
					else lo_i = i + 1;
				}
				return lo_i;
			};
			const std::size_t d0 = n * sl / slices, d1 = n * (sl + 1) / slices;
			const std::size_t i0 = corank(d0), i1 = corank(d1);
			std::merge(a + i0, a + i1, b + (d0 - i0), b + (d1 - i1), dst + cut[lo] + d0, less);
		});
		std::swap(src, dst);
	}
	if (src != perm.data()) std::copy(src, src + perm.size(), perm.data());
	return perm;
}
