#include "mine/worker.hpp"

inline namespace REFIM_WIDTHS {

namespace {
constexpr std::uint32_t kPrefetchTx = 12, kPrefetchData = 5;
}  // namespace

Util Worker::project_child(const Level& db, int depth, std::size_t k, Level& child) {
	child.clear();
	Util utility = 0;
	++stats.nodes;
	if (keep_table && depth > 0) {
		const int* m = mark.data();
		for (std::size_t x = 0; x < mark.size(); ++x) keep[x] = m[x] >= depth ? 0xFF : 0;
	}

	const std::uint32_t occ_end = db.occ_begin[k + 1], shift = db.occ_shift;
	for (std::uint32_t o = db.occ_begin[k]; o < occ_end; ++o) {
		// Occurrences point at scattered transactions: pull the descriptor a few steps ahead and
		// the item/utility data (whose address needs the descriptor) a little closer.
		if (o + kPrefetchTx < occ_end) __builtin_prefetch(&db.txs[occurrence_tid(db.occ[o + kPrefetchTx], shift)], 0, 3);
		if (o + kPrefetchData < occ_end) {
			const Occurrence next = db.occ[o + kPrefetchData];
			const Transaction& nt = db.txs[occurrence_tid(next, shift)];
			__builtin_prefetch(nt.items + occurrence_pos(next, shift), 0, 3);
			__builtin_prefetch(nt.utils + occurrence_pos(next, shift), 0, 3);
		}
		const Transaction& t = db.txs[occurrence_tid(db.occ[o], shift)];
		const std::uint32_t pos = occurrence_pos(db.occ[o], shift);
		const Util curr = t.utils[pos] + t.prefix;
		utility += curr;
		if (pos + 1 == t.len) continue;

		const Item* items = t.items + pos + 1;
		const Util* utils = t.utils + pos + 1;
		const std::uint32_t n = t.len - pos - 1;
		const Item* kept_items = items;
		const Util* kept_utils = utils;
		std::uint32_t kept = n;
		Bound* __restrict b = bound.data();
		if (fused) {
			// Backward pass: drop items that are no longer extensions, pack the survivors at the
			// end of the scratch buffers and add each one's suffix utility to its subtree bound
			// (subtree[x] += curr + utility of x and everything after it).
			if (new_key.size() < n) {
				new_key.resize(2 * n);
				new_val.resize(2 * n);
			}
			Item* const out_items = new_key.data();
			Util* const out_utils = new_val.data();
			const bool filtering = depth > 0;
			std::uint32_t at = n;
			Util suffix = 0;
			for (std::uint32_t i = n; i-- > 0;) {
				const Item x = items[i];
				if (filtering && mark[x] < depth) continue;
				const Util u = utils[i];
				suffix += u;
				--at;
				out_items[at] = x;
				out_utils[at] = u;
				b[x].subtree += curr + suffix;
			}
			kept = n - at;
			if (kept == 0) continue;
			if (kept != n) {
				kept_items = out_items + at;
				kept_utils = out_utils + at;
			}
			const Util base = suffix + curr;
			for (std::uint32_t i = 0; i < kept; ++i) {
				const Item x = kept_items[i];
				if (b[x].local == 0) touched.push_back(x);
				b[x].local += base;
			}
		} else {
			// Items that are no longer extensions (not secondary at this depth) are dropped.
			Util s = 0;
			if (depth > 0) {
				if (new_key.size() < n + kSimdSlack) {
					new_key.resize(2 * n + kSimdSlack);
					new_val.resize(2 * n + kSimdSlack);
				}
				kept = filter(items, utils, n, mark.data(), depth, keep, new_key.data(), new_val.data(), &s);
				if (kept == 0) continue;
				if (kept != n) {
					kept_items = new_key.data();
					kept_utils = new_val.data();
				}
			} else {
				for (std::uint32_t i = 0; i < n; ++i) s += utils[i];
			}
			const Util base = s + curr;
			Util temp = 0;
			for (std::uint32_t i = 0; i < kept; ++i) {
				const Item x = kept_items[i];
				if (b[x].local == 0) touched.push_back(x);
				b[x].local += base;
				b[x].subtree += base - temp;
				temp += kept_utils[i];
			}
		}
		const bool view = kept == n;

		if (view)
			child.add(items, utils, n, curr, true);
		else
			child.add(kept_items, kept_utils, kept, curr, false);
	}
	child.finalize();
	return utility;
}

}  // namespace REFIM_WIDTHS
