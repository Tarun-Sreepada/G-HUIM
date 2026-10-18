#include "mine/worker.hpp"

#include <algorithm>
#include <chrono>
#include <string_view>

namespace {
constexpr std::uint32_t kPrefetchTx = 12, kPrefetchData = 5;
}  // namespace

Worker::Worker(int id, std::size_t num_items, Utility min_util, int threads, Scheduler& scheduler, bool collect_patterns,
			   int refine, bool compact, double refine_min_drop, bool leaf)
	: id(id), min_util(min_util), threads(threads), scheduler(scheduler), collect_patterns(collect_patterns), refine(refine),
	  compact(compact), refine_min_drop(refine_min_drop), leaf(leaf),
	  mark(num_items, -1), bound(num_items),
	  slot(num_items, -1), filter(filter_kernel()),
	  fused(std::string_view(simd_name()) == "scalar") {}

void Worker::run() {
	Task task;
	while (scheduler.pop(id, task)) {
		const auto start = std::chrono::steady_clock::now();
		execute(task);
		busy_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
		task = {};  // release the database before reporting completion
		scheduler.done();
	}
}

void Worker::execute(const Task& task) {
	if (collect_patterns) prefix_items = task.prefix;
	if (task.depth > 0)
		for (const Item x : task.level->secondary) mark[x] = task.depth;
	expand(*task.level, task.depth, task.slot);
	if (task.depth > 0)
		for (const Item x : task.level->secondary) mark[x] = -1;
}

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

void Worker::compact_child(Level& child, int depth) {
	while (spare.size() <= static_cast<std::size_t>(depth)) spare.push_back(std::make_unique<Level>());
	Level& out = *spare[depth];
	out.clear();
	++stats.compacted;
	stats.compact_tx_in += child.txs.size();
	for (const auto& t : child.txs) {
		stats.compact_items_in += t.len;
		if (new_key.size() < t.len) {
			new_key.resize(2 * t.len);
			new_val.resize(2 * t.len);
		}
		std::uint32_t kept = 0;
		for (std::uint32_t i = 0; i < t.len; ++i) {
			if (mark[t.items[i]] != depth) continue;
			new_key[kept] = t.items[i];
			new_val[kept++] = t.utils[i];
		}
		if (kept == 0) continue;
		// An unchanged transaction stays a view of the old storage (an ancestor's, or child's own,
		// which moves to spare[depth] below and so outlives it).
		if (kept == t.len)
			out.add(t.items, t.utils, kept, t.prefix, true);
		else
			out.add(new_key.data(), new_val.data(), kept, t.prefix, false);
	}
	out.finalize();
	stats.compact_tx_out += out.txs.size();
	for (const auto& t : out.txs) stats.compact_items_out += t.len;
	std::swap(child, out);
}

Level& Worker::level_at(std::size_t depth) {
	while (levels.size() <= depth) levels.push_back(std::make_unique<Level>());
	return *levels[depth];
}

void Worker::expand(const Level& db, int depth, std::size_t k) {
	const Item item = db.primary[k];
	// Leaf: no extension of the prefix comes after `item` (the root keeps every item, the largest
	// id last), so the projection would be empty and only the itemset's utility is needed.
	const Item top = depth == 0 ? static_cast<Item>(bound.size() - 1) : db.secondary.back();
	if (leaf && item >= top) {
		Utility utility = 0;
		for (std::uint32_t o = db.occ_begin[k]; o < db.occ_begin[k + 1]; ++o) {
			const Transaction& t = db.txs[db.occ[o].tid];
			utility += t.prefix + t.utils[db.occ[o].pos];
		}
		++stats.leaves;
		if (utility >= min_util) {
			++pattern_count;
			if (collect_patterns) {
				patterns.items.insert(patterns.items.end(), prefix_items.begin(), prefix_items.end());
				patterns.items.push_back(item);
				patterns.begin.push_back(patterns.items.size());
				patterns.utils.push_back(utility);
			}
		}
		return;
	}
	Level& child = level_at(depth + 1);
	child.clear();
	Utility utility = 0;
	++stats.nodes;

	const std::uint32_t occ_end = db.occ_begin[k + 1];
	for (std::uint32_t o = db.occ_begin[k]; o < occ_end; ++o) {
		// Occurrences point at scattered transactions: pull the descriptor a few steps ahead and
		// the item/utility data (whose address needs the descriptor) a little closer.
		if (o + kPrefetchTx < occ_end) __builtin_prefetch(&db.txs[db.occ[o + kPrefetchTx].tid], 0, 3);
		if (o + kPrefetchData < occ_end) {
			const Occurrence& next = db.occ[o + kPrefetchData];
			const Transaction& nt = db.txs[next.tid];
			__builtin_prefetch(nt.items + next.pos, 0, 3);
			__builtin_prefetch(nt.utils + next.pos, 0, 3);
		}
		const Transaction& t = db.txs[db.occ[o].tid];
		const std::uint32_t pos = db.occ[o].pos;
		const Utility curr = t.utils[pos] + t.prefix;
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
			Utility suffix = 0;
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
			const Utility base = suffix + curr;
			for (std::uint32_t i = 0; i < kept; ++i) {
				const Item x = kept_items[i];
				if (b[x].local == 0) touched.push_back(x);
				b[x].local += base;
			}
		} else {
			// Items that are no longer extensions (not secondary at this depth) are dropped.
			Utility s = 0;
			if (depth > 0) {
				if (new_key.size() < n + kSimdSlack) {
					new_key.resize(2 * n + kSimdSlack);
					new_val.resize(2 * n + kSimdSlack);
				}
				kept = filter(items, utils, n, mark.data(), depth, new_key.data(), new_val.data(), &s);
				if (kept == 0) continue;
				if (kept != n) {
					kept_items = new_key.data();
					kept_utils = new_val.data();
				}
			} else {
				for (std::uint32_t i = 0; i < n; ++i) s += utils[i];
			}
			const Utility base = s + curr;
			Utility temp = 0;
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

	if (collect_patterns) prefix_items.push_back(item);
	if (utility >= min_util) {
		++pattern_count;
		if (collect_patterns) {
			patterns.items.insert(patterns.items.end(), prefix_items.begin(), prefix_items.end());
			patterns.begin.push_back(patterns.items.size());
			patterns.utils.push_back(utility);
		}
	}

	select_extensions(child, depth);

	if (!child.primary.empty()) {
		child.index(slot);

		const int next = depth + 1;
		const std::uint64_t work = std::uint64_t{child.occ.size()} * child.primary.size();
		const bool split = threads > 1 && ((next <= kMaxSplitDepth && work >= kSplitWork) ||
											(next <= kMaxHungryDepth && work >= kHungryWork && scheduler.hungry()));
		if (split) {
			std::shared_ptr<const Level> shared = child.materialize();
			std::vector<Task> tasks;
			tasks.reserve(child.primary.size());
			for (std::uint32_t c = 0; c < child.primary.size(); ++c)
				tasks.push_back({shared, next, c, prefix_items});
			scheduler.push(id, std::move(tasks));
		} else {
			for (const Item x : child.secondary) mark[x] = next;
			for (std::size_t c = 0; c < child.primary.size(); ++c) {
				// An idle worker exists: hand it the siblings not started yet, at any depth and size,
				// so one deep subtree of many small databases cannot keep a single thread busy.
				// Measured on BMS: a size threshold here left most workers idle.
				const std::size_t left = child.primary.size() - c;
				if (threads > 1 && left > 1 && scheduler.hungry()) {
					std::shared_ptr<const Level> shared = child.materialize();
					std::vector<Task> tasks;
					tasks.reserve(left - 1);
					for (std::size_t r = c + 1; r < child.primary.size(); ++r)
						tasks.push_back({shared, next, static_cast<std::uint32_t>(r), prefix_items});
					scheduler.push(id, std::move(tasks));
					expand(child, next, c);
					break;
				}
				expand(child, next, c);
			}
			for (const Item x : child.secondary) mark[x] = depth > 0 ? depth : -1;
		}
	}
	if (collect_patterns) prefix_items.pop_back();
}
