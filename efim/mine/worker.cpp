#include "mine/worker.hpp"

#include <algorithm>

namespace {
constexpr std::uint32_t kPrefetchTx = 12, kPrefetchData = 5;
}  // namespace

Worker::Worker(int id, std::size_t num_items, Utility min_util, int threads, Scheduler& scheduler, bool collect_patterns)
	: id(id), min_util(min_util), threads(threads), scheduler(scheduler), collect_patterns(collect_patterns), mark(num_items, -1), local(num_items, 0),
	  subtree(num_items, 0), seen(num_items, 0), slot(num_items, -1), filter(filter_kernel()) {}

void Worker::run() {
	Task task;
	while (scheduler.pop(id, task)) {
		execute(task);
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

Level& Worker::level_at(std::size_t depth) {
	while (levels.size() <= depth) levels.push_back(std::make_unique<Level>());
	return *levels[depth];
}

void Worker::expand(const Level& db, int depth, std::size_t k) {
	const Item item = db.primary[k];
	Level& child = level_at(depth + 1);
	child.clear();
	Utility utility = 0;

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
		// Items that are no longer extensions (not secondary at this depth) are dropped.
		const Item* kept_items = items;
		const Util* kept_utils = utils;
		std::uint32_t kept = n;
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
		const bool view = kept == n;

		const Utility base = s + curr;
		Utility temp = 0;
		for (std::uint32_t i = 0; i < kept; ++i) {
			const Item x = kept_items[i];
			if (!seen[x]) {
				seen[x] = 1;
				touched.push_back(x);
			}
			local[x] += base;
			subtree[x] += base - temp;
			temp += kept_utils[i];
		}
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

	for (const Item x : touched) {
		if (subtree[x] >= min_util) child.primary.push_back(x);
		if (local[x] >= min_util) child.secondary.push_back(x);
		local[x] = subtree[x] = 0;
		seen[x] = 0;
	}
	touched.clear();

	if (!child.primary.empty()) {
		std::sort(child.primary.begin(), child.primary.end());
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
