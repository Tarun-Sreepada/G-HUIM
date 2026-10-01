#include "worker.hpp"

#include <algorithm>

Worker::Worker(std::size_t num_items, Utility min_util, int threads, Scheduler& scheduler)
	: min_util(min_util), threads(threads), scheduler(scheduler), mark(num_items, -1), local(num_items, 0),
	  subtree(num_items, 0), seen(num_items, 0), slot(num_items, -1) {}

void Worker::run() {
	Task task;
	while (scheduler.pop(task)) {
		execute(task);
		task = {};  // release the database before reporting completion
		scheduler.done();
	}
}

void Worker::execute(const Task& task) {
	prefix_items = task.prefix;
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

	for (std::uint32_t o = db.occ_begin[k]; o < db.occ_begin[k + 1]; ++o) {
		const Transaction& t = db.txs[db.occ[o].tid];
		const std::uint32_t pos = db.occ[o].pos;
		const Utility curr = t.utils[pos] + t.prefix;
		utility += curr;
		if (pos + 1 == t.len) continue;

		const Item* items = t.items + pos + 1;
		const Utility* utils = t.utils + pos + 1;
		const std::uint32_t n = t.len - pos - 1;
		Utility s = 0;
		std::uint32_t kept = 0;
		for (std::uint32_t i = 0; i < n; ++i) {
			if (depth > 0 && mark[items[i]] < depth) continue;
			s += utils[i];
			++kept;
		}
		if (kept == 0) continue;

		const bool view = kept == n;
		if (!view) {
			new_key.clear();
			new_val.clear();
		}
		Utility temp = 0;
		for (std::uint32_t i = 0; i < n; ++i) {
			const Item x = items[i];
			if (depth > 0 && mark[x] < depth) continue;
			if (!seen[x]) {
				seen[x] = 1;
				touched.push_back(x);
			}
			local[x] += s + curr;
			subtree[x] += s + curr - temp;
			temp += utils[i];
			if (!view) {
				new_key.push_back(x);
				new_val.push_back(utils[i]);
			}
		}
		if (view)
			child.add(items, utils, n, curr, true);
		else
			child.add(new_key.data(), new_val.data(), kept, curr, false);
	}
	child.finalize();

	prefix_items.push_back(item);
	if (utility >= min_util) {
		patterns.items.insert(patterns.items.end(), prefix_items.begin(), prefix_items.end());
		patterns.begin.push_back(patterns.items.size());
		patterns.utils.push_back(utility);
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
		if (threads > 1 && next <= kMaxSplitDepth && work >= kSplitWork) {
			std::shared_ptr<const Level> shared = child.materialize();
			std::vector<Task> tasks;
			tasks.reserve(child.primary.size());
			for (std::uint32_t c = 0; c < child.primary.size(); ++c)
				tasks.push_back({shared, next, c, prefix_items});
			scheduler.push(std::move(tasks));
		} else {
			for (const Item x : child.secondary) mark[x] = next;
			for (std::size_t c = 0; c < child.primary.size(); ++c) expand(child, next, c);
			for (const Item x : child.secondary) mark[x] = depth > 0 ? depth : -1;
		}
	}
	prefix_items.pop_back();
}
