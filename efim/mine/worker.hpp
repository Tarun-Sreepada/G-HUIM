#pragma once
// Per-thread search state and the depth-first expansion of one subtree.

#include "mine/scheduler.hpp"
#include "mine/simd.hpp"

#include <memory>
#include <vector>

// Per-thread search state. Scratch arrays are indexed by item id and reset
// after each projection via the `touched` list.
struct Worker {
	Worker(int id, std::size_t num_items, Utility min_util, int threads, Scheduler& scheduler, bool collect_patterns);

	void run();

	Patterns patterns;
	std::size_t pattern_count = 0;
	double busy_seconds = 0;  // time spent executing tasks (vs. waiting for one)

private:
	void execute(const Task& task);
	Level& level_at(std::size_t depth);
	// Projects `db` (the database at `depth`) on db.primary[k], records the pattern
	// and recurses or hands the subtree to other threads.
	void expand(const Level& db, int depth, std::size_t k);

	int id;
	Utility min_util;
	int threads;
	Scheduler& scheduler;
	const bool collect_patterns;
	// mark[x] = deepest depth on the current path at which x is secondary, -1 if none.
	// The root (depth 0) keeps every item, so it is never marked.
	std::vector<int> mark;
	std::vector<Utility> local, subtree;
	std::vector<char> seen;
	std::vector<std::int32_t> slot;
	std::vector<Item> touched, prefix_items, new_key;
	std::vector<Util> new_val;
	FilterFn filter;  // projection kernel (AVX-512 or scalar)
	std::vector<std::unique_ptr<Level>> levels;  // levels[d] = reusable database at depth d
};
