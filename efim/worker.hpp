#pragma once
// Per-thread search state and the depth-first expansion of one subtree.

#include "scheduler.hpp"

#include <memory>
#include <vector>

// Per-thread search state. Scratch arrays are indexed by item id and reset
// after each projection via the `touched` list.
struct Worker {
	Worker(std::size_t num_items, Utility min_util, int threads, Scheduler& scheduler);

	void run();

	Patterns patterns;

private:
	void execute(const Task& task);
	Level& level_at(std::size_t depth);
	// Projects `db` (the database at `depth`) on db.primary[k], records the pattern
	// and recurses or hands the subtree to other threads.
	void expand(const Level& db, int depth, std::size_t k);

	Utility min_util;
	int threads;
	Scheduler& scheduler;
	// mark[x] = deepest depth on the current path at which x is secondary, -1 if none.
	// The root (depth 0) keeps every item, so it is never marked.
	std::vector<int> mark;
	std::vector<Utility> local, subtree;
	std::vector<char> seen;
	std::vector<std::int32_t> slot;
	std::vector<Item> touched, prefix_items, new_key;
	std::vector<Utility> new_val;
	std::vector<std::unique_ptr<Level>> levels;  // levels[d] = reusable database at depth d
};
