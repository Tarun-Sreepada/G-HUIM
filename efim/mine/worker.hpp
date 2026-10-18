#pragma once
// Per-thread search state and the depth-first expansion of one subtree.

#include "mine/scheduler.hpp"
#include "mine/simd.hpp"

#include <memory>
#include <vector>

// Search counters, summed over workers for --stats.
struct SearchStats {
	std::uint64_t nodes = 0;            // projections
	std::uint64_t primary = 0;          // extensions recursed into
	std::uint64_t secondary = 0;        // items kept for deeper projections
	std::uint64_t refine_rounds = 0;    // rescans of a child
	std::uint64_t refine_secondary = 0; // items dropped from secondary by refinement
	std::uint64_t refine_primary = 0;   // items dropped from primary by refinement

	SearchStats& operator+=(const SearchStats& o) {
		nodes += o.nodes;
		primary += o.primary;
		secondary += o.secondary;
		refine_rounds += o.refine_rounds;
		refine_secondary += o.refine_secondary;
		refine_primary += o.refine_primary;
		return *this;
	}
};

// Per-thread search state. Scratch arrays are indexed by item id and reset
// after each projection via the `touched` list.
struct Worker {
	Worker(int id, std::size_t num_items, Utility min_util, int threads, Scheduler& scheduler, bool collect_patterns,
		   int refine = -1);

	void run();

	Patterns patterns;
	std::size_t pattern_count = 0;
	double busy_seconds = 0;  // time spent executing tasks (vs. waiting for one)
	SearchStats stats;

private:
	void execute(const Task& task);
	Level& level_at(std::size_t depth);
	// Projects `db` (the database at `depth`) on db.primary[k], records the pattern
	// and recurses or hands the subtree to other threads.
	void expand(const Level& db, int depth, std::size_t k);
	// Chooses child.primary/secondary from the bounds of the `touched` items (resetting them),
	// after `refine` rounds of rescanning `child` (the database at depth + 1) without the items ruled out.
	void select_extensions(Level& child, int depth);

	int id;
	Utility min_util;
	int threads;
	Scheduler& scheduler;
	const bool collect_patterns;
	const int refine;  // rounds of recomputing a child's bounds without the items it rules out (-1 = until none is)
	std::vector<std::pair<Item, Utility>> chosen;  // scratch for select_extensions
	// mark[x] = deepest depth on the current path at which x is secondary, -1 if none.
	// The root (depth 0) keeps every item, so it is never marked.
	std::vector<int> mark;
	// Per-item bounds for the child being built, interleaved so an update touches one cache line.
	// local == 0 marks an item not yet touched (utilities are >= 0; a zero-utility item can enter
	// `touched` twice, which is harmless since its bounds stay 0 < minUtil).
	struct Bound {
		Utility local = 0, subtree = 0;
	};
	std::vector<Bound> bound;
	std::vector<std::int32_t> slot;
	std::vector<Item> touched, prefix_items, new_key;
	std::vector<Util> new_val;
	FilterFn filter;  // projection kernel (AVX-512 or scalar)
	bool fused;       // scalar mode: filter and bound updates in one backward pass instead of `filter`
	std::vector<std::unique_ptr<Level>> levels;  // levels[d] = reusable database at depth d
};
