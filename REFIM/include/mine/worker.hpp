#pragma once
// Per-thread search state and the depth-first expansion of one subtree.

#include "mine/scheduler.hpp"
#include "mine/simd.hpp"

#include <memory>
#include <vector>

inline namespace REFIM_WIDTHS {

// Search counters, summed over workers for --stats.
struct SearchStats {
	std::uint64_t nodes = 0;            // projections
	std::uint64_t primary = 0;          // extensions recursed into
	std::uint64_t secondary = 0;        // items kept for deeper projections
	std::uint64_t refine_rounds = 0;    // rescans of a child
	std::uint64_t refine_secondary = 0; // items dropped from secondary by refinement
	std::uint64_t refine_primary = 0;   // items dropped from primary by refinement
	std::uint64_t leaves = 0;           // extensions with nothing after them, summed without projecting
	std::uint64_t single = 0;           // children of one transaction, counted without searching
	std::uint64_t single_patterns = 0;  // patterns those counts found
	std::uint64_t full_lower = 0;       // children whose transactions holding every extension qualify them all
	std::uint64_t full_exact = 0;       // children whose transactions all hold every extension, counted exactly
	std::uint64_t full_patterns = 0;    // patterns those two found
	std::uint64_t compacted = 0;        // children rewritten without their ruled-out items
	std::uint64_t compact_tx_in = 0, compact_tx_out = 0;      // their transactions before/after
	std::uint64_t compact_items_in = 0, compact_items_out = 0;  // and item occurrences

	SearchStats& operator+=(const SearchStats& o) {
		nodes += o.nodes;
		primary += o.primary;
		secondary += o.secondary;
		refine_rounds += o.refine_rounds;
		refine_secondary += o.refine_secondary;
		refine_primary += o.refine_primary;
		leaves += o.leaves;
		single += o.single;
		single_patterns += o.single_patterns;
		full_lower += o.full_lower;
		full_exact += o.full_exact;
		full_patterns += o.full_patterns;
		compacted += o.compacted;
		compact_tx_in += o.compact_tx_in;
		compact_tx_out += o.compact_tx_out;
		compact_items_in += o.compact_items_in;
		compact_items_out += o.compact_items_out;
		return *this;
	}
};

// Per-thread search state. Scratch arrays are indexed by item id and reset
// after each projection via the `touched` list.
struct Worker {
	Worker(int id, std::size_t num_items, std::uint64_t min_util, int threads, Scheduler& scheduler, bool collect_patterns,
		   int refine = -1, bool compact = true, double refine_min_drop = 0, bool leaf = true);

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
	// expand(), step by step:
	// sums the utility of db.primary[k] from the occurrence index, without projecting.
	Util leaf_utility(const Level& db, std::size_t k) const;
	// fills `child` with the projection of db on db.primary[k] and bounds the extensions; returns the
	// utility of the itemset.
	Util project_child(const Level& db, int depth, std::size_t k, Level& child);
	// counts the itemset (and keeps it with --output) if its utility reaches min_util.
	void record_pattern(Util utility);
	// expands each primary item of `child`, or hands siblings to idle threads.
	void descend(Level& child, int depth);
	// Chooses child.primary/secondary from the bounds of the `touched` items (resetting them),
	// after `refine` rounds of rescanning `child` (the database at depth + 1) without the items ruled out,
	// then (with `compact`) rewrites child without those items.
	void select_extensions(Level& child, int depth);
	// Counts child's patterns directly when it holds a single transaction (and patterns are not kept);
	// returns false, leaving child untouched, otherwise.
	bool count_single(Level& child);
	// After select_extensions: counts child's whole subtree when the transactions holding every secondary
	// item already qualify all extensions, or when every transaction holds them all; false otherwise.
	bool count_full(const Level& child);
	// Nonempty subsets of `tail` (utilities, reordered) reaching min_util - prefix.
	std::uint64_t count_tail(Util prefix);
	// Replaces child's transactions by their items with mark[x] == depth, merging equal neighbours.
	void compact_child(Level& child, int depth);

	int id;
	std::uint64_t min_util;
	int threads;
	Scheduler& scheduler;
	const bool collect_patterns;
	const int refine;  // rounds of recomputing a child's bounds without the items it rules out (-1 = until none is)
	const bool compact;
	const double refine_min_drop;
	const bool leaf;  // sum leaf extensions from the occurrence index instead of projecting them
	std::vector<std::pair<Item, Util>> chosen;  // scratch for select_extensions
	// mark[x] = deepest depth on the current path at which x is secondary, -1 if none.
	// The root (depth 0) keeps every item, so it is never marked.
	std::vector<int> mark;
	// Per-item bounds for the child being built, interleaved so an update touches one cache line.
	// local == 0 marks an item not yet touched (utilities are >= 0; a zero-utility item can enter
	// `touched` twice, which is harmless since its bounds stay 0 < minUtil).
	struct Bound {
		Util local = 0, subtree = 0;
	};
	std::vector<Bound> bound;
	std::vector<std::int32_t> slot;
	std::vector<Item> touched, prefix_items, new_key;
	std::vector<Util> new_val, tail;
	std::vector<std::uint64_t> tail_rest;  // scratch for count_single
	FilterFn filter;  // projection kernel (AVX-512 VBMI, AVX-512, AVX2 or scalar)
	bool keep_table;  // `filter` reads `keep`: keep[x] = 0xFF if mark[x] >= depth, for the projection in progress
	alignas(64) std::uint8_t keep[256] = {};
	bool fused;       // scalar mode: filter and bound updates in one backward pass instead of `filter`
	std::vector<std::unique_ptr<Level>> levels;  // levels[d] = reusable database at depth d
	// spare[d] = the database levels[d] was compacted from; its storage backs levels[d]'s views
	// until the next sibling at depth d is compacted.
	std::vector<std::unique_ptr<Level>> spare;
	std::uint32_t occ_shift = 0;  // position bits of an occurrence (the same for every database of the search)
};

}  // namespace REFIM_WIDTHS
