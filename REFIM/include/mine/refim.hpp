// REFIM (refined EFIM): EFIM's search space and bounds with a different search engine. Started as a
// port of efim.py.
//
// Reference: Zida, S., Fournier-Viger, P., Lin, J.CW. et al. EFIM: a fast and memory
// efficient algorithm for high-utility itemset mining. Knowl Inf Syst 51, 595-625 (2017).
//
// Input format (one transaction per line, items/utilities split by `sep`):
//     a c d:8:5 1 2
//
// Usage: refim -i <input> -m <minUtil> [-o <output>] [-s space|tab|<literal>] [-t <threads>] [-r <rounds>]
//             [-n on|off] [-v auto|avx512|scalar] [-l <level>] [-L <file>]  (refim --help lists all; G-HUIM shares
//             -i -m -o -s -r -l -L)
//        (legacy positional form: refim <input> <output> <minUtil> [sep] [threads])
//     threads defaults to 1 (0 = all) and is capped at the hardware thread count.
//
// Widths: item ids and utilities are 8/16/32 and 16/32/64 bits, the narrowest that hold the kept items and
// the largest TWU (--key-bits/--util-bits override; mine/widths.hpp). Utilities must be non-negative and every
// TWU must fit 64 bits; the reader fails with an error otherwise.
//
// Differences from efim.py: the search is depth-first instead of level-by-level
// (same patterns, far less memory). Workers are pinned one per core (spread over L3 groups and
// both NUMA nodes), allocate their own memory after pinning (first touch = node-local, huge
// pages for big arrays), and steal large subtrees from each other nearest-first. Input parsing,
// sorting, the root build and the output are parallel.
//
// Optimizations (beyond EFIM's TWU pruning, local/subtree utility bounds and
// utility-bin arrays):
//   - Transactions are sorted in reverse-lexicographic order, which projection
//     preserves, so identical projected transactions are adjacent and merge in
//     one comparison with the previous transaction (no hashing).
//   - Pseudo-projection: a projected transaction points into its parent's storage
//     and is only copied when items are filtered out or it is merged.
//   - Each database keeps, for every item it can be extended with, the list of
//     (transaction, position) occurrences, so a projection only visits
//     transactions containing the item and needs no binary search.
//   - Bounds are refined: a child's local and subtree utilities are recomputed without the items they
//     ruled out until none drops, and the child is compacted without them.
//   - Leaf extensions are summed from the occurrence index without projecting.
//   - Per-depth buffers are reused across the search, so the hot path does not
//     allocate once they have grown.
//   - The input is read in one block and parsed in place with std::from_chars.
//   - Patterns are counted; flat arrays are populated only when output is requested.

#pragma once

#include "mine/search.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

class Refim {
public:
	Refim(std::string input_file, std::int64_t min_util, std::string sep = "\t", int threads = 1)
		: input_file_(std::move(input_file)), min_util_(min_util), sep_(std::move(sep)) {
		if (sep_.empty()) throw std::invalid_argument("separator must not be empty");
		search_.threads = std::clamp(threads, 1, max_threads());
	}

	// Hardware thread count; hardware_concurrency() may return 0 when unknown.
	static int max_threads() {
		const unsigned hw = std::thread::hardware_concurrency();
		return hw ? static_cast<int>(hw) : 1;
	}

	// Pin workers to cores, place memory node-locally and steal nearest-first (default on).
	void set_numa(bool on) { search_.numa = on; }
	// Rounds of recomputing each child's bounds without the items it rules out (-1 = until none is).
	void set_refine(int rounds) { search_.refine = rounds; }
	// Rewrite each child without the items it rules out (default on).
	void set_compact(bool on) { search_.compact = on; }
	// Sum leaf extensions from the occurrence index instead of projecting them (default on).
	void set_leaf(bool on) { search_.leaf = on; }
	// Skip a refinement round unless the last pass dropped at least this fraction of the candidates.
	void set_refine_min_drop(double fraction) { search_.refine_min_drop = fraction; }
	// Item id and utility widths (0 = the narrowest that holds the dataset).
	void set_widths(int key_bits, int util_bits) {
		key_bits_ = key_bits;
		util_bits_ = util_bits;
	}

	void set_stats(bool on) { search_.stats = on; }
	int threads() const { return search_.threads; }
	// Count by default; collection must be enabled before mining to call save().
	void mine(bool collect_patterns = false);
	void save(const std::string& out_file) const;
	std::size_t pattern_count() const;
	void print_results() const;
	double runtime() const { return runtime_; }
	std::size_t memory_rss() const { return memory_rss_; }

private:
	std::string input_file_;
	std::int64_t min_util_;
	std::string sep_;
	SearchOptions search_;
	int key_bits_ = 0, util_bits_ = 0;
	std::size_t pattern_count_ = 0;
	std::vector<std::string> rename_;         // rename_[id] = original item name; index 0 unused
	std::unique_ptr<FoundPatterns> patterns_;  // with collect_patterns
	double runtime_ = 0;
	std::size_t memory_rss_ = 0;
};
