// EFIM (Efficient high-utility itemset mining), ported from efim.py.
//
// Reference: Zida, S., Fournier-Viger, P., Lin, J.CW. et al. EFIM: a fast and memory
// efficient algorithm for high-utility itemset mining. Knowl Inf Syst 51, 595-625 (2017).
//
// Input format (one transaction per line, items/utilities split by `sep`):
//     a c d:8:5 1 2
//
// Usage: efim -i <input> -m <minUtil> [-o <output>] [--separator tab|space|<literal>] [-t <threads>]
//             [--numa on|off] [--simd auto|avx512|scalar] [--log-level L] [--log-file F]
//        (legacy positional form: efim <input> <output> <minUtil> [sep] [threads])
//     threads defaults to 1 (0 = all) and is capped at the hardware thread count.
//
// Limits: items are uint16 (at most 65535 after TWU pruning) and stored utilities int32 (no item's
// total utility may exceed INT32_MAX); the reader fails with an error otherwise.
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
//   - Per-depth buffers are reused across the search, so the hot path does not
//     allocate once they have grown.
//   - The input is read in one block and parsed in place with std::from_chars.
//   - Patterns are counted; flat arrays are populated only when output is requested.

#pragma once

#include "mine/level.hpp"
#include "mine/scheduler.hpp"
#include "mine/worker.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

class Efim {
public:
	Efim(std::string input_file, Utility min_util, std::string sep = "\t", int threads = 1)
		: input_file_(std::move(input_file)), min_util_(min_util), sep_(std::move(sep)),
		  threads_(std::clamp(threads, 1, max_threads())) {
		if (sep_.empty()) throw std::invalid_argument("separator must not be empty");
	}

	// Hardware thread count; hardware_concurrency() may return 0 when unknown.
	static int max_threads() {
		const unsigned hw = std::thread::hardware_concurrency();
		return hw ? static_cast<int>(hw) : 1;
	}

	// Pin workers to cores, place memory node-locally and steal nearest-first (default on).
	void set_numa(bool on) { numa_ = on; }
	// Rounds of recomputing each child's bounds without the items it rules out (-1 = until none is).
	void set_refine(int rounds) { refine_ = rounds; }
	// Rewrite each child without the items it rules out (default on).
	void set_compact(bool on) { compact_ = on; }
	// Skip a refinement round unless the last pass dropped at least this fraction of the candidates.
	void set_leaf(bool on) { leaf_ = on; }
	void set_refine_min_drop(double fraction) { refine_min_drop_ = fraction; }

	void set_stats(bool on) { stats_ = on; }
	int threads() const { return threads_; }
	// Count by default; collection must be enabled before mining to call save().
	void mine(bool collect_patterns = false);
	void save(const std::string& out_file) const;
	std::size_t pattern_count() const;
	void print_results() const;
	double runtime() const { return runtime_; }
	std::size_t memory_rss() const { return memory_rss_; }

private:
	std::string input_file_;
	Utility min_util_;
	std::string sep_;
	int threads_;
	bool numa_ = true;
	int refine_ = -1;
	bool stats_ = false, compact_ = true, leaf_ = true;
	double refine_min_drop_ = 0;
	bool collect_patterns_ = false;
	std::size_t pattern_count_ = 0;
	std::vector<std::string> rename_;  // rename_[id] = original item name; index 0 unused
	std::vector<Patterns> results_;    // one per worker
	double runtime_ = 0;
	std::size_t memory_rss_ = 0;
};
