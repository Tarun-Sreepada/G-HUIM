// EFIM (Efficient high-utility itemset mining), ported from efim.py.
//
// Reference: Zida, S., Fournier-Viger, P., Lin, J.CW. et al. EFIM: a fast and memory
// efficient algorithm for high-utility itemset mining. Knowl Inf Syst 51, 595-625 (2017).
//
// Input format (one transaction per line, items/utilities split by `sep`):
//     a c d:8:5 1 2
//
// Usage: efim <inputFile> <outputFile> <minUtil> [sep] [threads]
//     sep: "tab" (default) or "space", or the literal separator, e.g. ";" or ",".
//     threads defaults to 1 and is capped at the hardware thread count.
//
// Differences from efim.py: the search is depth-first instead of level-by-level
// (same patterns, far less memory), and threads share a task stack that large
// subtrees are split into.
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
//   - Patterns are stored in flat arrays.

#pragma once

#include "level.hpp"
#include "scheduler.hpp"

#include <algorithm>
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

	int threads() const { return threads_; }
	void mine();
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
	std::vector<std::string> rename_;  // rename_[id] = original item name; index 0 unused
	std::vector<Patterns> results_;    // one per worker
	double runtime_ = 0;
	std::size_t memory_rss_ = 0;
};
