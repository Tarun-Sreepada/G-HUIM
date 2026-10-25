#pragma once
// One search over the renamed transactions, by the build for the chosen widths (mine/widths.hpp).

#include "io/reader.hpp"
#include "mine/widths.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Root tasks shared with other engines (GREFIM): the width-specific RootShare (mine/share.hpp) derives from this.
struct RootShareBase {
	virtual ~RootShareBase() = default;
};

struct SearchOptions {
	std::uint64_t min_util = 0;
	int threads = 1;
	bool numa = true;            // pin workers to cores, place memory node-locally and steal nearest-first
	int refine = -1;             // rounds of recomputing each child's bounds without the items it rules out (-1 = until none is)
	bool compact = true;         // rewrite each child without the items it rules out
	bool leaf = true;            // sum leaf extensions from the occurrence index instead of projecting them
	double refine_min_drop = 0;  // skip a refinement round unless the last pass dropped this fraction of the candidates
	bool stats = false;
	bool collect_patterns = false;
	// Optional, called with the search's (item bits, utility bits): workers then take root tasks from the RootShare
	// it returns instead of being seeded with all of them.
	std::function<std::unique_ptr<RootShareBase>(int, int)> make_share;
};

// The itemsets a search kept (SearchOptions::collect_patterns), in its widths.
class FoundPatterns {
public:
	virtual ~FoundPatterns() = default;
	// Writes them to `path` (items joined by '\t', then ':' and the utility); returns the bytes written.
	virtual std::size_t write(const std::string& path, const std::vector<std::string>& names) const = 0;
};

struct SearchResult {
	std::size_t pattern_count = 0;
	std::unique_ptr<FoundPatterns> patterns;  // null unless collect_patterns
};

#ifdef REFIM_WIDTHS
inline namespace REFIM_WIDTHS {

// Builds the root database from `input` (freeing it) and mines it.
SearchResult search(Transactions& input, const SearchOptions& options);

}  // namespace REFIM_WIDTHS
#endif
