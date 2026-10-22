#include "mine/refim.hpp"

#include "io/reader.hpp"
#include "sys/perf.hpp"
#include "sys/util.hpp"

#include <chrono>
#include <iostream>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

// The searches of every built width pair (src/mine/search.cpp).
#define REFIM_DECLARE_SEARCH(item, util) \
	inline namespace REFIM_WIDTH_NAME(item, util) { SearchResult search(Transactions& input, const SearchOptions& options); }
REFIM_FOR_EACH_BUILT_WIDTH(REFIM_DECLARE_SEARCH)
#undef REFIM_DECLARE_SEARCH

namespace {

struct Widths {
	int key_bits, util_bits;
};

// Items keep ids 1..num_items; every utility is at most the largest TWU (mine/widths.hpp).
int key_bits_for(std::size_t num_items) { return num_items < (1u << 8) ? 8 : num_items < (1u << 16) ? 16 : 32; }
int util_bits_for(std::uint64_t max_twu) { return max_twu <= UINT16_MAX ? 16 : max_twu <= UINT32_MAX ? 32 : 64; }

// --key-bits/--util-bits when given (they must hold the dataset), else the narrowest built pair that holds
// it: fewest bytes per entry, then the narrower utilities.
Widths choose_widths(const Transactions& input, int key_bits, int util_bits) {
	const int key_needed = key_bits_for(input.num_items), util_needed = util_bits_for(input.max_twu);
	if (key_bits != 0 && key_bits < key_needed)
		throw std::runtime_error("--key-bits " + std::to_string(key_bits) + " is too narrow: " + std::to_string(input.num_items) +
								 " items need " + std::to_string(key_needed) + "-bit ids");
	if (util_bits != 0 && util_bits < util_needed)
		throw std::runtime_error("--util-bits " + std::to_string(util_bits) + " is too narrow: the largest TWU " +
								 std::to_string(input.max_twu) + " needs " + std::to_string(util_needed) + "-bit utilities");
	Widths best{0, 0};
	const auto consider = [&](int key, int util) {
		if (key_bits != 0 ? key != key_bits : key < key_needed) return;
		if (util_bits != 0 ? util != util_bits : util < util_needed) return;
		const int bits = key + util, best_bits = best.key_bits + best.util_bits;
		if (best.key_bits == 0 || bits < best_bits || (bits == best_bits && util < best.util_bits)) best = {key, util};
	};
#define REFIM_CONSIDER(item, util) consider(item, util);
	REFIM_FOR_EACH_BUILT_WIDTH(REFIM_CONSIDER)
#undef REFIM_CONSIDER
	if (best.key_bits == 0)
		throw std::runtime_error("no search was built for " + std::to_string(key_bits ? key_bits : key_needed) + "-bit ids and " +
								 std::to_string(util_bits ? util_bits : util_needed) + "-bit utilities (see CMake option REFIM_WIDTHS)");
	return best;
}

SearchResult search_with(const Widths& widths, Transactions& input, const SearchOptions& options) {
#define REFIM_DISPATCH(item, util) \
	if (widths.key_bits == item && widths.util_bits == util) return REFIM_WIDTH_NAME(item, util)::search(input, options);
	REFIM_FOR_EACH_BUILT_WIDTH(REFIM_DISPATCH)
#undef REFIM_DISPATCH
	throw std::logic_error("unreachable: chosen widths were not built");
}

}  // namespace

void Refim::mine(bool collect_patterns) {
	const auto start = std::chrono::steady_clock::now();
	search_.collect_patterns = collect_patterns;
	search_.min_util = static_cast<std::uint64_t>(min_util_);
	pattern_count_ = 0;
	patterns_.reset();
	rename_.clear();

	PerfCounters perf(bandwidth_);
	perf.start();
	Transactions input = read_transactions(input_file_, sep_, min_util_, rename_, search_.threads);
	perf.report("read");
	const Widths widths = choose_widths(input, key_bits_, util_bits_);
	spdlog::info("widths: {}-bit ids ({} items), {}-bit utilities (largest TWU {})", widths.key_bits, input.num_items,
				 widths.util_bits, input.max_twu);
	perf.start();
	SearchResult result = search_with(widths, input, search_);
	perf.report("mine");
	pattern_count_ = result.pattern_count;
	patterns_ = std::move(result.patterns);

	runtime_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	memory_rss_ = peak_rss_bytes();
}

void Refim::save(const std::string& out_file) const {
	if (!search_.collect_patterns) throw std::logic_error("cannot save patterns after count-only mining");
	const auto start = std::chrono::steady_clock::now();
	const std::size_t bytes = patterns_->write(out_file, rename_);
	spdlog::info("Wrote {} patterns ({} bytes) to {} in {:.3f} s", pattern_count(), bytes, out_file,
				 std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
}

std::size_t Refim::pattern_count() const {
	return pattern_count_;
}

void Refim::print_results() const {
	std::cout << "Total number of High Utility Patterns: " << pattern_count() << '\n';
	std::cout << "Peak Memory in RSS: " << memory_rss_ << '\n';
	std::cout << "Total ExecutionTime in seconds: " << runtime_ << '\n';
}
