#include "io/root.hpp"

#include "io/reader/root_steps.hpp"

#include <algorithm>
#include <chrono>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

inline namespace REFIM_WIDTHS {

void make_root(Transactions& input, std::uint64_t min_util, int threads, Level& root) {
	threads = std::max(threads, 1);
	auto t0 = std::chrono::steady_clock::now();
	const auto lap = [&](const char* what) {
		const auto t1 = std::chrono::steady_clock::now();
		spdlog::debug("reader: {} {:.3f} s", what, std::chrono::duration<double>(t1 - t0).count());
		t0 = t1;
	};

	const Flat flat = concatenate(input.chunks);
	lap("concatenate");

	const std::vector<std::uint32_t> order = sort_transactions(flat, threads);
	lap("sort transactions");

	build_root(flat, order, threads, root);
	lap("build root");

	find_primary(root, input.num_items, min_util, threads);
	std::vector<std::int32_t> slot(input.num_items + 1, -1);
	if (!occurrences_fit(input.num_transactions, input.max_length))
		throw std::runtime_error(std::to_string(input.num_transactions) + " transactions of up to " +
								 std::to_string(input.max_length) + " items do not fit 32-bit occurrences");
	root.index(slot, occurrence_pos_bits(input.max_length), threads);
	lap("subtree+index");
}

}  // namespace REFIM_WIDTHS
