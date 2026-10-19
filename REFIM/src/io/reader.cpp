#include "io/reader.hpp"

#include "io/reader/steps.hpp"

#include <algorithm>
#include <chrono>
#include <spdlog/spdlog.h>

void read_database(const std::string& input_file_, const std::string& sep_, Utility min_util_, Level& root,
                   std::vector<std::string>& rename_, int threads) {
	threads = std::max(threads, 1);
	auto t0 = std::chrono::steady_clock::now();
	const auto lap = [&](const char* what) {
		const auto t1 = std::chrono::steady_clock::now();
		spdlog::debug("reader: {} {:.3f} s", what, std::chrono::duration<double>(t1 - t0).count());
		t0 = t1;
	};

	const std::string buf = read_all(input_file_, threads);
	const std::string_view text(buf);
	lap("read file");

	std::vector<Chunk> chunks = split_into_chunks(text, threads);
	parse_chunks(text, sep_, chunks);
	lap("parse");

	const NameTable names = merge_names(chunks);
	const std::vector<Item> new_id = rank_items(names, min_util_, rename_);
	const std::size_t num_items = rename_.size() - 1;
	rename_transactions(chunks, new_id, num_items);
	check_utilities(chunks, num_items, rename_);
	lap("rename+sort items");

	const Flat flat = concatenate(chunks);
	lap("concatenate");

	const std::vector<std::uint32_t> order = sort_transactions(flat, threads);
	lap("sort transactions");

	build_root(flat, order, threads, root);
	lap("build root");

	find_primary(root, num_items, min_util_, threads);
	std::vector<std::int32_t> slot(num_items + 1, -1);
	root.index(slot, threads);
	lap("subtree+index");
}
