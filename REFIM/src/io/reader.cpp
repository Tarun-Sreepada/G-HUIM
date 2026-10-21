#include "io/reader.hpp"

#include "io/reader/steps.hpp"

#include <algorithm>
#include <chrono>
#include <spdlog/spdlog.h>

Transactions read_transactions(const std::string& input_file_, const std::string& sep_, std::int64_t min_util_,
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

	Transactions input;
	input.chunks = split_into_chunks(text, threads);
	parse_chunks(text, sep_, input.chunks);
	lap("parse");

	const NameTable names = merge_names(input.chunks);
	const std::vector<std::uint32_t> new_id = rank_items(names, min_util_, rename_);
	input.num_items = rename_.size() - 1;
	rename_transactions(input.chunks, new_id, input.num_items);
	input.max_twu = check_utilities(input.chunks, input.num_items);
	for (Chunk& c : input.chunks) c.table = {};  // its names point into `buf`
	lap("rename+sort items");
	return input;
}
