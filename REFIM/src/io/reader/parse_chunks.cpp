#include "io/reader/steps.hpp"

#include "io/reader/text.hpp"
#include "sys/parallel.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace {

void parse_chunk(std::string_view text, const std::string& sep, Chunk& c) {
	std::vector<std::string_view> item_tokens;
	for (std::size_t pos = c.begin; pos < c.end;) {
		const std::size_t line_start = pos;
		const void* hit = std::memchr(text.data() + pos, '\n', c.end - pos);
		const std::size_t nl = hit ? static_cast<std::size_t>(static_cast<const char*>(hit) - text.data()) : c.end;
		const std::string_view line = trim(text.substr(pos, nl - pos));
		pos = nl + 1;
		if (line.empty()) continue;
		try {
			const std::size_t c1 = line.find(':');
			const std::size_t c2 = c1 == std::string_view::npos ? c1 : line.find(':', c1 + 1);
			if (c2 == std::string_view::npos) throw std::runtime_error("expected items:total:utilities");

			std::string_view weight_token;
			for_each_token(sep, line.substr(c1 + 1, c2 - c1 - 1), [&](std::string_view tok) {
				if (weight_token.empty()) weight_token = tok;
			});
			const std::int64_t weight = parse_int(weight_token);

			item_tokens.clear();
			for_each_token(sep, line.substr(0, c1), [&](std::string_view tok) { item_tokens.push_back(tok); });
			std::size_t i = 0;
			for_each_token(sep, line.substr(c2 + 1), [&](std::string_view tok) {
				if (i >= item_tokens.size()) return;  // like zip() in efim.py
				const std::int32_t id = c.table.intern(item_tokens[i++]);
				c.table.twu[static_cast<std::size_t>(id)] += weight;
				c.raw_names.push_back(id);
				c.raw_utils.push_back(parse_int(tok));
			});
			c.raw_begin.push_back(c.raw_names.size());
		} catch (const std::exception& e) {
			c.failed = true;
			c.error_pos = line_start;
			c.error = e.what();
			return;
		}
	}
}

}  // namespace

void parse_chunks(std::string_view text, const std::string& sep, std::vector<Chunk>& chunks) {
	parallel_run(static_cast<int>(chunks.size()), [&](int i) { parse_chunk(text, sep, chunks[i]); });
	for (const Chunk& c : chunks) {
		if (!c.failed) continue;
		const auto line_no = 1 + std::count(text.begin(), text.begin() + static_cast<std::ptrdiff_t>(c.error_pos), '\n');
		throw std::runtime_error("line " + std::to_string(line_no) + ": " + c.error);
	}
}
