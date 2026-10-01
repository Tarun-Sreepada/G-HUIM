#include "reader.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace {

std::string_view trim(std::string_view s) {
	const auto b = s.find_first_not_of(" \t\r\n");
	if (b == std::string_view::npos) return {};
	return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

template <class F>
void for_each_token(const std::string& sep_, std::string_view s, F&& f) {
	std::size_t start = 0;
	while (true) {
		const std::size_t end = s.find(sep_, start);
		const std::string_view token = s.substr(start, end == std::string_view::npos ? end : end - start);
		if (!token.empty()) f(token);
		if (end == std::string_view::npos) break;
		start = end + sep_.size();
	}
}

Utility parse_int(std::string_view token, std::size_t line_no) {
	token = trim(token);
	Utility value = 0;
	const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
	if (ec != std::errc() || ptr != token.data() + token.size() || token.empty())
		throw std::runtime_error("line " + std::to_string(line_no) + ": invalid number '" +
								 std::string(token) + "'");
	return value;
}

std::string read_all(const std::string& path) {
	std::ifstream in(path, std::ios::binary | std::ios::ate);
	if (!in) throw std::runtime_error("cannot open input file: " + path);
	std::string buf(static_cast<std::size_t>(in.tellg()), '\0');
	in.seekg(0);
	in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
	return buf;
}

}  // namespace

void read_database(const std::string& input_file_, const std::string& sep_, Utility min_util_, Level& root,
                   std::vector<std::string>& rename_) {
	const std::string buf = read_all(input_file_);
	const std::string_view text(buf);

	std::vector<std::string_view> names;
	std::unordered_map<std::string_view, std::int32_t> name_index;
	std::vector<Utility> twu;
	std::vector<std::int32_t> raw_names;  // transactions back to back, as indices into `names`
	std::vector<Utility> raw_utils;
	std::vector<std::size_t> raw_begin{0};
	std::vector<std::string_view> item_tokens;

	std::size_t line_no = 0;
	for (std::size_t pos = 0; pos < text.size();) {
		std::size_t nl = text.find('\n', pos);
		if (nl == std::string_view::npos) nl = text.size();
		const std::string_view line = trim(text.substr(pos, nl - pos));
		pos = nl + 1;
		++line_no;
		if (line.empty()) continue;

		const std::size_t c1 = line.find(':');
		const std::size_t c2 = c1 == std::string_view::npos ? c1 : line.find(':', c1 + 1);
		if (c2 == std::string_view::npos)
			throw std::runtime_error("line " + std::to_string(line_no) + ": expected items:total:utilities");

		std::string_view weight_token;
		for_each_token(sep_, line.substr(c1 + 1, c2 - c1 - 1), [&](std::string_view tok) {
			if (weight_token.empty()) weight_token = tok;
		});
		const Utility weight = parse_int(weight_token, line_no);

		item_tokens.clear();
		for_each_token(sep_, line.substr(0, c1), [&](std::string_view tok) { item_tokens.push_back(tok); });
		std::size_t i = 0;
		for_each_token(sep_, line.substr(c2 + 1), [&](std::string_view tok) {
			if (i >= item_tokens.size()) return;  // like zip() in efim.py
			auto [it, inserted] = name_index.emplace(item_tokens[i++], static_cast<std::int32_t>(names.size()));
			if (inserted) {
				names.push_back(it->first);
				twu.push_back(0);
			}
			twu[it->second] += weight;
			raw_names.push_back(it->second);
			raw_utils.push_back(parse_int(tok, line_no));
		});
		raw_begin.push_back(raw_names.size());
	}

	// Keep promising items, sorted by TWU descending (stable, like Python's sorted).
	std::vector<std::int32_t> order;
	for (std::int32_t i = 0; i < static_cast<std::int32_t>(names.size()); ++i)
		if (twu[i] >= min_util_) order.push_back(i);
	std::stable_sort(order.begin(), order.end(), [&](std::int32_t a, std::int32_t b) { return twu[a] > twu[b]; });

	const Item num_items = static_cast<Item>(order.size());
	std::vector<Item> str_to_int(names.size(), 0);  // 0 = pruned
	rename_.assign(num_items + 1, "");
	Item next_id = num_items;
	for (const std::int32_t i : order) {
		str_to_int[i] = next_id;
		rename_[next_id] = std::string(names[i]);
		--next_id;
	}

	// Renamed transactions back to back, items ascending.
	std::vector<Item> keys;
	std::vector<Utility> vals;
	std::vector<std::size_t> begin{0};
	std::vector<std::pair<Item, Utility>> transaction;
	for (std::size_t t = 0; t + 1 < raw_begin.size(); ++t) {
		transaction.clear();
		for (std::size_t i = raw_begin[t]; i < raw_begin[t + 1]; ++i)
			if (const Item x = str_to_int[raw_names[i]]) transaction.emplace_back(x, raw_utils[i]);
		if (transaction.empty()) continue;
		std::sort(transaction.begin(), transaction.end(),
				  [](const auto& a, const auto& b) { return a.first < b.first; });
		for (const auto& [x, u] : transaction) {
			keys.push_back(x);
			vals.push_back(u);
		}
		begin.push_back(keys.size());
	}

	// Reverse-lexicographic order (compare from the last item). Projection keeps
	// this order, so identical projected transactions stay adjacent.
	std::vector<std::uint32_t> perm(begin.size() - 1);
	std::iota(perm.begin(), perm.end(), 0);
	std::sort(perm.begin(), perm.end(), [&](std::uint32_t a, std::uint32_t b) {
		return std::lexicographical_compare(
			std::make_reverse_iterator(keys.begin() + begin[a + 1]), std::make_reverse_iterator(keys.begin() + begin[a]),
			std::make_reverse_iterator(keys.begin() + begin[b + 1]), std::make_reverse_iterator(keys.begin() + begin[b]));
	});

	root.clear();
	for (const std::uint32_t t : perm)
		root.add(keys.data() + begin[t], vals.data() + begin[t],
				 static_cast<std::uint32_t>(begin[t + 1] - begin[t]), 0, false);
	root.finalize();

	std::vector<Utility> subtree(num_items + 1, 0);
	for (const auto& t : root.txs) {
		Utility remaining = 0;
		for (std::uint32_t i = 0; i < t.len; ++i) remaining += t.utils[i];
		for (std::uint32_t i = 0; i < t.len; ++i) {
			subtree[t.items[i]] += remaining;
			remaining -= t.utils[i];
		}
	}
	for (Item x = 1; x <= num_items; ++x)
		if (subtree[x] >= min_util_) root.primary.push_back(x);

	std::vector<std::int32_t> slot(num_items + 1, -1);
	root.index(slot);
}
