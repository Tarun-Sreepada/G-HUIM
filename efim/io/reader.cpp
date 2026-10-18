#include "io/reader.hpp"

#include "sys/parallel.hpp"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <numeric>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>

namespace {

std::string_view trim(std::string_view s) {
	const auto b = s.find_first_not_of(" \t\r\n");
	if (b == std::string_view::npos) return {};
	return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

template <class F>
void for_each_token(const std::string& sep, std::string_view s, F&& f) {
	std::size_t start = 0;
	while (true) {
		std::size_t end;
		if (sep.size() == 1) {
			const void* hit = start < s.size() ? std::memchr(s.data() + start, sep[0], s.size() - start) : nullptr;
			end = hit ? static_cast<std::size_t>(static_cast<const char*>(hit) - s.data()) : std::string_view::npos;
		} else {
			end = s.find(sep, start);
		}
		const std::string_view token = s.substr(start, end == std::string_view::npos ? end : end - start);
		if (!token.empty()) f(token);
		if (end == std::string_view::npos) break;
		start = end + sep.size();
	}
}

Utility parse_int(std::string_view token) {
	token = trim(token);
	Utility value = 0;
	const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
	if (ec != std::errc() || ptr != token.data() + token.size() || token.empty())
		throw std::runtime_error("invalid number '" + std::string(token) + "'");
	return value;
}

// Reads the whole file with parallel pread()s.
std::string read_all(const std::string& path, int threads) {
	const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
	if (fd < 0) throw std::runtime_error("cannot open input file: " + path);
	struct stat st{};
	if (::fstat(fd, &st) != 0) {
		::close(fd);
		throw std::runtime_error("cannot stat input file: " + path);
	}
	std::string buf(static_cast<std::size_t>(st.st_size), '\0');
	const std::size_t size = buf.size();
	const int n = size < (std::size_t{8} << 20) ? 1 : threads;
	try {
		parallel_run(n, [&](int i) {
			std::size_t pos = size / n * i;
			const std::size_t end = i + 1 == n ? size : size / n * (i + 1);
			while (pos < end) {
				const ssize_t r = ::pread(fd, buf.data() + pos, end - pos, static_cast<off_t>(pos));
				if (r < 0 && errno == EINTR) continue;
				if (r <= 0) throw std::runtime_error("read failed: " + path);
				pos += static_cast<std::size_t>(r);
			}
		});
	} catch (...) {
		::close(fd);
		throw;
	}
	::close(fd);
	return buf;
}

// Interning table for item names (open addressing, string_views into the input buffer).
struct NameTable {
	std::vector<std::int32_t> slots = std::vector<std::int32_t>(1024, -1);
	std::vector<std::string_view> names;  // first-appearance order
	std::vector<std::uint64_t> hashes;
	std::vector<Utility> twu;

	static std::uint64_t hash(std::string_view s) {
		std::uint64_t h = 1469598103934665603ull;
		for (const unsigned char c : s) h = (h ^ c) * 1099511628211ull;
		return h ^ (h >> 29);
	}

	std::int32_t intern(std::string_view s) {
		const std::uint64_t h = hash(s);
		std::size_t i = h & (slots.size() - 1);
		while (slots[i] >= 0) {
			const auto id = static_cast<std::size_t>(slots[i]);
			if (hashes[id] == h && names[id] == s) return slots[i];
			i = (i + 1) & (slots.size() - 1);
		}
		const auto id = static_cast<std::int32_t>(names.size());
		slots[i] = id;
		names.push_back(s);
		hashes.push_back(h);
		twu.push_back(0);
		if (names.size() * 2 > slots.size()) grow();
		return id;
	}

	void grow() {
		slots.assign(slots.size() * 2, -1);
		for (std::size_t id = 0; id < names.size(); ++id) {
			std::size_t i = hashes[id] & (slots.size() - 1);
			while (slots[i] >= 0) i = (i + 1) & (slots.size() - 1);
			slots[i] = static_cast<std::int32_t>(id);
		}
	}
};

// One thread's share of the input: a block of whole lines.
struct Chunk {
	std::size_t begin = 0, end = 0;  // byte range in the input
	NameTable table;
	std::vector<std::int32_t> raw_names;  // transactions back to back, as indices into table.names
	std::vector<Utility> raw_utils;
	std::vector<std::size_t> raw_begin{0};
	std::vector<std::int32_t> remap;  // local name id -> global name id
	// Renamed transactions (items ascending), pruned items removed.
	std::vector<Item> keys;
	std::vector<Util> vals;
	std::vector<Utility> item_sum;  // per renamed item: sum of |utility| over this chunk (overflow check)
	std::vector<std::size_t> tx_begin;  // tx_begin[t]..tx_begin[t + 1]; tx_begin[0] = 0
	bool failed = false;
	bool bad_util = false;
	std::size_t error_pos = 0;
	std::string error;
};

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
			const Utility weight = parse_int(weight_token);

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

	// Split at line boundaries so each thread parses whole lines.
	const int n = text.size() < (std::size_t{1} << 20) ? 1 : threads;
	std::vector<Chunk> chunks(static_cast<std::size_t>(n));
	{
		std::size_t at = 0;
		for (int i = 0; i < n; ++i) {
			std::size_t end = text.size();
			if (i + 1 < n) {
				const std::size_t nl = text.find('\n', std::max(at, text.size() / n * (i + 1)));
				end = nl == std::string_view::npos ? text.size() : nl + 1;
			}
			chunks[i].begin = at;
			chunks[i].end = end;
			at = end;
		}
	}
	parallel_run(n, [&](int i) { parse_chunk(text, sep_, chunks[i]); });
	for (const Chunk& c : chunks) {
		if (!c.failed) continue;
		const auto line_no = 1 + std::count(text.begin(), text.begin() + static_cast<std::ptrdiff_t>(c.error_pos), '\n');
		throw std::runtime_error("line " + std::to_string(line_no) + ": " + c.error);
	}
	lap("parse");

	// Merge the per-chunk name tables in chunk order (= first appearance in the file).
	NameTable global;
	for (Chunk& c : chunks) {
		c.remap.resize(c.table.names.size());
		for (std::size_t l = 0; l < c.table.names.size(); ++l) {
			const std::int32_t g = global.intern(c.table.names[l]);
			global.twu[static_cast<std::size_t>(g)] += c.table.twu[l];
			c.remap[l] = g;
		}
	}
	const auto& names = global.names;
	const auto& twu = global.twu;

	// Keep promising items, sorted by TWU descending (stable, like Python's sorted).
	std::vector<std::int32_t> order;
	for (std::int32_t i = 0; i < static_cast<std::int32_t>(names.size()); ++i)
		if (twu[i] >= min_util_) order.push_back(i);
	std::stable_sort(order.begin(), order.end(), [&](std::int32_t a, std::int32_t b) { return twu[a] > twu[b]; });

	if (order.size() > kMaxItems)
		throw std::runtime_error(std::to_string(order.size()) + " items survive TWU pruning at minutil " +
								 std::to_string(min_util_) + "; this build stores items in 16 bits (at most " +
								 std::to_string(kMaxItems) + "). Raise minutil.");
	const std::size_t num_items = order.size();
	std::vector<Item> str_to_int(names.size(), 0);  // 0 = pruned
	spdlog::debug("reader: {} distinct items, {} kept", names.size(), num_items);
	rename_.assign(num_items + 1, "");
	std::size_t next_id = num_items;
	for (const std::int32_t i : order) {
		str_to_int[i] = static_cast<Item>(next_id);
		rename_[next_id] = std::string(names[i]);
		--next_id;
	}

	// Renamed transactions per chunk, items ascending.
	parallel_run(n, [&](int ci) {
		Chunk& c = chunks[ci];
		std::vector<std::pair<Item, Util>> transaction;
		c.tx_begin.assign(1, 0);
		c.item_sum.assign(num_items + 1, 0);
		for (std::size_t t = 0; t + 1 < c.raw_begin.size(); ++t) {
			transaction.clear();
			for (std::size_t i = c.raw_begin[t]; i < c.raw_begin[t + 1]; ++i)
				if (const Item x = str_to_int[c.remap[c.raw_names[i]]]) {
					const Utility u = c.raw_utils[i];
					if (u < 0 || u > std::numeric_limits<Util>::max()) c.bad_util = true;
					c.item_sum[x] += u;
					transaction.emplace_back(x, static_cast<Util>(u));
				}
			if (transaction.empty()) continue;
			std::sort(transaction.begin(), transaction.end(),
					  [](const auto& a, const auto& b) { return a.first < b.first; });
			// An item listed twice in one transaction (kosarak) becomes one entry with the summed utility.
			const std::size_t first = c.keys.size();
			for (const auto& [x, u] : transaction) {
				if (c.keys.size() > first && c.keys.back() == x) {
					const Utility sum = Utility{c.vals.back()} + u;
					if (sum > std::numeric_limits<Util>::max()) c.bad_util = true;
					c.vals.back() = static_cast<Util>(sum);
					continue;
				}
				c.keys.push_back(x);
				c.vals.push_back(u);
			}
			c.tx_begin.push_back(c.keys.size());
		}
		std::vector<std::int32_t>().swap(c.raw_names);
		std::vector<Utility>().swap(c.raw_utils);
	});
	// Merging identical transactions adds utilities of one item, so 32 bits are safe if no item's
	// total utility over the whole database exceeds INT32_MAX.
	{
		std::vector<Utility> total(num_items + 1, 0);
		for (const Chunk& c : chunks) {
			if (c.bad_util) throw std::runtime_error("a utility is negative or does not fit in 32 bits");
			for (std::size_t x = 1; x <= num_items; ++x) total[x] += c.item_sum[x];
		}
		const auto worst = std::max_element(total.begin(), total.end());
		if (*worst > std::numeric_limits<Util>::max())
			throw std::runtime_error("item '" + rename_[worst - total.begin()] + "' has total utility " +
									 std::to_string(*worst) + " which overflows the 32-bit utility storage");
	}
	lap("rename+sort items");

	// Concatenate chunks into global arrays.
	std::vector<std::size_t> key_off(chunks.size() + 1, 0), tx_off(chunks.size() + 1, 0);
	for (std::size_t i = 0; i < chunks.size(); ++i) {
		key_off[i + 1] = key_off[i] + chunks[i].keys.size();
		tx_off[i + 1] = tx_off[i] + chunks[i].tx_begin.size() - 1;
	}
	const std::unique_ptr<Item[]> keys(new Item[key_off.back()]);  // uninitialised: filled in parallel below
	const std::unique_ptr<Util[]> vals(new Util[key_off.back()]);
	const std::unique_ptr<std::size_t[]> begin(new std::size_t[tx_off.back() + 1]);
	begin[0] = 0;
	parallel_run(n, [&](int i) {
		const Chunk& c = chunks[i];
		std::copy(c.keys.begin(), c.keys.end(), keys.get() + key_off[i]);
		std::copy(c.vals.begin(), c.vals.end(), vals.get() + key_off[i]);
		for (std::size_t t = 1; t < c.tx_begin.size(); ++t) begin[tx_off[i] + t] = key_off[i] + c.tx_begin[t];
	});
	chunks.clear();
	chunks.shrink_to_fit();
	lap("concatenate");

	// Reverse-lexicographic order (compare from the last item). Projection keeps
	// this order, so identical projected transactions stay adjacent. Sorted in
	// parallel slices, then merged pairwise.
	std::vector<std::uint32_t> perm(tx_off.back());
	std::iota(perm.begin(), perm.end(), 0);
	const auto less = [&](std::uint32_t a, std::uint32_t b) {
		return std::lexicographical_compare(
			std::make_reverse_iterator(keys.get() + begin[a + 1]), std::make_reverse_iterator(keys.get() + begin[a]),
			std::make_reverse_iterator(keys.get() + begin[b + 1]), std::make_reverse_iterator(keys.get() + begin[b]));
	};
	{
		const int parts = perm.size() < 4096 ? 1 : threads;
		std::vector<std::size_t> cut(static_cast<std::size_t>(parts) + 1);
		for (int i = 0; i <= parts; ++i) cut[i] = perm.size() / parts * i;
		cut[parts] = perm.size();
		const auto at = [&](int i) { return perm.begin() + static_cast<std::ptrdiff_t>(cut[i]); };
		parallel_run(parts, [&](int i) { std::sort(at(i), at(i + 1), less); });
		// Each round merges pairs of runs into `buf`. Every merge is split into slices of the
		// output by co-ranking, so all threads stay busy even in the last (single-pair) round.
		std::vector<std::uint32_t> buf(perm.size());
		std::uint32_t *src = perm.data(), *dst = buf.data();
		for (int width = 1; width < parts; width *= 2) {
			const int merges = (parts + 2 * width - 1) / (2 * width);
			const int slices = std::max(1, threads / merges);
			parallel_run(merges * slices, [&](int job) {
				const int m = job / slices, sl = job % slices;
				const int lo = m * 2 * width, mid = std::min(lo + width, parts), hi = std::min(lo + 2 * width, parts);
				const std::uint32_t *a = src + cut[lo], *b = src + cut[mid];
				const std::size_t na = cut[mid] - cut[lo], nb = cut[hi] - cut[mid], n = na + nb;
				// Smallest i with i + j = d such that the first i of `a` and first j of `b` are
				// the first d outputs of a stable merge (ties take `a` first).
				const auto corank = [&](std::size_t d) {
					std::size_t lo_i = d > nb ? d - nb : 0, hi_i = std::min(d, na);
					while (lo_i < hi_i) {
						const std::size_t i = (lo_i + hi_i) / 2, j = d - i - 1;
						if (less(b[j], a[i])) hi_i = i;
						else lo_i = i + 1;
					}
					return lo_i;
				};
				const std::size_t d0 = n * sl / slices, d1 = n * (sl + 1) / slices;
				const std::size_t i0 = corank(d0), i1 = corank(d1);
				std::merge(a + i0, a + i1, b + (d0 - i0), b + (d1 - i1), dst + cut[lo] + d0, less);
			});
			std::swap(src, dst);
		}
		if (src != perm.data()) std::copy(src, src + perm.size(), perm.data());
	}
	lap("sort transactions");

	// Build the root in parallel ranges of the sorted order (merging equal neighbours inside
	// each range), then concatenate. Range cuts move past runs of equal transactions so no
	// merge has to cross a boundary.
	root.clear();
	{
		const auto same = [&](std::uint32_t a, std::uint32_t b) {
			return begin[a + 1] - begin[a] == begin[b + 1] - begin[b] &&
				   std::equal(keys.get() + begin[a], keys.get() + begin[a + 1], keys.get() + begin[b]);
		};
		const int parts = perm.size() < 4096 ? 1 : threads;
		std::vector<std::size_t> cut(static_cast<std::size_t>(parts) + 1);
		for (int i = 0; i <= parts; ++i) cut[i] = perm.size() / parts * i;
		cut[parts] = perm.size();
		for (int i = 1; i < parts; ++i) {
			cut[i] = std::max(cut[i], cut[i - 1]);
			while (cut[i] < perm.size() && cut[i] > 0 && same(perm[cut[i] - 1], perm[cut[i]])) ++cut[i];
		}
		std::vector<Level> piece(static_cast<std::size_t>(parts));
		parallel_run(parts, [&](int p) {
			for (std::size_t i = cut[p]; i < cut[p + 1]; ++i) {
				const std::uint32_t t = perm[i];
				piece[p].add(keys.get() + begin[t], vals.get() + begin[t], static_cast<std::uint32_t>(begin[t + 1] - begin[t]), 0, false);
			}
		});
		std::vector<std::size_t> item_off(piece.size() + 1, 0), tx_off2(piece.size() + 1, 0);
		for (std::size_t p = 0; p < piece.size(); ++p) {
			item_off[p + 1] = item_off[p] + piece[p].items.size();
			tx_off2[p + 1] = tx_off2[p] + piece[p].txs.size();
		}
		root.items.resize(item_off.back());
		root.utils.resize(item_off.back());
		root.txs.resize(tx_off2.back());
		parallel_run(parts, [&](int p) {
			Level& src = piece[p];
			std::copy(src.items.begin(), src.items.end(), root.items.begin() + static_cast<std::ptrdiff_t>(item_off[p]));
			std::copy(src.utils.begin(), src.utils.end(), root.utils.begin() + static_cast<std::ptrdiff_t>(item_off[p]));
			for (std::size_t i = 0; i < src.txs.size(); ++i) {
				Transaction t = src.txs[i];
				t.own += static_cast<std::int64_t>(item_off[p]);
				t.items = root.items.data() + t.own;
				t.utils = root.utils.data() + t.own;
				root.txs[tx_off2[p] + i] = t;
			}
			src = Level();  // release the piece
		});
	}
	lap("build root");

	std::vector<Utility> subtree(num_items + 1, 0);
	{
		const int parts = root.txs.size() < 4096 ? 1 : threads;
		std::vector<std::vector<Utility>> partial(static_cast<std::size_t>(parts), std::vector<Utility>(num_items + 1, 0));
		parallel_run(parts, [&](int p) {
			const std::size_t lo = root.txs.size() / parts * p, hi = p + 1 == parts ? root.txs.size() : root.txs.size() / parts * (p + 1);
			for (std::size_t ti = lo; ti < hi; ++ti) {
				const auto& t = root.txs[ti];
				Utility remaining = 0;
				for (std::uint32_t i = 0; i < t.len; ++i) remaining += t.utils[i];
				for (std::uint32_t i = 0; i < t.len; ++i) {
					partial[p][t.items[i]] += remaining;
					remaining -= t.utils[i];
				}
			}
		});
		for (const auto& v : partial)
			for (std::size_t x = 0; x < subtree.size(); ++x) subtree[x] += v[x];
	}
	for (std::size_t x = 1; x <= num_items; ++x)
		if (subtree[x] >= min_util_) root.primary.push_back(static_cast<Item>(x));

	std::vector<std::int32_t> slot(num_items + 1, -1);
	root.index(slot, threads);
	lap("subtree+index");
}
