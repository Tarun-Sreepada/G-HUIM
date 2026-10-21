#pragma once
// Databases of the EFIM search: transactions, and the occurrence index used to project them.

#include "mine/widths.hpp"
#include "sys/alloc.hpp"
#include "sys/parallel.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

inline namespace REFIM_WIDTHS {

// A transaction of a (projected) database, items sorted ascending. A view points into
// an ancestor's storage; an owned transaction lives in its own Level's storage.
struct Transaction {
	const Item* items;
	const Util* utils;
	std::uint32_t len;
	Util prefix;       // utility of the current prefix itemset in this transaction
	std::int64_t own;  // offset into Level::items/utils, or -1 for a view
};

// An occurrence index entry: transaction `tid` and position `pos` of an item, packed as tid << shift | pos
// into 32 bits. The shift is occurrence_pos_bits(longest input transaction) for the whole search.
using Occurrence = std::uint32_t;

inline Occurrence make_occurrence(std::uint32_t tid, std::uint32_t pos, std::uint32_t shift) {
	return static_cast<Occurrence>(Occurrence{tid} << shift | pos);
}
inline std::uint32_t occurrence_tid(Occurrence o, std::uint32_t shift) { return static_cast<std::uint32_t>(o >> shift); }
inline std::uint32_t occurrence_pos(Occurrence o, std::uint32_t shift) {
	return static_cast<std::uint32_t>(o & ((Occurrence{1} << shift) - 1));
}

// A (projected) database plus the occurrence index used to project it further.
struct Level {
	HugeVector<Item> items;  // storage for owned transactions
	HugeVector<Util> utils;
	HugeVector<Transaction> txs;
	std::vector<Item> primary;    // items to extend with (subtree utility >= minUtil), ascending
	std::vector<Item> secondary;  // items kept in deeper projections (local utility >= minUtil)
	std::vector<std::uint32_t> occ_begin;  // primary[k] occurs at occ[occ_begin[k], occ_begin[k + 1])
	HugeVector<Occurrence> occ;
	std::uint32_t occ_shift = 0;        // position bits of an occurrence
	std::vector<std::uint32_t> cursor;  // scratch for the serial index()

	void clear() {
		items.clear();
		utils.clear();
		txs.clear();
		primary.clear();
		secondary.clear();
		occ_begin.clear();
		occ.clear();
	}

	const Item* items_of(const Transaction& t) const { return t.own >= 0 ? items.data() + t.own : t.items; }

	// Appends a transaction, merging it into the previous one when their items are
	// equal. A view must point outside this Level; otherwise the data is copied. n >= 1.
	void add(const Item* it, const Util* ut, std::uint32_t n, Util prefix, bool view) {
		if (!txs.empty()) {
			Transaction& last = txs.back();
			// Transactions arrive in reverse-lexicographic order, so a different neighbour usually
			// differs in its last item: test that before comparing the whole suffix.
			const Item* last_items = items_of(last);
			if (last.len == n && last_items[n - 1] == it[n - 1] && std::equal(it, it + n - 1, last_items)) {
				if (last.own < 0) {
					last.own = static_cast<std::int64_t>(items.size());
					append(last.items, last.utils, n);
				}
				Util* dst = utils.data() + last.own;
				for (std::uint32_t i = 0; i < n; ++i) dst[i] += ut[i];
				last.prefix += prefix;
				return;
			}
		}
		if (view) {
			txs.push_back({it, ut, n, prefix, -1});
		} else {
			txs.push_back({nullptr, nullptr, n, prefix, static_cast<std::int64_t>(items.size())});
			append(it, ut, n);
		}
	}

	// Appends n items/utilities to the owned storage. resize() leaves the new elements
	// uninitialised (HugeAllocator default-initialises), so this is a capacity check and two copies.
	void append(const Item* it, const Util* ut, std::uint32_t n) {
		const std::size_t at = items.size();
		if (at + n > items.capacity()) {
			const std::size_t cap = std::max<std::size_t>(2 * items.capacity(), at + n);
			items.reserve(cap);
			utils.reserve(cap);
		}
		items.resize(at + n);
		utils.resize(at + n);
		std::copy(it, it + n, items.data() + at);
		std::copy(ut, ut + n, utils.data() + at);
	}

	// Turns owned offsets into pointers; call once every transaction has been added.
	void finalize() {
		for (auto& t : txs) {
			if (t.own < 0) continue;
			t.items = items.data() + t.own;
			t.utils = utils.data() + t.own;
		}
	}

	// Builds the occurrence index for `primary`, packing positions into the low `shift` bits. `slot` is
	// indexed by item and must be all -1; it is left that way. With threads > 1 the transactions are
	// counted and placed in parallel ranges; occurrences stay in ascending transaction order.
	void index(std::vector<std::int32_t>& slot, std::uint32_t shift, int threads = 1) {
		occ_shift = shift;
		for (std::size_t k = 0; k < primary.size(); ++k) slot[primary[k]] = static_cast<std::int32_t>(k);
		const std::size_t K = primary.size();
		if (threads <= 1 || txs.size() < 4096) {
			occ_begin.assign(K + 1, 0);
			for (const auto& t : txs)
				for (std::uint32_t i = 0; i < t.len; ++i)
					if (const std::int32_t s = slot[t.items[i]]; s >= 0) ++occ_begin[s + 1];
			std::partial_sum(occ_begin.begin(), occ_begin.end(), occ_begin.begin());
			occ.resize(occ_begin.back());
			cursor.assign(occ_begin.begin(), occ_begin.end() - 1);
			for (std::uint32_t tid = 0; tid < txs.size(); ++tid) {
				const auto& t = txs[tid];
				for (std::uint32_t i = 0; i < t.len; ++i)
					if (const std::int32_t s = slot[t.items[i]]; s >= 0) occ[cursor[s]++] = make_occurrence(tid, i, shift);
			}
			for (const Item x : primary) slot[x] = -1;
			return;
		}
		const int parts = threads;
		std::vector<std::size_t> cut(static_cast<std::size_t>(parts) + 1);
		for (int p = 0; p <= parts; ++p) cut[p] = txs.size() / parts * p;
		cut[parts] = txs.size();
		std::vector<std::vector<std::uint32_t>> count(parts, std::vector<std::uint32_t>(K, 0));
		parallel_run(parts, [&](int p) {
			for (std::size_t tid = cut[p]; tid < cut[p + 1]; ++tid)
				for (std::uint32_t i = 0; i < txs[tid].len; ++i)
					if (const std::int32_t s = slot[txs[tid].items[i]]; s >= 0) ++count[p][s];
		});
		occ_begin.assign(K + 1, 0);
		for (std::size_t s = 0; s < K; ++s) {
			std::uint32_t at = occ_begin[s];
			for (int p = 0; p < parts; ++p) {
				const std::uint32_t c = count[p][s];
				count[p][s] = at;  // this part's first slot for item s
				at += c;
			}
			occ_begin[s + 1] = at;
		}
		occ.resize(occ_begin.back());
		parallel_run(parts, [&](int p) {
			for (std::size_t tid = cut[p]; tid < cut[p + 1]; ++tid) {
				const auto& t = txs[tid];
				for (std::uint32_t i = 0; i < t.len; ++i)
					if (const std::int32_t s = slot[t.items[i]]; s >= 0)
						occ[count[p][s]++] = make_occurrence(static_cast<std::uint32_t>(tid), i, shift);
			}
		});
		for (const Item x : primary) slot[x] = -1;
	}

	// Parallel copy of a Level whose transactions are all owned (the root): shape_copy() sizes the
	// arrays (left uninitialised), then fill_part(src, p, parts) for p in [0, parts) copies slices,
	// each from its own thread, so the copy is first-touched by the threads that will use it.
	std::shared_ptr<Level> shape_copy() const {
		auto copy = std::make_shared<Level>();
		copy->items.resize(items.size());
		copy->utils.resize(utils.size());
		copy->txs.resize(txs.size());
		copy->occ.resize(occ.size());
		copy->occ_shift = occ_shift;
		copy->primary = primary;
		copy->secondary = secondary;
		copy->occ_begin = occ_begin;
		return copy;
	}
	void fill_part(const Level& src, int p, int parts) {
		const auto slice = [&](std::size_t n) { return std::pair{n / parts * p, p + 1 == parts ? n : n / parts * (p + 1)}; };
		const auto [i0, i1] = slice(src.items.size());
		std::copy(src.items.begin() + i0, src.items.begin() + i1, items.begin() + i0);
		std::copy(src.utils.begin() + i0, src.utils.begin() + i1, utils.begin() + i0);
		const auto [o0, o1] = slice(src.occ.size());
		std::copy(src.occ.begin() + o0, src.occ.begin() + o1, occ.begin() + o0);
		const auto [t0, t1] = slice(src.txs.size());
		for (std::size_t i = t0; i < t1; ++i) {
			Transaction t = src.txs[i];
			t.items = items.data() + t.own;
			t.utils = utils.data() + t.own;
			txs[i] = t;
		}
	}

	// A self-contained copy whose transactions all point into its own storage.
	std::shared_ptr<Level> materialize() const {
		auto copy = std::make_shared<Level>();
		std::size_t total = 0;
		for (const auto& t : txs) total += t.len;
		copy->items.reserve(total);
		copy->utils.reserve(total);
		copy->txs.reserve(txs.size());
		for (const auto& t : txs) {
			copy->txs.push_back({nullptr, nullptr, t.len, t.prefix, static_cast<std::int64_t>(copy->items.size())});
			copy->items.insert(copy->items.end(), t.items, t.items + t.len);
			copy->utils.insert(copy->utils.end(), t.utils, t.utils + t.len);
		}
		copy->finalize();
		copy->primary = primary;
		copy->secondary = secondary;
		copy->occ_begin = occ_begin;
		copy->occ = occ;
		copy->occ_shift = occ_shift;
		return copy;
	}
};

}  // namespace REFIM_WIDTHS
