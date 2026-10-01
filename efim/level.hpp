#pragma once
// Databases of the EFIM search: transactions, and the occurrence index used to project them.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

using Item = std::int32_t;  // renamed item id, 1..numItems (larger id = larger TWU)
using Utility = std::int64_t;

// A transaction of a (projected) database, items sorted ascending. A view points into
// an ancestor's storage; an owned transaction lives in its own Level's storage.
struct Transaction {
	const Item* items;
	const Utility* utils;
	std::uint32_t len;
	Utility prefix;    // utility of the current prefix itemset in this transaction
	std::int64_t own;  // offset into Level::items/utils, or -1 for a view
};

struct Occurrence {
	std::uint32_t tid;
	std::uint32_t pos;
};

// A (projected) database plus the occurrence index used to project it further.
struct Level {
	std::vector<Item> items;  // storage for owned transactions
	std::vector<Utility> utils;
	std::vector<Transaction> txs;
	std::vector<Item> primary;    // items to extend with (subtree utility >= minUtil), ascending
	std::vector<Item> secondary;  // items kept in deeper projections (local utility >= minUtil)
	std::vector<std::uint32_t> occ_begin;  // primary[k] occurs at occ[occ_begin[k], occ_begin[k + 1])
	std::vector<Occurrence> occ;
	std::vector<std::uint32_t> cursor;  // scratch for index()

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
	// equal. A view must point outside this Level; otherwise the data is copied.
	void add(const Item* it, const Utility* ut, std::uint32_t n, Utility prefix, bool view) {
		if (!txs.empty()) {
			Transaction& last = txs.back();
			if (last.len == n && std::equal(it, it + n, items_of(last))) {
				if (last.own < 0) {
					last.own = static_cast<std::int64_t>(items.size());
					items.insert(items.end(), last.items, last.items + n);
					utils.insert(utils.end(), last.utils, last.utils + n);
				}
				Utility* dst = utils.data() + last.own;
				for (std::uint32_t i = 0; i < n; ++i) dst[i] += ut[i];
				last.prefix += prefix;
				return;
			}
		}
		if (view) {
			txs.push_back({it, ut, n, prefix, -1});
		} else {
			txs.push_back({nullptr, nullptr, n, prefix, static_cast<std::int64_t>(items.size())});
			items.insert(items.end(), it, it + n);
			utils.insert(utils.end(), ut, ut + n);
		}
	}

	// Turns owned offsets into pointers; call once every transaction has been added.
	void finalize() {
		for (auto& t : txs) {
			if (t.own < 0) continue;
			t.items = items.data() + t.own;
			t.utils = utils.data() + t.own;
		}
	}

	// Builds the occurrence index for `primary`. `slot` is indexed by item and must be
	// all -1; it is left that way.
	void index(std::vector<std::int32_t>& slot) {
		for (std::size_t k = 0; k < primary.size(); ++k) slot[primary[k]] = static_cast<std::int32_t>(k);
		occ_begin.assign(primary.size() + 1, 0);
		for (const auto& t : txs)
			for (std::uint32_t i = 0; i < t.len; ++i)
				if (const std::int32_t s = slot[t.items[i]]; s >= 0) ++occ_begin[s + 1];
		std::partial_sum(occ_begin.begin(), occ_begin.end(), occ_begin.begin());
		occ.resize(occ_begin.back());
		cursor.assign(occ_begin.begin(), occ_begin.end() - 1);
		for (std::uint32_t tid = 0; tid < txs.size(); ++tid) {
			const auto& t = txs[tid];
			for (std::uint32_t i = 0; i < t.len; ++i)
				if (const std::int32_t s = slot[t.items[i]]; s >= 0) occ[cursor[s]++] = {tid, i};
		}
		for (const Item x : primary) slot[x] = -1;
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
		return copy;
	}
};
