#include "mine/worker.hpp"

inline namespace REFIM_WIDTHS {

void Worker::compact_child(Level& child, int depth) {
	while (spare.size() <= static_cast<std::size_t>(depth)) spare.push_back(std::make_unique<Level>());
	Level& out = *spare[depth];
	out.clear();
	++stats.compacted;
	stats.compact_tx_in += child.txs.size();
	for (const auto& t : child.txs) {
		stats.compact_items_in += t.len;
		if (new_key.size() < t.len) {
			new_key.resize(2 * t.len);
			new_val.resize(2 * t.len);
		}
		std::uint32_t kept = 0;
		for (std::uint32_t i = 0; i < t.len; ++i) {
			if (mark[t.items[i]] != depth) continue;
			new_key[kept] = t.items[i];
			new_val[kept++] = t.utils[i];
		}
		if (kept == 0) continue;
		// An unchanged transaction stays a view of the old storage (an ancestor's, or child's own,
		// which moves to spare[depth] below and so outlives it).
		if (kept == t.len)
			out.add(t.items, t.utils, kept, t.prefix, true);
		else
			out.add(new_key.data(), new_val.data(), kept, t.prefix, false);
	}
	out.finalize();
	stats.compact_tx_out += out.txs.size();
	for (const auto& t : out.txs) stats.compact_items_out += t.len;
	std::swap(child, out);
}

}  // namespace REFIM_WIDTHS
