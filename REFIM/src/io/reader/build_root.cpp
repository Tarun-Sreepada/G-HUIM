#include "io/reader/root_steps.hpp"

#include "sys/parallel.hpp"

#include <algorithm>

inline namespace REFIM_WIDTHS {

// Builds the root in parallel ranges of the sorted order (merging equal neighbours inside
// each range), then concatenates. Range cuts move past runs of equal transactions so no
// merge has to cross a boundary.
void build_root(const Flat& flat, const std::vector<std::uint32_t>& perm, int threads, Level& root) {
	root.clear();
	const auto same = [&](std::uint32_t a, std::uint32_t b) {
		return flat.begin[a + 1] - flat.begin[a] == flat.begin[b + 1] - flat.begin[b] &&
			   std::equal(flat.keys.get() + flat.begin[a], flat.keys.get() + flat.begin[a + 1], flat.keys.get() + flat.begin[b]);
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
			piece[p].add(flat.keys.get() + flat.begin[t], flat.vals.get() + flat.begin[t], static_cast<std::uint32_t>(flat.begin[t + 1] - flat.begin[t]), 0, false);
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

}  // namespace REFIM_WIDTHS
