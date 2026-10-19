#include "mine/worker.hpp"

Utility Worker::leaf_utility(const Level& db, std::size_t k) const {
	Utility utility = 0;
	for (std::uint32_t o = db.occ_begin[k]; o < db.occ_begin[k + 1]; ++o) {
		const Transaction& t = db.txs[db.occ[o].tid];
		utility += t.prefix + t.utils[db.occ[o].pos];
	}
	return utility;
}
