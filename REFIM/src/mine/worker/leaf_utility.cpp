#include "mine/worker.hpp"

inline namespace REFIM_WIDTHS {

Util Worker::leaf_utility(const Level& db, std::size_t k) const {
	Util utility = 0;
	for (std::uint32_t o = db.occ_begin[k]; o < db.occ_begin[k + 1]; ++o) {
		const Transaction& t = db.txs[occurrence_tid(db.occ[o], db.occ_shift)];
		utility += t.prefix + t.utils[occurrence_pos(db.occ[o], db.occ_shift)];
	}
	return utility;
}

}  // namespace REFIM_WIDTHS
