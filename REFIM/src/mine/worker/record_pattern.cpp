#include "mine/worker.hpp"

inline namespace REFIM_WIDTHS {

void Worker::record_pattern(Util utility) {
	if (utility < min_util) return;
	++pattern_count;
	if (!collect_patterns) return;
	patterns.items.insert(patterns.items.end(), prefix_items.begin(), prefix_items.end());
	patterns.begin.push_back(patterns.items.size());
	patterns.utils.push_back(utility);
}

}  // namespace REFIM_WIDTHS
