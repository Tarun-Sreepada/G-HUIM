#include "mine/worker.hpp"

void Worker::record_pattern(Utility utility) {
	if (utility < min_util) return;
	++pattern_count;
	if (!collect_patterns) return;
	patterns.items.insert(patterns.items.end(), prefix_items.begin(), prefix_items.end());
	patterns.begin.push_back(patterns.items.size());
	patterns.utils.push_back(utility);
}
