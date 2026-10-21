#include "mine/worker.hpp"

inline namespace REFIM_WIDTHS {

void Worker::descend(Level& child, int depth) {
	if (child.primary.empty()) return;
	child.index(slot, occ_shift);

	const int next = depth + 1;
	const std::uint64_t work = std::uint64_t{child.occ.size()} * child.primary.size();
	const bool split = threads > 1 && ((next <= kMaxSplitDepth && work >= kSplitWork) ||
										(next <= kMaxHungryDepth && work >= kHungryWork && scheduler.hungry()));
	if (split) {
		std::shared_ptr<const Level> shared = child.materialize();
		std::vector<Task> tasks;
		tasks.reserve(child.primary.size());
		for (std::uint32_t c = 0; c < child.primary.size(); ++c)
			tasks.push_back({shared, next, c, prefix_items});
		scheduler.push(id, std::move(tasks));
	} else {
		for (const Item x : child.secondary) mark[x] = next;
		for (std::size_t c = 0; c < child.primary.size(); ++c) {
			// An idle worker exists: hand it the siblings not started yet, at any depth and size,
			// so one deep subtree of many small databases cannot keep a single thread busy.
			// Measured on BMS: a size threshold here left most workers idle.
			const std::size_t left = child.primary.size() - c;
			if (threads > 1 && left > 1 && scheduler.hungry()) {
				std::shared_ptr<const Level> shared = child.materialize();
				std::vector<Task> tasks;
				tasks.reserve(left - 1);
				for (std::size_t r = c + 1; r < child.primary.size(); ++r)
					tasks.push_back({shared, next, static_cast<std::uint32_t>(r), prefix_items});
				scheduler.push(id, std::move(tasks));
				expand(child, next, c);
				break;
			}
			expand(child, next, c);
		}
		for (const Item x : child.secondary) mark[x] = depth > 0 ? depth : -1;
	}
}

}  // namespace REFIM_WIDTHS
