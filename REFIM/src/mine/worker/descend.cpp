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
		// frames may grow (and move) while the siblings recurse, so it is indexed, not referenced.
		const std::size_t f = frames.size();
		frames.push_back({&child, next, prefix_items.size(), 0, child.primary.size()});
		for (std::size_t c = 0; c < frames[f].end; ++c) {
			frames[f].next = c + 1;
			if (threads > 1 && scheduler.hungry()) offer();
			expand(child, next, c);
		}
		frames.pop_back();
		for (const Item x : child.secondary) mark[x] = depth > 0 ? depth : -1;
	}
}

void Worker::offer() {
	// An idle worker exists: hand it the siblings not started yet of the shallowest open loop, at any
	// depth and size, so one deep subtree of many small databases cannot keep a single thread busy.
	// The shallowest loop holds the largest subtrees left: offering the innermost one (as before)
	// handed out tasks of a few microseconds and BMS spent most of its time stealing them.
	for (Frame& frame : frames) {
		if (frame.next >= frame.end) continue;
		std::shared_ptr<const Level> shared = frame.level->materialize();
		const std::vector<Item> prefix(prefix_items.begin(), prefix_items.begin() + frame.prefix_len);
		std::vector<Task> tasks;
		tasks.reserve(frame.end - frame.next);
		for (std::size_t r = frame.next; r < frame.end; ++r)
			tasks.push_back({shared, frame.depth, static_cast<std::uint32_t>(r), prefix, true});
		frame.end = frame.next;
		scheduler.push(id, std::move(tasks));
		return;
	}
}

}  // namespace REFIM_WIDTHS
