#include "mine/worker.hpp"

void Worker::expand(const Level& db, int depth, std::size_t k) {
	const Item item = db.primary[k];
	// Leaf: no extension of the prefix comes after `item` (the root keeps every item, the largest
	// id last), so the projection would be empty and only the itemset's utility is needed.
	const Item top = depth == 0 ? static_cast<Item>(bound.size() - 1) : db.secondary.back();
	if (leaf && item >= top) {
		++stats.leaves;
		if (collect_patterns) prefix_items.push_back(item);
		record_pattern(leaf_utility(db, k));
		if (collect_patterns) prefix_items.pop_back();
		return;
	}
	Level& child = level_at(depth + 1);
	if (collect_patterns) prefix_items.push_back(item);
	record_pattern(project_child(db, depth, k, child));
	select_extensions(child, depth);
	descend(child, depth);
	if (collect_patterns) prefix_items.pop_back();
}
