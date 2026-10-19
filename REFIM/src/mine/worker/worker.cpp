#include "mine/worker.hpp"

#include <chrono>
#include <string_view>

Worker::Worker(int id, std::size_t num_items, Utility min_util, int threads, Scheduler& scheduler, bool collect_patterns,
			   int refine, bool compact, double refine_min_drop, bool leaf)
	: id(id), min_util(min_util), threads(threads), scheduler(scheduler), collect_patterns(collect_patterns), refine(refine),
	  compact(compact), refine_min_drop(refine_min_drop), leaf(leaf),
	  mark(num_items, -1), bound(num_items),
	  slot(num_items, -1), filter(filter_kernel()),
	  fused(std::string_view(simd_name()) == "scalar") {}

void Worker::run() {
	Task task;
	while (scheduler.pop(id, task)) {
		const auto start = std::chrono::steady_clock::now();
		execute(task);
		busy_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
		task = {};  // release the database before reporting completion
		scheduler.done();
	}
}

void Worker::execute(const Task& task) {
	if (collect_patterns) prefix_items = task.prefix;
	if (task.depth > 0)
		for (const Item x : task.level->secondary) mark[x] = task.depth;
	expand(*task.level, task.depth, task.slot);
	if (task.depth > 0)
		for (const Item x : task.level->secondary) mark[x] = -1;
}

Level& Worker::level_at(std::size_t depth) {
	while (levels.size() <= depth) levels.push_back(std::make_unique<Level>());
	return *levels[depth];
}
