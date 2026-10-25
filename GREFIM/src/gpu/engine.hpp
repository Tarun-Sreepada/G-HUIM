#pragma once
// GPU engine: mines whole subtrees of the root on one Vulkan device, level by level, like G-HUIM's BFS.
//
// A database lives in the arena (32-bit words; the root at word 0) as
//   n_trans, n_items, n_entries, 0 | names[n_items] | (begin, length, prefix utility)[n_trans] |
//   items[n_entries] | utils[n_entries]
// with items named 0..n_items-1 in ascending order (names[] maps them to REFIM's ids). A task is
// (database word, item); shaders/expand.comp runs one per workgroup and writes each child database, compacted
// to its own items, and the child's tasks. Tasks come in two classes: on small databases, with workgroups of
// two subgroups, and on big ones (at least a big workgroup's worth of transactions), with the largest
// workgroup the device runs. The root carries an occurrence index, so a root task visits only the
// transactions holding its item.
//
// A level of tasks is one dispatch and one wait. The arena is double-ended, like G-HUIM's two regions:
// even levels grow up from the bottom, odd levels down from the top, so a level's children are written
// facing it. Once every task of a level is done its databases are dead, and its grandchildren overwrite
// them: memory holds about two levels. A pass may fill half the space between it and the facing level;
// tasks that get no room are retried once the children that fit are mined (adaptive partitioning).

#include "vk/context.hpp"

#include <cstdint>
#include <deque>
#include <vector>

namespace grefim {

// The root database in the engine's layout (built from REFIM's root); primary[k] is the item of root slot k.
struct GpuRoot {
	std::vector<std::uint32_t> words;  // the root database, with its occurrence index, as stored at arena word 0
	std::vector<std::uint32_t> primary;
	std::uint32_t items = 0;  // of the root database
};

class Engine {
public:
	Engine(vk::Device& device, const GpuRoot& root, std::uint32_t min_util);
	~Engine();
	Engine(const Engine&) = delete;
	Engine& operator=(const Engine&) = delete;

	// Mines the subtrees of the root slots (the itemsets starting with root.primary[slot]) completely.
	void mine(const std::vector<std::uint32_t>& slots);

	std::uint64_t pattern_count = 0;
	std::uint64_t tasks = 0, passes = 0, retried = 0;

private:
	// One level's task lists: per class, the input and the retries of a pass (they swap roles).
	struct Level {
		vk::Buffer lists[2][2];
	};
	struct Control {
		std::uint64_t patterns, used, next[2];
		std::uint32_t retry[2];
	};
	// The push constants of shaders/expand.comp.
	struct Push {
		std::uint64_t arena, tasks, next, next_big, retry, control;
		std::uint32_t count, min_util, task_cap, start, down, limit, groups_x, task_class, big;
	};
	// Arena words [lo, hi).
	struct Span {
		std::uint64_t lo, hi;
	};

	// Mines `count` tasks of depth `depth`, whose databases fill `mine`. Their children go from `from`
	// towards `frontier` (down for odd depths + 1, i.e. levels on the top end).
	void mine_level(std::size_t depth, const std::uint32_t (&count)[2], Span mine, std::uint64_t from, std::uint64_t frontier);
	Level& level(std::size_t depth);

	vk::Device& device_;
	const std::uint32_t min_util_;
	std::uint32_t workgroup_[2] = {}, task_cap_ = 0;
	std::uint64_t root_words_ = 0, arena_words_ = 0;
	std::vector<std::uint32_t> primary_;
	vk::Buffer arena_, control_, readback_;
	VkPipeline expand_[2];  // per task class
	std::deque<Level> levels_;  // levels_[d]: tasks of depth d (deque: references survive growth)
};

}  // namespace grefim
