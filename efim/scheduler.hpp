#pragma once
// Work shared between search threads: tasks (subtrees to expand) and the stack that holds them.

#include "level.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

// Subtrees are split into tasks for other threads only near the top of the search
// and only when their database * extension count is large enough to be worth the copy.
constexpr int kMaxSplitDepth = 2;
constexpr std::uint64_t kSplitWork = std::uint64_t{1} << 20;

// Patterns stored back to back: pattern p is items[begin[p], begin[p + 1]).
struct Patterns {
	std::vector<Item> items;
	std::vector<std::size_t> begin{0};
	std::vector<Utility> utils;

	std::size_t size() const { return utils.size(); }
};

// Extend `prefix` with level->primary[slot], where `level` is the database at `depth`.
struct Task {
	std::shared_ptr<const Level> level;
	int depth;
	std::uint32_t slot;
	std::vector<Item> prefix;
};

// LIFO task stack shared by the workers; pop() returns false once all work is done.
class Scheduler {
public:
	void push(std::vector<Task>&& tasks) {
		{
			std::lock_guard<std::mutex> lock(mutex_);
			outstanding_ += tasks.size();
			// Reversed so the lowest slot is popped first.
			for (auto it = tasks.rbegin(); it != tasks.rend(); ++it) stack_.push_back(std::move(*it));
		}
		cv_.notify_all();
	}

	bool pop(Task& out) {
		std::unique_lock<std::mutex> lock(mutex_);
		cv_.wait(lock, [&] { return !stack_.empty() || outstanding_ == 0; });
		if (stack_.empty()) return false;
		out = std::move(stack_.back());
		stack_.pop_back();
		return true;
	}

	void done() {
		std::lock_guard<std::mutex> lock(mutex_);
		if (--outstanding_ == 0) cv_.notify_all();
	}

private:
	std::mutex mutex_;
	std::condition_variable cv_;
	std::vector<Task> stack_;
	std::size_t outstanding_ = 0;  // queued + running tasks
};
