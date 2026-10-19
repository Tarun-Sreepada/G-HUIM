#pragma once
// Work shared between search threads: tasks (subtrees to expand) and the stack that holds them.

#include "mine/level.hpp"

#include <atomic>
#include <cstddef>
#include <deque>
#include <memory>
#include <thread>
#include <vector>

// Subtrees are split into tasks for other threads only when their database * extension count
// is large enough to be worth the copy: always near the top of the search, and deeper whenever
// some worker is idle (so the tail of the run stays balanced).
constexpr int kMaxSplitDepth = 2;
constexpr std::uint64_t kSplitWork = std::uint64_t{1} << 20;
constexpr int kMaxHungryDepth = 8;
constexpr std::uint64_t kHungryWork = std::uint64_t{1} << 14;

// Patterns stored back to back: pattern p is items[begin[p], begin[p + 1]).
struct Patterns {
	HugeVector<Item> items;
	HugeVector<std::size_t> begin{0};
	HugeVector<Utility> utils;

	std::size_t size() const { return utils.size(); }
};

// Extend `prefix` with level->primary[slot], where `level` is the database at `depth`.
struct Task {
	std::shared_ptr<const Level> level;
	int depth;
	std::uint32_t slot;
	std::vector<Item> prefix;
};

// Work-stealing scheduler: one deque per worker. The owner pushes and pops at the back (depth
// first, cache-warm); thieves take from the front, i.e. the oldest and so largest subtree, trying
// victims in the order given (same L3 group, same NUMA node, then remote). Tasks are coarse
// (a split materialises a database), so a spin lock per deque is enough.
class Scheduler {
public:
	// victims[w] = the other workers in the order w should steal from them.
	explicit Scheduler(std::vector<std::vector<int>> victims)
		: victims_(std::move(victims)), queues_(victims_.size()), steals_(victims_.size()) {}

	// Seeding before the workers start.
	void seed(int worker, Task&& task) {
		outstanding_.fetch_add(1, std::memory_order_relaxed);
		queues_[worker].q.push_back(std::move(task));
		queues_[worker].size.store(queues_[worker].q.size(), std::memory_order_relaxed);
	}

	void push(int self, std::vector<Task>&& tasks) {
		outstanding_.fetch_add(tasks.size(), std::memory_order_relaxed);
		{
			Queue& q = queues_[self];
			q.lock();
			// Reversed so the lowest slot is popped first.
			for (auto it = tasks.rbegin(); it != tasks.rend(); ++it) q.q.push_back(std::move(*it));
			q.size.store(q.q.size(), std::memory_order_relaxed);
			q.unlock();
		}
		epoch_.fetch_add(1);
		if (idle_.load() > 0) epoch_.notify_all();
	}

	// Next task for `self`: its own, else stolen, else wait. Returns false once all work is done.
	bool pop(int self, Task& out) {
		while (true) {
			if (try_get(self, out)) return true;
			idle_.fetch_add(1);
			bool got = false;
			for (int spin = 0; spin < 64 && !got; ++spin) {
				if (outstanding_.load() == 0) break;
				got = try_get(self, out);
			}
			if (!got && outstanding_.load() != 0) {
				const unsigned e = epoch_.load();
				got = try_get(self, out);
				if (!got && outstanding_.load() != 0) epoch_.wait(e);
			}
			idle_.fetch_sub(1);
			if (got) return true;
			if (outstanding_.load() == 0) return false;
		}
	}

	void done() {
		if (outstanding_.fetch_sub(1) == 1) {
			epoch_.fetch_add(1);
			epoch_.notify_all();
		}
	}

	// True while some worker is looking for work.
	bool hungry() const { return idle_.load(std::memory_order_relaxed) > 0; }
	std::size_t steals(int worker) const { return steals_[worker].n; }

private:
	struct alignas(64) Queue {
		std::atomic_flag flag = ATOMIC_FLAG_INIT;
		std::atomic<std::size_t> size{0};
		std::deque<Task> q;
		void lock() {
			while (flag.test_and_set(std::memory_order_acquire)) {
				while (flag.test(std::memory_order_relaxed)) std::this_thread::yield();
			}
		}
		void unlock() { flag.clear(std::memory_order_release); }
	};
	struct alignas(64) Counter {
		std::size_t n = 0;
	};

	bool try_get(int self, Task& out) {
		Queue& own = queues_[self];
		if (own.size.load(std::memory_order_relaxed)) {
			own.lock();
			if (!own.q.empty()) {
				out = std::move(own.q.back());
				own.q.pop_back();
				own.size.store(own.q.size(), std::memory_order_relaxed);
				own.unlock();
				return true;
			}
			own.unlock();
		}
		for (const int v : victims_[self]) {
			Queue& q = queues_[v];
			if (!q.size.load(std::memory_order_relaxed)) continue;
			q.lock();
			if (!q.q.empty()) {
				out = std::move(q.q.front());
				q.q.pop_front();
				q.size.store(q.q.size(), std::memory_order_relaxed);
				q.unlock();
				++steals_[self].n;
				return true;
			}
			q.unlock();
		}
		return false;
	}

	std::vector<std::vector<int>> victims_;
	std::vector<Queue> queues_;
	std::vector<Counter> steals_;
	std::atomic<std::size_t> outstanding_{0};  // queued + running tasks
	std::atomic<unsigned> epoch_{0};           // bumped on every push / completion, for waiting
	std::atomic<int> idle_{0};
};
