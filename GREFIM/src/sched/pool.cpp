#include "sched/pool.hpp"

#include <algorithm>
#include <numeric>

namespace grefim {

Pool::Pool(std::vector<std::uint32_t> slots, std::vector<double> work, int cpu_workers, std::vector<bool> fast)
	: slots_(std::move(slots)), work_(std::move(work)), cpu_workers_(cpu_workers), fast_(std::move(fast)), retired_(fast_.size()),
	  gpu_work_(fast_.size()), gpu_seconds_(fast_.size()) {
	total_ = left_ = std::accumulate(work_.begin(), work_.end(), 0.0);
	end_ = slots_.size();
}

double Pool::fast_rate() const {
	double rate = 0;
	for (std::size_t g = 0; g < fast_.size(); ++g)
		if (fast_[g] && !retired_[g]) rate += rate_of(g);
	return rate;
}

bool Pool::fast_waiting() const {
	bool running = false;
	for (std::size_t g = 0; g < fast_.size(); ++g)
		if (fast_[g] && !retired_[g]) {
			if (rate_of(g) > 0) return false;
			running = true;
		}
	return running;
}

bool Pool::worth(double work, double rate) const {
	const double fast = fast_rate();
	if (fast == 0) return true;  // no fast GPU (left): every engine takes what it can
	return work / rate <= (left_ - work) / fast;
}

std::size_t Pool::take_back() {
	const std::size_t i = --end_;
	left_ -= work_[i];
	return i;
}

std::size_t Pool::take_front() {
	const std::size_t i = next_++;
	left_ -= work_[i];
	return i;
}

bool Pool::take_one(int worker, std::uint32_t& slot) {
	std::unique_lock lock(mutex_);
	if (taken_.size() <= static_cast<std::size_t>(worker)) taken_.resize(static_cast<std::size_t>(worker) + 1);
	const Clock::time_point now = Clock::now();
	if (Taken& mine = taken_[static_cast<std::size_t>(worker)]; mine.open) {
		cpu_done_ += mine.work;
		cpu_seconds_ += std::chrono::duration<double>(now - mine.at).count();
		mine.open = false;
	}
	if (empty()) return false;
	const std::size_t i = take_front();
	slot = slots_[i];
	cpu_work_ += work_[i];
	++cpu_slots_;
	taken_[static_cast<std::size_t>(worker)] = {work_[i], Clock::now(), true};
	changed_.notify_all();
	return true;
}

std::vector<std::uint32_t> Pool::take_batch(int gpu, double& work) {
	std::unique_lock lock(mutex_);
	const auto g = static_cast<std::size_t>(gpu);
	work = 0;
	std::vector<std::uint32_t> batch;
	if (!fast_[g]) {
		// Slow GPU: one slot to measure itself, then (once the fast GPUs have a rate) from the back while it
		// is worth it, up to half its share.
		changed_.wait(lock, [&] { return empty() || !fast_waiting(); });
		const double own = rate_of(g);
		const double target = own > 0 ? left_ * own / (own + fast_rate() + cpu_rate() * cpu_workers_) / 2 : 0;
		while (!empty()) {
			const double w = work_[end_ - 1];
			if (own > 0 && (!worth(work + w, own) || (!batch.empty() && work + w > target))) break;
			const std::size_t i = take_back();
			batch.push_back(slots_[i]);
			work += work_[i];
			if (own == 0) break;
		}
		if (batch.empty()) retired_[g] = true;
		changed_.notify_all();
		return batch;
	}
	if (empty()) return batch;
	// Fast GPU: guided self-scheduling over the rates of every engine still taking work.
	const double own = rate_of(g);
	double share = 1.0 / static_cast<double>(cpu_workers_ + static_cast<int>(fast_.size()));
	if (own > 0) {
		double all = cpu_rate() * cpu_workers_;
		for (std::size_t h = 0; h < fast_.size(); ++h)
			if (!retired_[h]) all += rate_of(h) > 0 ? rate_of(h) : (fast_[h] ? own : 0);
		share = own / std::max(all, own);
	}
	const double target = left_ * share / 2;
	do {
		const std::size_t i = take_back();
		batch.push_back(slots_[i]);
		work += work_[i];
	} while (next_ < end_ && work + work_[end_ - 1] <= target);
	changed_.notify_all();
	return batch;
}

void Pool::report(int gpu, double work, double seconds) {
	const std::lock_guard lock(mutex_);
	gpu_work_[static_cast<std::size_t>(gpu)] += work;
	gpu_seconds_[static_cast<std::size_t>(gpu)] += seconds;
	changed_.notify_all();
}

void Pool::report_setup(double seconds) {
	const std::lock_guard lock(mutex_);
	fast_setup_ = std::max(fast_setup_, seconds);
}

bool Pool::join(int gpu) {
	std::unique_lock lock(mutex_);
	changed_.wait(lock, [&] { return empty() || !fast_waiting(); });
	const double fast = fast_rate();
	if (fast == 0 || (!empty() && left_ / fast > 8 * fast_setup_)) return true;
	retired_[static_cast<std::size_t>(gpu)] = true;
	return false;
}

void Pool::retire(int gpu) {
	const std::lock_guard lock(mutex_);
	retired_[static_cast<std::size_t>(gpu)] = true;
	changed_.notify_all();
}

}  // namespace grefim
