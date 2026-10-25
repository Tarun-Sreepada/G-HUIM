#pragma once
// The root tasks of a GREFIM run, largest first, for every engine.
//
// CPU workers take one slot at a time from the front: the largest subtrees, which REFIM splits over all its
// workers. Small subtrees are the CPU's worst case (a whole root projection for few itemsets) and a GPU's
// best (the projection is one wide pass, and many of them share each level's passes).
//
// GPUs take batches from the back. A fast GPU (discrete, or any GPU if none is) takes by guided
// self-scheduling: half its share of the estimated work left, its share being its measured rate over the
// rate of every engine. A slow GPU (integrated, next to a discrete one) does not even set up until the fast
// GPUs have measured themselves, and joins only if they still have more than eight of their own setup times
// of work left (a slow GPU must set up, measure itself and still win back what it costs the others in shared
// memory bandwidth). It then measures itself on one slot,
// then takes only work it will finish before the fast GPUs would empty the pool:
// work / own rate <= (work left - work) / fast rate, and stops once that fails. Both GPU rates come from
// slots of the same end, in estimated work per second, so they compare.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <vector>

namespace grefim {

class Pool {
public:
	// slots[i] with estimated work work[i], in the order to hand out (largest first); fast[g]: GPU g is fast.
	Pool(std::vector<std::uint32_t> slots, std::vector<double> work, int cpu_workers, std::vector<bool> fast);

	// Next slot for CPU worker `worker`, which finished the last one it took; false once the pool is empty.
	bool take_one(int worker, std::uint32_t& slot);
	// A batch for GPU `gpu` (empty once it should stop), with its estimated work.
	std::vector<std::uint32_t> take_batch(int gpu, double& work);
	// GPU `gpu` mined `work` in `seconds`.
	void report(int gpu, double work, double seconds);
	// GPU `gpu` takes no more (done or failed).
	void retire(int gpu);
	// A fast GPU's setup took `seconds`.
	void report_setup(double seconds);
	// Slow GPU `gpu`: waits until joining can be decided; false (and retired) if it should not.
	bool join(int gpu);

	double total_work() const { return total_; }
	double cpu_work() const { return cpu_work_; }
	std::size_t cpu_slots() const { return cpu_slots_; }

private:
	using Clock = std::chrono::steady_clock;
	struct Taken {
		double work = 0;
		Clock::time_point at;
		bool open = false;
	};

	double rate_of(std::size_t g) const { return gpu_seconds_[g] > 0 ? gpu_work_[g] / gpu_seconds_[g] : 0; }
	double fast_rate() const;   // of the running fast GPUs that measured one
	bool fast_waiting() const;  // some fast GPU still runs and none has a rate yet
	bool empty() const { return next_ == end_; }
	double cpu_rate() const { return cpu_seconds_ > 0 ? cpu_done_ / cpu_seconds_ : 0; }  // per worker
	// Whether a slow engine of `rate` should take `work` more.
	bool worth(double work, double rate) const;
	std::size_t take_back();
	std::size_t take_front();

	std::mutex mutex_;
	std::condition_variable changed_;
	std::vector<std::uint32_t> slots_;
	std::vector<double> work_;
	std::size_t next_ = 0, end_ = 0;  // slots [next_, end_) are left
	double total_ = 0, left_ = 0;
	// CPU: work taken and slots; work finished and the worker time it took.
	double cpu_work_ = 0, cpu_done_ = 0, cpu_seconds_ = 0;
	std::size_t cpu_slots_ = 0;
	int cpu_workers_;
	std::vector<Taken> taken_;  // per CPU worker
	std::vector<bool> fast_, retired_;
	std::vector<double> gpu_work_, gpu_seconds_;
	double fast_setup_ = 0;
};

}  // namespace grefim
