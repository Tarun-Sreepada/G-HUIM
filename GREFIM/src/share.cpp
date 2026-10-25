#include "share.hpp"

#include "gpu/engine.hpp"
#include "mine/share.hpp"
#include "sched/pool.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <future>
#include <limits>
#include <numeric>
#include <thread>

inline namespace REFIM_WIDTHS {

namespace {

// The root in the engines' layout (gpu/engine.hpp): REFIM's ids name the items, so names[] is the identity.
grefim::GpuRoot to_gpu(const Level& root, std::size_t num_items) {
	grefim::GpuRoot g;
	g.items = static_cast<std::uint32_t>(num_items + 1);
	std::size_t entries = 0;
	for (const Transaction& t : root.txs) entries += t.len;
	const std::size_t n = root.txs.size();
	std::vector<std::uint32_t>& w = g.words;
	w.reserve(4 + g.items + 3 * n + 2 * entries);
	w.insert(w.end(), {static_cast<std::uint32_t>(n), g.items, static_cast<std::uint32_t>(entries), 0u});
	for (std::uint32_t x = 0; x < g.items; ++x) w.push_back(x);
	std::uint32_t begin = 0;
	for (const Transaction& t : root.txs) {
		w.insert(w.end(), {begin, t.len, static_cast<std::uint32_t>(t.prefix)});
		begin += t.len;
	}
	for (const Transaction& t : root.txs) w.insert(w.end(), t.items, t.items + t.len);
	for (const Transaction& t : root.txs) w.insert(w.end(), t.utils, t.utils + t.len);
	// Occurrence index: occ_begin[n_items + 1], then the transactions holding each item, ascending.
	w[3] = static_cast<std::uint32_t>(w.size());
	std::vector<std::uint32_t> count(g.items + 1, 0);
	for (const Transaction& t : root.txs)
		for (std::uint32_t j = 0; j < t.len; ++j) ++count[t.items[j] + 1];
	for (std::uint32_t x = 0; x < g.items; ++x) count[x + 1] += count[x];
	w.insert(w.end(), count.begin(), count.end());
	const std::size_t lists = w.size();
	w.resize(lists + count[g.items]);
	for (std::uint32_t tid = 0; tid < n; ++tid)
		for (std::uint32_t j = 0; j < root.txs[tid].len; ++j) w[lists + count[root.txs[tid].items[j]]++] = tid;
	g.primary.assign(root.primary.begin(), root.primary.end());
	return g;
}

class Share final : public RootShare {
public:
	explicit Share(const grefim::ShareConfig& config) : config_(config) {
		// The engines keep every utility in 32 bits and count itemsets without keeping them.
		const char* why = nullptr;
		if (std::numeric_limits<Util>::digits > 32) why = "utilities need 64 bits";
		if (config_.collect_patterns) why = "itemsets are written out (--output)";
		if (why && !config_.devices.empty()) {
			spdlog::warn("GPUs not used: {}", why);
			config_.devices.clear();
		}
		if (config_.devices.empty() && config_.cpu_workers == 0) config_.cpu_workers = 1;  // someone has to mine
		results_.resize(config_.devices.size());
	}

	~Share() override {
		for (std::thread& t : threads_)
			if (t.joinable()) t.join();
	}

	void start(std::shared_ptr<const Level> root, std::size_t num_items) override {
		start_ = std::chrono::steady_clock::now();
		// Work of slot k, estimated by the entries its projection reads: the items after it in every
		// transaction holding it. Slots are ordered largest first.
		const std::size_t roots = root->primary.size();
		std::vector<double> work(roots, 0);
		for (std::size_t k = 0; k < roots; ++k)
			for (std::uint32_t o = root->occ_begin[k]; o < root->occ_begin[k + 1]; ++o) {
				const Transaction& t = root->txs[occurrence_tid(root->occ[o], root->occ_shift)];
				work[k] += static_cast<double>(t.len - occurrence_pos(root->occ[o], root->occ_shift));
			}
		std::vector<std::uint32_t> order(roots);
		std::iota(order.begin(), order.end(), 0u);
		std::stable_sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) { return work[a] > work[b]; });
		std::vector<double> ordered_work(roots);
		for (std::size_t i = 0; i < roots; ++i) ordered_work[i] = work[order[i]];
		// Fast GPUs: the discrete ones, or every one if none is.
		std::vector<bool> fast;
		const bool any_discrete = std::any_of(config_.devices.begin(), config_.devices.end(),
											  [](const grefim::vk::Device* g) { return !g->info().integrated; });
		for (const grefim::vk::Device* g : config_.devices) fast.push_back(!any_discrete || !g->info().integrated);
		fast_ = fast;
		pool_ = std::make_unique<grefim::Pool>(std::move(order), std::move(ordered_work), config_.cpu_workers, std::move(fast));
		if (config_.devices.empty()) return;

		// The GPUs' copy of the root is built off the CPU workers' critical path, while they start mining.
		std::shared_future<std::shared_ptr<const grefim::GpuRoot>> gpu_root =
				std::async(std::launch::async, [root, num_items] { return std::make_shared<const grefim::GpuRoot>(to_gpu(*root, num_items)); });
		for (std::size_t d = 0; d < config_.devices.size(); ++d)
			threads_.emplace_back([this, d, gpu_root] { run_gpu(d, gpu_root.get()); });
	}

	bool next(int worker, std::uint32_t& slot) override { return config_.cpu_workers > 0 && pool_->take_one(worker, slot); }

	std::size_t finish(std::vector<Patterns>&) override {
		const double cpu_done = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
		for (std::thread& t : threads_) t.join();
		threads_.clear();
		std::size_t count = 0;
		const double total = std::max(pool_->total_work(), 1.0);
		for (std::size_t d = 0; d < results_.size(); ++d) {
			const Result& r = results_[d];
			if (r.error) std::rethrow_exception(r.error);
			if (r.slots == 0) {
				spdlog::info("gpu {} ({}): not used (the others finish sooner than it would help)", d, config_.devices[d]->info().name);
				continue;
			}
			spdlog::info("gpu {} ({}): {} root slots ({:.1f}% of the work), {} patterns, {} tasks ({} retried) in {} passes, "
						 "setup {:.3f} s, mining {:.3f} s, done at {:.3f} s",
						 d, config_.devices[d]->info().name, r.slots, 100 * r.work / total, r.count, r.tasks, r.retried, r.passes,
						 r.setup, r.seconds, r.done);
			count += r.count;
		}
		if (config_.cpu_workers > 0)
			spdlog::info("cpu: {} root slots ({:.1f}% of the work), done at {:.3f} s", pool_->cpu_slots(), 100 * pool_->cpu_work() / total,
						 cpu_done);
		return count;
	}

private:
	struct Result {
		std::size_t count = 0, slots = 0;
		double work = 0, seconds = 0, setup = 0, done = 0;  // done: seconds after start()
		std::uint64_t tasks = 0, passes = 0, retried = 0;
		std::exception_ptr error;
	};

	void run_gpu(std::size_t d, std::shared_ptr<const grefim::GpuRoot> root) {
		Result& r = results_[d];
		try {
			if (!fast_[d] && !pool_->join(static_cast<int>(d))) {
				spdlog::debug("gpu {}: not worth joining", d);
				return;
			}
			const auto t0 = std::chrono::steady_clock::now();
			grefim::Engine engine(*config_.devices[d], *root, static_cast<std::uint32_t>(config_.min_util));
			root.reset();
			r.setup = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
			if (fast_[d]) pool_->report_setup(r.setup);
			for (;;) {
				double work = 0;
				const std::vector<std::uint32_t> batch = pool_->take_batch(static_cast<int>(d), work);
				if (batch.empty()) break;
				const auto t1 = std::chrono::steady_clock::now();
				engine.mine(batch);
				const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
				pool_->report(static_cast<int>(d), work, seconds);
				r.slots += batch.size();
				r.work += work;
				r.seconds += seconds;
				spdlog::debug("gpu {}: {} slots, work {:.3g} in {:.3f} s", d, batch.size(), work, seconds);
			}
			r.count = engine.pattern_count;
			r.tasks = engine.tasks;
			r.passes = engine.passes;
			r.retried = engine.retried;
			r.done = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
		} catch (...) {
			r.error = std::current_exception();
		}
		pool_->retire(static_cast<int>(d));
	}

	grefim::ShareConfig config_;
	std::unique_ptr<grefim::Pool> pool_;
	std::vector<Result> results_;
	std::vector<std::thread> threads_;
	std::chrono::steady_clock::time_point start_;
	std::vector<bool> fast_;
};

}  // namespace

std::unique_ptr<RootShareBase> make_share(const grefim::ShareConfig& config) {
	return std::make_unique<Share>(config);
}

}  // namespace REFIM_WIDTHS
