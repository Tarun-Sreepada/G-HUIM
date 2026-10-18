#include "mine/efim.hpp"

#include "io/output.hpp"
#include "io/reader.hpp"
#include "mine/worker.hpp"
#include "sys/alloc.hpp"
#include "sys/numa.hpp"
#include "sys/parallel.hpp"
#include "sys/util.hpp"

#include <barrier>
#include <chrono>
#include <iostream>
#include <spdlog/spdlog.h>
#include <memory>
#include <stdexcept>

void Efim::mine(bool collect_patterns) {
	const auto start = std::chrono::steady_clock::now();
	collect_patterns_ = collect_patterns;
	pattern_count_ = 0;
	results_.clear();
	rename_.clear();

	auto root = std::make_shared<Level>();
	read_database(input_file_, sep_, min_util_, *root, rename_, threads_);


	// Placement: pin worker w to cpus[w] and group workers by NUMA node.
	std::vector<CpuInfo> cpus;
	int nodes = 1;
	if (numa_ && threads_ > 1) {
		const Topology topology = detect_topology();
		if (topology.valid) {
			cpus = pick_cpus(topology, threads_);
			if (static_cast<int>(cpus.size()) == threads_) {
				nodes = topology.nodes;
				spdlog::debug("numa: {} nodes, {} L3 groups, pinning {} workers", topology.nodes, topology.l3_groups, threads_);
			} else {
				cpus.clear();
			}
		}
		if (cpus.empty()) spdlog::warn("numa: topology unavailable, workers are not pinned");
	}
	const auto node_of = [&](int w) { return cpus.empty() ? 0 : cpus[w].node; };

	// Steal order per worker: same L3 group, then same node, then remote; nearest index first.
	std::vector<std::vector<int>> victims(threads_);
	for (int w = 0; w < threads_; ++w) {
		for (int k = 1; k < threads_; ++k) victims[w].push_back((w + k) % threads_);
		if (!cpus.empty())
			std::stable_sort(victims[w].begin(), victims[w].end(), [&](int a, int b) {
				const auto dist = [&](int v) { return cpus[v].l3 == cpus[w].l3 && cpus[v].node == cpus[w].node ? 0 : cpus[v].node == cpus[w].node ? 1 : 2; };
				return dist(a) < dist(b);
			});
	}

	Scheduler scheduler(std::move(victims));
	std::vector<std::unique_ptr<Worker>> workers(threads_);
	std::vector<std::shared_ptr<const Level>> replicas(nodes);
	std::barrier sync(threads_);
	const std::size_t roots = root->primary.size();

	const auto mine_start = std::chrono::steady_clock::now();
	const auto body = [&](int w) {
		if (!cpus.empty() && !pin_this_thread(cpus[w].cpu)) spdlog::warn("numa: cannot pin worker {} to cpu {}", w, cpus[w].cpu);
		// Big arrays of this worker come from (and return to) the arena of its L3 group.
		const ArenaBinding binding(cpus.empty() ? nullptr : Arena::get(cpus[w].l3));
		workers[w] = std::make_unique<Worker>(w, rename_.size(), min_util_, threads_, scheduler, collect_patterns_, refine_);

		// One copy of the root per node, shaped by the first worker of that node and filled by
		// all of the node's workers (first touch, in parallel).
		const int node = node_of(w);
		int rank = 0, peers = 0;
		for (int v = 0; v < threads_; ++v) {
			if (node_of(v) != node) continue;
			if (v < w) ++rank;
			++peers;
		}
		if (nodes > 1) {
			if (rank == 0) replicas[node] = root->shape_copy();
			sync.arrive_and_wait();
			std::const_pointer_cast<Level>(replicas[node])->fill_part(*root, rank, peers);
		}
		sync.arrive_and_wait();
		const std::shared_ptr<const Level>& mine = nodes > 1 ? replicas[node] : root;
		for (std::size_t k = roots; k-- > 0;)
			if (k % threads_ == static_cast<std::size_t>(w)) scheduler.seed(w, {mine, 0, static_cast<std::uint32_t>(k), {}});
		sync.arrive_and_wait();
		if (w == 0) {
			spdlog::debug("mine: setup {:.3f} s", std::chrono::duration<double>(std::chrono::steady_clock::now() - mine_start).count());
			root.reset();
			replicas.clear();
		}
		workers[w]->run();
	};
	parallel_run(threads_, body);
	const double mine_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - mine_start).count();
	SearchStats stats;
	for (auto& w : workers) {
		stats += w->stats;
		pattern_count_ += w->pattern_count;
		if (collect_patterns_) results_.push_back(std::move(w->patterns));
	}
	if (stats_)
		spdlog::info("stats: nodes {} primary {} secondary {} | refine rounds {} secondary -{} primary -{}", stats.nodes,
					 stats.primary, stats.secondary, stats.refine_rounds, stats.refine_secondary, stats.refine_primary);
	if (spdlog::should_log(spdlog::level::debug)) {
		const Arena::Stats a = Arena::stats();
		spdlog::debug("arenas: {} chunks mapped ({} MiB), {} reuses", a.mapped, a.bytes_mapped >> 20, a.reused);
	}
	if (spdlog::should_log(spdlog::level::debug)) {
		double busy = 0;
		for (int w = 0; w < threads_; ++w) {
			spdlog::debug("worker {}: {} patterns, {} steals, busy {:.3f} s", w, workers[w]->pattern_count, scheduler.steals(w), workers[w]->busy_seconds);
			busy += workers[w]->busy_seconds;
		}
		spdlog::debug("mine: {:.3f} s wall, workers busy {:.0f}%", mine_seconds, 100 * busy / (mine_seconds * threads_));
	}

	runtime_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	memory_rss_ = peak_rss_bytes();
}

void Efim::save(const std::string& out_file) const {
	if (!collect_patterns_) throw std::logic_error("cannot save patterns after count-only mining");
	const auto start = std::chrono::steady_clock::now();
	const std::size_t bytes = write_patterns(out_file, results_, rename_);
	spdlog::info("Wrote {} patterns ({} bytes) to {} in {:.3f} s", pattern_count(), bytes, out_file,
				 std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
}

std::size_t Efim::pattern_count() const {
	return pattern_count_;
}

void Efim::print_results() const {
	std::cout << "Total number of High Utility Patterns: " << pattern_count() << '\n';
	std::cout << "Peak Memory in RSS: " << memory_rss_ << '\n';
	std::cout << "Total ExecutionTime in seconds: " << runtime_ << '\n';
}
