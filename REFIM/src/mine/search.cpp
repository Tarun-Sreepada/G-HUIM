#include "mine/search.hpp"

#include "io/output.hpp"
#include "io/root.hpp"
#include "mine/worker.hpp"
#include "sys/alloc.hpp"
#include "sys/numa.hpp"
#include "sys/parallel.hpp"

#include <barrier>
#include <chrono>
#include <memory>
#include <spdlog/spdlog.h>

inline namespace REFIM_WIDTHS {

namespace {

// One Patterns per worker.
class WorkerPatterns final : public FoundPatterns {
public:
	std::vector<Patterns> results;

	std::size_t write(const std::string& path, const std::vector<std::string>& names) const override {
		return write_patterns(path, results, names);
	}
};

}  // namespace

SearchResult search(Transactions& input, const SearchOptions& options) {
	const int threads = options.threads;
	const std::size_t num_items = input.num_items;
	auto root = std::make_shared<Level>();
	make_root(input, options.min_util, threads, *root);

	SearchResult result;
	auto patterns = std::make_unique<WorkerPatterns>();

	// Placement: pin worker w to cpus[w] and group workers by NUMA node.
	std::vector<CpuInfo> cpus;
	int nodes = 1;
	if (options.numa && threads > 1) {
		const Topology topology = detect_topology();
		if (topology.valid) {
			cpus = pick_cpus(topology, threads);
			if (static_cast<int>(cpus.size()) == threads) {
				nodes = topology.nodes;
				spdlog::debug("numa: {} nodes, {} L3 groups, pinning {} workers", topology.nodes, topology.l3_groups, threads);
			} else {
				cpus.clear();
			}
		}
		if (cpus.empty()) spdlog::warn("numa: topology unavailable, workers are not pinned");
	}
	const auto node_of = [&](int w) { return cpus.empty() ? 0 : cpus[w].node; };

	// Steal order per worker: same L3 group, then same node, then remote; next index first within each.
	// Unpinned workers have no known placement and count as one L3 group.
	std::vector<std::vector<Victim>> victims(threads);
	for (int w = 0; w < threads; ++w) {
		const auto distance = [&](int v) {
			if (cpus.empty()) return kSameL3;
			if (cpus[v].node != cpus[w].node) return kRemote;
			return cpus[v].l3 == cpus[w].l3 ? kSameL3 : kSameNode;
		};
		for (int k = 1; k < threads; ++k) victims[w].push_back({(w + k) % threads, distance((w + k) % threads)});
		std::stable_sort(victims[w].begin(), victims[w].end(), [](const Victim& a, const Victim& b) { return a.distance < b.distance; });
	}

	Scheduler scheduler(std::move(victims));
	std::vector<std::unique_ptr<Worker>> workers(threads);
	std::vector<std::shared_ptr<const Level>> replicas(nodes);
	std::barrier sync(threads);
	const std::size_t roots = root->primary.size();

	const auto mine_start = std::chrono::steady_clock::now();
	const auto body = [&](int w) {
		if (!cpus.empty() && !pin_this_thread(cpus[w].cpu)) spdlog::warn("numa: cannot pin worker {} to cpu {}", w, cpus[w].cpu);
		// Big arrays of this worker come from (and return to) the arena of its L3 group.
		const ArenaBinding binding(cpus.empty() ? nullptr : Arena::get(cpus[w].l3));
		workers[w] = std::make_unique<Worker>(w, num_items + 1, options.min_util, threads, scheduler, options.collect_patterns,
											  options.refine, options.compact, options.refine_min_drop, options.leaf);

		// One copy of the root per node, shaped by the first worker of that node and filled by
		// all of the node's workers (first touch, in parallel).
		const int node = node_of(w);
		int rank = 0, peers = 0;
		for (int v = 0; v < threads; ++v) {
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
			if (k % threads == static_cast<std::size_t>(w)) scheduler.seed(w, {mine, 0, static_cast<std::uint32_t>(k), {}});
		sync.arrive_and_wait();
		if (w == 0) {
			spdlog::debug("mine: setup {:.3f} s", std::chrono::duration<double>(std::chrono::steady_clock::now() - mine_start).count());
			root.reset();
			replicas.clear();
		}
		workers[w]->run();
	};
	parallel_run(threads, body);
	const double mine_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - mine_start).count();
	SearchStats stats;
	for (auto& w : workers) {
		stats += w->stats;
		result.pattern_count += w->pattern_count;
		if (options.collect_patterns) patterns->results.push_back(std::move(w->patterns));
	}
	if (options.stats)
		spdlog::info("stats: nodes {} primary {} secondary {} | refine rounds {} secondary -{} primary -{} | "
					 "compacted {} txs {} -> {} items {} -> {} | leaves {} | single {} patterns {} | full lower {} exact {} patterns {}",
					 stats.nodes, stats.primary, stats.secondary, stats.refine_rounds, stats.refine_secondary, stats.refine_primary,
					 stats.compacted, stats.compact_tx_in, stats.compact_tx_out, stats.compact_items_in, stats.compact_items_out,
					 stats.leaves, stats.single, stats.single_patterns, stats.full_lower, stats.full_exact, stats.full_patterns);
	if (spdlog::should_log(spdlog::level::debug)) {
		const Arena::Stats a = Arena::stats();
		spdlog::debug("arenas: {} chunks mapped ({} MiB), {} reuses", a.mapped, a.bytes_mapped >> 20, a.reused);
	}
	std::size_t steals[3] = {}, offered = 0;
	for (int w = 0; w < threads; ++w) {
		for (int d = 0; d < 3; ++d) steals[d] += scheduler.steals(w, d);
		offered += scheduler.offered_steals(w);
	}
	if (threads > 1)
		spdlog::info("steals: {} same L3, {} same node, {} remote; {} split tasks, {} offered", steals[kSameL3], steals[kSameNode],
					 steals[kRemote], steals[kSameL3] + steals[kSameNode] + steals[kRemote] - offered, offered);
	if (spdlog::should_log(spdlog::level::debug)) {
		double busy = 0;
		for (int w = 0; w < threads; ++w) {
			spdlog::debug("worker {}: {} patterns, steals {} same L3 / {} same node / {} remote ({} offered), busy {:.3f} s",
						  w, workers[w]->pattern_count, scheduler.steals(w, kSameL3), scheduler.steals(w, kSameNode),
						  scheduler.steals(w, kRemote), scheduler.offered_steals(w), workers[w]->busy_seconds);
			busy += workers[w]->busy_seconds;
		}
		spdlog::debug("mine: {:.3f} s wall, workers busy {:.0f}%", mine_seconds, 100 * busy / (mine_seconds * threads));
	}

	if (options.collect_patterns) result.patterns = std::move(patterns);
	return result;
}

}  // namespace REFIM_WIDTHS
