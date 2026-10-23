#include "app/run.hpp"

#include "mine/refim.hpp"
#include "mine/simd_mode.hpp"
#include "sys/alloc.hpp"
#include "sys/numa.hpp"

#include <spdlog/spdlog.h>

#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

// "tab" and "space" are names for separators that are awkward to type in a shell.
std::string parse_sep(const std::string& arg) {
	if (arg == "tab") return "\t";
	if (arg == "space") return " ";
	return arg;
}

// Measured crossover: huge pages win up to ~32 of 48 threads, then compaction stalls dominate.
bool use_huge_pages(const std::string& thp, int threads) {
	return thp == "on" || (thp == "auto" && 4 * threads <= 3 * Refim::max_threads());
}

// Restricts the process to --nodes before any thread starts; returns the CPUs left (0 = all nodes, unchanged).
int restrict_nodes(const std::string& arg) {
	if (arg == "all") return 0;
	std::vector<int> nodes;
	std::stringstream in(arg);
	for (std::string id; std::getline(in, id, ',');) {
		try {
			nodes.push_back(std::stoi(id));
		} catch (const std::exception&) {
			throw std::runtime_error("--nodes " + arg + ": expected all or a comma-separated list of node ids");
		}
	}
	const int cpus = restrict_to_nodes(nodes);
	if (cpus == 0) throw std::runtime_error("--nodes " + arg + ": no usable CPU on those nodes");
	spdlog::debug("nodes: {} ({} cpus)", arg, cpus);
	return cpus;
}

void select_vector_kernels(const std::string& simd) {
	if (!select_simd(simd)) throw std::runtime_error("--simd " + simd + " is not supported by this CPU");
	spdlog::debug("simd: {} (cpu supports {})", simd_name(), simd_supported());
}

void configure(Refim& miner, const Options& o) {
	miner.set_numa(o.numa == "on");
	miner.set_refine(o.refine);
	miner.set_compact(o.compact == "on");
	miner.set_refine_min_drop(o.refine_min_drop);
	miner.set_leaf(o.leaf == "on");
	miner.set_stats(o.stats);
	miner.set_bandwidth(o.bandwidth);
	miner.set_widths(o.key_bits, o.util_bits);
}

}  // namespace

void run(const Options& o) {
	const int node_cpus = restrict_nodes(o.nodes);
	int threads = o.threads;
	if (threads == 0) threads = node_cpus ? node_cpus : Refim::max_threads();
	if (node_cpus && threads > node_cpus)
		spdlog::warn("{} threads on {} cpus of --nodes {}: workers are not pinned", threads, node_cpus, o.nodes);
	const bool huge_pages = use_huge_pages(o.thp, threads);
	Arena::set_huge_pages(huge_pages);
	Refim miner(o.input, o.min_util, parse_sep(o.sep), threads);
	if (miner.threads() != threads)
		spdlog::warn("threads {} capped to {} (allowed range 1-{})", threads, miner.threads(), Refim::max_threads());
	select_vector_kernels(o.simd);
	spdlog::debug("thp: {}", huge_pages ? "on" : "off");
	configure(miner, o);
	const bool collect_patterns = !o.output.empty() && o.output != "/dev/null";
	miner.mine(collect_patterns);
	if (collect_patterns) miner.save(o.output);
	miner.print_results();
}
