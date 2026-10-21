#include "app/run.hpp"

#include "mine/refim.hpp"
#include "mine/simd_mode.hpp"
#include "sys/alloc.hpp"

#include <spdlog/spdlog.h>

#include <stdexcept>

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

void select_vector_kernels(const std::string& simd) {
	if (!select_simd(simd)) throw std::runtime_error("--simd " + simd + " is not supported by this CPU");
	spdlog::debug("simd: {}", simd_name());
}

void configure(Refim& miner, const Options& o) {
	miner.set_numa(o.numa == "on");
	miner.set_refine(o.refine);
	miner.set_compact(o.compact == "on");
	miner.set_refine_min_drop(o.refine_min_drop);
	miner.set_leaf(o.leaf == "on");
	miner.set_stats(o.stats);
	miner.set_widths(o.key_bits, o.util_bits);
}

}  // namespace

void run(const Options& o) {
	int threads = o.threads;
	if (threads == 0) threads = Refim::max_threads();
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
