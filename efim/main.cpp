#include "mine/efim.hpp"
#include "mine/simd.hpp"
#include "sys/alloc.hpp"

#include <CLI/CLI.hpp>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <iostream>
#include <stdexcept>
#include <string>

// "tab" and "space" are names for separators that are awkward to type in a shell.
static std::string parse_sep(const std::string& arg) {
	if (arg == "tab") return "\t";
	if (arg == "space") return " ";
	return arg;
}

static void init_logging(const std::string& level, const std::string& file) {
	std::vector<spdlog::sink_ptr> sinks{std::make_shared<spdlog::sinks::stderr_color_sink_mt>()};
	if (!file.empty()) sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(file, false));
	auto logger = std::make_shared<spdlog::logger>("efim", sinks.begin(), sinks.end());
	logger->set_pattern("[%l] %v");
	logger->set_level(spdlog::level::from_str(level));
	logger->flush_on(spdlog::level::err);
	spdlog::set_default_logger(std::move(logger));
}

int main(int argc, char** argv) {
	std::string input, output, sep = "tab", log_level = "info", log_file, numa = "on", simd = "auto", thp = "auto";
	long long min_util = 0;
	int threads = 1, refine = -1;
	bool stats = false;

	// Legacy form: efim <input> <output> <minUtil> [sep] [threads]
	std::vector<std::string> legacy;
	if (argc >= 4 && argv[1][0] != '-') {
		for (int i = 1; i < argc; ++i) legacy.emplace_back(argv[i]);
		input = legacy[0];
		output = legacy[1];
		try {
			min_util = std::stoll(legacy[2]);
			if (min_util < 1) throw std::invalid_argument("minUtil must be >= 1");
			if (legacy.size() > 3) sep = legacy[3];
			if (legacy.size() > 4) threads = std::stoi(legacy[4]);
		} catch (const std::exception&) {
			std::cerr << "Usage: " << argv[0] << " <inputFile> <outputFile> <minUtil> [sep] [threads]\n";
			return 1;
		}
	} else {
		CLI::App app{"EFIM high-utility itemset miner (CPU)"};
		app.add_option("-i,--input", input, "Input file (items:total:utilities per line)")
			->required()
			->check(CLI::ExistingFile);
		app.add_option("-m,--minutil", min_util, "Minimum utility threshold (>= 1)")->required()->check(CLI::PositiveNumber);
		app.add_option("-o,--output", output, "Output file (omit or use /dev/null to only count patterns)");
		app.add_option("--separator", sep, "Field separator: tab, space or a literal")->capture_default_str();
		app.add_option("-t,--threads", threads, "Worker threads (0 = all hardware threads)")
			->capture_default_str()
			->check(CLI::NonNegativeNumber);
		app.add_option("--numa", numa, "Pin workers, place memory per node, steal nearest-first")
			->capture_default_str()
			->check(CLI::IsMember({"on", "off"}));
		app.add_option("--simd", simd, "Vector kernels: auto, avx512 or scalar")
			->capture_default_str()
			->check(CLI::IsMember({"auto", "avx512", "scalar"}));
		app.add_option("--thp", thp, "Transparent huge pages for big arrays: auto (off above 3/4 of the hardware threads), on or off")
			->capture_default_str()
			->check(CLI::IsMember({"auto", "on", "off"}));
		app.add_option("--refine", refine, "Rounds of recomputing each child's bounds without the items it rules out (-1 = until none is)")
			->capture_default_str()
			->check(CLI::Range(-1, 1000));
		app.add_flag("--stats", stats, "Log search counters");
		app.add_option("--log-level", log_level, "trace, debug, info, warn, error, critical or off")
			->capture_default_str()
			->check(CLI::IsMember({"trace", "debug", "info", "warn", "error", "critical", "off"}));
		app.add_option("--log-file", log_file, "Also write the log to this file");
		try {
			app.parse(argc, argv);
		} catch (const CLI::ParseError& error) {
			return app.exit(error);
		}
	}

	init_logging(log_level, log_file);
	try {
		if (threads == 0) threads = Efim::max_threads();
		// Measured crossover: huge pages win up to ~32 of 48 threads, then compaction stalls dominate.
		const bool huge_pages = thp == "on" || (thp == "auto" && 4 * threads <= 3 * Efim::max_threads());
		Arena::set_huge_pages(huge_pages);
		Efim miner(input, min_util, parse_sep(sep), threads);
		if (miner.threads() != threads)
			spdlog::warn("threads {} capped to {} (allowed range 1-{})", threads, miner.threads(), Efim::max_threads());
		if (!select_simd(simd)) throw std::runtime_error("--simd " + simd + " is not supported by this CPU");
		spdlog::debug("simd: {}", simd_name());
		spdlog::debug("thp: {}", huge_pages ? "on" : "off");
		miner.set_numa(numa == "on");
		miner.set_refine(refine);
		miner.set_stats(stats);
		const bool collect_patterns = !output.empty() && output != "/dev/null";
		miner.mine(collect_patterns);
		if (collect_patterns) miner.save(output);
		miner.print_results();
	} catch (const std::exception& e) {
		spdlog::error("{}", e.what());
		return 1;
	}
}
