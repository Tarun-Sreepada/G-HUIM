#include "app/options.hpp"

#include <CLI/CLI.hpp>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

// Legacy form: refim <input> <output> <minUtil> [sep] [threads]
bool parse_legacy(int argc, char** argv, Options& o) {
	std::vector<std::string> legacy;
	for (int i = 1; i < argc; ++i) legacy.emplace_back(argv[i]);
	o.input = legacy[0];
	o.output = legacy[1];
	try {
		o.min_util = std::stoll(legacy[2]);
		if (o.min_util < 1) throw std::invalid_argument("minUtil must be >= 1");
		if (legacy.size() > 3) o.sep = legacy[3];
		if (legacy.size() > 4) o.threads = std::stoi(legacy[4]);
	} catch (const std::exception&) {
		std::cerr << "Usage: " << argv[0] << " <inputFile> <outputFile> <minUtil> [sep] [threads]\n";
		return false;
	}
	return true;
}

void declare_options(CLI::App& app, Options& o) {
	app.add_option("-i,--input", o.input, "Input file (items:total:utilities per line)")
		->required()
		->check(CLI::ExistingFile);
	app.add_option("-m,--minutil", o.min_util, "Minimum utility threshold (>= 1)")->required()->check(CLI::PositiveNumber);
	app.add_option("-o,--output", o.output, "Output file (omit or use /dev/null to only count patterns)");
	app.add_option("-s,--separator", o.sep, "Field separator: tab, space or a literal")->capture_default_str();
	app.add_option("-t,--threads", o.threads, "Worker threads (0 = all hardware threads)")
		->capture_default_str()
		->check(CLI::NonNegativeNumber);
	app.add_option("-n,--numa", o.numa, "Pin workers, place memory per node, steal nearest-first")
		->capture_default_str()
		->check(CLI::IsMember({"on", "off"}));
	app.add_option("-v,--simd", o.simd, "Vector kernels: auto, avx512 or scalar")
		->capture_default_str()
		->check(CLI::IsMember({"auto", "avx512", "scalar"}));
	app.add_option("-p,--thp", o.thp, "Transparent huge pages for big arrays: auto (off above 3/4 of the hardware threads), on or off")
		->capture_default_str()
		->check(CLI::IsMember({"auto", "on", "off"}));
	app.add_option("-r,--refine", o.refine, "Rounds of recomputing each child's bounds without the items it rules out (-1 = until none is)")
		->capture_default_str()
		->check(CLI::Range(-1, 1000));
	app.add_option("-d,--refine-min-drop", o.refine_min_drop, "Refine only while a pass drops at least this fraction of the candidates")
		->capture_default_str()
		->check(CLI::Range(0.0, 1.0));
	app.add_option("-f,--leaf", o.leaf, "Sum leaf extensions from the occurrence index instead of projecting them")
		->capture_default_str()
		->check(CLI::IsMember({"on", "off"}));
	app.add_option("-c,--compact", o.compact, "Rewrite each child without the items it rules out")
		->capture_default_str()
		->check(CLI::IsMember({"on", "off"}));
	app.add_flag("-S,--stats", o.stats, "Log search counters");
	app.add_option("-l,--log-level", o.log_level, "trace, debug, info, warn, error, critical or off")
		->capture_default_str()
		->check(CLI::IsMember({"trace", "debug", "info", "warn", "error", "critical", "off"}));
	app.add_option("-L,--log-file", o.log_file, "Also write the log to this file");
}

}  // namespace

bool parse_command_line(int argc, char** argv, Options& o, int& exit_code) {
	if (argc >= 4 && argv[1][0] != '-') {
		exit_code = 1;
		return parse_legacy(argc, argv, o);
	}
	CLI::App app{"REFIM high-utility itemset miner (CPU; refined EFIM)"};
	declare_options(app, o);
	try {
		app.parse(argc, argv);
	} catch (const CLI::ParseError& error) {
		exit_code = app.exit(error);
		return false;
	}
	return true;
}
