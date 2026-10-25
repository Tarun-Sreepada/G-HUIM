// GREFIM: REFIM's search on every CPU core and every GPU at once. The CPU reads the dataset and builds the
// root database (REFIM); its root tasks (the subtrees of the single items) then go to a shared pool that
// REFIM's workers and one Vulkan engine per GPU draw from (sched/pool.hpp, gpu/engine.hpp). Any GPU with a
// Vulkan driver works: NVIDIA, AMD, Intel, discrete or integrated.
//
// Usage: GREFIM -i <input> -m <minUtil> [-o <output>] [-s space|tab|<literal>] [-t <threads>] [-g all|none|<i,...>]
//               [-c on|off] [--list-devices] [-l <level>] [-L <file>]

#include "app/logging.hpp"
#include "mine/built_widths.hpp"
#include "mine/refim.hpp"
#include "mine/simd_mode.hpp"
#include "mine/widths.hpp"
#include "share.hpp"
#include "sys/alloc.hpp"
#include "vk/context.hpp"

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define GREFIM_DECLARE_SHARE(item, util) \
	inline namespace REFIM_WIDTH_NAME(item, util) { std::unique_ptr<RootShareBase> make_share(const grefim::ShareConfig& config); }
REFIM_FOR_EACH_BUILT_WIDTH(GREFIM_DECLARE_SHARE)
#undef GREFIM_DECLARE_SHARE

namespace {

struct Options {
	std::string input, output, sep = "space", gpus = "all", cpu = "on", numa = "on", log_level = "info", log_file;
	long long min_util = 0;
	int threads = 0;
	bool list_devices = false, stats = false;
};

bool parse(int argc, char** argv, Options& o, int& exit_code) {
	CLI::App app{"GREFIM high-utility itemset miner (REFIM on every CPU core and GPU)"};
	app.add_option("-i,--input", o.input, "Input file (items:total:utilities per line)")->check(CLI::ExistingFile);
	app.add_option("-m,--minutil", o.min_util, "Minimum utility threshold (>= 1)")->check(CLI::PositiveNumber);
	app.add_option("-o,--output", o.output, "Output file (omit or use /dev/null to only count patterns)");
	app.add_option("-s,--separator", o.sep, "Field separator: tab, space or a literal")->capture_default_str();
	app.add_option("-t,--threads", o.threads, "CPU worker threads (0 = all hardware threads but one per GPU)")
		->capture_default_str()
		->check(CLI::NonNegativeNumber);
	app.add_option("-g,--gpus", o.gpus, "GPUs to use: all, none, or a comma-separated list of indices from --list-devices")
		->capture_default_str();
	app.add_option("-c,--cpu", o.cpu, "Mine on the CPU too (off: the CPU only builds the root)")
		->capture_default_str()
		->check(CLI::IsMember({"on", "off"}));
	app.add_option("-n,--numa", o.numa, "Pin CPU workers, place memory per node, steal nearest-first")
		->capture_default_str()
		->check(CLI::IsMember({"on", "off"}));
	app.add_flag("--list-devices", o.list_devices, "List the GPUs and exit");
	app.add_flag("-S,--stats", o.stats, "Log search counters");
	app.add_option("-l,--log-level", o.log_level, "trace, debug, info, warn, error, critical or off")
		->capture_default_str()
		->check(CLI::IsMember({"trace", "debug", "info", "warn", "error", "critical", "off"}));
	app.add_option("-L,--log-file", o.log_file, "Also write the log to this file");
	try {
		app.parse(argc, argv);
		if (!o.list_devices && (o.input.empty() || o.min_util < 1)) throw CLI::RequiredError("--input and --minutil");
	} catch (const CLI::ParseError& error) {
		exit_code = app.exit(error);
		return false;
	}
	return true;
}

std::string parse_sep(const std::string& arg) {
	if (arg == "tab") return "\t";
	if (arg == "space") return " ";
	return arg;
}

// Indices of the GPUs --gpus selects among `count`.
std::vector<std::size_t> pick_gpus(const std::string& arg, std::size_t count) {
	std::vector<std::size_t> picked;
	if (arg == "none") return picked;
	if (arg == "all") {
		for (std::size_t i = 0; i < count; ++i) picked.push_back(i);
		return picked;
	}
	std::stringstream in(arg);
	for (std::string id; std::getline(in, id, ',');) {
		std::size_t i;
		try {
			i = std::stoul(id);
		} catch (const std::exception&) {
			throw std::runtime_error("--gpus " + arg + ": expected all, none or a comma-separated list of indices");
		}
		if (i >= count) throw std::runtime_error("--gpus " + arg + ": there is no GPU " + id + " (see --list-devices)");
		picked.push_back(i);
	}
	return picked;
}

std::unique_ptr<RootShareBase> make_share_for(int key_bits, int util_bits, const grefim::ShareConfig& config) {
#define GREFIM_DISPATCH(item, util) \
	if (key_bits == item && util_bits == util) return REFIM_WIDTH_NAME(item, util)::make_share(config);
	REFIM_FOR_EACH_BUILT_WIDTH(GREFIM_DISPATCH)
#undef GREFIM_DISPATCH
	throw std::logic_error("unreachable: no share for the search's widths");
}

void run(const Options& o) {
	// GPUs: every one with a Vulkan driver, usable ones selected by --gpus.
	std::optional<grefim::vk::Instance> instance;
	std::vector<grefim::vk::GpuInfo> gpus;
	if (o.gpus != "none" || o.list_devices) {
		try {
			instance.emplace();
			gpus = grefim::vk::list_gpus(instance->get());
		} catch (const std::exception& e) {
			if (o.list_devices) throw;
			spdlog::warn("no Vulkan ({}): mining on the CPU only", e.what());
		}
	}
	if (o.list_devices) {
		for (std::size_t i = 0; i < gpus.size(); ++i)
			std::cout << i << ": " << gpus[i].name << " (" << (gpus[i].integrated ? "integrated" : "discrete") << ", " << gpus[i].driver
					  << ", subgroup " << gpus[i].subgroup_size << ")" << (gpus[i].missing.empty() ? "" : " unusable, lacks " + gpus[i].missing)
					  << '\n';
		if (gpus.empty()) std::cout << "no GPU with a Vulkan driver\n";
		std::cout << "cpu: " << Refim::max_threads() << " hardware threads\n";
		return;
	}
	std::vector<std::unique_ptr<grefim::vk::Device>> devices;
	for (const std::size_t i : pick_gpus(o.gpus, gpus.size())) {
		if (!gpus[i].missing.empty()) {
			spdlog::warn("gpu {} ({}) lacks {}: skipped", i, gpus[i].name, gpus[i].missing);
			continue;
		}
		devices.push_back(std::make_unique<grefim::vk::Device>(gpus[i]));
	}
	const bool cpu = o.cpu == "on";
	if (!cpu && devices.empty()) throw std::runtime_error("--cpu off and no usable GPU: nothing would mine");

	// All hardware threads but one per GPU, which its engine's host thread needs to keep it fed.
	const int threads = o.threads ? o.threads : std::max(1, Refim::max_threads() - static_cast<int>(devices.size()));
	Arena::set_huge_pages(4 * threads <= 3 * Refim::max_threads());  // REFIM's --thp auto
	if (!select_simd("auto")) throw std::runtime_error("no vector kernel for this CPU");
	Refim miner(o.input, o.min_util, parse_sep(o.sep), threads);
	miner.set_numa(o.numa == "on");
	miner.set_stats(o.stats);

	grefim::ShareConfig config;
	for (auto& d : devices) config.devices.push_back(d.get());
	config.min_util = static_cast<std::uint64_t>(o.min_util);
	config.collect_patterns = !o.output.empty() && o.output != "/dev/null";
	config.cpu_workers = cpu ? miner.threads() : 0;
	std::string names;
	for (auto& d : devices) names += ", " + d->info().name;
	spdlog::info("GREFIM: {}{}", cpu ? std::to_string(miner.threads()) + " CPU threads" : "CPU builds the root only", names);
	miner.set_share([config](int key_bits, int util_bits) { return make_share_for(key_bits, util_bits, config); });

	miner.mine(config.collect_patterns);
	if (config.collect_patterns) miner.save(o.output);
	miner.print_results();
}

}  // namespace

int main(int argc, char** argv) {
	Options options;
	int exit_code = 0;
	if (!parse(argc, argv, options, exit_code)) return exit_code;
	init_logging(options.log_level, options.log_file);
	try {
		run(options);
	} catch (const std::exception& e) {
		spdlog::error("{}", e.what());
		return 1;
	}
}
