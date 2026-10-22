#include "sys/perf.hpp"

#include <spdlog/spdlog.h>

#include <cerrno>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#if defined(__linux__)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace {

#if defined(__linux__)
struct Event {
	const char* name;
	std::uint32_t type;
	std::uint64_t config;
	bool kernel_only = false;
	// Scheduled together with the event before it, so their ratio holds even while both are multiplexed.
	bool grouped = false;
};

constexpr std::uint64_t cache(std::uint64_t id, std::uint64_t op, std::uint64_t result) { return id | (op << 8) | (result << 16); }

// Hardware events beyond the PMU's counter slots are time-multiplexed; counts are scaled to the full phase.
constexpr Event kEvents[] = {
	{"task-clock-ns", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK},
	{"context-switches", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES},
	{"cpu-migrations", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS},
	{"minor-faults", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MIN},
	{"major-faults", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MAJ},
	{"cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
	{"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, false, true},
	{"ref-cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_REF_CPU_CYCLES},
	{"kernel-cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, true},
	{"stalled-frontend", PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_FRONTEND},
	{"stalled-backend", PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_BACKEND},
	{"branches", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS},
	{"branch-misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES},
	{"cache-references", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES},
	{"cache-misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES},
	{"L1d-loads", PERF_TYPE_HW_CACHE, cache(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS)},
	{"L1d-load-misses", PERF_TYPE_HW_CACHE, cache(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS)},
	{"L1i-load-misses", PERF_TYPE_HW_CACHE, cache(PERF_COUNT_HW_CACHE_L1I, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS)},
	{"LLC-loads", PERF_TYPE_HW_CACHE, cache(PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS)},
	{"LLC-load-misses", PERF_TYPE_HW_CACHE, cache(PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS)},
	{"LLC-stores", PERF_TYPE_HW_CACHE, cache(PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_OP_WRITE, PERF_COUNT_HW_CACHE_RESULT_ACCESS)},
	{"LLC-store-misses", PERF_TYPE_HW_CACHE, cache(PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_OP_WRITE, PERF_COUNT_HW_CACHE_RESULT_MISS)},
	{"dTLB-loads", PERF_TYPE_HW_CACHE, cache(PERF_COUNT_HW_CACHE_DTLB, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS)},
	{"dTLB-load-misses", PERF_TYPE_HW_CACHE, cache(PERF_COUNT_HW_CACHE_DTLB, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS)},
	{"iTLB-load-misses", PERF_TYPE_HW_CACHE, cache(PERF_COUNT_HW_CACHE_ITLB, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS)},
	{"node-loads", PERF_TYPE_HW_CACHE, cache(PERF_COUNT_HW_CACHE_NODE, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS)},
	{"node-load-misses", PERF_TYPE_HW_CACHE, cache(PERF_COUNT_HW_CACHE_NODE, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS)},
};

// AMD's perf driver has no generic ref-cycles, stalled-backend, LLC or node events; on Zen 4 these raw core events stand in
// (encodings from perf's amdzen4 tables). Fills count lines brought into L1d, by where they came from after missing L2.
constexpr std::uint64_t zen(std::uint64_t event, std::uint64_t umask) { return (event & 0xff) | (umask << 8) | ((event >> 8) << 32); }
constexpr Event kZen4Events[] = {
	{"backend-stall-slots", PERF_TYPE_RAW, zen(0x1a0, 0x1e)},
	{"fills-local-l2", PERF_TYPE_RAW, zen(0x44, 0x01)},
	{"fills-local-l3", PERF_TYPE_RAW, zen(0x44, 0x02)},
	{"fills-other-ccx", PERF_TYPE_RAW, zen(0x44, 0x04)},
	{"fills-local-dram", PERF_TYPE_RAW, zen(0x44, 0x08)},
	{"fills-remote-cache", PERF_TYPE_RAW, zen(0x44, 0x10)},
	{"fills-remote-dram", PERF_TYPE_RAW, zen(0x44, 0x40)},
	// Demand fills only (no prefetches): the misses a load actually waited for.
	{"demand-beyond-l2", PERF_TYPE_RAW, zen(0x43, 0xfe)},
	{"demand-remote", PERF_TYPE_RAW, zen(0x43, 0x50)},
};
// Zen 4 dispatches up to 6 ops per cycle; backend-stall-slots counts the slots left empty because the backend was full.
constexpr double kZen4DispatchWidth = 6;
// What kZen4Events (and the msr PMU's mperf) cover, so they are not reported as missing.
constexpr const char* kZen4Covers[] = {"ref-cycles", "stalled-backend", "LLC-loads", "LLC-load-misses",
									   "LLC-stores", "LLC-store-misses", "node-loads", "node-load-misses"};

std::string read_line(const std::string& path) {
	std::ifstream in(path);
	std::string line;
	std::getline(in, line);
	return line;
}

// Family 0x19 is Zen 3 and Zen 4; perf's mapfile gives Zen 3 models 0x00-0x0f and 0x20-0x5f, the rest are Zen 4.
bool detect_zen4() {
	std::ifstream in("/proc/cpuinfo");
	std::string line, vendor;
	int family = -1, model = -1;
	while (std::getline(in, line) && !line.empty()) {
		const auto colon = line.find(':');
		if (colon == std::string::npos) continue;
		const std::string key = line.substr(0, line.find_last_not_of(" \t", colon - 1) + 1);
		const std::string value = line.substr(std::min(colon + 2, line.size()));
		if (key == "vendor_id") vendor = value;
		else if (key == "cpu family") family = std::stoi(value);
		else if (key == "model") model = std::stoi(value);
	}
	return vendor == "AuthenticAMD" && family == 0x19 && model >= 0x10 && !(model >= 0x20 && model <= 0x5f);
}

bool is_zen4() {
	static const bool zen4 = detect_zen4();
	return zen4;
}

// An event of a dynamic PMU (msr, power) named in sysfs; these PMUs take no exclude bits. pid -1 counts a whole CPU.
int open_sysfs_event(const std::string& pmu, const std::string& event, int pid, int cpu) {
	const std::string dir = "/sys/bus/event_source/devices/" + pmu + "/";
	const std::string type = read_line(dir + "type"), config = read_line(dir + "events/" + event);
	if (type.empty() || config.rfind("event=", 0) != 0) return -1;
	perf_event_attr attr{};
	attr.size = sizeof(attr);
	attr.type = static_cast<std::uint32_t>(std::stoul(type));
	attr.config = std::stoull(config.substr(6), nullptr, 0);
	attr.disabled = 1;
	attr.inherit = pid == 0;
	attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
	return static_cast<int>(syscall(SYS_perf_event_open, &attr, pid, cpu, -1, 0));
}

// "0,24" or "0-3,8".
std::vector<int> parse_cpu_list(const std::string& list) {
	std::vector<int> cpus;
	std::stringstream in(list);
	std::string part;
	while (std::getline(in, part, ',')) {
		if (part.empty()) continue;
		const auto dash = part.find('-');
		const int first = std::stoi(part), last = dash == std::string::npos ? first : std::stoi(part.substr(dash + 1));
		for (int c = first; c <= last; ++c) cpus.push_back(c);
	}
	return cpus;
}

double seconds(const timeval& t) { return static_cast<double>(t.tv_sec) + static_cast<double>(t.tv_usec) * 1e-6; }

// Resident bytes of this process on each NUMA node, from the N<node>=<pages> fields of /proc/self/numa_maps.
std::vector<std::uint64_t> node_resident_bytes() {
	std::vector<std::uint64_t> bytes;
	std::ifstream in("/proc/self/numa_maps");
	std::string line;
	while (std::getline(in, line)) {
		std::uint64_t page = 4096;
		if (const auto at = line.find("kernelpagesize_kB="); at != std::string::npos) page = std::stoull(line.substr(at + 18)) * 1024;
		for (std::size_t at = line.find(" N"); at != std::string::npos; at = line.find(" N", at + 1)) {
			const auto eq = line.find('=', at);
			if (eq == std::string::npos) break;
			const std::size_t node = std::stoul(line.substr(at + 2, eq - at - 2));
			if (bytes.size() <= node) bytes.resize(node + 1);
			bytes[node] += std::stoull(line.substr(eq + 1)) * page;
		}
	}
	return bytes;
}

// A "Key:  <n> kB" field of /proc/self/smaps_rollup, in bytes.
std::uint64_t smaps_bytes(const std::string& key) {
	std::ifstream in("/proc/self/smaps_rollup");
	std::string line;
	while (std::getline(in, line))
		if (line.rfind(key + ":", 0) == 0) return std::stoull(line.substr(key.size() + 1)) * 1024;
	return 0;
}

// Memory bandwidth monitoring (resctrl, Zen's MBM): bytes the L3s fetched from memory, summed over L3 domains, as
// counted for the default group, which holds every task not placed in another group (so the whole system, not only
// this process). mbm_local_bytes is the part from the L3's own node.
bool read_mbm(std::uint64_t& total, std::uint64_t& local) {
	namespace fs = std::filesystem;
	std::error_code ec;
	total = local = 0;
	bool any = false;
	for (const auto& d : fs::directory_iterator("/sys/fs/resctrl/mon_data", ec)) {
		const std::string t = read_line(d.path() / "mbm_total_bytes"), l = read_line(d.path() / "mbm_local_bytes");
		if (t.empty() || l.empty() || t == "Unavailable" || l == "Unavailable") continue;
		total += std::stoull(t);
		local += std::stoull(l);
		any = true;
	}
	return any;
}

// User and kernel space (system calls, page faults, huge-page compaction) need perf_event_paranoid <= 1;
// above that, user_only drops the kernel. A group member (group_fd >= 0) starts enabled and counts whenever its leader does.
int open_event(const Event& e, bool user_only, int group_fd = -1) {
	perf_event_attr attr{};
	attr.size = sizeof(attr);
	attr.type = e.type;
	attr.config = e.config;
	attr.disabled = group_fd < 0;
	attr.inherit = 1;
	attr.exclude_kernel = user_only;
	attr.exclude_user = e.kernel_only;
	attr.exclude_hv = 1;
	attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
	return static_cast<int>(syscall(SYS_perf_event_open, &attr, 0, -1, group_fd, 0));
}

std::string paranoid_level() {
	std::ifstream in("/proc/sys/kernel/perf_event_paranoid");
	std::string level;
	in >> level;
	return level;
}
#endif

double ratio(double num, double den) { return den > 0 ? num / den : 0; }

}  // namespace

PerfCounters::PerfCounters(bool bandwidth) : bandwidth_(bandwidth) {
#if defined(__linux__)
	if (bandwidth_ && !std::filesystem::exists("/sys/fs/resctrl/mon_data"))
		spdlog::warn("bandwidth: resctrl is not mounted; run `sudo mount -t resctrl resctrl /sys/fs/resctrl` to count it");
	if (!spdlog::should_log(spdlog::level::debug)) return;
	int first_error = 0;
	std::vector<const char*> missing;
	if (const int probe = open_event(kEvents[0], false); probe >= 0)
		close(probe);
	else if (const int user_probe = open_event(kEvents[0], true); user_probe >= 0) {
		close(user_probe);
		user_only_ = true;
	}
	if (user_only_) spdlog::warn("perf: counting user space only; kernel.perf_event_paranoid is {}, set it to 1 or lower to count the kernel too", paranoid_level());
	for (const Event& e : kEvents) {
		if (e.kernel_only && user_only_) continue;
		int leader = -1;
		if (e.grouped && !counters_.empty() && counters_.back().name == (&e)[-1].name) leader = counters_.back().fd;
		int fd = open_event(e, user_only_, leader);
		// Counts on its own if the PMU cannot schedule the pair together.
		if (fd < 0 && leader >= 0) fd = open_event(e, user_only_);
		if (fd >= 0) {
			counters_.push_back({e.name, fd});
			continue;
		}
		if (!first_error) first_error = errno;
		if (errno == ENOENT || errno == EOPNOTSUPP || errno == EINVAL) missing.push_back(e.name);
	}
	if (!counters_.empty() && is_zen4()) {
		for (const Event& e : kZen4Events)
			if (const int fd = open_event(e, user_only_); fd >= 0) counters_.push_back({e.name, fd});
		if (const int fd = open_sysfs_event("msr", "mperf", 0, -1); fd >= 0) counters_.push_back({"ref-cycles", fd});
		std::erase_if(missing, [](const char* name) { return std::ranges::any_of(kZen4Covers, [&](const char* c) { return std::strcmp(c, name) == 0; }); });
	}
	// RAPL counts the whole package (every process on it), one counter per socket.
	for (const int cpu : parse_cpu_list(read_line("/sys/bus/event_source/devices/power/cpumask")))
		if (const int fd = open_sysfs_event("power", "energy-pkg", -1, cpu); fd >= 0) counters_.push_back({"energy-pkg", fd});
	if (const std::string scale = read_line("/sys/bus/event_source/devices/power/events/energy-pkg.scale"); !scale.empty())
		energy_scale_ = std::stod(scale);
	if (!bandwidth_ && !std::filesystem::exists("/sys/fs/resctrl/mon_data"))
		spdlog::debug("perf: no memory bandwidth; mount resctrl (`sudo mount -t resctrl resctrl /sys/fs/resctrl`) to count it");
	std::string unsupported;
	for (const char* name : missing) unsupported += unsupported.empty() ? name : std::string(", ") + name;
	if (counters_.empty())
		spdlog::warn("perf: no counters ({}); kernel.perf_event_paranoid is {}, run `sudo sysctl kernel.perf_event_paranoid=-1` to allow them",
					 std::strerror(first_error), paranoid_level());
	else if (!unsupported.empty())
		spdlog::debug("perf: not supported on this CPU: {}", unsupported);
#endif
}

PerfCounters::~PerfCounters() {
#if defined(__linux__)
	for (const Counter& c : counters_) close(c.fd);
#endif
}

void PerfCounters::start() {
#if defined(__linux__)
	if (!bandwidth_ && !spdlog::should_log(spdlog::level::debug)) return;
	mbm_ = read_mbm(mbm_total_, mbm_local_);
	rusage usage{};
	getrusage(RUSAGE_SELF, &usage);
	user_seconds_ = seconds(usage.ru_utime);
	sys_seconds_ = seconds(usage.ru_stime);
	for (const Counter& c : counters_) {
		ioctl(c.fd, PERF_EVENT_IOC_RESET, 0);
		ioctl(c.fd, PERF_EVENT_IOC_ENABLE, 0);
	}
#endif
}

void PerfCounters::report(const std::string& phase) {
#if defined(__linux__)
	if (std::uint64_t total = 0, local = 0; mbm_ && read_mbm(total, local)) {
		const double t = static_cast<double>(total - mbm_total_), l = static_cast<double>(local - mbm_local_);
		spdlog::log(bandwidth_ ? spdlog::level::info : spdlog::level::debug,
					"bandwidth {}: {:.2f} GiB from memory, {:.1f}% from the remote node (whole system)", phase, t / (1u << 30),
					100 * ratio(t - l, t));
	}
	if (!spdlog::should_log(spdlog::level::debug)) return;
	std::map<std::string, double> v;
	for (const Counter& c : counters_) {
		ioctl(c.fd, PERF_EVENT_IOC_DISABLE, 0);
		std::uint64_t values[3]{};
		if (read(c.fd, values, sizeof(values)) != sizeof(values) || values[2] == 0) continue;
		const double scale = static_cast<double>(values[1]) / static_cast<double>(values[2]);
		v[c.name] += static_cast<double>(values[0]) * scale;  // energy-pkg has one counter per socket
		spdlog::trace("perf {}: {} {:.0f} (counted {:.0f}% of the time)", phase, c.name, static_cast<double>(values[0]) * scale, 100 / scale);
	}
	const auto has = [&](const char* name) { return v.count(name) != 0; };
	if (has("backend-stall-slots") && !has("stalled-backend")) v["stalled-backend"] = v["backend-stall-slots"] / kZen4DispatchWidth;
	const auto pct = [&](const char* num, const char* den) { return 100 * ratio(v[num], v[den]); };
	// Leaves out what this CPU or perf_event_paranoid did not let us count instead of logging it as 0.
	const auto opt = [&](const char* fmt, const char* num, const char* den) {
		return has(num) ? fmt::format(fmt::runtime(fmt), pct(num, den)) : std::string();
	};
	if (has("task-clock-ns"))
		spdlog::debug("perf {}: cpu {:.3f} s, {:.0f} minor / {:.0f} major faults{}", phase, v["task-clock-ns"] * 1e-9, v["minor-faults"],
					  v["major-faults"],
					  user_only_ ? std::string()
								 : fmt::format(", {:.0f} context switches, {:.0f} migrations", v["context-switches"], v["cpu-migrations"]));
	if (has("cycles") && has("instructions"))
		spdlog::debug("perf {}: {:.4g} cycles{}{}, {:.4g} instructions, IPC {:.2f}{}{}", phase, v["cycles"],
					  opt(" ({:.1f}% in kernel)", "kernel-cycles", "cycles"),
					  has("ref-cycles") ? fmt::format(", clock {:.2f}x base", ratio(v["cycles"], v["ref-cycles"])) : std::string(),
					  v["instructions"], ratio(v["instructions"], v["cycles"]), opt(", stalled frontend {:.1f}%", "stalled-frontend", "cycles"),
					  opt(has("backend-stall-slots") ? ", backend-bound {:.1f}% of dispatch slots" : ", stalled backend {:.1f}%", "stalled-backend",
						  "cycles"));
	if (has("branches"))
		spdlog::debug("perf {}: {:.4g} branches, {:.2f}% missed ({:.2f} per 1k instructions)", phase, v["branches"],
					  pct("branch-misses", "branches"), 1000 * ratio(v["branch-misses"], v["instructions"]));
	if (has("cache-references"))
		spdlog::debug("perf {}: {:.4g} cache references, {:.2f}% missed ({:.2f} per 1k instructions)", phase, v["cache-references"],
					  pct("cache-misses", "cache-references"), 1000 * ratio(v["cache-misses"], v["instructions"]));
	if (has("L1d-loads"))
		spdlog::debug("perf {}: L1d {:.4g} loads, {:.2f}% missed; L1i {:.4g} misses", phase, v["L1d-loads"],
					  pct("L1d-load-misses", "L1d-loads"), v["L1i-load-misses"]);
	if (has("LLC-loads"))
		spdlog::debug("perf {}: LLC {:.4g} loads, {:.2f}% missed{}", phase, v["LLC-loads"], pct("LLC-load-misses", "LLC-loads"),
					  has("LLC-stores") ? fmt::format("; {:.4g} stores, {:.2f}% missed", v["LLC-stores"], pct("LLC-store-misses", "LLC-stores"))
										: std::string());
	if (has("dTLB-loads"))
		spdlog::debug("perf {}: dTLB {:.4g} loads, {:.3f}% missed; iTLB {:.4g} misses", phase, v["dTLB-loads"],
					  pct("dTLB-load-misses", "dTLB-loads"), v["iTLB-load-misses"]);
	if (has("node-loads"))
		spdlog::debug("perf {}: NUMA {:.4g} node loads, {:.2f}% remote", phase, v["node-loads"], pct("node-load-misses", "node-loads"));
	if (has("fills-local-l2")) {
		// Every L1d fill comes from one of these; past local L2 is an L2 miss, past the local L3 is an L3 (LLC) miss.
		const double l2 = v["fills-local-l2"], l3 = v["fills-local-l3"], ccx = v["fills-other-ccx"], dram = v["fills-local-dram"];
		const double rcache = v["fills-remote-cache"], rdram = v["fills-remote-dram"];
		const double l2_misses = l3 + ccx + dram + rcache + rdram, l3_misses = ccx + dram + rcache + rdram;
		spdlog::debug("perf {}: L1d fills {:.4g}, {:.4g} from beyond L2 ({:.2f} per 1k instructions); of those L3 {:.1f}%, other CCX {:.1f}%, "
					  "local DRAM {:.1f}%, remote cache {:.1f}%, remote DRAM {:.1f}%",
					  phase, l2 + l2_misses, l2_misses, 1000 * ratio(l2_misses, v["instructions"]), 100 * ratio(l3, l2_misses),
					  100 * ratio(ccx, l2_misses), 100 * ratio(dram, l2_misses), 100 * ratio(rcache, l2_misses), 100 * ratio(rdram, l2_misses));
		spdlog::debug("perf {}: LLC {:.4g} misses ({:.2f} per 1k instructions); NUMA {:.1f}% of LLC misses remote", phase, l3_misses,
					  1000 * ratio(l3_misses, v["instructions"]), 100 * ratio(rcache + rdram, l3_misses));
	}
	if (has("demand-beyond-l2"))
		spdlog::debug("perf {}: demand fills from beyond L2 {:.4g} ({:.2f} per 1k instructions, {:.0f}% of all such fills), {:.1f}% of them remote",
					  phase, v["demand-beyond-l2"], 1000 * ratio(v["demand-beyond-l2"], v["instructions"]),
					  100 * ratio(v["demand-beyond-l2"], v["fills-local-l3"] + v["fills-other-ccx"] + v["fills-local-dram"] +
															 v["fills-remote-cache"] + v["fills-remote-dram"]),
					  100 * ratio(v["demand-remote"], v["demand-beyond-l2"]));

	rusage usage{};
	getrusage(RUSAGE_SELF, &usage);
	spdlog::debug("time {}: user {:.3f} s, system {:.3f} s", phase, seconds(usage.ru_utime) - user_seconds_,
				  seconds(usage.ru_stime) - sys_seconds_);
	std::string nodes;
	const std::vector<std::uint64_t> resident = node_resident_bytes();
	std::uint64_t resident_total = 0;
	for (const std::uint64_t b : resident) resident_total += b;
	for (std::size_t n = 0; n < resident.size(); ++n)
		nodes += fmt::format("{}node {} {:.0f} MiB ({:.0f}%)", n ? ", " : "", n, resident[n] / 1048576.0, 100 * ratio(resident[n], resident_total));
	spdlog::debug("mem {}: peak RSS {:.0f} MiB, huge pages {:.0f} MiB; resident now {}", phase, usage.ru_maxrss / 1024.0,
				  smaps_bytes("AnonHugePages") / 1048576.0, nodes.empty() ? std::string("unknown") : nodes);
	if (has("energy-pkg"))
		spdlog::debug("energy {}: {:.1f} J over all packages (includes other processes)", phase, v["energy-pkg"] * energy_scale_);

#else
	(void)phase;
#endif
}

ThreadCounters::ThreadCounters() {
#if defined(__linux__)
	if (!spdlog::should_log(spdlog::level::debug)) return;
	const Event events[4] = {{"cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
							 {"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
							 {"demand-beyond-l2", PERF_TYPE_RAW, zen(0x43, 0xfe)},
							 {"demand-remote", PERF_TYPE_RAW, zen(0x43, 0x50)}};
	for (int k = 0; k < (is_zen4() ? 4 : 2); ++k) {
		perf_event_attr attr{};
		attr.size = sizeof(attr);
		attr.type = events[k].type;
		attr.config = events[k].config;
		attr.exclude_hv = 1;
		attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
		fds_[k] = static_cast<int>(syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0));
		if (fds_[k] < 0) {
			attr.exclude_kernel = 1;
			fds_[k] = static_cast<int>(syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0));
		}
	}
#endif
}

ThreadCounters::~ThreadCounters() {
#if defined(__linux__)
	for (const int fd : fds_)
		if (fd >= 0) close(fd);
#endif
}

ThreadCounters::Values ThreadCounters::read() {
	Values out;
#if defined(__linux__)
	double value[4] = {};
	bool counted[4] = {};  // got PMU time; a short-lived thread may get none while counters are multiplexed
	for (int k = 0; k < 4; ++k) {
		if (fds_[k] < 0) continue;
		ioctl(fds_[k], PERF_EVENT_IOC_DISABLE, 0);
		std::uint64_t values[3]{};
		if (::read(fds_[k], values, sizeof(values)) != sizeof(values) || values[2] == 0) continue;
		value[k] = static_cast<double>(values[0]) * static_cast<double>(values[1]) / static_cast<double>(values[2]);
		counted[k] = true;
	}
	out = {value[0], value[1], value[2], value[3], counted[0] && counted[1], counted[1] && counted[2] && counted[3]};
#endif
	return out;
}
