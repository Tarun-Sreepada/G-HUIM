#include "sys/perf.hpp"

#include <spdlog/spdlog.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <string>

#if defined(__linux__)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
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
	{"ref-cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_REF_CPU_CYCLES},
	{"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
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

// User and kernel space (system calls, page faults, huge-page compaction) need perf_event_paranoid <= 1;
// above that, user_only drops the kernel.
int open_event(const Event& e, bool user_only) {
	perf_event_attr attr{};
	attr.size = sizeof(attr);
	attr.type = e.type;
	attr.config = e.config;
	attr.disabled = 1;
	attr.inherit = 1;
	attr.exclude_kernel = user_only;
	attr.exclude_user = e.kernel_only;
	attr.exclude_hv = 1;
	attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
	return static_cast<int>(syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0));
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

PerfCounters::PerfCounters() {
#if defined(__linux__)
	if (!spdlog::should_log(spdlog::level::debug)) return;
	int first_error = 0;
	std::string unsupported;
	if (const int probe = open_event(kEvents[0], false); probe >= 0)
		close(probe);
	else if (const int user_probe = open_event(kEvents[0], true); user_probe >= 0) {
		close(user_probe);
		user_only_ = true;
	}
	if (user_only_) spdlog::warn("perf: counting user space only; kernel.perf_event_paranoid is {}, set it to 1 or lower to count the kernel too", paranoid_level());
	for (const Event& e : kEvents) {
		if (e.kernel_only && user_only_) continue;
		const int fd = open_event(e, user_only_);
		if (fd >= 0) {
			counters_.push_back({e.name, fd});
			continue;
		}
		if (!first_error) first_error = errno;
		if (errno == ENOENT || errno == EOPNOTSUPP || errno == EINVAL) unsupported += unsupported.empty() ? e.name : std::string(", ") + e.name;
	}
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
	for (const Counter& c : counters_) {
		ioctl(c.fd, PERF_EVENT_IOC_RESET, 0);
		ioctl(c.fd, PERF_EVENT_IOC_ENABLE, 0);
	}
#endif
}

void PerfCounters::report(const std::string& phase) {
#if defined(__linux__)
	if (counters_.empty()) return;
	std::map<std::string, double> v;
	for (const Counter& c : counters_) {
		ioctl(c.fd, PERF_EVENT_IOC_DISABLE, 0);
		std::uint64_t values[3]{};
		if (read(c.fd, values, sizeof(values)) != sizeof(values) || values[2] == 0) continue;
		const double scale = static_cast<double>(values[1]) / static_cast<double>(values[2]);
		v[c.name] = static_cast<double>(values[0]) * scale;
		spdlog::trace("perf {}: {} {:.0f} (counted {:.0f}% of the time)", phase, c.name, v[c.name], 100 / scale);
	}
	const auto has = [&](const char* name) { return v.count(name) != 0; };
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
		spdlog::debug("perf {}: {:.4g} cycles{}, {:.4g} instructions, IPC {:.2f}{}{}", phase, v["cycles"],
					  opt(" ({:.1f}% in kernel)", "kernel-cycles", "cycles"), v["instructions"], ratio(v["instructions"], v["cycles"]),
					  opt(", stalled frontend {:.1f}%", "stalled-frontend", "cycles"), opt(", stalled backend {:.1f}%", "stalled-backend", "cycles"));
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
#else
	(void)phase;
#endif
}
