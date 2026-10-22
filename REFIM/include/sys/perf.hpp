// CPU hardware counters (perf_event_open) for one phase of the run, logged at debug level. Counters cover the
// calling thread and every thread it starts afterwards (their counts are added in when they exit), user and kernel space
// (user space only when kernel.perf_event_paranoid is above 1). Each phase also logs process-wide figures: CPU time,
// peak RSS, transparent huge pages, resident memory per NUMA node, package energy (RAPL) and, when resctrl is mounted,
// memory bandwidth from the L3 monitors.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

class PerfCounters {
public:
	// Opens nothing unless the log level is debug or trace. With `bandwidth`, the memory bandwidth line is logged at info
	// level whatever the log level.
	explicit PerfCounters(bool bandwidth = false);
	~PerfCounters();
	PerfCounters(const PerfCounters&) = delete;
	PerfCounters& operator=(const PerfCounters&) = delete;

	// Zeroes the counters and starts counting.
	void start();
	// Stops counting and logs "perf <phase>: ..." lines.
	void report(const std::string& phase);

private:
	struct Counter {
		const char* name;
		int fd;
	};
	std::vector<Counter> counters_;
	bool user_only_ = false;
	double energy_scale_ = 0;  // joules per RAPL count
	// Snapshots taken by start().
	double user_seconds_ = 0, sys_seconds_ = 0;
	std::uint64_t mbm_total_ = 0, mbm_local_ = 0;
	bool mbm_ = false, bandwidth_ = false;
};

// Counters for the calling thread alone (no inherit), to tell workers apart: cycles, instructions and, on Zen 4,
// demand fills from beyond L2 and from the other socket. Opens nothing unless the log level is debug or trace.
class ThreadCounters {
public:
	struct Values {
		double cycles = 0, instructions = 0, demand_beyond_l2 = 0, demand_remote = 0;
		bool valid = false, fills = false;
	};

	ThreadCounters();
	~ThreadCounters();
	ThreadCounters(const ThreadCounters&) = delete;
	ThreadCounters& operator=(const ThreadCounters&) = delete;

	// Stops counting and returns the counts so far.
	Values read();

private:
	int fds_[4] = {-1, -1, -1, -1};
};
