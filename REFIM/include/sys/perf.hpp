// CPU hardware counters (perf_event_open) for one phase of the run, logged at debug level. Counters cover the
// calling thread and every thread it starts afterwards (their counts are added in when they exit), user and kernel space
// (user space only when kernel.perf_event_paranoid is above 1).
#pragma once

#include <string>
#include <vector>

class PerfCounters {
public:
	// Opens nothing unless the log level is debug or trace.
	PerfCounters();
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
};
