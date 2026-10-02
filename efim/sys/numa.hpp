#pragma once
// CPU topology (NUMA node / L3 group = CCD on EPYC / core) read from sysfs, and thread pinning.
// Memory placement relies on first touch: a worker is pinned before it allocates, so its pages
// land on its own node. No libnuma needed.

#include <vector>

struct CpuInfo {
	int cpu;
	int node;  // compact NUMA node index
	int l3;    // compact L3-sharing group index (one CCD on Zen)
	int rank;  // 0 = first hardware thread of its core, 1 = SMT sibling
};

struct Topology {
	std::vector<CpuInfo> cpus;  // CPUs this process may run on
	int nodes = 1;
	int l3_groups = 1;
	bool valid = false;
};

Topology detect_topology();

// The first `n` CPUs to use: one hardware thread per core before any SMT sibling, spreading over
// L3 groups round-robin (alternating nodes) so each worker gets its own cache slice and both
// sockets' memory bandwidth. Fewer than `n` entries are returned if the CPU set is smaller.
std::vector<CpuInfo> pick_cpus(const Topology& topology, int n);

// Pins the calling thread to one CPU; returns false if the kernel refused.
bool pin_this_thread(int cpu);
