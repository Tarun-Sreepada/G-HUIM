#include "sys/numa.hpp"

#include <algorithm>
#include <dirent.h>
#include <fstream>
#include <linux/mempolicy.h>
#include <map>
#include <sched.h>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/syscall.h>
#include <unistd.h>

namespace {

std::string read_line(const std::string& path) {
	std::ifstream in(path);
	std::string line;
	std::getline(in, line);
	return line;
}

int node_of(int cpu) {
	const std::string dir = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
	DIR* d = ::opendir(dir.c_str());
	if (!d) return 0;
	int node = 0;
	while (const dirent* e = ::readdir(d))
		if (std::string_view(e->d_name).substr(0, 4) == "node" && e->d_name[4] >= '0' && e->d_name[4] <= '9')
			node = std::stoi(e->d_name + 4);
	::closedir(d);
	return node;
}

}  // namespace

Topology detect_topology() {
	Topology t;
	cpu_set_t allowed;
	CPU_ZERO(&allowed);
	if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return t;

	std::map<int, int> node_ids;          // raw node -> compact
	std::map<std::string, int> l3_ids;    // shared_cpu_list -> compact
	std::map<std::string, int> core_seen; // thread_siblings_list -> threads already seen
	for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
		if (!CPU_ISSET(cpu, &allowed)) continue;
		const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
		CpuInfo info{cpu, 0, 0, 0};
		info.node = node_ids.emplace(node_of(cpu), static_cast<int>(node_ids.size())).first->second;
		std::string l3 = read_line(base + "/cache/index3/shared_cpu_list");
		if (l3.empty()) l3 = "node" + std::to_string(info.node);
		info.l3 = l3_ids.emplace(l3, static_cast<int>(l3_ids.size())).first->second;
		const std::string siblings = read_line(base + "/topology/thread_siblings_list");
		info.rank = siblings.empty() ? 0 : core_seen[siblings]++;
		t.cpus.push_back(info);
	}
	t.nodes = std::max<int>(1, static_cast<int>(node_ids.size()));
	t.l3_groups = std::max<int>(1, static_cast<int>(l3_ids.size()));
	t.valid = !t.cpus.empty();
	return t;
}

std::vector<CpuInfo> pick_cpus(const Topology& topology, int n) {
	// Groups ordered node-alternating: node0/l3a, node1/l3c, node0/l3b, ...
	std::map<int, std::vector<std::vector<CpuInfo>>> by_node;  // node -> groups
	std::map<std::pair<int, int>, std::size_t> group_index;
	for (const CpuInfo& c : topology.cpus) {
		auto& groups = by_node[c.node];
		const auto key = std::make_pair(c.node, c.l3);
		auto it = group_index.find(key);
		if (it == group_index.end()) it = group_index.emplace(key, groups.size()).first, groups.emplace_back();
		groups[it->second].push_back(c);
	}
	std::vector<std::vector<CpuInfo>*> order;
	for (std::size_t round = 0;; ++round) {
		bool any = false;
		for (auto& [node, groups] : by_node)
			if (round < groups.size()) order.push_back(&groups[round]), any = true;
		if (!any) break;
	}
	std::vector<CpuInfo> picked;
	for (int rank = 0; static_cast<int>(picked.size()) < n; ++rank) {
		std::vector<std::size_t> next(order.size(), 0);
		bool any = false;
		// Round-robin over groups, taking only CPUs of this SMT rank.
		for (bool progress = true; progress && static_cast<int>(picked.size()) < n;) {
			progress = false;
			for (std::size_t g = 0; g < order.size() && static_cast<int>(picked.size()) < n; ++g) {
				auto& list = *order[g];
				while (next[g] < list.size() && list[next[g]].rank != rank) ++next[g];
				if (next[g] < list.size()) picked.push_back(list[next[g]++]), progress = any = true;
			}
		}
		if (!any) break;
	}
	return picked;
}

bool pin_this_thread(int cpu) {
	cpu_set_t set;
	CPU_ZERO(&set);
	CPU_SET(cpu, &set);
	return sched_setaffinity(0, sizeof(set), &set) == 0;
}

int restrict_to_nodes(const std::vector<int>& nodes) {
	cpu_set_t allowed, set;
	CPU_ZERO(&set);
	if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return 0;
	for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
		if (CPU_ISSET(cpu, &allowed) && std::find(nodes.begin(), nodes.end(), node_of(cpu)) != nodes.end()) CPU_SET(cpu, &set);
	const int count = CPU_COUNT(&set);
	if (count == 0) return 0;
	unsigned long mask[16] = {};  // up to 1024 nodes
	for (const int node : nodes)
		if (node >= 0 && node < 1024) mask[node / 64] |= 1ul << (node % 64);
	if (::syscall(SYS_set_mempolicy, MPOL_BIND, mask, 1024 + 1) != 0) return 0;
	return sched_setaffinity(0, sizeof(set), &set) == 0 ? count : 0;
}
