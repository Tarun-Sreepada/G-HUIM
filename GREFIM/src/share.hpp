#pragma once
// GREFIM's RootShare (REFIM mine/share.hpp): the root tasks go to a Pool that REFIM's CPU workers and one
// GPU engine per device draw from. Built once per REFIM width pair, like the search.

#include "mine/search.hpp"
#include "vk/context.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace grefim {

struct ShareConfig {
	std::vector<vk::Device*> devices;
	std::uint64_t min_util = 0;
	bool collect_patterns = false;
	int cpu_workers = 0;  // 0: the CPU only builds the root, the GPUs mine
};

}  // namespace grefim

#ifdef REFIM_WIDTHS
inline namespace REFIM_WIDTHS {

std::unique_ptr<RootShareBase> make_share(const grefim::ShareConfig& config);

}  // namespace REFIM_WIDTHS
#endif
