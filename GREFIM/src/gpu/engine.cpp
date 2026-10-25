#include "gpu/engine.hpp"

#include "shaders.hpp"  // generated: expand_spv()

#include <spdlog/spdlog.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <stdexcept>

namespace grefim {

namespace {

constexpr std::uint32_t kNone = 0xffffffffu;
constexpr std::uint32_t kMaxGroupsX = 65535;  // the guaranteed maxComputeWorkGroupCount[0]
constexpr std::uint32_t kMaxSubgroups = 32;   // of shaders/expand.comp

}  // namespace

Engine::Engine(vk::Device& device, const GpuRoot& root, std::uint32_t min_util)
	: device_(device), min_util_(min_util), root_words_(root.words.size()), primary_(root.primary) {
	// Small tasks: two subgroups per workgroup. Big ones: the largest workgroup (a power of two). Shared bins
	// for every item of the largest database, the root.
	const std::uint32_t subgroup = device.info().subgroup_size;
	workgroup_[0] = 2 * subgroup;
	workgroup_[1] = std::max(workgroup_[0], std::bit_floor(std::min(device.info().max_workgroup, kMaxSubgroups * subgroup)));
	const std::uint32_t max_items = std::bit_ceil(std::max(root.items, 32u));
	if (3 * 4 * std::uint64_t{max_items} + 1024 > device.info().shared_memory)
		throw std::runtime_error(device.info().name + ": " + std::to_string(root.items) + " items do not fit shared memory");
	for (int c = 0; c < 2; ++c) expand_[c] = device.pipeline(expand_spv(), {workgroup_[c], max_items});

	// Memory: a discrete GPU takes three quarters of what is free. An integrated one shares system memory (and
	// its bandwidth) with the CPU workers, so it takes at most a quarter, and no more than eight roots' worth:
	// it only ever gets a small share of the work. An eighth of that holds the task lists: two per level for
	// as many levels as the longest itemset, which has at most as many items as the root.
	const VkDeviceSize free = device.free_device_memory();
	const VkDeviceSize budget = device.info().integrated ? std::min(free / 4, 8 * VkDeviceSize{root_words_} * 4) : free / 4 * 3;
	const VkDeviceSize per_list = budget / 8 / (2 * VkDeviceSize{std::max(root.items, 1u)});
	task_cap_ = static_cast<std::uint32_t>(std::bit_floor(std::min<VkDeviceSize>(per_list / 8, kNone / 2)));
	const VkDeviceSize arena_bytes = std::min({budget - budget / 8, device.max_buffer(), VkDeviceSize{kNone} * 4});
	arena_words_ = arena_bytes / 4;
	if (task_cap_ == 0 || arena_words_ < 2 * root_words_) throw std::runtime_error(device.info().name + ": not enough memory");
	arena_ = device.create(arena_words_ * 4, vk::Memory::kDevice);
	device.upload(arena_, root.words.data(), root_words_ * 4);
	control_ = device.create(sizeof(Control), vk::Memory::kDevice);
	readback_ = device.create(sizeof(Control), vk::Memory::kReadback);
	device.run([&](VkCommandBuffer cmd) { vkCmdFillBuffer(cmd, control_.buffer, 0, VK_WHOLE_SIZE, 0); });
	spdlog::debug("{}: workgroups {} and {}, arena {} MiB (root {} MiB), {} tasks per list", device.info().name, workgroup_[0],
				  workgroup_[1], arena_bytes >> 20, (root_words_ * 4) >> 20, task_cap_);
}

Engine::~Engine() {
	for (Level& l : levels_)
		for (auto& lists : l.lists)
			for (vk::Buffer& b : lists) device_.destroy(b);
	for (vk::Buffer* b : {&arena_, &control_, &readback_}) device_.destroy(*b);
}

Engine::Level& Engine::level(std::size_t depth) {
	while (levels_.size() <= depth) {
		Level& l = levels_.emplace_back();
		for (auto& lists : l.lists)
			for (vk::Buffer& b : lists) b = device_.create(VkDeviceSize{task_cap_} * 8, vk::Memory::kDevice);
	}
	return levels_[depth];
}

void Engine::mine(const std::vector<std::uint32_t>& slots) {
	std::vector<std::uint32_t> roots;
	for (const std::uint32_t s : slots) roots.insert(roots.end(), {0u, primary_[s]});
	const vk::Buffer& list = level(0).lists[1][0];  // the root is big
	device_.run([&](VkCommandBuffer cmd) {
		for (std::size_t at = 0; at < roots.size(); at += 16384)  // vkCmdUpdateBuffer takes up to 64 KiB
			vkCmdUpdateBuffer(cmd, list.buffer, at * 4, std::min<std::size_t>(16384, roots.size() - at) * 4, roots.data() + at);
	});
	const std::uint32_t count[2] = {0, static_cast<std::uint32_t>(slots.size())};
	mine_level(0, count, {0, root_words_}, arena_words_, root_words_);
	pattern_count = readback_.as<Control>()->patterns;
}

void Engine::mine_level(std::size_t depth, const std::uint32_t (&tasks_in)[2], Span mine, std::uint64_t from,
						std::uint64_t frontier) {
	Level& here = level(depth);
	const Level& below = level(depth + 1);
	const bool down = depth % 2 == 0;  // children (odd depth) on the top end
	const std::uint64_t space = down ? from - frontier : frontier - from;
	std::uint32_t count[2] = {tasks_in[0], tasks_in[1]};
	int input[2] = {0, 0};
	bool all_free = false;  // the last pass placed nothing in half the space: offer all of it
	while (count[0] + count[1] != 0) {
		const std::uint64_t limit = all_free ? space : space / 2;
		const std::uint64_t reset[4] = {0, 0, 0, 0};  // used, next[2], retry[2]
		const auto t0 = std::chrono::steady_clock::now();
		device_.run([&](VkCommandBuffer cmd) {
			vkCmdUpdateBuffer(cmd, control_.buffer, offsetof(Control, used), sizeof reset, reset);
			vk::Device::barrier(cmd);
			// Both classes at once: they share the arena and the next level's lists, through atomics.
			for (std::uint32_t c = 0; c < 2; ++c) {
				if (count[c] == 0) continue;
				const std::uint32_t groups_x = std::min(count[c], kMaxGroupsX);
				const Push push{arena_.address, here.lists[c][input[c]].address, below.lists[0][0].address, below.lists[1][0].address,
								here.lists[c][input[c] ^ 1].address, control_.address, count[c], min_util_, task_cap_,
								static_cast<std::uint32_t>(from), down ? 1u : 0u, static_cast<std::uint32_t>(limit), groups_x, c,
								workgroup_[1]};
				device_.dispatch(cmd, expand_[c], &push, sizeof push, groups_x, (count[c] + groups_x - 1) / groups_x);
			}
			vk::Device::barrier(cmd);
			const VkBufferCopy all{0, 0, sizeof(Control)};
			vkCmdCopyBuffer(cmd, control_.buffer, readback_.buffer, 1, &all);
		});
		const Control done = *readback_.as<Control>();
		const std::uint32_t retry = done.retry[0] + done.retry[1], ran = count[0] + count[1];
		spdlog::trace("{}: depth {} pass: {}+{} tasks, {} retry, {}+{} next, {} MiB used of {} MiB, {:.1f} ms", device_.info().name, depth,
					  count[0], count[1], retry, done.next[0], done.next[1], (std::min(done.used, limit) * 4) >> 20, (limit * 4) >> 20,
					  1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
		++passes;
		tasks += ran - retry;
		retried += retry;
		if (retry == ran) {
			if (all_free) throw std::runtime_error(device_.info().name + ": a child database does not fit the arena");
			all_free = true;
		} else {
			all_free = false;
			const std::uint32_t next[2] = {static_cast<std::uint32_t>(std::min<std::uint64_t>(done.next[0], task_cap_)),
										   static_cast<std::uint32_t>(std::min<std::uint64_t>(done.next[1], task_cap_))};
			const std::uint64_t used = std::min(done.used, limit);
			const Span kids = down ? Span{from - used, from} : Span{from, from + used};
			// Grandchildren go on this level's end: over this level once it is all done (the root never is),
			// else past it. They grow towards the children.
			const bool dead = retry == 0 && depth > 0;
			const std::uint64_t grand_from = down ? (dead ? mine.lo : mine.hi) : (dead ? mine.hi : mine.lo);
			if (next[0] + next[1] != 0) mine_level(depth + 1, next, kids, grand_from, down ? kids.lo : kids.hi);
		}
		for (int c = 0; c < 2; ++c) {
			if (count[c] != 0) input[c] ^= 1;
			count[c] = done.retry[c];
		}
	}
}

}  // namespace grefim
