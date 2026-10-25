#include "vk/context.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace grefim::vk {

void check(VkResult result, const char* what) {
	if (result != VK_SUCCESS) throw std::runtime_error(std::string("vulkan: ") + what + " failed (VkResult " + std::to_string(result) + ")");
}

Instance::Instance() {
	VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.pApplicationName = "GREFIM";
	app.apiVersion = VK_API_VERSION_1_3;
	VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	info.pApplicationInfo = &app;
	check(vkCreateInstance(&info, nullptr, &instance_), "create instance");
}

Instance::~Instance() {
	if (instance_) vkDestroyInstance(instance_, nullptr);
}

namespace {

bool has_extension(VkPhysicalDevice physical, const char* name) {
	std::uint32_t n = 0;
	vkEnumerateDeviceExtensionProperties(physical, nullptr, &n, nullptr);
	std::vector<VkExtensionProperties> all(n);
	vkEnumerateDeviceExtensionProperties(physical, nullptr, &n, all.data());
	return std::any_of(all.begin(), all.end(), [&](const VkExtensionProperties& e) { return std::strcmp(e.extensionName, name) == 0; });
}

}  // namespace

std::vector<GpuInfo> list_gpus(VkInstance instance) {
	std::uint32_t n = 0;
	check(vkEnumeratePhysicalDevices(instance, &n, nullptr), "enumerate devices");
	std::vector<VkPhysicalDevice> physical(n);
	check(vkEnumeratePhysicalDevices(instance, &n, physical.data()), "enumerate devices");

	std::vector<GpuInfo> gpus;
	for (VkPhysicalDevice p : physical) {
		VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
		VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, &subgroup};
		VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driver};
		vkGetPhysicalDeviceProperties2(p, &props);
		const VkPhysicalDeviceType type = props.properties.deviceType;
		if (type == VK_PHYSICAL_DEVICE_TYPE_CPU) continue;  // llvmpipe and the like: the CPU workers do better

		GpuInfo gpu;
		gpu.physical = p;
		gpu.name = props.properties.deviceName;
		gpu.driver = driver.driverName;
		gpu.integrated = type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
		gpu.subgroup_size = subgroup.subgroupSize;
		gpu.shared_memory = props.properties.limits.maxComputeSharedMemorySize;
		gpu.max_workgroup = std::min(props.properties.limits.maxComputeWorkGroupSize[0], props.properties.limits.maxComputeWorkGroupInvocations);

		VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
		VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v12};
		vkGetPhysicalDeviceFeatures2(p, &features);
		std::vector<std::string> missing;
		if (props.properties.apiVersion < VK_API_VERSION_1_2) missing.emplace_back("Vulkan 1.2");
		if (!features.features.shaderInt64) missing.emplace_back("shaderInt64");
		if (!v12.bufferDeviceAddress) missing.emplace_back("bufferDeviceAddress");
		if (!v12.shaderBufferInt64Atomics) missing.emplace_back("shaderBufferInt64Atomics");
		const VkSubgroupFeatureFlags ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
		if ((subgroup.supportedOperations & ops) != ops || !(subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT))
			missing.emplace_back("subgroup arithmetic in compute");

		std::uint32_t families = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(p, &families, nullptr);
		std::vector<VkQueueFamilyProperties> family(families);
		vkGetPhysicalDeviceQueueFamilyProperties(p, &families, family.data());
		// A compute-only family (no graphics) if there is one: it does not share the queue with the desktop.
		int pick = -1;
		for (std::uint32_t f = 0; f < families; ++f) {
			if (!(family[f].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
			if (pick < 0 || !(family[f].queueFlags & VK_QUEUE_GRAPHICS_BIT)) pick = static_cast<int>(f);
		}
		if (pick < 0) missing.emplace_back("a compute queue");
		gpu.queue_family = pick < 0 ? 0 : static_cast<std::uint32_t>(pick);

		for (std::size_t i = 0; i < missing.size(); ++i) gpu.missing += (i ? ", " : "") + missing[i];
		gpus.push_back(std::move(gpu));
	}
	return gpus;
}

Device::Device(const GpuInfo& info) : info_(info) {
	const float priority = 1;
	VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	queue.queueFamilyIndex = info.queue_family;
	queue.queueCount = 1;
	queue.pQueuePriorities = &priority;

	VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
	v12.bufferDeviceAddress = VK_TRUE;
	v12.shaderBufferInt64Atomics = VK_TRUE;
	VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v12};
	features.features.shaderInt64 = VK_TRUE;

	std::vector<const char*> extensions;
	budget_ = has_extension(info.physical, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
	if (budget_) extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);

	VkDeviceCreateInfo create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &features};
	create.queueCreateInfoCount = 1;
	create.pQueueCreateInfos = &queue;
	create.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
	create.ppEnabledExtensionNames = extensions.data();
	check(vkCreateDevice(info.physical, &create, nullptr, &device_), "create device");
	vkGetDeviceQueue(device_, info.queue_family, 0, &queue_);

	VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
	pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	pool.queueFamilyIndex = info.queue_family;
	check(vkCreateCommandPool(device_, &pool, nullptr, &pool_), "create command pool");
	VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
	alloc.commandPool = pool_;
	alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	alloc.commandBufferCount = 1;
	check(vkAllocateCommandBuffers(device_, &alloc, &cmd_), "allocate command buffer");
	VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	check(vkCreateFence(device_, &fence, nullptr, &fence_), "create fence");

	VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes};
	VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	layout.pushConstantRangeCount = 1;
	layout.pPushConstantRanges = &push;
	check(vkCreatePipelineLayout(device_, &layout, nullptr, &layout_), "create pipeline layout");

	vkGetPhysicalDeviceMemoryProperties(info.physical, &memory_);
	VkPhysicalDeviceMaintenance4Properties m4{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES};
	VkPhysicalDeviceMaintenance3Properties m3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES, &m4};
	VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &m3};
	vkGetPhysicalDeviceProperties2(info.physical, &props);
	max_buffer_ = m3.maxMemoryAllocationSize;
	if (props.properties.apiVersion >= VK_API_VERSION_1_3 && m4.maxBufferSize) max_buffer_ = std::min(max_buffer_, m4.maxBufferSize);
	device_type_ = memory_type(Memory::kDevice);
	upload_type_ = memory_type(Memory::kUpload);
	readback_type_ = memory_type(Memory::kReadback);
}

Device::~Device() {
	if (!device_) return;
	vkDeviceWaitIdle(device_);
	for (VkPipeline p : pipelines_) vkDestroyPipeline(device_, p, nullptr);
	for (VkShaderModule m : modules_) vkDestroyShaderModule(device_, m, nullptr);
	vkDestroyPipelineLayout(device_, layout_, nullptr);
	vkDestroyFence(device_, fence_, nullptr);
	vkDestroyCommandPool(device_, pool_, nullptr);
	vkDestroyDevice(device_, nullptr);
}

std::uint32_t Device::memory_type(Memory memory) const {
	const auto flags = [&](std::uint32_t t) { return memory_.memoryTypes[t].propertyFlags; };
	const auto heap = [&](std::uint32_t t) { return memory_.memoryHeaps[memory_.memoryTypes[t].heapIndex].size; };
	constexpr VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	int best = -1;
	// Higher is better; negative: unusable.
	const auto score = [&](std::uint32_t t) -> int {
		const VkMemoryPropertyFlags f = flags(t);
		switch (memory) {
			case Memory::kDevice:
				// An integrated GPU's device-local heap may be a small carve-out: any type of its largest heap will do.
				if (info_.integrated) return 1;
				return f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ? 1 : -1;
			case Memory::kUpload:
				if ((f & host) != host) return -1;
				return f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT ? 1 : 2;  // write-combined is best for writes
			case Memory::kReadback:
				if ((f & host) != host) return -1;
				return f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT ? 2 : 1;
		}
		return -1;
	};
	for (std::uint32_t t = 0; t < memory_.memoryTypeCount; ++t) {
		if (score(t) < 0) continue;
		if (best < 0 || score(t) > score(best) || (score(t) == score(best) && heap(t) > heap(best))) best = static_cast<int>(t);
	}
	if (best < 0) throw std::runtime_error(info_.name + ": no suitable memory type");
	return static_cast<std::uint32_t>(best);
}

VkDeviceSize Device::free_device_memory() const {
	const std::uint32_t heap = memory_.memoryTypes[device_type_].heapIndex;
	if (!budget_) return memory_.memoryHeaps[heap].size;
	VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
	VkPhysicalDeviceMemoryProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &budget};
	vkGetPhysicalDeviceMemoryProperties2(info_.physical, &props);
	return budget.heapBudget[heap] > budget.heapUsage[heap] ? budget.heapBudget[heap] - budget.heapUsage[heap] : 0;
}

Buffer Device::create(VkDeviceSize size, Memory memory) {
	Buffer b;
	b.size = std::max<VkDeviceSize>(size, 16);
	VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	info.size = b.size;
	info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
				 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	check(vkCreateBuffer(device_, &info, nullptr, &b.buffer), "create buffer");
	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements(device_, b.buffer, &req);
	const std::uint32_t type = memory == Memory::kDevice ? device_type_ : memory == Memory::kUpload ? upload_type_ : readback_type_;
	if (!(req.memoryTypeBits & (1u << type))) throw std::runtime_error(info_.name + ": buffer cannot use the chosen memory type");
	VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
	flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
	VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
	alloc.allocationSize = req.size;
	alloc.memoryTypeIndex = type;
	check(vkAllocateMemory(device_, &alloc, nullptr, &b.memory), "allocate memory");
	check(vkBindBufferMemory(device_, b.buffer, b.memory, 0), "bind buffer memory");
	if (memory_.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
		check(vkMapMemory(device_, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped), "map memory");
	VkBufferDeviceAddressInfo address{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
	address.buffer = b.buffer;
	b.address = vkGetBufferDeviceAddress(device_, &address);
	return b;
}

void Device::destroy(Buffer& b) {
	if (b.buffer) vkDestroyBuffer(device_, b.buffer, nullptr);
	if (b.memory) vkFreeMemory(device_, b.memory, nullptr);
	b = {};
}

void Device::upload(const Buffer& dst, const void* data, VkDeviceSize size) {
	if (size == 0) return;
	if (dst.mapped) {
		std::memcpy(dst.mapped, data, size);
		return;
	}
	Buffer staging = create(size, Memory::kUpload);
	std::memcpy(staging.mapped, data, size);
	run([&](VkCommandBuffer cmd) {
		const VkBufferCopy region{0, 0, size};
		vkCmdCopyBuffer(cmd, staging.buffer, dst.buffer, 1, &region);
	});
	destroy(staging);
}

VkPipeline Device::pipeline(const std::vector<std::uint32_t>& spirv, const std::vector<std::uint32_t>& spec_values) {
	VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	module.codeSize = spirv.size() * sizeof(std::uint32_t);
	module.pCode = spirv.data();
	VkShaderModule m;
	check(vkCreateShaderModule(device_, &module, nullptr, &m), "create shader module");
	modules_.push_back(m);

	std::vector<VkSpecializationMapEntry> entries;
	for (std::uint32_t i = 0; i < spec_values.size(); ++i)
		entries.push_back({i, static_cast<std::uint32_t>(i * sizeof(std::uint32_t)), sizeof(std::uint32_t)});
	VkSpecializationInfo spec{static_cast<std::uint32_t>(entries.size()), entries.data(), spec_values.size() * sizeof(std::uint32_t),
							  spec_values.data()};
	VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
	info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
	info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	info.stage.module = m;
	info.stage.pName = "main";
	info.stage.pSpecializationInfo = &spec;
	info.layout = layout_;
	VkPipeline p;
	check(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &info, nullptr, &p), "create compute pipeline");
	pipelines_.push_back(p);
	return p;
}

void Device::run(const std::function<void(VkCommandBuffer)>& record) {
	VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	check(vkBeginCommandBuffer(cmd_, &begin), "begin command buffer");
	// Submission order alone does not make earlier submissions' writes visible to these commands.
	VkMemoryBarrier earlier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
	earlier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
	earlier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
	vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &earlier, 0, nullptr, 0,
						 nullptr);
	record(cmd_);
	VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
	host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
	host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
	vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
						 &host, 0, nullptr, 0, nullptr);
	check(vkEndCommandBuffer(cmd_), "end command buffer");
	VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &cmd_;
	check(vkQueueSubmit(queue_, 1, &submit, fence_), "submit");
	check(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX), "wait for fence");
	check(vkResetFences(device_, 1, &fence_), "reset fence");
	check(vkResetCommandBuffer(cmd_, 0), "reset command buffer");
}

void Device::dispatch(VkCommandBuffer cmd, VkPipeline pipeline, const void* push, std::uint32_t bytes, std::uint32_t groups_x,
					  std::uint32_t groups_y) const {
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, bytes, push);
	vkCmdDispatch(cmd, groups_x, groups_y, 1);
}

void Device::barrier(VkCommandBuffer cmd) {
	VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
	b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
	b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
						 0, 1, &b, 0, nullptr, 0, nullptr);
}

}  // namespace grefim::vk
