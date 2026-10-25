#pragma once
// Vulkan compute: the instance, the GPUs the miner can use, buffers in their memory and compute pipelines.
// The kernels address memory through buffer device addresses (pointers in push constants), so a pipeline
// has no descriptor sets: one layout with a push-constant block serves every kernel.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace grefim::vk {

// Throws std::runtime_error naming `what` unless `result` is VK_SUCCESS.
void check(VkResult result, const char* what);

class Instance {
public:
	Instance();
	~Instance();
	Instance(const Instance&) = delete;
	Instance& operator=(const Instance&) = delete;
	VkInstance get() const { return instance_; }

private:
	VkInstance instance_ = VK_NULL_HANDLE;
};

struct GpuInfo {
	VkPhysicalDevice physical = VK_NULL_HANDLE;
	std::string name, driver;
	bool integrated = false;  // shares system memory with the CPU
	std::uint32_t subgroup_size = 0;
	std::uint32_t shared_memory = 0;  // bytes per workgroup
	std::uint32_t max_workgroup = 0;  // invocations per workgroup (1-D)
	std::uint32_t queue_family = 0;
	std::string missing;  // what the kernels need and the device lacks; empty if usable
};

// Every GPU of the instance (software rasterisers excluded), usable or not.
std::vector<GpuInfo> list_gpus(VkInstance instance);

// Where a buffer lives: device memory, or host-visible memory the host writes (kUpload) or reads (kReadback).
enum class Memory { kDevice, kUpload, kReadback };

struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
	VkDeviceSize size = 0;
	VkDeviceAddress address = 0;
	void* mapped = nullptr;  // host-visible buffers only

	template <class T>
	T* as() const {
		return static_cast<T*>(mapped);
	}
};

constexpr std::uint32_t kPushBytes = 128;  // push-constant block of every kernel (the guaranteed minimum)

class Device {
public:
	explicit Device(const GpuInfo& info);
	~Device();
	Device(const Device&) = delete;
	Device& operator=(const Device&) = delete;

	const GpuInfo& info() const { return info_; }
	// Bytes still free in the heap of device buffers (the driver's budget when it reports one).
	VkDeviceSize free_device_memory() const;
	// Largest single buffer the driver allows.
	VkDeviceSize max_buffer() const { return max_buffer_; }

	Buffer create(VkDeviceSize size, Memory memory);
	void destroy(Buffer& buffer);
	// Copies `size` bytes into `dst` from the host (through a staging buffer unless dst is host-visible).
	void upload(const Buffer& dst, const void* data, VkDeviceSize size);

	// A compute pipeline from SPIR-V; spec[i] sets specialisation constant i (0: the workgroup size).
	VkPipeline pipeline(const std::vector<std::uint32_t>& spirv, const std::vector<std::uint32_t>& spec);
	// Records commands with `record`, submits them and waits until they finish; device writes are then
	// visible to the host.
	void run(const std::function<void(VkCommandBuffer)>& record);
	void dispatch(VkCommandBuffer cmd, VkPipeline pipeline, const void* push, std::uint32_t bytes, std::uint32_t groups_x,
				  std::uint32_t groups_y = 1) const;
	// Compute and transfer writes before it are visible to compute reads after it.
	static void barrier(VkCommandBuffer cmd);

private:
	std::uint32_t memory_type(Memory memory) const;

	GpuInfo info_;
	VkDevice device_ = VK_NULL_HANDLE;
	VkQueue queue_ = VK_NULL_HANDLE;
	VkCommandPool pool_ = VK_NULL_HANDLE;
	VkCommandBuffer cmd_ = VK_NULL_HANDLE;
	VkFence fence_ = VK_NULL_HANDLE;
	VkPipelineLayout layout_ = VK_NULL_HANDLE;
	std::vector<VkPipeline> pipelines_;
	std::vector<VkShaderModule> modules_;
	VkPhysicalDeviceMemoryProperties memory_{};
	bool budget_ = false;  // VK_EXT_memory_budget
	VkDeviceSize max_buffer_ = 0;
	std::uint32_t device_type_ = 0, upload_type_ = 0, readback_type_ = 0;
};

}  // namespace grefim::vk
