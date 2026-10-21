#pragma once
// Device memory for the projected-database arena, from the VMM API: cuMemCreate maps tens of GB in a few ms
// where cudaMalloc takes hundreds. With `compress` (and device support) the allocation is also
// CU_MEM_ALLOCATION_COMP_GENERIC, so the L2 stores lines that compress (zeros) in fewer DRAM bytes
// (microbenchmark 14); the arena's data barely compresses, so it is off by default.

#include "cuda/buffer.cuh"

#include <cuda.h>
#include <utility>

namespace ghuim {

inline void check_cu(CUresult result, const char* operation)
{
    if (result != CUDA_SUCCESS) {
        const char* text = nullptr;
        cuGetErrorString(result, &text);
        throw std::runtime_error(std::string(operation) + ": " + (text ? text : "unknown driver error"));
    }
}

class ArenaBuffer
{
  public:
    ArenaBuffer() = default;
    ArenaBuffer(size_t bytes, bool compress)
    {
        if (bytes == 0)
            return;
        int device = 0, supported = 0;
        check_cuda(cudaGetDevice(&device), "get device");
        if (compress)
            check_cu(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, device),
                     "query compression support");
        CUmemAllocationProp prop{};
        prop.type                       = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type              = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id                = device;
        if (supported)
            prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
        size_t granularity = 0;
        check_cu(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
                 "query allocation granularity");
        size_ = (bytes + granularity - 1) / granularity * granularity;
        check_cu(cuMemCreate(&handle_, size_, &prop, 0), "allocate arena memory");
        CUmemAllocationProp granted{};
        check_cu(cuMemGetAllocationPropertiesFromHandle(&granted, handle_), "query allocation");
        compressed_ = granted.allocFlags.compressionType == CU_MEM_ALLOCATION_COMP_GENERIC;
        check_cu(cuMemAddressReserve(&address_, size_, granularity, 0, 0), "reserve device addresses");
        check_cu(cuMemMap(address_, size_, 0, handle_, 0), "map device memory");
        CUmemAccessDesc access{};
        access.location = prop.location;
        access.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        check_cu(cuMemSetAccess(address_, size_, &access, 1), "enable device memory access");
    }
    ~ArenaBuffer() { release(); }
    ArenaBuffer(ArenaBuffer&& other) noexcept { *this = std::move(other); }
    ArenaBuffer& operator=(ArenaBuffer&& other) noexcept
    {
        std::swap(address_, other.address_);
        std::swap(size_, other.size_);
        std::swap(handle_, other.handle_);
        std::swap(compressed_, other.compressed_);
        return *this;
    }
    ArenaBuffer(const ArenaBuffer&)            = delete;
    ArenaBuffer& operator=(const ArenaBuffer&) = delete;

    char* get() const { return reinterpret_cast<char*>(address_); }
    bool  compressed() const { return compressed_; }

  private:
    void release()
    {
        if (address_ == 0)
            return;
        cuMemUnmap(address_, size_);
        cuMemAddressFree(address_, size_);
        cuMemRelease(handle_);
        address_ = 0;
    }

    CUdeviceptr                  address_ = 0;
    size_t                       size_    = 0;
    CUmemGenericAllocationHandle handle_{};
    bool                         compressed_ = false;
};

} // namespace ghuim
