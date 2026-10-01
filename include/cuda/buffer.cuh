#pragma once

#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ghuim {

inline void check_cuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

// Owns a device allocation of `count` elements.
template <typename T> class DeviceBuffer
{
  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) : count_(count)
    {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::overflow_error("device buffer size overflow");
        if (count != 0)
            check_cuda(cudaMalloc(&data_, count * sizeof(T)), "allocate device memory");
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0))
    {
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        std::swap(data_, other.data_);
        std::swap(count_, other.count_);
        return *this;
    }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T*     get() const { return data_; }
    size_t size() const { return count_; }

  private:
    T*     data_  = nullptr;
    size_t count_ = 0;
};

} // namespace ghuim
