#pragma once

#include <cstring>
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

// Reports the first error of the launches issued since the last check. Launch failures are
// reported here rather than at the next synchronization.
inline void check_launch(const char* operation)
{
    check_cuda(cudaGetLastError(), operation);
}

// Waits for all device work and reports any error it hit.
inline void synchronize(const char* operation)
{
    check_cuda(cudaDeviceSynchronize(), operation);
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

// A small zero-initialized object in managed memory, for control blocks the host reads between
// launches. Bulk data stays in DeviceBuffer; managed memory is never oversubscribed.
template <typename T> class ManagedObject
{
  public:
    ManagedObject()
    {
        check_cuda(cudaMallocManaged(&object_, sizeof(T)), "allocate managed control block");
        std::memset(static_cast<void*>(object_), 0, sizeof(T));
    }
    ~ManagedObject() { cudaFree(object_); }
    ManagedObject(ManagedObject&& other) noexcept : object_(std::exchange(other.object_, nullptr)) {}
    ManagedObject& operator=(ManagedObject&& other) noexcept
    {
        std::swap(object_, other.object_);
        return *this;
    }
    ManagedObject(const ManagedObject&)            = delete;
    ManagedObject& operator=(const ManagedObject&) = delete;

    T* get() const { return object_; }
    T* operator->() const { return object_; }
    T& operator*() const { return *object_; }

  private:
    T* object_ = nullptr;
};

} // namespace ghuim
