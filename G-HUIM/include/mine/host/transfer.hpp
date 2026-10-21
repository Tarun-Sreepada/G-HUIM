#pragma once
// Host <-> device copies and timing helpers for the miner.

#include "cuda/buffer.cuh"

#include <chrono>
#include <type_traits>
#include <vector>

namespace ghuim {

template <typename T> void upload(T* device, const T* host, size_t count)
{
    if (count != 0)
        check_cuda(cudaMemcpy(device, host, count * sizeof(T), cudaMemcpyHostToDevice), "upload");
}

// Uploads host values of another type converted to T (the dataset is 32/64-bit, device entries may be narrower).
template <typename T, typename S> void upload_as(T* device, const S* host, size_t count)
{
    if constexpr (std::is_same_v<T, S>)
        upload(device, host, count);
    else {
        const std::vector<T> converted(host, host + count);
        upload(device, converted.data(), count);
    }
}

template <typename T> void download(T* host, const T* device, size_t count)
{
    if (count != 0)
        check_cuda(cudaMemcpy(host, device, count * sizeof(T), cudaMemcpyDeviceToHost), "download");
}

inline double ms_since(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

} // namespace ghuim
