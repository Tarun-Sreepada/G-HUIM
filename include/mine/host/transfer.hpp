#pragma once
// Host <-> device copies and timing helpers for the miner.

#include "cuda/buffer.cuh"

#include <chrono>

namespace ghuim {

template <typename T> void upload(T* device, const T* host, size_t count)
{
    if (count != 0)
        check_cuda(cudaMemcpy(device, host, count * sizeof(T), cudaMemcpyHostToDevice), "upload");
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
