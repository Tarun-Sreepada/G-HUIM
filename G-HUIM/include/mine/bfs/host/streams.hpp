#pragma once
// The CUDA streams the BFS miner runs its bins on.

#include "cuda/buffer.cuh"
#include "mine/widths.hpp"

#include <array>
#include <cstddef>
#include <cuda_runtime.h>

namespace ghuim::inline GHUIM_WIDTHS::bfs {

constexpr uint32_t kStreams = 4;

class Streams
{
  public:
    Streams()
    {
        for (auto& stream : streams_)
            check_cuda(cudaStreamCreate(&stream), "create stream");
    }
    ~Streams()
    {
        for (auto stream : streams_)
            cudaStreamDestroy(stream);
    }
    Streams(const Streams&)                = delete;
    Streams&     operator=(const Streams&) = delete;
    cudaStream_t operator[](size_t i) const
    {
        return streams_[i % kStreams];
    }

  private:
    std::array<cudaStream_t, kStreams> streams_{};
};

} // namespace ghuim::inline GHUIM_WIDTHS::bfs
