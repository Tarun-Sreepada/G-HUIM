#pragma once
// Thread groups that cooperatively process one task (static-parallelism kernels).

#include "mine/widths.hpp"

#include <cuda_runtime.h>
#include <cstdint>

namespace ghuim::inline GHUIM_WIDTHS::dfs::baseline {

struct BlockGroup
{
    __device__ uint32_t rank() const { return threadIdx.x; }
    __device__ uint32_t size() const { return blockDim.x; }
    __device__ void     sync() const { __syncthreads(); }
};

struct WarpGroup
{
    __device__ uint32_t rank() const { return threadIdx.x & 31; }
    __device__ uint32_t size() const { return 32; }
    __device__ void     sync() const { __syncwarp(); }
};

} // namespace ghuim::inline GHUIM_WIDTHS::dfs::baseline
