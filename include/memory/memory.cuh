#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#define gpuErrchk(ans)                                                                                                 \
    {                                                                                                                  \
        gpuAssert((ans), __FILE__, __LINE__);                                                                          \
    }
// Reports CUDA errors and optionally aborts when a call fails.
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    }
}

namespace allocators {
namespace atomic {

// Declaration only
extern __managed__ char*              allocator_base;
extern __managed__ unsigned long long allocator_offset;
extern __managed__ unsigned long long allocator_total;

// Allocates a global buffer used by the custom device-side allocator.
void init_global_allocator(unsigned long long int total_memory, bool managed = false);
// Releases the global buffer owned by the custom allocator.
void free_global_allocator();

// Device-side allocation using the global allocator.
__device__ void* gmalloc(unsigned long long size);
// Host-accessible allocation backed by the global allocator.
__host__ void*   gmalloc_host(unsigned long long size);
// Device-side free for memory from the global allocator.
__device__ void  gfree(void* ptr);

} // namespace atomic
} // namespace allocators

using namespace allocators::atomic;
