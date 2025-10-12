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
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        //   if (abort) exit(code);
    }
}

namespace allocators {
namespace atomic {

// Declaration only
extern __managed__ char*              allocator_base;
extern __managed__ unsigned long long allocator_offset;
extern __managed__ unsigned long long allocator_total;

void init_global_allocator(unsigned long long int total_memory, bool managed = false);
void free_global_allocator();

void reclaim_memory();
void record_memory_usage();

__device__ void* gmalloc(unsigned long long size);
__host__ void*   gmalloc_host(unsigned long long size);
__device__ void  gfree(void* ptr);

} // namespace atomic
} // namespace allocators

using namespace allocators::atomic;
