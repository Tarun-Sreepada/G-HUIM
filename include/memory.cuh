#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <iostream>


namespace allocators
{
    namespace atomic
    {

        // Use __managed__ so the variables are accessible from both host and device.
        __managed__ char *allocator_base = nullptr;
        __managed__ unsigned long long allocator_offset = 0;
        __managed__ unsigned long long allocator_total = 0;

        // Initializes the global allocator with the requested total memory.
        // The second parameter is unused (dummy) but provided to match the given function signature.
        void init_global_allocator(uint64_t total_memory, bool managed = false)
        {
            allocator_total = total_memory;
            // cudaError_t err = cudaMalloc(&allocator_base, total_memory);
            if (allocator_base)
            {
                std::fprintf(stderr, "Global allocator already initialized.\n");
                std::exit(EXIT_FAILURE);
            }

            // Allocate memory on the device
            if (managed)
            {
                printf("Using managed memory\n");
                cudaError_t err = cudaMallocManaged(&allocator_base, total_memory);
                if (err != cudaSuccess)
                {
                    std::fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
                    std::exit(EXIT_FAILURE);
                }
            }
            else
            {
                printf("Allocating memory on device\n");
                cudaError_t err = cudaMalloc(&allocator_base, total_memory);
                if (err != cudaSuccess)
                {
                    std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
                    std::exit(EXIT_FAILURE);
                }
            }

            allocator_offset = 0;
        }

        // Frees the entire allocated memory.
        void free_global_allocator()
        {
            if (allocator_base)
            {
                cudaFree(allocator_base);
                allocator_base = nullptr;
            }
        }

        // Returns a pointer to a block of memory of the requested size from the global pool.
        // Alignment defaults to 8 bytes. On failure, returns nullptr.
        __device__ void *gmalloc(unsigned long long size)
        {

            // Round up to the next multiple of 8
            size = (size + 7) & ~7;

            unsigned long long ret = atomicAdd(&allocator_offset, size);
            return allocator_base + ret;
        }

        // Bump allocators typically do not support freeing individual allocations.
        // This function is provided to match the API but is a no-op.
        __device__ void gfree(void * /*ptr*/)
        {
            // No-op: individual allocations cannot be freed in a bump allocator.
        }

    }
}

using namespace allocators::atomic;
