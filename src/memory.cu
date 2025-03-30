#include "memory.cuh"

namespace allocators {
    namespace atomic {

        // Definitions
        __managed__ char *allocator_base = nullptr;
        __managed__ unsigned long long allocator_offset = 0;
        __managed__ unsigned long long allocator_total = 0;

        void init_global_allocator(uint64_t total_memory, bool managed)
        {
            allocator_total = total_memory;
            if (allocator_base)
            {
                std::fprintf(stderr, "Global allocator already initialized.\n");
                std::exit(EXIT_FAILURE);
            }

            if (managed)
            {
                #ifdef DEBUG
                printf("Using managed memory\n");
                #endif
                cudaError_t err = cudaMallocManaged(&allocator_base, total_memory);
                if (err != cudaSuccess)
                {
                    std::fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
                    std::exit(EXIT_FAILURE);
                }
            }
            else
            {
                #ifdef DEBUG
                printf("Using device memory\n");
                #endif
                cudaError_t err = cudaMalloc(&allocator_base, total_memory);
                if (err != cudaSuccess)
                {
                    std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
                    std::exit(EXIT_FAILURE);
                }
            }
            allocator_offset = 0;
        }

        void free_global_allocator()
        {
            if (allocator_base)
            {
                cudaFree(allocator_base);
                allocator_base = nullptr;
            }
        }

        __device__ void *gmalloc(unsigned long long size)
        {
            size = (size + 7) & ~7;
            unsigned long long ret = atomicAdd(&allocator_offset, size);
            return allocator_base + ret;
        }

        __host__ void *gmalloc_host(unsigned long long size)
        {
            size = (size + 7) & ~7;
            unsigned long long ret = allocator_offset;
            allocator_offset += size;
            return allocator_base + ret;
        }

        __device__ void gfree(void * /*ptr*/)
        {
            // No-op
        }

    } // namespace atomic
} // namespace allocators
