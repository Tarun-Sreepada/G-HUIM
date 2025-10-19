#include "memory/memory.cuh"

namespace allocators {
namespace atomic {

__managed__ char*                  allocator_base   = nullptr;
__managed__ unsigned long long int allocator_offset = 0;
__managed__ unsigned long long int allocator_total  = 0;

// Reserves device or managed memory for the global allocator.
void init_global_allocator(unsigned long long int total_memory, bool managed)
{
    allocator_total = total_memory;

    if (managed) {
#ifdef DEBUG
        printf("Using managed memory\n");
#endif
        cudaError_t err = cudaMallocManaged(&allocator_base, total_memory);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
            std::exit(EXIT_FAILURE);
        }
    }
    else {
#ifdef DEBUG
        printf("Using device memory\n");
#endif
        cudaError_t err = cudaMalloc(&allocator_base, total_memory);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
            std::exit(EXIT_FAILURE);
        }
    }
    allocator_offset = 0;
}

// Releases allocator storage and resets bookkeeping.
void free_global_allocator()
{
    if (allocator_base) {
        cudaFree(allocator_base);
        allocator_base = nullptr;
    }
}

// Allocates aligned memory from the global allocator.
__device__ void* gmalloc(unsigned long long int size)
{
    size                       = (size + 7) & ~7ULL;
    unsigned long long int ret = atomicAdd(&allocator_offset, size);

    return allocator_base + ret;
}

// No-op placeholder for freeing memory from the allocator.
__device__ void gfree(void* /*ptr*/)
{
}

} // namespace atomic
} // namespace allocators