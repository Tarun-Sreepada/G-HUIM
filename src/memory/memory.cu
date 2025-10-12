#include "memory/memory.cuh"

namespace allocators {
namespace atomic {

// Definitions
__managed__ char*                  allocator_base   = nullptr;
__managed__ unsigned long long int allocator_offset = 0;
__managed__ unsigned long long int allocator_total  = 0;

std::vector<unsigned long long> allocation_sizes;

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

void free_global_allocator()
{
    if (allocator_base) {
        // std::cout << "Memory Usage Summary:\n";
        // Note: Calculating used memory is more complex with wrapping.
        // This is a simplified view.
        // std::cout << "  Total: " << (allocator_total / (1024 * 1024)) << " MB\n";

        cudaFree(allocator_base);
        allocator_base = nullptr;
    }
}

__device__ void* gmalloc(unsigned long long int size)
{
    size                       = (size + 7) & ~7ULL; // align to 8 bytes
    unsigned long long int ret = atomicAdd(&allocator_offset, size);

    return allocator_base + ret;
}

__device__ void gfree(void* /*ptr*/)
{
    // No-op
}

} // namespace atomic
} // namespace allocators