#include "memory/memory.cuh"

namespace allocators {
    namespace atomic {

        // Definitions
        __managed__ char *allocator_base = nullptr;
        __managed__ unsigned long long int allocator_offset = 0;
        __managed__ unsigned long long int allocator_total = 0;

        std::vector<unsigned long long> allocation_sizes;


        void init_global_allocator(unsigned long long int total_memory, bool managed)
        {
            allocator_total = total_memory;

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


        void record_memory_usage()
        {
            allocation_sizes.push_back(allocator_offset);
        }

        void reclaim_memory()
        {
            if (allocation_sizes.size() < 3)
            {
                // Not enough data to reclaim memory
                return;
            }

            // get the boundary
            uint64_t boundary = allocation_sizes[allocation_sizes.size() - 3];
            uint64_t front = allocator_total - allocator_offset;
            uint64_t back = boundary;

            if (front < back){
                allocator_offset = 0;
                printf("Reclaiming memory. New offset: %llu\n", allocator_offset);
                for (size_t i = 0; i < allocation_sizes.size(); ++i)
                    std::cout << "Allocation " << i << ": " << allocation_sizes[i] << " bytes\n";
                allocation_sizes.clear();
                cudaMemset(allocator_base, 0, boundary);
                printf("Memory reclaimed up to boundary: %llu bytes\n", boundary);
            }

            // debug 

        }

        void free_global_allocator()
        {
            if (allocator_base)
            {
                std::cout << "Memory Usage Summary:\n";
                // Note: Calculating used memory is more complex with wrapping.
                // This is a simplified view.
                std::cout << "  Total: " << (allocator_total / (1024 * 1024)) << " MB\n";

                cudaFree(allocator_base);
                allocator_base = nullptr;
            }
        }

        __device__ void* gmalloc(unsigned long long int size)
        {
            size = (size + 7) & ~7ULL;  // align to 8 bytes
            unsigned long long int ret = atomicAdd(&allocator_offset, size);
        
            return allocator_base + ret;
        }

        __device__ void gfree(void * /*ptr*/)
        {
            // No-op
        }

    } // namespace atomic
} // namespace allocators