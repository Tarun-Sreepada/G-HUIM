#include <cstdio>
#include <cstdlib>
#include <iostream>

namespace allocators {
    namespace atomic {

        // Definitions
        __managed__ char *allocator_base = nullptr;
        __managed__ unsigned long long int allocator_offset = 0;
        __managed__ unsigned long long int allocator_total = 0;
        
        // This marks the end of the memory used by the previous iteration.
        __managed__ unsigned long long int current_reclaim_boundary = 0;
        // This will mark the end of the memory for the current iteration.
        __managed__ unsigned long long int next_reclaim_boundary = 0;

        __managed__ bool allocator_wrapped = false;

        void print_memory_map()
        {
            printf("--- GPU Memory Map ---\n");
            // All values converted to MB for readability
            unsigned long long int total_mb = allocator_total / (1024 * 1024);
            unsigned long long int current_b_mb = current_reclaim_boundary / (1024 * 1024);
            unsigned long long int next_b_mb = next_reclaim_boundary / (1024 * 1024);
            unsigned long long int offset_mb = allocator_offset / (1024 * 1024);

            // This logic handles both the normal (linear) and wrapped-around cases.
            // By sorting the points of interest, we can iterate through the segments correctly.
            std::vector<std::pair<unsigned long long int, std::string>> points;
            points.push_back({0, "Start"});
            points.push_back({current_reclaim_boundary, "Current Boundary"});
            points.push_back({next_reclaim_boundary, "Next Boundary"});
            points.push_back({allocator_offset, "Offset"});
            points.push_back({allocator_total, "End"});

            // Remove duplicate points and sort
            std::sort(points.begin(), points.end());
            points.erase(std::unique(points.begin(), points.end()), points.end());

            for (size_t i = 0; i < points.size() - 1; ++i)
            {
                unsigned long long int start = points[i].first;
                unsigned long long int end = points[i+1].first;

                if (start == end) continue; // Skip zero-sized segments

                // Determine the status of the segment between start and end
                std::string status = "UNKNOWN";
                if (!allocator_wrapped) { // Linear allocation
                    if (start >= 0 && end <= current_reclaim_boundary) status = "FREE (Reclaimable)";
                    else if (start >= current_reclaim_boundary && end <= next_reclaim_boundary) status = "USED (Prev Iter)";
                    else if (start >= next_reclaim_boundary && end <= allocator_offset) status = "USED (Curr Iter)";
                    else if (start >= allocator_offset && end <= allocator_total) status = "FREE (Unallocated)";
                } else { // Wrapped around
                    if (start >= 0 && end <= allocator_offset) status = "USED (Curr Iter) -- WRAPPED";
                    else if (start >= allocator_offset && end <= current_reclaim_boundary) status = "FREE (Reclaimable)";
                    else if (start >= current_reclaim_boundary && end <= next_reclaim_boundary) status = "USED (Prev Iter)";
                    else if (start >= next_reclaim_boundary && end <= allocator_total) status = "FREE (Reclaimable)";
                }
                
                unsigned long long start_mb = start / (1024*1024);
                unsigned long long end_mb = end / (1024*1024);
                printf("  [%llu -> %llu MB] (%llu MB): %s\n", start_mb, end_mb, end_mb - start_mb, status.c_str());
            }
            printf("----------------------\n");
        }

        void init_global_allocator(uint64_t total_memory, bool managed)
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
            current_reclaim_boundary = 0;
            next_reclaim_boundary = 0;
            allocator_wrapped = false;
        }

        void reclaim_memory()
        {
            // 1. Calculate state from the iteration that just finished
            unsigned long long int used_in_last_iteration;
            if (allocator_offset >= next_reclaim_boundary) {
                used_in_last_iteration = allocator_offset - next_reclaim_boundary;
            } else {
                used_in_last_iteration = (allocator_total - next_reclaim_boundary) + allocator_offset;
            }

            // 2. Advance the boundaries
            current_reclaim_boundary = next_reclaim_boundary;
            next_reclaim_boundary = allocator_offset;

            // 3. Define free-space regions
            unsigned long long int space_at_end = allocator_total - allocator_offset;
            unsigned long long int space_at_start = current_reclaim_boundary;

            // 4. Safer decision logic
            bool should_wrap = (space_at_start > space_at_end) && (space_at_start >= used_in_last_iteration);

            // 5. Execute decision and memset
            if (should_wrap)
            {
                if (space_at_start > 0) {
                    // FIX: Use the asynchronous version to avoid stalling the CPU.
                    cudaMemset(allocator_base, 0, space_at_start);
                }
                allocator_offset = 0;
                allocator_wrapped = true;
            }
            else
            {
                if (space_at_end > 0) {
                    cudaMemset(allocator_base + allocator_offset, 0, space_at_end);
                }
            }

            #ifdef DEBUG
                // This debug output can remain as is.
            #endif
        }

        void reset_allocator_wrapped_flag()
        {
            allocator_wrapped = false;
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
            // TODO: Add a check to ensure ret + size <= allocator_total if not wrapping,
            // or ret + size <= current_reclaim_boundary if it is wrapping.
            // This would require kernel-side knowledge of the boundaries.
            return allocator_base + ret;
        }

        __device__ void gfree(void * /*ptr*/)
        {
            // No-op
        }

    } // namespace atomic
} // namespace allocators