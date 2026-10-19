// 12: device malloc/free vs a bump-pointer arena, one allocation per thread.
//   malloc_free: in-kernel malloc, touch the block, free it
//   bump_thread: atomicAdd on a global cursor per thread (no free; the arena is reset between runs)
//   bump_warp  : one atomicAdd per warp, lanes take consecutive slices
// Sweep: allocation count and size. Configurations needing more than 512 MB are skipped.

#include "bench.cuh"

constexpr size_t kArenaBytes = 512u << 20;
constexpr size_t kHeapBytes  = 768u << 20;

// ctl[0] = allocation failures, ctl[1] = readback mismatches
__device__ inline void touch(uint32_t* p, uint32_t words, uint32_t id, unsigned* ctl)
{
    p[0]         = id;
    p[words - 1] = id;
    __threadfence_block();
    if (p[0] != id || p[words - 1] != id)
        atomicAdd(&ctl[1], 1u);
}

__global__ void malloc_free_kernel(uint32_t bytes, unsigned* ctl)
{
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    auto*          p  = static_cast<uint32_t*>(malloc(bytes));
    if (!p) {
        atomicAdd(&ctl[0], 1u);
        return;
    }
    touch(p, bytes / 4, id, ctl);
    free(p);
}

__global__ void bump_thread_kernel(char* arena, u64* cursor, uint32_t bytes, unsigned* ctl)
{
    const uint32_t id  = blockIdx.x * blockDim.x + threadIdx.x;
    const u64      off = atomicAdd(cursor, (u64)bytes);
    if (off + bytes > kArenaBytes) {
        atomicAdd(&ctl[0], 1u);
        return;
    }
    touch(reinterpret_cast<uint32_t*>(arena + off), bytes / 4, id, ctl);
}

__global__ void bump_warp_kernel(char* arena, u64* cursor, uint32_t bytes, unsigned* ctl)
{
    const uint32_t id   = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane = threadIdx.x % 32;
    u64            base = 0;
    if (lane == 0)
        base = atomicAdd(cursor, (u64)bytes * 32);
    base = __shfl_sync(0xffffffffu, base, 0);
    const u64 off = base + (u64)lane * bytes;
    if (off + bytes > kArenaBytes) {
        atomicAdd(&ctl[0], 1u);
        return;
    }
    touch(reinterpret_cast<uint32_t*>(arena + off), bytes / 4, id, ctl);
}

int main()
{
    print_device();
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, kHeapBytes)); // must precede the first malloc kernel

    DeviceBuffer<char>     arena(kArenaBytes);
    DeviceBuffer<u64>      cursor(1);
    DeviceBuffer<unsigned> ctl(2);

    std::printf("allocs,bytes,variant,ms,mallocs_per_us,ok\n");
    for (uint32_t count : {1u << 10, 1u << 14, 1u << 18, 1u << 20}) {
        for (uint32_t bytes : {64u, 256u, 1024u, 4096u}) {
            if ((size_t)count * bytes > kArenaBytes)
                continue;
            const uint32_t blocks = count / 256;
            auto           reset  = [&] {
                CUDA_CHECK(cudaMemsetAsync(cursor.get(), 0, sizeof(u64)));
                CUDA_CHECK(cudaMemsetAsync(ctl.get(), 0, 2 * sizeof(unsigned)));
            };
            auto clean = [&] {
                const std::vector<unsigned> c = ctl.download();
                return c[0] == 0 && c[1] == 0;
            };

            float ms = time_ms([&] {
                reset();
                malloc_free_kernel<<<blocks, 256>>>(bytes, ctl.get());
            });
            std::printf("%u,%u,malloc_free,%.4f,%.3f,%d\n", count, bytes, ms, count / ms / 1e3, clean());

            ms = time_ms([&] {
                reset();
                bump_thread_kernel<<<blocks, 256>>>(arena.get(), cursor.get(), bytes, ctl.get());
            });
            std::printf("%u,%u,bump_thread,%.4f,%.3f,%d\n", count, bytes, ms, count / ms / 1e3,
                        clean() && read_scalar(cursor.get()) == (u64)count * bytes);

            ms = time_ms([&] {
                reset();
                bump_warp_kernel<<<blocks, 256>>>(arena.get(), cursor.get(), bytes, ctl.get());
            });
            std::printf("%u,%u,bump_warp,%.4f,%.3f,%d\n", count, bytes, ms, count / ms / 1e3,
                        clean() && read_scalar(cursor.get()) == (u64)count * bytes);
        }
    }
    return 0;
}
