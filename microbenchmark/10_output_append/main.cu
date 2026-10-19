// 10: appending selected items to a compacted output buffer.
//   atomic_per_item: every selected thread does atomicAdd(counter, 1)
//   warp_aggregated: ballot + popc, one atomic per warp, lane offset from the ballot mask
//   block_scan     : warp ballots + shared-memory prefix over warps, one atomic per block iteration
// Sweep: density of selected items (1% .. 100%).

#include "bench.cuh"

constexpr uint32_t kItems  = 1u << 25;
constexpr uint32_t kBlocks = 1024; // 1024 * 256 divides kItems: loops are block-uniform
constexpr uint32_t kBlock  = 256;

__device__ inline bool selected(uint32_t i, uint32_t density_pct) { return hash32(i) % 100u < density_pct; }

__global__ void reference_kernel(uint32_t density_pct, u64* count, u64* sum)
{
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < kItems; i += gridDim.x * blockDim.x)
        if (selected(i, density_pct)) {
            atomicAdd(count, 1ull);
            atomicAdd(sum, (u64)i);
        }
}

__global__ void atomic_kernel(uint32_t* out, uint32_t* counter, uint32_t density_pct)
{
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < kItems; i += gridDim.x * blockDim.x)
        if (selected(i, density_pct))
            out[atomicAdd(counter, 1u)] = i;
}

__global__ void warp_kernel(uint32_t* out, uint32_t* counter, uint32_t density_pct)
{
    const uint32_t lane = threadIdx.x % 32;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < kItems; i += gridDim.x * blockDim.x) {
        const bool     flag = selected(i, density_pct);
        const unsigned m    = __ballot_sync(0xffffffffu, flag);
        if (m) {
            const int leader = __ffs(m) - 1;
            uint32_t  base   = 0;
            if (lane == static_cast<uint32_t>(leader))
                base = atomicAdd(counter, static_cast<uint32_t>(__popc(m)));
            base = __shfl_sync(0xffffffffu, base, leader);
            if (flag)
                out[base + __popc(m & ((1u << lane) - 1u))] = i;
        }
    }
}

__global__ void block_kernel(uint32_t* out, uint32_t* counter, uint32_t density_pct)
{
    __shared__ uint32_t warp_base[kBlock / 32];
    __shared__ uint32_t block_base;
    const uint32_t      lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < kItems; i += gridDim.x * blockDim.x) {
        const bool     flag = selected(i, density_pct);
        const unsigned m    = __ballot_sync(0xffffffffu, flag);
        if (lane == 0)
            warp_base[warp] = __popc(m);
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t total = 0;
            for (uint32_t w = 0; w < kBlock / 32; ++w) {
                const uint32_t c = warp_base[w];
                warp_base[w]     = total;
                total += c;
            }
            block_base = total ? atomicAdd(counter, total) : 0;
        }
        __syncthreads();
        if (flag)
            out[block_base + warp_base[warp] + __popc(m & ((1u << lane) - 1u))] = i;
        __syncthreads(); // warp_base / block_base are rewritten next iteration
    }
}

int main()
{
    print_device();
    DeviceBuffer<uint32_t> out(kItems), counter(1);
    DeviceBuffer<u64>      ref(2);

    std::printf("density_pct,variant,ms,gitems_per_s,ok\n");
    for (uint32_t density : {1u, 5u, 25u, 50u, 100u}) {
        ref.zero();
        reference_kernel<<<kBlocks, kBlock>>>(density, ref.get(), ref.get() + 1);
        CUDA_CHECK(cudaGetLastError());
        const std::vector<u64> expected = ref.download();

        auto report = [&](const char* name, float ms) {
            const u64 count = read_scalar(counter.get());
            const u64 sum   = count == expected[0] ? device_sum_u32(out.get(), count) : 0;
            std::printf("%u,%s,%.4f,%.3f,%d\n", density, name, ms, kItems / ms / 1e6,
                        count == expected[0] && sum == expected[1]);
        };
        auto reset = [&] { CUDA_CHECK(cudaMemsetAsync(counter.get(), 0, sizeof(uint32_t))); };

        report("atomic_per_item", time_ms([&] {
                   reset();
                   atomic_kernel<<<kBlocks, kBlock>>>(out.get(), counter.get(), density);
               }));
        report("warp_aggregated", time_ms([&] {
                   reset();
                   warp_kernel<<<kBlocks, kBlock>>>(out.get(), counter.get(), density);
               }));
        report("block_scan", time_ms([&] {
                   reset();
                   block_kernel<<<kBlocks, kBlock>>>(out.get(), counter.get(), density);
               }));
    }
    return 0;
}
