// 07: occupancy vs block size and dynamic shared memory for a latency-bound kernel.
// Every thread follows a chain of dependent random loads, so throughput depends on how many warps are resident
// to hide latency. The shared-memory request per block limits resident blocks; the CUDA occupancy API gives the
// theoretical value to compare against the measured time.

#include "bench.cuh"

constexpr uint32_t kLogTable = 24; // 64 MB table
constexpr uint32_t kHops     = 16;
constexpr uint32_t kThreads  = 1u << 22;
constexpr int      kMaxSmem  = 160 * 1024;

__global__ void chase_kernel(const uint32_t* __restrict__ table, uint32_t mask, uint32_t hops, uint32_t smem_words,
                             uint32_t* out)
{
    extern __shared__ uint32_t sm[];
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t       idx = hash32(gid) & mask;
    uint32_t       acc = 0;
    for (uint32_t h = 0; h < hops; ++h) {
        idx = table[idx] & mask; // table holds random words; masking keeps the chain in range
        acc ^= idx;
    }
    if (smem_words) { // keeps the allocation alive; smem_words >= blockDim for every nonzero size tested
        sm[threadIdx.x] = acc;
        __syncthreads();
        acc ^= sm[(threadIdx.x + 1) % blockDim.x];
    }
    out[gid] = acc;
}

int main()
{
    print_device();
    const uint32_t         mask = (1u << kLogTable) - 1;
    DeviceBuffer<uint32_t> table((size_t)1 << kLogTable);
    DeviceBuffer<uint32_t> out(kThreads);
    fill_hash_u32(table.get(), table.size(), 5u);

    CUDA_CHECK(cudaFuncSetAttribute(chase_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxSmem));

    std::printf("block,smem_bytes,blocks_per_sm,occupancy_pct,ms,gloads_per_s\n");
    for (int block : {64, 128, 256, 512, 1024}) {
        for (int smem : {0, 4096, 8192, 16384, 32768, 49152, 65536, 98304, 131072, 163840}) {
            int per_sm = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, chase_kernel, block, smem));
            if (per_sm == 0) {
                std::printf("%d,%d,0,0,nan,nan\n", block, smem);
                continue;
            }
            const uint32_t blocks = kThreads / block;
            const float    ms     = time_ms([&] {
                chase_kernel<<<blocks, block, smem>>>(table.get(), mask, kHops, smem / 4, out.get());
            });
            std::printf("%d,%d,%d,%.1f,%.4f,%.3f\n", block, smem, per_sm, 100.0 * per_sm * block / 2048.0, ms,
                        (double)kThreads * kHops / ms / 1e6);
        }
    }
    return 0;
}
