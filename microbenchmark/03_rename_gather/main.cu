// 03: cost of the rename[] indirection (random 4-byte gather) as the table grows.
// Table sizes 4 KB .. 64 MB cross the L1 / L2 / DRAM boundaries; the shared-memory variant copies the table
// into each block first (only possible up to 128 KB).
//   random   : every lookup index is independent
//   clustered: runs of 64 nearby ascending indices, like keys of a sorted transaction

#include "bench.cuh"

constexpr size_t kLookups   = 1u << 25;
constexpr int    kMaxLogT   = 24;
constexpr int    kMaxSmemLogT = 15; // 128 KB table

__global__ void make_keys(uint32_t* keys, size_t n, uint32_t mask, int pattern)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint32_t idx = static_cast<uint32_t>(i);
        keys[i] = pattern == 0 ? (hash32(idx) & mask) : ((hash32(idx >> 6) + (idx & 63u) * 7u) & mask);
    }
}

template <bool kShared>
__global__ void gather_kernel(const uint32_t* __restrict__ keys, const uint32_t* __restrict__ table, uint32_t t_words,
                              size_t n, u64* total)
{
    extern __shared__ uint32_t s_table[];
    if (kShared) {
        for (uint32_t i = threadIdx.x; i < t_words; i += blockDim.x)
            s_table[i] = table[i];
        __syncthreads();
    }
    u64 acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint32_t k = keys[i];
        acc += kShared ? s_table[k] : __ldg(&table[k]);
    }
    atomicAdd(total, acc);
}

int main()
{
    print_device();
    DeviceBuffer<uint32_t> keys(kLookups);
    DeviceBuffer<uint32_t> table((size_t)1 << kMaxLogT);
    DeviceBuffer<u64>      total(1);
    fill_hash_u32(table.get(), table.size(), 0x1234u);

    CUDA_CHECK(cudaFuncSetAttribute(gather_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 128 * 1024));
    const int blocks = sm_count() * 4;

    std::printf("pattern,variant,table_bytes,ms,glookups_per_s,ok\n");
    for (int pattern = 0; pattern < 2; ++pattern) {
        for (int log_t = 10; log_t <= kMaxLogT; ++log_t) {
            const uint32_t t_words = 1u << log_t;
            make_keys<<<1024, 256>>>(keys.get(), kLookups, t_words - 1, pattern);
            CUDA_CHECK(cudaGetLastError());

            total.zero();
            const float g_ms = time_ms([&] {
                CUDA_CHECK(cudaMemsetAsync(total.get(), 0, sizeof(u64)));
                gather_kernel<false><<<blocks, 256>>>(keys.get(), table.get(), t_words, kLookups, total.get());
            });
            const u64 g_sum = read_scalar(total.get());
            std::printf("%s,global,%u,%.4f,%.3f,1\n", pattern == 0 ? "random" : "clustered", t_words * 4, g_ms,
                        kLookups / g_ms / 1e6);

            if (log_t <= kMaxSmemLogT) {
                const float s_ms = time_ms([&] {
                    CUDA_CHECK(cudaMemsetAsync(total.get(), 0, sizeof(u64)));
                    gather_kernel<true><<<blocks, 256, t_words * 4>>>(keys.get(), table.get(), t_words, kLookups,
                                                                       total.get());
                });
                std::printf("%s,shared,%u,%.4f,%.3f,%d\n", pattern == 0 ? "random" : "clustered", t_words * 4, s_ms,
                            kLookups / s_ms / 1e6, read_scalar(total.get()) == g_sum);
            }
        }
    }
    return 0;
}
