// 09: accumulating utilities into C counters under contention.
//   atomic32 : atomicAdd on a 32-bit global counter per update
//   atomic64 : atomicAdd on a 64-bit global counter per update
//   warp_agg : __match_any_sync groups lanes hitting the same counter, __reduce_add_sync sums them, the leader
//              issues one atomic
//   smem     : per-block shared-memory counters flushed at the end (C <= 4096)
// Sweep: counter count C (contention) and index distribution (uniform vs skewed towards counter 0).

#include "bench.cuh"

constexpr size_t   kUpdates = 1u << 25;
constexpr uint32_t kBlocks  = 1024; // 1024 * 256 = 2^18 threads divides kUpdates: loops are warp-uniform
constexpr uint32_t kBlock   = 256;

__device__ inline uint32_t pick(uint32_t i, uint32_t C, int dist)
{
    const uint32_t r = hash32(i) % C;
    return dist == 0 ? r : static_cast<uint32_t>((u64)r * r / C);
}

__device__ inline uint32_t value_of(uint32_t i) { return (hash32(i ^ 0x5bd1e995u) & 15u) + 1u; }

__global__ void atomic32_kernel(uint32_t* cnt, uint32_t C, int dist)
{
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < kUpdates; i += stride)
        atomicAdd(&cnt[pick(i, C, dist)], value_of(i));
}

__global__ void atomic64_kernel(u64* cnt, uint32_t C, int dist)
{
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < kUpdates; i += stride)
        atomicAdd(&cnt[pick(i, C, dist)], (u64)value_of(i));
}

__global__ void warp_agg_kernel(uint32_t* cnt, uint32_t C, int dist)
{
    const uint32_t stride = gridDim.x * blockDim.x;
    const uint32_t lane   = threadIdx.x % 32;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < kUpdates; i += stride) {
        const uint32_t idx  = pick(i, C, dist);
        const unsigned mask = __match_any_sync(0xffffffffu, idx);
        const uint32_t sum  = __reduce_add_sync(mask, value_of(i));
        if (lane == static_cast<uint32_t>(__ffs(mask) - 1))
            atomicAdd(&cnt[idx], sum);
    }
}

__global__ void smem_kernel(uint32_t* cnt, uint32_t C, int dist)
{
    extern __shared__ uint32_t s_cnt[];
    for (uint32_t i = threadIdx.x; i < C; i += blockDim.x)
        s_cnt[i] = 0;
    __syncthreads();
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < kUpdates; i += stride)
        atomicAdd(&s_cnt[pick(i, C, dist)], value_of(i));
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < C; i += blockDim.x)
        if (s_cnt[i])
            atomicAdd(&cnt[i], s_cnt[i]);
}

template <typename T> __global__ void mismatch_kernel(const T* a, const uint32_t* ref, uint32_t n, unsigned* bad)
{
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        if (a[i] != static_cast<T>(ref[i]))
            atomicAdd(bad, 1u);
}

int main()
{
    print_device();
    constexpr uint32_t kMaxC = 1u << 20;
    DeviceBuffer<uint32_t> c32(kMaxC), ref(kMaxC);
    DeviceBuffer<u64>      c64(kMaxC);
    DeviceBuffer<unsigned> bad(1);

    std::printf("dist,counters,variant,ms,gupdates_per_s,ok\n");
    for (int dist = 0; dist < 2; ++dist) {
        for (uint32_t C : {1u, 16u, 256u, 4096u, 65536u, kMaxC}) {
            const char* dname = dist == 0 ? "uniform" : "skewed";
            auto        rate  = [&](float ms) { return kUpdates / ms / 1e6; };
            auto        check32 = [&] {
                CUDA_CHECK(cudaMemset(bad.get(), 0, sizeof(unsigned)));
                mismatch_kernel<uint32_t><<<256, 256>>>(c32.get(), ref.get(), C, bad.get());
                CUDA_CHECK(cudaGetLastError());
                return read_scalar(bad.get()) == 0;
            };

            const float a32 = time_ms([&] {
                CUDA_CHECK(cudaMemsetAsync(c32.get(), 0, C * sizeof(uint32_t)));
                atomic32_kernel<<<kBlocks, kBlock>>>(c32.get(), C, dist);
            });
            CUDA_CHECK(cudaMemcpy(ref.get(), c32.get(), C * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
            std::printf("%s,%u,atomic32,%.4f,%.3f,1\n", dname, C, a32, rate(a32));

            const float a64 = time_ms([&] {
                CUDA_CHECK(cudaMemsetAsync(c64.get(), 0, C * sizeof(u64)));
                atomic64_kernel<<<kBlocks, kBlock>>>(c64.get(), C, dist);
            });
            CUDA_CHECK(cudaMemset(bad.get(), 0, sizeof(unsigned)));
            mismatch_kernel<u64><<<256, 256>>>(c64.get(), ref.get(), C, bad.get());
            std::printf("%s,%u,atomic64,%.4f,%.3f,%d\n", dname, C, a64, rate(a64), read_scalar(bad.get()) == 0);

            const float wa = time_ms([&] {
                CUDA_CHECK(cudaMemsetAsync(c32.get(), 0, C * sizeof(uint32_t)));
                warp_agg_kernel<<<kBlocks, kBlock>>>(c32.get(), C, dist);
            });
            std::printf("%s,%u,warp_agg,%.4f,%.3f,%d\n", dname, C, wa, rate(wa), check32());

            if (C <= 4096) {
                const float sm = time_ms([&] {
                    CUDA_CHECK(cudaMemsetAsync(c32.get(), 0, C * sizeof(uint32_t)));
                    smem_kernel<<<kBlocks, kBlock, C * sizeof(uint32_t)>>>(c32.get(), C, dist);
                });
                std::printf("%s,%u,smem,%.4f,%.3f,%d\n", dname, C, sm, rate(sm), check32());
            }
        }
    }
    return 0;
}
