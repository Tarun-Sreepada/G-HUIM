// 02: ways to find an item in one sorted transaction, one warp per transaction.
//   thread_binary : each lane runs its own binary search for a different query
//   thread_linear : each lane scans linearly for its own query
//   warp_ballot   : the warp walks the array 32 keys at a time and ballots on key >= item
//   warp_two_level: binary search over 32-key chunks, then one ballot on equality inside the chunk
// Sweep: transaction length L and queries per transaction Q. Q < 32 leaves lanes idle in the thread_* methods.

#include "bench.cuh"

constexpr size_t kTotalKeys = 1u << 25;

__host__ __device__ inline uint32_t key_value(uint32_t tx, uint32_t k)
{
    return 2u * k + (hash32(tx * 2654435761u + k) & 1u); // strictly increasing
}

__global__ void init_kernel(uint32_t* keys, uint32_t n_tx, uint32_t len)
{
    const size_t n = (size_t)n_tx * len;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        keys[i] = key_value(static_cast<uint32_t>(i / len), static_cast<uint32_t>(i % len));
}

__device__ inline int bsearch_thread(const uint32_t* a, uint32_t len, uint32_t item)
{
    int low = 0, high = static_cast<int>(len) - 1;
    while (low <= high) {
        const int      mid = (low + high) >> 1;
        const uint32_t key = a[mid];
        if (key == item)
            return mid;
        if (key < item)
            low = mid + 1;
        else
            high = mid - 1;
    }
    return -1;
}

__device__ inline int linear_thread(const uint32_t* a, uint32_t len, uint32_t item)
{
    for (uint32_t k = 0; k < len; ++k) {
        const uint32_t key = a[k];
        if (key >= item)
            return key == item ? static_cast<int>(k) : -1;
    }
    return -1;
}

// Whole warp cooperates; item is warp-uniform.
__device__ inline int ballot_scan(const uint32_t* a, uint32_t len, uint32_t item, uint32_t lane)
{
    for (uint32_t base = 0; base < len; base += 32) {
        const uint32_t i = base + lane;
        const uint32_t v = i < len ? a[i] : 0xffffffffu;
        const unsigned m = __ballot_sync(0xffffffffu, v >= item);
        if (m) {
            const int      first = __ffs(m) - 1;
            const uint32_t key   = __shfl_sync(0xffffffffu, v, first);
            return key == item ? static_cast<int>(base) + first : -1;
        }
    }
    return -1;
}

__device__ inline int two_level(const uint32_t* a, uint32_t len, uint32_t item, uint32_t lane)
{
    int low = 0, high = static_cast<int>((len + 31) / 32) - 1;
    while (low < high) {
        const int mid = (low + high + 1) >> 1;
        if (a[mid * 32] <= item)
            low = mid;
        else
            high = mid - 1;
    }
    const uint32_t i = static_cast<uint32_t>(low) * 32 + lane;
    const unsigned m = __ballot_sync(0xffffffffu, i < len && a[i] == item);
    return m ? low * 32 + __ffs(m) - 1 : -1;
}

enum Method { kThreadBinary, kThreadLinear, kWarpBallot, kWarpTwoLevel };

template <Method M>
__global__ void search_kernel(const uint32_t* keys, uint32_t n_tx, uint32_t len, uint32_t queries, unsigned* bad)
{
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const uint32_t lane = threadIdx.x % 32;
    if (warp >= n_tx)
        return;
    const uint32_t* a = keys + (size_t)warp * len;

    if (M == kThreadBinary || M == kThreadLinear) {
        for (uint32_t q = lane; q < queries; q += 32) {
            const uint32_t pos  = hash32(warp * 977u + q) % len;
            const uint32_t item = key_value(warp, pos);
            const int      r    = M == kThreadBinary ? bsearch_thread(a, len, item) : linear_thread(a, len, item);
            if (r != static_cast<int>(pos))
                atomicAdd(bad, 1u);
        }
    } else {
        for (uint32_t q = 0; q < queries; ++q) {
            const uint32_t pos  = hash32(warp * 977u + q) % len;
            const uint32_t item = key_value(warp, pos);
            const int      r    = M == kWarpBallot ? ballot_scan(a, len, item, lane) : two_level(a, len, item, lane);
            if (lane == 0 && r != static_cast<int>(pos))
                atomicAdd(bad, 1u);
        }
    }
}

template <Method M>
void run(const char* name, const uint32_t* keys, unsigned* bad, uint32_t n_tx, uint32_t len, uint32_t queries)
{
    const uint32_t blocks = (n_tx * 32 + 127) / 128;
    CUDA_CHECK(cudaMemset(bad, 0, sizeof(unsigned)));
    const float ms = time_ms([&] { search_kernel<M><<<blocks, 128>>>(keys, n_tx, len, queries, bad); });
    std::printf("%s,%u,%u,%.4f,%.2f,%d\n", name, len, queries, ms, (double)n_tx * queries / ms / 1e3,
                read_scalar(bad) == 0);
}

int main()
{
    print_device();
    DeviceBuffer<uint32_t> keys(kTotalKeys);
    DeviceBuffer<unsigned> bad(1);

    std::printf("method,len,queries,ms,mqueries_per_s,ok\n");
    for (uint32_t len : {8u, 16u, 32u, 64u, 128u, 256u, 512u, 1024u, 2048u, 4096u}) {
        const uint32_t n_tx = static_cast<uint32_t>(std::min<size_t>(1u << 17, kTotalKeys / len));
        init_kernel<<<1024, 256>>>(keys.get(), n_tx, len);
        CUDA_CHECK(cudaGetLastError());
        for (uint32_t queries : {1u, 4u, 16u, 64u}) {
            run<kThreadBinary>("thread_binary", keys.get(), bad.get(), n_tx, len, queries);
            run<kThreadLinear>("thread_linear", keys.get(), bad.get(), n_tx, len, queries);
            run<kWarpBallot>("warp_ballot", keys.get(), bad.get(), n_tx, len, queries);
            run<kWarpTwoLevel>("warp_two_level", keys.get(), bad.get(), n_tx, len, queries);
        }
    }
    return 0;
}
