// 01: tiled vs contiguous key layout.
// S transactions are interleaved in groups: element k of transaction t lives at
//   (t / S) * S * len + k * S + (t % S)
// S = 1 is the contiguous layout, S = 32 is the warp-wide tile used by G-HUIM (lane = transaction).
// Thread-per-transaction scan and binary search (find_item_tiled) are timed for each S and length.

#include "bench.cuh"

constexpr size_t   kTotalKeys = 1u << 26; // 256 MB of keys regardless of length
constexpr uint32_t kQueries   = 8;

__host__ __device__ inline uint32_t key_value(uint32_t tx, uint32_t k)
{
    return 4u * k + (hash32(tx * 0x9E3779B1u ^ k) & 3u); // strictly increasing in k
}

template <uint32_t S> __device__ inline size_t base_slot(uint32_t tx, uint32_t len)
{
    return (size_t)(tx / S) * S * len + (tx % S);
}

template <uint32_t S> __global__ void init_kernel(uint32_t* keys, uint32_t n_tx, uint32_t len)
{
    const size_t n = (size_t)n_tx * len;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint32_t tx = static_cast<uint32_t>(i / len), k = static_cast<uint32_t>(i % len);
        keys[base_slot<S>(tx, len) + (size_t)k * S] = key_value(tx, k);
    }
}

template <uint32_t S> __global__ void scan_kernel(const uint32_t* keys, u64* out, uint32_t n_tx, uint32_t len)
{
    const uint32_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx >= n_tx)
        return;
    const uint32_t* col = keys + base_slot<S>(tx, len);
    u64             sum = 0;
    for (uint32_t k = 0; k < len; ++k)
        sum += col[(size_t)k * S];
    out[tx] = sum;
}

template <uint32_t S>
__global__ void search_kernel(const uint32_t* keys, u64* out, uint32_t n_tx, uint32_t len, unsigned* bad)
{
    const uint32_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx >= n_tx)
        return;
    const uint32_t* col = keys + base_slot<S>(tx, len);
    u64             sum = 0;
    for (uint32_t q = 0; q < kQueries; ++q) {
        const uint32_t pos  = hash32(tx * 31u + q) % len;
        const uint32_t item = key_value(tx, pos);
        int            low = 0, high = static_cast<int>(len) - 1, found = -1;
        while (low <= high) {
            const int      mid = (low + high) >> 1;
            const uint32_t key = col[(size_t)mid * S];
            if (key == item) {
                found = mid;
                break;
            }
            if (key < item)
                low = mid + 1;
            else
                high = mid - 1;
        }
        if (found != static_cast<int>(pos))
            atomicAdd(bad, 1u);
        sum += found;
    }
    out[tx] = sum;
}

template <uint32_t S>
void run_stride(uint32_t* keys, u64* out, unsigned* bad, uint32_t len, u64& scan_ref)
{
    const uint32_t n_tx   = static_cast<uint32_t>(kTotalKeys / len);
    const uint32_t blocks = (n_tx + 255) / 256;

    init_kernel<S><<<1024, 256>>>(keys, n_tx, len);
    CUDA_CHECK(cudaGetLastError());

    const float scan_ms = time_ms([&] { scan_kernel<S><<<blocks, 256>>>(keys, out, n_tx, len); });
    const u64   scan_sum = device_sum_u64(out, n_tx);
    if (scan_ref == 0)
        scan_ref = scan_sum;
    std::printf("%u,%u,scan,%.4f,%.2f,GB/s,%d\n", S, len, scan_ms, kTotalKeys * 4.0 / scan_ms / 1e6,
                scan_sum == scan_ref);

    CUDA_CHECK(cudaMemset(bad, 0, sizeof(unsigned)));
    const float search_ms = time_ms([&] { search_kernel<S><<<blocks, 256>>>(keys, out, n_tx, len, bad); });
    std::printf("%u,%u,search,%.4f,%.2f,Mqueries/s,%d\n", S, len, search_ms,
                (double)n_tx * kQueries / search_ms / 1e3, read_scalar(bad) == 0);
}

int main()
{
    print_device();
    DeviceBuffer<uint32_t> keys(kTotalKeys);
    DeviceBuffer<u64>      out(kTotalKeys / 8);
    DeviceBuffer<unsigned> bad(1);

    std::printf("stride,len,mode,ms,rate,unit,ok\n");
    for (uint32_t len : {8u, 32u, 128u, 512u, 2048u}) {
        u64 scan_ref = 0;
        run_stride<1>(keys.get(), out.get(), bad.get(), len, scan_ref);
        run_stride<8>(keys.get(), out.get(), bad.get(), len, scan_ref);
        run_stride<16>(keys.get(), out.get(), bad.get(), len, scan_ref);
        run_stride<32>(keys.get(), out.get(), bad.get(), len, scan_ref);
    }
    return 0;
}
