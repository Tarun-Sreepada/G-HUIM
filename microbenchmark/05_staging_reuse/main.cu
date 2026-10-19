// 05: when does staging a tile into shared memory beat reading it straight from global memory?
// Every block processes tiles of W words and re-reads each tile R times (reuse factor). Modes:
//   direct  : all R passes read global memory (served by L1 / L2 after the first pass)
//   staged  : plain loads into shared memory once, then R passes from shared memory
//   cp_async: same, using cp.async for the copy
// The crossover in R is the break-even reuse.

#include "bench.cuh"

constexpr size_t kTotalWords = 1u << 26; // 256 MB

enum Mode { kDirect, kStaged, kCpAsync };

template <Mode M>
__global__ void tile_kernel(const uint32_t* __restrict__ data, uint32_t tile_words, uint32_t n_tiles, int reuse,
                            uint32_t* out)
{
    extern __shared__ __align__(16) uint32_t s_tile[];
    uint32_t acc = 0;
    for (uint32_t t = blockIdx.x; t < n_tiles; t += gridDim.x) {
        const uint32_t* g = data + (size_t)t * tile_words;
        const uint32_t* src = g;
        if (M == kStaged) {
            for (uint32_t i = threadIdx.x; i < tile_words; i += blockDim.x)
                s_tile[i] = g[i];
            __syncthreads();
            src = s_tile;
        } else if (M == kCpAsync) {
            for (uint32_t i = threadIdx.x; i < tile_words / 4; i += blockDim.x)
                cp_async16(s_tile + 4 * i, g + 4 * i);
            cp_commit();
            cp_wait<0>();
            __syncthreads();
            src = s_tile;
        }
        for (int r = 0; r < reuse; ++r)
            for (uint32_t i = threadIdx.x; i < tile_words; i += blockDim.x) {
                const uint32_t j = (i + r * 257u) & (tile_words - 1);
                acc = acc * 31u + (src[j] ^ static_cast<uint32_t>(r));
            }
        if (M != kDirect)
            __syncthreads(); // tile is overwritten by the next iteration
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

int main()
{
    print_device();
    DeviceBuffer<uint32_t> data(kTotalWords);
    fill_hash_u32(data.get(), kTotalWords, 99u);
    const int              blocks = sm_count() * 4;
    DeviceBuffer<uint32_t> out((size_t)blocks * 256);

    std::printf("tile_bytes,reuse,mode,ms,gb_per_s,ok\n");
    for (uint32_t tile_words : {1024u, 4096u, 8192u}) {
        const uint32_t n_tiles = static_cast<uint32_t>(kTotalWords / tile_words);
        const size_t   smem    = (size_t)tile_words * 4;
        for (int reuse : {1, 2, 4, 8, 16}) {
            const float d_ms  = time_ms([&] { tile_kernel<kDirect><<<blocks, 256>>>(data.get(), tile_words, n_tiles, reuse, out.get()); });
            const u64   ref   = device_sum_u32(out.get(), out.size());
            std::printf("%u,%d,direct,%.4f,%.2f,1\n", tile_words * 4, reuse, d_ms, kTotalWords * 4.0 / d_ms / 1e6);

            const float s_ms = time_ms([&] { tile_kernel<kStaged><<<blocks, 256, smem>>>(data.get(), tile_words, n_tiles, reuse, out.get()); });
            const u64   s_sum = device_sum_u32(out.get(), out.size());
            std::printf("%u,%d,staged,%.4f,%.2f,%d\n", tile_words * 4, reuse, s_ms, kTotalWords * 4.0 / s_ms / 1e6, s_sum == ref);

            const float c_ms = time_ms([&] { tile_kernel<kCpAsync><<<blocks, 256, smem>>>(data.get(), tile_words, n_tiles, reuse, out.get()); });
            const u64   c_sum = device_sum_u32(out.get(), out.size());
            std::printf("%u,%d,cp_async,%.4f,%.2f,%d\n", tile_words * 4, reuse, c_ms, kTotalWords * 4.0 / c_ms / 1e6, c_sum == ref);
        }
    }
    return 0;
}
