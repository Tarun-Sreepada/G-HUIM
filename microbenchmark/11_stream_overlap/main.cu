// 11: copy / compute overlap with chunked cudaMemcpyAsync and a consumer kernel on 1..4 streams.
// A pinned 256 MB host buffer is sent to the GPU in chunks; each chunk is consumed by a kernel doing `rounds`
// of ALU work per word. Rows report copy-only, kernel-only and the combined pipeline, so the overlap quality
// (combined vs max(copy, kernel)) can be read off directly.

#include "bench.cuh"

constexpr size_t kTotalBytes = 256u << 20;

__global__ void consume_kernel(const uint32_t* buf, size_t words, int rounds, u64* total)
{
    u64 acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < words; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = buf[i];
        for (int r = 0; r < rounds; ++r)
            x = x * 1664525u + 1013904223u + (x >> 7);
        acc += x;
    }
    atomicAdd(total, acc);
}

int main()
{
    print_device();
    const size_t words = kTotalBytes / 4;

    uint32_t* host = nullptr;
    CUDA_CHECK(cudaMallocHost(&host, kTotalBytes));
    for (size_t i = 0; i < words; ++i)
        host[i] = static_cast<uint32_t>(i) * 2654435761u;

    DeviceBuffer<u64> total(1);
    const int         blocks = sm_count() * 8;

    std::printf("chunk_mb,streams,rounds,mode,ms,gb_per_s,ok\n");
    u64 ref[2] = {0, 0}; // checksum per rounds value; every chunk/stream config must agree
    for (size_t chunk_mb : {1, 4, 16, 64}) {
        const size_t chunk_bytes = chunk_mb << 20, chunk_words = chunk_bytes / 4;
        const size_t n_chunks = kTotalBytes / chunk_bytes;

        for (int n_streams : {1, 2, 3, 4}) {
            std::vector<cudaStream_t>        streams(n_streams);
            std::vector<uint32_t*>           dev(n_streams);
            for (int s = 0; s < n_streams; ++s) {
                CUDA_CHECK(cudaStreamCreate(&streams[s])); // blocking: ordered against the default stream
                CUDA_CHECK(cudaMalloc(&dev[s], chunk_bytes));
                CUDA_CHECK(cudaMemset(dev[s], 0, chunk_bytes));
            }

            auto pipeline = [&](bool copy, bool kernel, int rounds) {
                CUDA_CHECK(cudaMemsetAsync(total.get(), 0, sizeof(u64)));
                for (size_t c = 0; c < n_chunks; ++c) {
                    const int s = static_cast<int>(c % n_streams);
                    if (copy)
                        CUDA_CHECK(cudaMemcpyAsync(dev[s], host + c * chunk_words, chunk_bytes,
                                                   cudaMemcpyHostToDevice, streams[s]));
                    if (kernel)
                        consume_kernel<<<blocks, 256, 0, streams[s]>>>(dev[s], chunk_words, rounds, total.get());
                }
            };

            const float copy_ms = time_ms([&] { pipeline(true, false, 0); });
            std::printf("%zu,%d,0,copy,%.4f,%.2f,1\n", chunk_mb, n_streams, copy_ms, kTotalBytes / copy_ms / 1e6);

            for (int rounds : {16, 256}) {
                const float kernel_ms = time_ms([&] { pipeline(false, true, rounds); });
                std::printf("%zu,%d,%d,kernel,%.4f,%.2f,1\n", chunk_mb, n_streams, rounds, kernel_ms,
                            kTotalBytes / kernel_ms / 1e6);

                const float both_ms = time_ms([&] { pipeline(true, true, rounds); });
                const u64   t       = read_scalar(total.get());
                u64&        r       = ref[rounds == 16 ? 0 : 1];
                if (r == 0)
                    r = t;
                std::printf("%zu,%d,%d,both,%.4f,%.2f,%d\n", chunk_mb, n_streams, rounds, both_ms,
                            kTotalBytes / both_ms / 1e6, t == r);
            }

            CUDA_CHECK(cudaDeviceSynchronize());
            for (int s = 0; s < n_streams; ++s) {
                CUDA_CHECK(cudaStreamDestroy(streams[s]));
                CUDA_CHECK(cudaFree(dev[s]));
            }
        }
    }
    CUDA_CHECK(cudaFreeHost(host));
    return 0;
}
