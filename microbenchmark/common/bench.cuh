#pragma once
// Shared helpers for the G-HUIM micro-benchmarks: error checks, timing, hashing, RAII buffers, and a few
// tiny utility kernels. Every benchmark prints CSV to stdout and device info to stderr.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

using u64 = unsigned long long;

#define CUDA_CHECK(call)                                                                                       \
    do {                                                                                                       \
        cudaError_t err_ = (call);                                                                             \
        if (err_ != cudaSuccess) {                                                                             \
            std::fprintf(stderr, "CUDA error: %s at %s:%d: %s\n", #call, __FILE__, __LINE__,                   \
                         cudaGetErrorString(err_));                                                            \
            std::exit(1);                                                                                      \
        }                                                                                                      \
    } while (0)

// ---- hashing --------------------------------------------------------------------------------------------

__host__ __device__ inline uint32_t hash32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__host__ __device__ inline u64 hash64(u64 x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// ---- device buffer --------------------------------------------------------------------------------------

template <typename T> class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) { CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T))); }
    ~DeviceBuffer() { cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T*     get() const { return ptr_; }
    size_t size() const { return count_; }

    void zero() { CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T))); }
    void upload(const std::vector<T>& v)
    {
        CUDA_CHECK(cudaMemcpy(ptr_, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const
    {
        std::vector<T> v(count_);
        CUDA_CHECK(cudaMemcpy(v.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }

private:
    T*     ptr_   = nullptr;
    size_t count_ = 0;
};

template <typename T> T read_scalar(const T* device_ptr)
{
    T v{};
    CUDA_CHECK(cudaMemcpy(&v, device_ptr, sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}

// ---- device info / timing -------------------------------------------------------------------------------

inline int sm_count()
{
    int dev = 0, n = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
    return n;
}

inline void print_device()
{
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    std::fprintf(stderr, "# device %d: %s, %d SMs, sm_%d%d\n", dev, p.name, p.multiProcessorCount, p.major, p.minor);
}

// Median kernel time in ms over `reps` runs after `warmup` runs. `launch` enqueues all work for one run.
// Events are recorded on the default stream, so work on blocking streams is covered as well.
template <typename F> float time_ms(F&& launch, int warmup = 3, int reps = 10)
{
    for (int i = 0; i < warmup; ++i)
        launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t a{}, b{};
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    std::vector<float> t(reps);
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaEventRecord(a, 0));
        launch();
        CUDA_CHECK(cudaEventRecord(b, 0));
        CUDA_CHECK(cudaEventSynchronize(b));
        CUDA_CHECK(cudaEventElapsedTime(&t[r], a, b));
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    std::sort(t.begin(), t.end());
    return t[reps / 2];
}

// ---- small utility kernels ------------------------------------------------------------------------------

__global__ void fill_hash_u32_kernel(uint32_t* v, size_t n, uint32_t seed)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        v[i] = hash32(static_cast<uint32_t>(i) ^ seed);
}

__global__ void sum_u32_kernel(const uint32_t* v, size_t n, u64* total)
{
    u64 acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        acc += v[i];
    atomicAdd(total, acc);
}

__global__ void sum_u64_kernel(const u64* v, size_t n, u64* total)
{
    u64 acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        acc += v[i];
    atomicAdd(total, acc);
}

inline void fill_hash_u32(uint32_t* d, size_t n, uint32_t seed = 0)
{
    fill_hash_u32_kernel<<<1024, 256>>>(d, n, seed);
    CUDA_CHECK(cudaGetLastError());
}

inline u64 device_sum_u32(const uint32_t* d, size_t n)
{
    DeviceBuffer<u64> total(1);
    total.zero();
    sum_u32_kernel<<<512, 256>>>(d, n, total.get());
    CUDA_CHECK(cudaGetLastError());
    return read_scalar(total.get());
}

inline u64 device_sum_u64(const u64* d, size_t n)
{
    DeviceBuffer<u64> total(1);
    total.zero();
    sum_u64_kernel<<<512, 256>>>(d, n, total.get());
    CUDA_CHECK(cudaGetLastError());
    return read_scalar(total.get());
}

// ---- cp.async (sm_80+) ----------------------------------------------------------------------------------

__device__ inline void cp_async16(void* smem_dst, const void* gmem_src)
{
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_src));
}

__device__ inline void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N> __device__ inline void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
