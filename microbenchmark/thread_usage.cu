// sync_vs_blocksize.cu
#include <cstdio>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void compute_only(std::uint64_t total_work, int *dummy)
{
    // compute n times
    std::uint64_t local = 0;
    for (std::uint64_t i = threadIdx.x; i < total_work; i += blockDim.x)
        local += (i ^ (i >> 1));

    if (threadIdx.x == 0) dummy[0] = static_cast<int>(local); // keep compiler honest
}

__global__ void sync_only(int sync_iters, int *dummy)
{
    // sync n times
    extern __shared__ int scratch[];             // size = blockDim.x
    for (int i = 0; i < sync_iters; ++i) {
        scratch[threadIdx.x] = i;                // inflate barrier traffic
        __syncthreads();
    }
    if (threadIdx.x == 0) dummy[0] = scratch[0];
}


__global__ void mixed_2(std::uint64_t total_work, int sync_iters, int *dummy)
{
    // compute n times and sync n times

    extern __shared__ int scratch[];
    std::uint64_t local = 0;
    for (std::uint64_t i = threadIdx.x; i < total_work; i += blockDim.x)
        local += (i ^ (i >> 1));

    for (int i = 0; i < sync_iters; ++i) {
        scratch[threadIdx.x] = static_cast<int>(local);
        __syncthreads();
    }
    if (threadIdx.x == 0) dummy[0] = static_cast<int>(local);
}

__global__ void mixed(std::uint64_t total_work,
                      int           sync_iters,
                      int          *dummy)
{

    extern __shared__ int scratch[];
    std::uint64_t local = 0;

    const std::uint64_t chunk = (total_work + sync_iters - 1) / sync_iters;

    for (int iter = 0; iter < sync_iters; ++iter) {

        const std::uint64_t start = iter * chunk;
        const std::uint64_t stop  = min(total_work, start + chunk);

        for (std::uint64_t i = start + threadIdx.x; i < stop; i += blockDim.x)
        {
            local += (i ^ (i >> 1));   
        }

        scratch[threadIdx.x] = static_cast<int>(local);
        __syncthreads();        
    }

    if (threadIdx.x == 0) dummy[0] = static_cast<int>(local);
}


static float launch_and_time(const void *kernel,
                             dim3            block,
                             size_t          shmem,
                             void          **args)
{
    cudaEvent_t start{}, stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaLaunchKernel(kernel,               
                     dim3(1),             
                     block,               
                     args,                
                     shmem,               
                     0);             
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main()
{
    const std::vector<int> blocksizes{32, 64, 128, 256, 512, 1024};

    const std::uint64_t TOTAL_WORK = 1ull << 20; // keep this big enough
    const int           SYNC_ITERS = 1ull << 13;      // barrier loops

    int *d_dummy{};
    cudaMalloc(&d_dummy, sizeof(int));

    printf("block_size,compute_ms,sync_ms,mixed_ms\n");

    for (int bs : blocksizes) {
        dim3 block(bs);

        // ---- compute-only -------------------------------------------------
        std::uint64_t per_thread_work = TOTAL_WORK;
        void *args1[]{ &per_thread_work, &d_dummy };
        float t_compute = launch_and_time(
            reinterpret_cast<const void *>(compute_only),
            block, 0, args1);

        // ---- sync-only ----------------------------------------------------
        int sync_iters = SYNC_ITERS; // Create a non-const variable
        void *args2[]{ &sync_iters, &d_dummy };
        float t_sync = launch_and_time(
            reinterpret_cast<const void *>(sync_only),
            block, sizeof(int) * bs, args2);

        // ---- mixed --------------------------------------------------------
        int sync_iters_var = SYNC_ITERS; // Create a non-const variable
        void *args3[]{ &per_thread_work, &sync_iters_var, &d_dummy };
        float t_mixed = launch_and_time(
            reinterpret_cast<const void *>(mixed),
            block, sizeof(int) * bs, args3);

        printf("%d,%.6f,%.6f,%.6f\n", bs, t_compute, t_sync, t_mixed);
    }

    cudaFree(d_dummy);
    return 0;
}

// nvcc -O3 -arch=sm_90 thread_usage.cu -o thread_usage
// ./thread_usage > results.csv


// nvcc -O3 -arch=sm_90 thread_usage.cu -o thread_usage && rm results.csv &&  ./thread_usage >> results.csv && /export/home1/ltarun/miniforge3/envs/rapids-25.02/bin/python /export/home1/ltarun/cuDPEFIM/microbenchmark/plot.py