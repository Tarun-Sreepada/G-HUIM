// 08: scheduling skewed task sizes. Task i has size ~ 1 / rank^alpha (Zipf), normalised so the total work is
// constant (2^26 units) for every alpha; alpha = 0 is uniform. Strategies:
//   block_per_task   : one 128-thread block per task, hardware block scheduler
//   persistent_static: persistent blocks, task = bid + j * grid
//   queue            : persistent blocks pulling tasks from an atomic counter (shuffled order)
//   queue_lpt        : same, largest tasks first
//   warp_queue       : persistent warps pulling tasks (shuffled order)
//   warp_queue_lpt   : same, largest tasks first

#include <cmath>
#include <numeric>
#include <random>

#include "bench.cuh"

constexpr u64      kTotalUnits = 1ull << 26;
constexpr uint32_t kTasks      = 1u << 14;
constexpr int      kBlock      = 128;

__device__ inline uint32_t work_unit(uint32_t task, uint32_t i)
{
    uint32_t x = i ^ (task * 0x9E3779B9u);
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        x ^= x >> 15;
        x *= 0x2c1b3c6du;
        x ^= x >> 12;
    }
    return x;
}

__device__ inline u64 run_task_block(uint32_t task, uint32_t size)
{
    u64 acc = 0;
    for (uint32_t i = threadIdx.x; i < size; i += blockDim.x)
        acc += work_unit(task, i);
    return acc;
}

__global__ void block_per_task(const uint32_t* sizes, u64* total)
{
    const uint32_t task = blockIdx.x;
    atomicAdd(total, run_task_block(task, sizes[task]));
}

__global__ void persistent_static(const uint32_t* sizes, uint32_t n_tasks, u64* total)
{
    u64 acc = 0;
    for (uint32_t task = blockIdx.x; task < n_tasks; task += gridDim.x)
        acc += run_task_block(task, sizes[task]);
    atomicAdd(total, acc);
}

__global__ void queue_kernel(const uint32_t* sizes, const uint32_t* order, uint32_t n_tasks, uint32_t* counter, u64* total)
{
    __shared__ uint32_t cur;
    u64                 acc = 0;
    for (;;) {
        if (threadIdx.x == 0)
            cur = atomicAdd(counter, 1u);
        __syncthreads();
        const uint32_t slot = cur;
        __syncthreads(); // everyone has read cur before thread 0 overwrites it
        if (slot >= n_tasks)
            break;
        const uint32_t task = order[slot];
        acc += run_task_block(task, sizes[task]);
    }
    atomicAdd(total, acc);
}

__global__ void warp_queue_kernel(const uint32_t* sizes, const uint32_t* order, uint32_t n_tasks, uint32_t* counter, u64* total)
{
    const uint32_t lane = threadIdx.x % 32;
    u64            acc  = 0;
    for (;;) {
        uint32_t slot = 0;
        if (lane == 0)
            slot = atomicAdd(counter, 1u);
        slot = __shfl_sync(0xffffffffu, slot, 0);
        if (slot >= n_tasks)
            break;
        const uint32_t task = order[slot], size = sizes[task];
        for (uint32_t i = lane; i < size; i += 32)
            acc += work_unit(task, i);
    }
    atomicAdd(total, acc);
}

int main()
{
    print_device();
    const int persistent_blocks = sm_count() * 8;

    DeviceBuffer<uint32_t> d_sizes(kTasks), d_shuffled(kTasks), d_lpt(kTasks), d_counter(1);
    DeviceBuffer<u64>      d_total(1);

    std::printf("alpha,max_task_share,strategy,ms,gunits_per_s,ok\n");
    for (double alpha : {0.0, 0.5, 1.0, 1.5}) {
        std::vector<double> w(kTasks);
        double              wsum = 0;
        for (uint32_t r = 0; r < kTasks; ++r)
            wsum += (w[r] = std::pow(double(r + 1), -alpha));
        std::vector<uint32_t> sizes(kTasks);
        u64                   expected = 0;
        for (uint32_t r = 0; r < kTasks; ++r) {
            sizes[r] = std::max<uint32_t>(1, static_cast<uint32_t>(std::llround(kTotalUnits * w[r] / wsum)));
            expected += sizes[r];
        }
        const double max_share = double(sizes[0]) / double(expected);

        std::mt19937 rng(12345);
        std::shuffle(sizes.begin(), sizes.end(), rng);
        std::vector<uint32_t> shuffled_order(kTasks), lpt_order(kTasks);
        std::iota(shuffled_order.begin(), shuffled_order.end(), 0u);
        std::iota(lpt_order.begin(), lpt_order.end(), 0u);
        std::sort(lpt_order.begin(), lpt_order.end(), [&](uint32_t a, uint32_t b) { return sizes[a] > sizes[b]; });
        d_sizes.upload(sizes);
        d_shuffled.upload(shuffled_order);
        d_lpt.upload(lpt_order);

        u64  ref = 0; // checksum of the first strategy; all must agree
        auto report = [&](const char* name, float ms) {
            const u64 t = read_scalar(d_total.get());
            if (ref == 0)
                ref = t;
            std::printf("%.1f,%.4f,%s,%.4f,%.3f,%d\n", alpha, max_share, name, ms, expected / ms / 1e6, t == ref);
        };
        auto reset = [&] {
            CUDA_CHECK(cudaMemsetAsync(d_counter.get(), 0, sizeof(uint32_t)));
            CUDA_CHECK(cudaMemsetAsync(d_total.get(), 0, sizeof(u64)));
        };

        report("block_per_task", time_ms([&] {
                   reset();
                   block_per_task<<<kTasks, kBlock>>>(d_sizes.get(), d_total.get());
               }));
        report("persistent_static", time_ms([&] {
                   reset();
                   persistent_static<<<persistent_blocks, kBlock>>>(d_sizes.get(), kTasks, d_total.get());
               }));
        report("queue", time_ms([&] {
                   reset();
                   queue_kernel<<<persistent_blocks, kBlock>>>(d_sizes.get(), d_shuffled.get(), kTasks, d_counter.get(), d_total.get());
               }));
        report("queue_lpt", time_ms([&] {
                   reset();
                   queue_kernel<<<persistent_blocks, kBlock>>>(d_sizes.get(), d_lpt.get(), kTasks, d_counter.get(), d_total.get());
               }));
        report("warp_queue", time_ms([&] {
                   reset();
                   warp_queue_kernel<<<persistent_blocks, kBlock>>>(d_sizes.get(), d_shuffled.get(), kTasks, d_counter.get(), d_total.get());
               }));
        report("warp_queue_lpt", time_ms([&] {
                   reset();
                   warp_queue_kernel<<<persistent_blocks, kBlock>>>(d_sizes.get(), d_lpt.get(), kTasks, d_counter.get(), d_total.get());
               }));
    }
    return 0;
}
