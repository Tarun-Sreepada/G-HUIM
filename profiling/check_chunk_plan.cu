#include "cuda/buffer.cuh"
#include "mine/bfs/kernels.cuh"

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
using namespace ghuim;
using namespace ghuim::bfs;
void check(const std::vector<uint64_t>& sizes, uint32_t max_tasks, uint64_t cap)
{
    std::vector<uint64_t> prefix(1, 0);
    for (auto n : sizes)
        prefix.push_back(prefix.back() + n);
    DeviceBuffer<uint64_t>  d(prefix.size());
    DeviceBuffer<ChunkPlan> plan(1);
    check_cuda(cudaMemcpy(d.get(), prefix.data(), prefix.size() * 8, cudaMemcpyHostToDevice), "upload");
    uint32_t begin = 0;
    do {
        launch_plan_chunks(d.get(), begin, sizes.size(), max_tasks, cap, plan.get(), nullptr);
        ChunkPlan p{};
        check_cuda(cudaMemcpy(&p, plan.get(), sizeof(p), cudaMemcpyDeviceToHost), "download");
        if (p.cuts[0] != begin)
            throw std::runtime_error("initial cut");
        for (uint32_t i = 0; i < p.count; ++i) {
            uint32_t end = begin;
            while (end < sizes.size() && end - begin < max_tasks && prefix[end + 1] - prefix[begin] <= cap)
                ++end;
            if (end == begin || p.cuts[i + 1] != end || p.bases[i] != prefix[begin])
                throw std::runtime_error("wrong cut/base");
            begin = end;
        }
        if (p.oversized) {
            if (begin >= sizes.size() || sizes[begin] <= cap)
                throw std::runtime_error("false overflow");
            return;
        }
        if (begin < sizes.size() && p.count != kChunkPlanPage)
            throw std::runtime_error("missing chunks");
    } while (begin < sizes.size());
}
int main()
{
    check({}, 10, 100);
    check({100}, 10, 100);
    check({101}, 10, 100);
    check({0, 0, 100, 0, 1, 99}, 3, 100);
    check({50, 50, 101}, 10, 100);
    check(std::vector<uint64_t>(2000, 1), 1, 1);
    std::mt19937 rng(42);
    for (int k = 0; k < 100; ++k) {
        std::vector<uint64_t> sizes(1 + rng() % 10000);
        for (auto& x : sizes)
            x = rng() % 101;
        check(sizes, 1 + rng() % 500, 100 + rng() % 10000);
    }
    std::cout << "106 chunk-plan cases passed, including multiple pages and oversized tasks\n";
}
