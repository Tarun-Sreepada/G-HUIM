#include "cuda/buffer.cuh"
#include "mine/dfs/kernels.cuh"
#include "mine/dfs/single_task.cuh"

namespace ghuim::dfs {
namespace {

__global__ void block_tasks(MineArgs args)
{
    const Routing routing{args.distribution != nullptr, args.distribution, args.singles};
    mine_block_task(*args.control, *args.current, *args.next, routing, args.minimum_utility);
}

__global__ void single_tasks(MineArgs args, uint32_t tasks)
{
    __shared__ Transaction projected[kSingleBlock * kSingleThreadTransactions];
    const uint32_t         task = blockIdx.x * blockDim.x + threadIdx.x;
    if (task >= tasks)
        return;
    Task current;
    if (!args.current->get(current, task))
        return;
    mine_single_task(*args.control, current, *args.next, &projected[threadIdx.x * kSingleThreadTransactions],
                     args.minimum_utility);
}

} // namespace

void launch_block_tasks(const MineArgs& args, uint32_t tasks, uint32_t threads)
{
    if (tasks == 0)
        return;
    block_tasks<<<tasks, threads>>>(args);
    check_launch("launch block-task kernel");
}

void launch_single_tasks(const MineArgs& args, uint32_t tasks)
{
    if (tasks == 0)
        return;
    single_tasks<<<(tasks + kSingleBlock - 1) / kSingleBlock, kSingleBlock>>>(args, tasks);
    check_launch("launch single-thread kernel");
}

} // namespace ghuim::dfs
