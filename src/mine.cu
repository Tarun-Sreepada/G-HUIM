#include "mine.cuh"


/************************************************************
 * Helper function to get the next work item from the stack.
 * Only thread 0 tries to pop; if no item is found, it leaves
 * outItem empty. Then all threads sync and check the result.
 ************************************************************/
__device__ bool fetchWorkItem(stack *work_stack, WorkItem &outItem, int thread_id)
{
    if (thread_id == 0)
    {
        outItem.clear();
        // Keep popping until we get a non-empty item or stack is empty
        while (outItem.is_empty() && work_stack->may_pop())
        {
            work_stack->pop(outItem);
        }
    }
    __syncthreads();
    // If outItem is empty here, there's no more work
    return !outItem.is_empty();
}

/************************************************************
 * Helper function to process the WorkItem.
 * Replace with your real logic.
 ************************************************************/
__device__ void processWorkItem(const WorkItem &item, int thread_id)
{
    // Example: do something with item
    // e.g., "process(curr, new)" from your snippet
    if (thread_id == 0)
    {
        printf("Processing item with primary=%u, utility=%u\n",
               item.primary, item.utility);
    }
    __syncthreads();
}

/************************************************************
 * Helper function to finish the task on the stack.
 * Only thread 0 calls 'finish_task'.
 ************************************************************/
__device__ void finishTask(stack *work_stack, int thread_id)
{
    if (thread_id == 0)
    {
        work_stack->finish_task();
    }
    __syncthreads();
}


/************************************************************
 * Example usage in your kernel
 ************************************************************/
__global__ void mine(stack *work_stack, uint32_t *patterns, uint32_t min_util)
{
    int thread_id = threadIdx.x;
    WorkItem curr_work_item;

    // While there is potentially another item to pop...
    while (work_stack->may_pop())
    {
        // Attempt to fetch an item
        bool gotItem = fetchWorkItem(work_stack, curr_work_item, thread_id);

        if (!gotItem)
        {
            // No more work available -> exit loop
            break;
        }

        // We do have a valid item now, so process it
        processWorkItem(curr_work_item, thread_id);

        // Indicate we're done with this item
        finishTask(work_stack, thread_id);
    }
}
