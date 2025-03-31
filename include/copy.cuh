/**
 * @file initialize.cuh
 * @brief CUDA kernel and initializer for preparing GPU memory and copying input data.
 *
 * Declares the `copy_work` kernel and `initialize_copy` function used to transfer
 * transaction data and primary items from host to device memory.
 */

 #pragma once

 #include "data_types.cuh"
 #include "stack.cuh"
 #include "memory.cuh"
 #include "data_types.hpp"
 #include "build.hpp"
 #include "timer.hpp"
 #include <numeric>
 
 /**
  * @brief CUDA kernel to copy transactions and primary items to the GPU.
  *
  * @param work_stack Pointer to the work stack on device.
  * @param db Pointer to the database on device.
  * @param primary Array of primary items on device.
  * @param primary_size Number of primary items.
  * @param max_item Maximum item ID used (for bounds).
  */
 __global__ void copy_work(stack *work_stack, Database *db, uint32_t *primary, uint32_t primary_size, uint32_t max_item);
 
 /**
  * @brief Allocates and initializes GPU memory for transactions and primary items.
  *
  * Transfers data from the `built_file` structure to device memory and prepares it
  * for kernel execution. Also copies initial work stack state.
  *
  * @param bf The processed transaction data from host.
  * @param s Pointer to the stack object (host).
  * @return A tuple containing:
  *         - Database* pointer to the allocated device database.
  *         - uint32_t* pointer to the device array of primary items.
  */
 std::tuple<Database *, uint32_t *> initialize_copy(built_file &bf, stack *s, Timer &timer);
 