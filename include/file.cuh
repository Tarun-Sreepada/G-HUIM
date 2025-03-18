#pragma once

#include <iostream>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cufile.h>
#include <memory>
#include <cub/cub.cuh>
#include "database.cuh"
#include "functions.cuh"
#include "timer.h"

#define HLL_PRECISION 14              // Precision (2^14 = 16384 buckets)
#define HLL_SIZE (1 << HLL_PRECISION) // Number of registers

#define INV_LOADFACTOR 2 // load factor inverse

#define file_ptr_additional 2 // For start and end of file

#define BLOCK_SIZE 1024

#define GPU_ID 0

#define new_line 0x0A

#define digits_u32 11 // 10 digits + 1 for null terminator

#define ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))

// Error checking macros
#define CUDA_CHECK(call)                                                                    \
    {                                                                                       \
        cudaError_t err = call;                                                             \
        if (err != cudaSuccess)                                                             \
        {                                                                                   \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err \
                      << " (" << cudaGetErrorString(err) << ")" << std::endl;               \
            exit(1);                                                                        \
        }                                                                                   \
    }

#define CUFILE_CHECK(call)                                                                        \
    {                                                                                             \
        CUfileError_t err = call;                                                                 \
        if (err.err != CU_FILE_SUCCESS)                                                           \
        {                                                                                         \
            std::cerr << "cuFile error at " << __FILE__ << ":" << __LINE__ << " code=" << err.err \
                      << std::endl;                                                               \
            exit(1);                                                                              \
        }                                                                                         \
    }

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

struct FileBuffer
{
    char *data;
    size_t size;

    uint64_t *lines;
    unsigned int line_arr_size;

    unsigned int num_lines;
    unsigned int num_items;
};

__device__ void hll(int *buckets, int key)
{
    int hash = murmur_hash3(&key, sizeof(int));
    int index = hash & (HLL_SIZE - 1);
    int value = __clz(hash << HLL_PRECISION | 1) + 1;
    atomicMax(&buckets[index], value);
}

// technically we also count number of items + utilities but cba to filter.
__global__ void initial_pass(FileBuffer *fd, const char separator, int *buckets)
{
    __shared__ unsigned int num_lines;
    __shared__ unsigned int num_items;

    __shared__ char buffer[BLOCK_SIZE];

    char potential[digits_u32];

    int g_tid = blockIdx.x * blockDim.x + threadIdx.x; // global thread id
    int l_tid = threadIdx.x;                           // local thread id

    __shared__ int col_index;

    // asm set num_lines and num_items to 0
    if (l_tid == 0)
    {
        num_lines = 0;
        num_items = 0;
        col_index = 0;
    }

    __syncthreads();

    // Load data into shared memory and count number of items and lines
    if (g_tid < fd->size)
    {
        buffer[l_tid] = fd->data[g_tid];
        if (buffer[l_tid] == separator)
        {
            atomicAdd(&num_items, 1);
        }
        if (buffer[l_tid] == new_line)
        {
            atomicAdd(&num_lines, 1);
        }
        if (buffer[l_tid] == ':')
        {
            atomicMax(&col_index, l_tid);
        }
    }

    __syncthreads();

    // Unique Count
    if (l_tid == 0)
    {
        atomicAdd(&fd->num_lines, num_lines);
        atomicAdd(&fd->num_items, num_items);
    }

    __syncthreads();

    // HLL
    if (l_tid > col_index) return;

    // if 0th thread or item at tid is separator or : or \n we hash
    if (l_tid == 0 || buffer[l_tid] == separator || buffer[l_tid] == '\n' || buffer[l_tid] == ':')
    {
        // load 11 characters into potential
        int end = 0;
        for (int i = 0; i < digits_u32; i++)
        {
            potential[i] = buffer[l_tid + i + 1]; // +1 to skip the separator
            if (potential[i] == separator || potential[i] == '\n' || potential[i] == ':')
            {
                end = i;
                break;
            }
        }

        // convert potential to int
        int key = str_to_uint(potential, 0, end);
        hll(buckets, key);
    }

    return;
}

__global__ void second_pass(FileBuffer *fd, const char separator)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < fd->size)
    {
        if (fd->data[tid] == '\n')
        {
            // atomicAdd(&fd->num_items, 1);
            unsigned long long int ret = atomicAdd((unsigned long long int *)&fd->lines[0], 1);
            fd->lines[ret + 1] = tid;
        }
    }

    return;
}

__global__ void parse(FileBuffer *fd, Database *db, const char separator, HashTable<Utilities> *initial_utils)
{
    // Use shared integer indices instead of pointers.
    __shared__ int twu_start_index;
    __shared__ int twu_end_index;
    __shared__ int num_items;

    int bid = blockIdx.x;
    // int bid = 5; // debug
    int tid = threadIdx.x;

    int line_start = fd->lines[bid];
    int line_end = fd->lines[bid + 1];
    int line_length = line_end - line_start;

    extern __shared__ char line[];

    if (line_start != 0)
    {
        line_start++;
        line_length--;
    }

    if (tid == 0)
    {
        num_items = 1;
        twu_start_index = line_length;
        twu_end_index = -1;
    }


    __syncthreads();

    // Load the line into shared memory.
    for (int i = tid; i < line_length; i += blockDim.x)
    {
        line[i] = fd->data[line_start + i];
        if (line[i] == ':')
        {
            atomicMin(&twu_start_index, i);
            atomicMax(&twu_end_index, i);
        }
    }

    __syncthreads();

    // Get number of items
    for (int i = tid; i < twu_start_index; i += blockDim.x)
    {
        if (line[i] == separator)
        {
            atomicAdd(&num_items, 1);
        }
    }

    __syncthreads();

    if (tid == 0)
    {
        int ret = atomicAdd(&db->numTransactions, 1);
        db->d_transactions[ret].length = num_items;

        unsigned int twu = str_to_uint(line, twu_start_index + 1, twu_end_index);

        int item_ret = atomicAdd(&db->numItems, num_items);
        db->d_transactions[ret].data = &db->d_data[item_ret];

        // start from 0 go all the way to twu start ptr
        int item_index = 0;
        int idx = 0;
        int key = 0;

        for (int i = 0; i < twu_start_index; i++)
        {
            if (line[i] == separator)
            {
                key = str_to_uint(line, idx, i);
                db->d_data[item_ret + item_index].key = key;

                item_index++;
                idx = i + 1;

                // add twu to hash table
                // printf("Inserting key: %d, twu: %d\n", key, twu);
                initial_utils->insert(key, twu, Utilities::UtilityType::LOCAL_UTILITY);
            }
        }
        // do the last item
        key = str_to_uint(line, idx, twu_start_index); // not including the colon
        db->d_data[item_ret + item_index].key = key;
        initial_utils->insert(key, twu, Utilities::UtilityType::LOCAL_UTILITY);

        // start from twu end ptr go all the way to end of line
        item_index = 0;
        idx = twu_end_index + 1; // skip the colon
        for (int i = twu_end_index + 2; i < line_length; i++)
        {
            if (line[i] == separator || i == ':')
            {
                db->d_data[item_ret + item_index].util = str_to_uint(line, idx, i);
                idx = i + 1;
                item_index++;
            }
        }
        // do the last item
        db->d_data[item_ret + item_index].util = str_to_uint(line, idx, line_length);
    }
}

__global__ void computeDifferences(uint64_t *lines, uint64_t *differences, size_t num_lines)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_lines - 1)
    {
        differences[idx] = lines[idx + 1] - lines[idx];
    }
}

__global__ void valid_utils(HashTable<Utilities> *initial_utils, uint32_t min_util)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < initial_utils->tableSize)
    {
        if (initial_utils->table[idx].key != Utilities::EMPTY_KEY)
        {
            if (initial_utils->table[idx].local_utility >= min_util)
            {
                initial_utils->table[idx].subtree_utility = 1;
            }
        }
    }
}

uint64_t getMaxDifference(uint64_t *d_lines, size_t num_lines)
{
    // Allocate device memory for differences
    uint64_t *d_differences;
    cudaMalloc(&d_differences, (num_lines - 1) * sizeof(uint64_t));

    // Compute differences in parallel
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (num_lines - 1 + threadsPerBlock - 1) / threadsPerBlock;
    computeDifferences<<<blocksPerGrid, threadsPerBlock>>>(d_lines, d_differences, num_lines);
    cudaDeviceSynchronize();

    // Use CUB to find the maximum
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    uint64_t *d_max_diff;
    cudaMalloc(&d_max_diff, sizeof(uint64_t));

    // Determine storage size
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_differences, d_max_diff, num_lines - 1);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Compute max
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_differences, d_max_diff, num_lines - 1);

    // Copy result back to host
    uint64_t max_diff;
    cudaMemcpy(&max_diff, d_max_diff, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_differences);
    cudaFree(d_max_diff);
    cudaFree(d_temp_storage);

    return max_diff;
}

// CPU function: Estimate Unique Count
// https://agkn.wordpress.com/2012/10/25/sketch-of-the-day-hyperloglog-cornerstone-of-a-big-data-infrastructure/
// http://content.research.neustar.biz/blog/hll.html

double hll_estimate(int *h_hll_registers)
{
    int m = HLL_SIZE;
    double alpha;

    // Determine alpha based on the number of registers (m)
    if (m == 16)
    {
        alpha = 0.673;
    }
    else if (m == 32)
    {
        alpha = 0.697;
    }
    else if (m == 64)
    {
        alpha = 0.709;
    }
    else
    {
        alpha = 0.7213 / (1 + 1.079 / m);
    }

    // Multiply by m^2 to get alpha * m^2
    double alphaMM = alpha * m * m;

    double sum = 0.0;
    int zero_count = 0;

    // Compute the harmonic sum and count the zero registers
    for (int i = 0; i < m; i++)
    {
        sum += pow(2.0, -h_hll_registers[i]);
        if (h_hll_registers[i] == 0)
        {
            zero_count++;
        }
    }

    // Raw HyperLogLog estimate (DV_est)
    double raw_estimate = alphaMM / sum;
    double DV = raw_estimate;

    // Small range correction: if raw estimate is less than 2.5*m
    if (raw_estimate < 2.5 * m)
    {
        if (zero_count > 0)
            DV = m * log((double)m / zero_count);
        // If zero_count == 0, we leave DV as raw_estimate.
    }
    // Large range correction: if raw estimate is very high
    else if (raw_estimate > (1.0 / 30.0) * pow(2.0, 32))
    {
        DV = -pow(2.0, 32) * log(1.0 - raw_estimate / pow(2.0, 32));
    }
    // Otherwise (intermediate range) we use the raw estimate

    return DV;
}

FileBuffer *load_file(const std::string &filename, Timer &timer)
{
    // 1) Open file with O_DIRECT
    int fd = open(filename.c_str(), O_RDONLY | O_DIRECT);
    if (fd == -1)
    {
        std::cerr << "Error: Unable to open file '" << filename << "'\n";
        return nullptr;
    }

    // 2) Get file size
    struct stat sb;
    if (fstat(fd, &sb) == -1)
    {
        std::cerr << "Error: Unable to get file size\n";
        close(fd);
        return nullptr;
    }
    size_t file_size = sb.st_size;
    std::cout << "File size: " << file_size << " bytes\n";

    // 3) Align the size to 4 KB boundaries for O_DIRECT
    size_t aligned_size = (file_size + 4095) & ~4095;

    // 4) Allocate the FileBuffer struct in managed memory
    FileBuffer *data = nullptr;
    CUDA_CHECK(cudaMallocManaged(&data, sizeof(FileBuffer)));
    std::memset(data, 0, sizeof(FileBuffer));
    data->size = file_size;

    // 5) Allocate device memory for the file contents
    CUDA_CHECK(cudaMalloc(&data->data, aligned_size));

    // 6) Register GPU memory with cuFile
    cuFileBufRegister(data->data, aligned_size, 0);

    // 7) Create and register a cuFile handle
    CUfileHandle_t cufileHandle;
    {
        CUfileDescr_t cufileDescr = {};
        cufileDescr.handle.fd = fd;
        cufileDescr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        CUFILE_CHECK(cuFileHandleRegister(&cufileHandle, &cufileDescr));
    }

    // Record timing
    timer.recordPoint("File setup");

    // 8) Perform GDS read
    ssize_t bytes_read = cuFileRead(cufileHandle, data->data, file_size,
                                    /*file_offset=*/0, /*devPtr_offset=*/0);
    if (bytes_read != static_cast<ssize_t>(file_size))
    {
        std::cerr << "Error: cuFileRead failed, only " << bytes_read << " bytes read\n";

        // Cleanup so we don't leak resources:
        cuFileHandleDeregister(cufileHandle);
        cuFileBufDeregister(data->data);
        cudaFree(data->data);
        cudaFree(data);
        close(fd);
        return nullptr;
    }

    // Record second point
    timer.recordPoint("File Loaded");

    // 9) Show throughput
    double time_diff = timer.getTimeDifference("File setup", "File Loaded");    // ms
    double throughput = (file_size / (1024.0 * 1024.0)) / (time_diff / 1000.0); // MB/s
    std::cout << "Throughput: " << throughput << " MB/s\n";

    // 10) Cleanup file handle registration and close
    cuFileHandleDeregister(cufileHandle);
    close(fd);

    cuFileBufDeregister(data->data);

    return data;
}

void file_parse(FileBuffer *fd, const char separator, const uint32_t min_util, Timer &timer)
{
    // get file roud up file size to the nearest multiple of 2
    size_t file_size = fd->size;
    size_t thread_count = file_size + (file_size % 2);

    size_t block_size = BLOCK_SIZE;
    size_t block_count = (thread_count + block_size - 1) / block_size;

#ifdef DEBUG
    std::cout << "Thread count: " << thread_count << "\n";
    std::cout << "Block size: " << block_size << "\n";
    std::cout << "Block count: " << block_count << "\n";
#endif

    int *buckets;
    cudaMallocManaged(&buckets, HLL_SIZE * sizeof(int));
    cudaMemset(buckets, 0, HLL_SIZE * sizeof(int));

    initial_pass<<<block_count, block_size>>>(fd, separator, buckets);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaDeviceSynchronize();

#ifdef DEBUG
    std::cout << "Number of items: " << fd->num_items << "\n";
    std::cout << "Number of lines: " << fd->num_lines << "\n";
#endif

    timer.recordPoint("Initial Pass");

    double estimate = hll_estimate(buckets);
    std::cout << "Estimated Unique Count: " << estimate << "\n";

    fd->line_arr_size = fd->num_lines + file_ptr_additional;

    cudaMalloc(&fd->lines, fd->line_arr_size * sizeof(uint64_t));
    cudaMemset(fd->lines, 0, fd->line_arr_size * sizeof(uint64_t));

    // set the last line pointer to the end of the file
    cudaMemcpy(&fd->lines[fd->line_arr_size - 1], &fd->size, sizeof(uint64_t), cudaMemcpyHostToDevice);

    // get line pointers and number of unique items
    second_pass<<<block_count, block_size>>>(fd, separator);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    timer.recordPoint("Second Pass");

    // set the first line pointer to the start of the file
    cudaMemset(&fd->lines[0], 0, sizeof(uint64_t));

    // use cub sort to sort the line pointers
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = NULL;

    // Determine temporary device storage requirements
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, fd->lines, fd->lines, fd->line_arr_size);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Sort the keys
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, fd->lines, fd->lines, fd->num_lines + 1);

    // Cleanup
    cudaFree(d_temp_storage);

    timer.recordPoint("Sort");

    uint32_t max_diff = getMaxDifference(fd->lines, fd->line_arr_size) + 1;
    std::cout << "Max Difference: " << max_diff << "\n";

    timer.recordPoint("Max Difference");

    Database *db;
    cudaMallocManaged(&db, sizeof(Database));
    cudaMemset(db, 0, sizeof(Database));

    cudaMalloc(&db->d_transactions, fd->num_lines * sizeof(Transaction));
    cudaMalloc(&db->d_data, fd->num_items * sizeof(Item));
    cudaMemset(db->d_data, 0, fd->num_items * sizeof(Item));
    cudaMemset(db->d_transactions, 0, fd->num_lines * sizeof(Transaction));

    std::cout << "Num items: " << db->numItems << "\n";
    std::cout << "Num transactions: " << db->numTransactions << "\n";

    // Utilities *initial_utils;
    uint64_t hash_set_size = estimate * INV_LOADFACTOR;

    // Initial utilities
    HashTable<Utilities> *initial_utils;
    cudaMallocManaged(&initial_utils, sizeof(HashTable<Utilities>));
    initial_utils->tableSize = hash_set_size;
    cudaMallocManaged(&initial_utils->table, hash_set_size * sizeof(Utilities));
    cudaMemset(initial_utils->table, 0, hash_set_size * sizeof(Utilities));

    parse<<<(fd->num_lines + 1), BLOCK_SIZE>>>(fd, db, separator, initial_utils);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    timer.recordPoint("Parse");

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

#ifdef DEBUG
    std::cout << "Number of Transactions: " << db->numTransactions << "\n";
    std::cout << "Number of Items: " << db->numItems << "\n";
    std::cout << "Table Size: " << initial_utils->tableSize << "\n";
    // initial_utils->print();
#endif

    valid_utils<<<(hash_set_size + 1023) / 1024, 1024>>>(initial_utils, min_util);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

// #ifdef DEBUG
//     std::cout << "Valid Utilities\n";
//     initial_utils->print();
// #endif

    initial_utils->sort();

#ifdef DEBUG
    std::cout << "Sorted Utilities\n";
    initial_utils->print();
#endif

    // next we cumsum the subtree utilities to get the unique ID for each item
    // we add this to a hashset

    // cuda free everything except db
    cudaFree(fd->data);
    cudaFree(fd->lines);
    cudaFree(fd);

    return;
}