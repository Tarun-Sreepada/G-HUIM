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

    __shared__ int length;

    int bid = blockIdx.x;
    int g_tid = blockIdx.x * blockDim.x + threadIdx.x; // global thread id
    int l_tid = threadIdx.x;                           // local thread id

    // asm set num_lines and num_items to 0
    if (l_tid == 0)
    {
        num_lines = 0;
        num_items = 0;
        length = 0;
    }

    __syncthreads();

    // Load data into shared memory and count number of items and lines
    if (g_tid < fd->size)
    {
        buffer[l_tid] = fd->data[g_tid];
        atomicMax(&length, g_tid);
    }

    __syncthreads();

    // Count number of items and lines
    if (l_tid < length)
    {
        if (buffer[l_tid] == separator)
        {
            atomicAdd(&num_items, 1);
        }
        if (buffer[l_tid] == new_line)
        {
            atomicAdd(&num_lines, 1);
        }
    }

    __syncthreads();

    // Unique Count
    if (l_tid == 0)
    {
        // start from 0, go until we find either a separator or a newline or a colon, then we hash the key and update the hll
        int start = 0;
        int end = 0;
        for (int i = 0; i < length; i++)
        {
            if (buffer[i] == separator || buffer[i] == '\n' || buffer[i] == ':')
            {
                end = i;
                int key = str_to_uint(buffer, start, end);
                hll(buckets, key);
                start = i + 1;
            }
        }
        // do the last one
        int key = str_to_uint(buffer, start, length);
        hll(buckets, key);

        atomicAdd(&fd->num_lines, num_lines);
        atomicAdd(&fd->num_items, num_items);
    }

    __syncthreads();

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
    int fd = open(filename.c_str(), O_RDONLY | O_DIRECT);
    if (fd == -1)
    {
        std::cerr << "Error: Unable to open file '" << filename << "'\n";
        return nullptr;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1)
    {
        std::cerr << "Error: Unable to get file size\n";
        close(fd);
        return nullptr;
    }
    size_t file_size = sb.st_size;
    std::cout << "File size: " << file_size << " bytes\n";

    // Allocate FileBuffer structure on host (much lighter than cudaMallocManaged)
    FileBuffer *data;
    cudaMallocManaged(&data, sizeof(FileBuffer));
    memset(data, 0, sizeof(FileBuffer));
    data->size = file_size;
    CUDA_CHECK(cudaMalloc(&data->data, file_size));

    // Register GPU memory with cuFile
    CUFILE_CHECK(cuFileBufRegister(data->data, file_size, 0));

    // Open file with cuFile
    CUfileHandle_t cufileHandle;
    CUfileDescr_t cufileDescr = {};
    cufileDescr.handle.fd = fd;
    cufileDescr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUFILE_CHECK(cuFileHandleRegister(&cufileHandle, &cufileDescr));

    timer.recordPoint("File setup");

    // Measure transfer time using high resolution clock (or CUDA events if needed)
    ssize_t bytes_read = cuFileRead(cufileHandle, data->data, file_size, 0, 0);

    if (bytes_read != static_cast<ssize_t>(file_size))
    {
        std::cerr << "Error: cuFileRead failed, only " << bytes_read << " bytes read\n";
        cuFileHandleDeregister(cufileHandle);
        cuFileBufDeregister(data->data);
        close(fd);
        delete data;
        return nullptr;
    }

    timer.recordPoint("File Loaded");

    // get time difference between two points and calculate throughput
    double time_diff = timer.getTimeDifference("File setup", "File Loaded");
    double throughput = (file_size / (1024 * 1024)) / (time_diff / 1000);
    std::cout << "Throughput: " << throughput << " MB/s\n";

    // Cleanup: deregister and close file handle; leave driver open for reuse
    cuFileHandleDeregister(cufileHandle);
    cuFileBufDeregister(data->data);
    close(fd);

    return data;
}

void file_parse(FileBuffer *fd, const char separator, Timer &timer)
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
    cudaDeviceSynchronize();

    timer.recordPoint("Second Pass");

    // // set the first line pointer to the start of the file
    // cudaMemset(&fd->lines[0], 0, sizeof(uint64_t));

    // // use cub sort to sort the line pointers
    // size_t temp_storage_bytes = 0;
    // void *d_temp_storage = NULL;

    // // Determine temporary device storage requirements
    // cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, fd->lines, fd->lines, fd->line_arr_size);

    // // Allocate temporary storage
    // cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // // Sort the keys
    // cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, fd->lines, fd->lines, fd->num_lines + 1);

    // // Cleanup
    // cudaFree(d_temp_storage);

    // timer.recordPoint("Sort");

    return;
}