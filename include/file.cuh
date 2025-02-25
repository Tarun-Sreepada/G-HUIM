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

#define HLL_PRECISION 14              // Precision (2^14 = 16384 buckets)
#define HLL_SIZE (1 << HLL_PRECISION) // Number of registers

#define INV_LOADFACTOR 2 // load factor inverse

#define file_ptr_additional 2 // For start and end of file

#define BLOCK_SIZE 1024

#define GPU_ID 0

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

// technically we also count number of items + utilities but cba to filter.
__global__ void initial_pass(FileBuffer *fd, const char separator)
{
    __shared__ unsigned int num_lines;
    __shared__ unsigned int num_items;

    int block_tid = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_tid == 0)
    {
        num_lines = 0;
        num_items = 0;
    }

    __syncthreads();

    if (tid < fd->size)
    {
        if (fd->data[tid] == separator)
        {
            atomicAdd(&num_items, 1);
        }
        if (fd->data[tid] == '\n')
        {
            atomicAdd(&num_lines, 1);
        }
    }

    __syncthreads();

    if (block_tid == 0)
    {
        atomicAdd(&fd->num_lines, num_lines);
        atomicAdd(&fd->num_items, num_items);
    }

    __syncthreads();

    return;
}

// // get line pointers and number of unique items
// second_pass<<<block_count, block_size>>>(fd, separator, buckets);
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

__device__ unsigned int str_to_uint(const char *str, const int start, const int end)
{
    unsigned int ret = 0;
    for (int i = start; i < end; i++)
    {
        ret = ret * 10 + (str[i] - '0');
    }
    return ret;
}

__device__ void hll(int *buckets, int key)
{
    int hash = murmur_hash3(&key, sizeof(int));
    int index = hash & (HLL_SIZE - 1);
    int value = __clz(hash << HLL_PRECISION | 1) + 1;
    atomicMax(&buckets[index], value);
}

__global__ void parse(FileBuffer *fd, Database *db, const char separator, int *buckets)
{
    __shared__ char line[BLOCK_SIZE];
    // Use shared integer indices instead of pointers.
    __shared__ int twu_start_index;
    __shared__ int twu_end_index;
    __shared__ int num_items;

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int line_start = fd->lines[bid];
    int line_end = fd->lines[bid + 1];
    int line_length = line_end - line_start;
    if (line_length > BLOCK_SIZE) printf("ERROR: Line length is greater than block size\n"); // lazy to fix/make dynamic. should  be possible by fixing this kernel code

    if (line_start != 0)
    {
        line_start++;
        line_length--;
    }

    char register_copy;

    // Only one thread initializes the indices.
    if (tid == 0)
    {
        twu_start_index = -1;
        twu_end_index = -1;
        num_items = 1;
    }
    __syncthreads();

    // Load the line into shared memory.
    if (tid < line_length)
    {
        line[tid] = fd->data[line_start + tid];
        register_copy = line[tid];
    }
    __syncthreads();

    // Look for the separator (':') in the line.
    if (tid < line_length && register_copy == ':')
    {
        // Use atomicCAS to set the start index if not already set.
        if (atomicCAS(&twu_start_index, -1, tid + 1) == -1)
        {
            // First occurrence: store the index right after the colon.
        }
        else
        {
            // Subsequent occurrence: store the end index (the position just before the colon).
            atomicCAS(&twu_end_index, -1, tid - 1);
        }
    }
    __syncthreads();

    if (tid == 0)
    {
        // swap if end index is before start
        if (twu_end_index < twu_start_index)
        {
            int temp = twu_start_index;
            twu_start_index = twu_end_index;
            twu_end_index = temp;
        }
    }

    __syncthreads();

    // if tid is behind the start index check if it is a separator
    if (tid < twu_start_index)
    {
        if (register_copy == separator)
        {
            atomicAdd(&num_items, 1);
        }
    }

    __syncthreads();

    if (tid == 0)
    {
        int ret = atomicAdd(&db->numTransactions, 1);
        db->d_transactions[ret].length = num_items;

        unsigned int twu = str_to_uint(line, twu_start_index, twu_end_index + 1);
        db->d_transactions[ret].utility = twu;

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
                hll(buckets, key);

                item_index++;
                idx = i + 1;
            }
        }
        // do the last item
        key = str_to_uint(line, idx, twu_start_index - 1); // not including the colon
        db->d_data[item_ret + item_index].key = key;

        // start from twu end ptr go all the way to end of line
        item_index = 0;
        idx = twu_end_index + 2; // skip the colon
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

// CPU function: Estimate Unique Count
double hll_estimate(int *h_hll_registers)
{
    double alphaMM = 0.7213 / (1 + 1.079 / HLL_SIZE) * HLL_SIZE * HLL_SIZE;
    double sum = 0.0;
    int zero_count = 0;

    // uint8_t h_hll_registers[HLL_SIZE];
    // cudaMemcpyFromSymbol(h_hll_registers, d_hll_registers, sizeof(h_hll_registers));

    for (int i = 0; i < HLL_SIZE; i++) {
        sum += pow(2.0, -h_hll_registers[i]);
        if (h_hll_registers[i] == 0) {
            zero_count++;
        }
    }

    double raw_estimate = alphaMM / sum;
    
    // Small range correction
    if (raw_estimate <= 2.5 * HLL_SIZE && zero_count > 0) {
        return HLL_SIZE * log((double)HLL_SIZE / zero_count);
    }
    
    return raw_estimate;
}

// Load file directly into GPU memory using cuFile
FileBuffer *load_file(const std::string &filename)
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

    // Allocate GPU memory
    FileBuffer *data;
    cudaMallocManaged(&data, sizeof(FileBuffer));
    data->size = file_size;
    CUDA_CHECK(cudaMalloc(&data->data, file_size));

    // Initialize cuFile Driver
    CUFILE_CHECK(cuFileDriverOpen());

    // Register GPU memory with cuFile
    CUFILE_CHECK(cuFileBufRegister(data->data, file_size, 0));

    // Open file with cuFile
    CUfileHandle_t cufileHandle;
    CUfileDescr_t cufileDescr = {};
    cufileDescr.handle.fd = fd;
    cufileDescr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUFILE_CHECK(cuFileHandleRegister(&cufileHandle, &cufileDescr));

    // Perform direct I/O transfer from NVMe to GPU
    auto start = std::chrono::high_resolution_clock::now();
    ssize_t bytes_read = cuFileRead(cufileHandle, data->data, file_size, 0, 0);
    if (bytes_read != static_cast<ssize_t>(file_size))
    {
        std::cerr << "Error: cuFileRead failed, only " << bytes_read << " bytes read\n";
        cuFileHandleDeregister(cufileHandle);
        cuFileBufDeregister(data->data);
        cuFileDriverClose();
        close(fd);
        return nullptr;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "File loaded in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    // Cleanup
    cuFileHandleDeregister(cufileHandle);
    cuFileBufDeregister(data->data);
    cuFileDriverClose();
    close(fd);

    return data;
}

//
// file_parse(fd, parser.get_separator("--separator"));
void file_parse(FileBuffer *fd, const char separator)
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

    initial_pass<<<block_count, block_size>>>(fd, separator);

    cudaDeviceSynchronize();

    std::cout << "Number of lines: " << fd->num_lines << "\n";
#ifdef DEBUG
    std::cout << "Number of items: " << fd->num_items << "\n";
#endif

    fd->line_arr_size = fd->num_lines + file_ptr_additional;

    cudaMalloc(&fd->lines, fd->line_arr_size * sizeof(uint64_t));
    cudaMemset(fd->lines, 0, fd->line_arr_size * sizeof(uint64_t));

    // set the last line pointer to the end of the file
    cudaMemcpy(&fd->lines[fd->line_arr_size - 1], &fd->size, sizeof(uint64_t), cudaMemcpyHostToDevice);



    // get line pointers and number of unique items
    second_pass<<<block_count, block_size>>>(fd, separator);
    cudaDeviceSynchronize();

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

#ifdef DEBUG
    // std::vector<uint64_t> h_lines(fd->num_lines + 2);
    // cudaMemcpy(h_lines.data(), fd->lines, (fd->num_lines + 2) * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < fd->num_lines + 1; i++)
    // {
    //     std::cout << "Line " << i << ": " << h_lines[i] << "-" << h_lines[i + 1] << "\n";
    // }
#endif

    Database *db;
    cudaMallocManaged(&db, sizeof(Database));
    cudaMemset(db, 0, sizeof(Database));

    cudaMalloc(&db->d_transactions, fd->num_lines * sizeof(Transaction));
    cudaMalloc(&db->d_data, fd->num_items * sizeof(Item));
    cudaMemset(db->d_data, 0, fd->num_items * sizeof(Item));
    cudaMemset(db->d_transactions, 0, fd->num_lines * sizeof(Transaction));


    // buckets for HLL
    int *buckets;
    cudaMallocManaged(&buckets, HLL_SIZE * sizeof(int));
    cudaMemset(buckets, 0, HLL_SIZE * sizeof(int));
    // memadvise we are going to use this in the kernel of GPU id 0
    cudaMemAdvise(buckets, HLL_SIZE * sizeof(int), cudaMemAdviseSetPreferredLocation, GPU_ID);

    parse<<<(fd->num_lines + 1), BLOCK_SIZE>>>(fd, db, separator, buckets);
    cudaDeviceSynchronize();


    // cuda free everything except db
    cudaFree(fd->data);
    cudaFree(fd->lines);
    cudaFree(fd);

    // calculate the number of unique items
    for (int i = 0; i < HLL_SIZE; i++)
    {
        if (buckets[i] != 0)
        std::cout << "Bucket " << i << ": " << buckets[i] << "\n";
    }

    std::cout << "Number of unique items: " << hll_estimate(buckets) << "\n";
        

    

    return;
}