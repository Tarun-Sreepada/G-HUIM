#include "file/parse.cuh"

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

// Counts line breaks and delimiter occurrences across the file buffer.
__global__ void count_lines_items(const char* buf, size_t n, char delim, uint32_t* lineCount, uint32_t* itemCount)
{
    size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    unsigned int local_line = 0;
    unsigned int local_item = 0;

    for (size_t i = idx; i < n; i += stride) {
        char c = __ldg(&buf[i]);
        local_item += (c == delim);
        local_line += (c == '\n');
    }

    unsigned int lane = threadIdx.x & 31;
    for (int offset = 16; offset > 0; offset /= 2) {
        local_item += __shfl_down_sync(0xffffffff, local_item, offset);
        local_line += __shfl_down_sync(0xffffffff, local_line, offset);
    }

    if (lane == 0) {
        atomicAdd(itemCount, local_item);
        atomicAdd(lineCount, local_line);
    }
}

// Records the position of each newline in the file buffer.
__global__ void find_newline_positions(const char* buf, size_t n, uint32_t* positions)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && buf[idx] == '\n') {
        uint32_t pos       = atomicAdd(positions, 1u);
        positions[pos + 1] = static_cast<uint32_t>(idx);
    }
}

// Computes item counts and the maximum line length per input line.
__global__ void get_items_per_line(const char* buf, const uint32_t* positions, uint32_t* itemsPerLine, char delim,
                                   uint32_t* max_diff)
{
    uint32_t            line_id = blockIdx.x;
    __shared__ uint32_t start;
    uint32_t            end = positions[line_id + 1];

    __shared__ uint32_t colon_pos;
    __shared__ uint32_t count_items;

    if (threadIdx.x == 0) {
        start = positions[line_id];
        atomicMax(max_diff, end - start);
        if (line_id != 0) // skip \n
            start += 1;
        colon_pos   = end;
        count_items = 0;
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x + start; i < end; i += blockDim.x)
        if (buf[i] == ':')
            atomicMin(&colon_pos, i);

    __syncthreads();

    for (uint32_t i = threadIdx.x + start; i < colon_pos; i += blockDim.x)
        if (buf[i] == delim)
            atomicAdd(&count_items, 1u);

    __syncthreads();

    if (threadIdx.x == 0) {
        itemsPerLine[line_id] = count_items + 1;
    }
}

// Converts a substring to an unsigned integer.
__device__ uint32_t my_atou(const char* str, int len)
{
    uint32_t result = 0;
    int      i      = 0;

    while (i < len && str[i] == ' ') {
        i++;
    }

    for (; i < len; ++i) {
        if (str[i] < '0' || str[i] > '9') {
            break;
        }
        result = result * 10 + (str[i] - '0');
    }

    return result;
}
// Parses transactions and utilities into GPU-resident buffers.
__global__ void parse_file(const char* buf, const uint32_t* line_positions, const uint32_t* items_per_line,
                           Database* db, uint32_t* line_count, uint32_t* item_count, uint32_t* max_item, char delim)
{
    int line_id   = blockIdx.x;
    int thread_id = threadIdx.x;

    __shared__ uint32_t    start, end, shared_end;
    __shared__ Transaction trans;
    extern __shared__ char shared_buf[];
    __shared__ uint32_t    colon_pos[3];

    if (threadIdx.x == 0) {
        start = line_positions[line_id];
        if (line_id != 0) // Skip the preceding '\n' for all lines but the first
            start += 1;

        end = line_positions[line_id + 1];

        shared_end = end - start;

        colon_pos[0] = 0;
        colon_pos[1] = 0;
        colon_pos[2] = 0;
    }
    __syncthreads();

    for (int i = thread_id + start; i < end; i += blockDim.x) {
        shared_buf[i - start] = buf[i];
    }
    __syncthreads();

    for (int i = thread_id; i < shared_end; i += blockDim.x) {
        if (shared_buf[i] == ':') {
            uint32_t pos_idx = atomicAdd(&colon_pos[0], 1);
            if (pos_idx < 2) {
                colon_pos[pos_idx + 1] = i;
            }
        }
    }
    __syncthreads();

    if (thread_id == 0) {
        if (colon_pos[1] > colon_pos[2]) {
            uint32_t temp = colon_pos[1];
            colon_pos[1]  = colon_pos[2];
            colon_pos[2]  = temp;
        }

        uint32_t items_start      = 0;
        uint32_t utility_start    = colon_pos[1] + 1;
        uint32_t item_utils_start = colon_pos[2] + 1;

        trans.data    = &db->data[atomicAdd(&db->num_items, items_per_line[line_id])];
        trans.length  = items_per_line[line_id];
        trans.utility = my_atou(shared_buf + utility_start, item_utils_start - utility_start - 1);

        uint32_t item_idx       = 0;
        uint32_t prev           = items_start;
        uint32_t local_max_item = 0;

        for (int i = items_start; i < utility_start; i++) {
            if (shared_buf[i] == delim || shared_buf[i] == ':') {
                uint32_t item            = my_atou(shared_buf + prev, i - prev);
                local_max_item           = max(local_max_item, item);
                trans.data[item_idx].key = item;
                item_idx++;
                prev = i + 1;
            }
        }
        atomicMax(max_item, local_max_item);

        item_idx = 0;
        prev     = item_utils_start;

        for (int i = item_utils_start; i < shared_end; i++) {
            if (shared_buf[i] == delim) {
                trans.data[item_idx].util = my_atou(shared_buf + prev, i - prev);
                item_idx++;
                prev = i + 1;
            }
        }
        if (prev < shared_end) {
            trans.data[item_idx].util = my_atou(shared_buf + prev, shared_end - prev);
        }

        db->transactions[line_id] = trans;
    }
}

// Returns the size of a file in bytes.
uint64_t get_file_size(const std::string& filename)
{
    struct stat stat_buf;
    int         rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

// Parses a high-utility itemset dataset on the GPU.
Parsed gpuParse(const std::string& filename, const char separator)
{
    Parsed pf;

    uint64_t file_size = get_file_size(filename);

    if (file_size <= 0) {
        std::cerr << "File is empty or does not exist.\n";
        pf.db       = nullptr;
        pf.max_item = 0;
        return pf;
    }
    std::cout << "File size: " << file_size << " bytes\n";

    char* buf;
    cudaMallocManaged(&buf, file_size);

    int fd = open(filename.c_str(), O_RDONLY);
    if (fd < 0) {
        cudaFree(buf);
        return pf;
    }
    int ret = read(fd, buf, file_size);
    if (ret < 0 || static_cast<uint64_t>(ret) != file_size) {
        cudaFree(buf);
        close(fd);
        return pf;
    }
    close(fd);

    cudaMemAdvise(buf, file_size, cudaMemAdviseSetReadMostly, 0);
    cudaMemPrefetchAsync(buf, file_size, 0);
    cudaDeviceSynchronize();

    uint32_t* line_count;
    uint32_t* item_count;
    CUDA_CHECK(cudaMallocManaged(&line_count, sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocManaged(&item_count, sizeof(uint32_t)));

    *line_count = 0;
    *item_count = 0;

    dim3 blockSize(1024);
    dim3 gridSize((file_size + blockSize.x - 1) / blockSize.x);

    count_lines_items<<<gridSize, blockSize>>>(buf, file_size, separator, line_count, item_count);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (file_size > 0 && buf[file_size - 1] != '\n') {
        (*line_count)++;
    }

    if (*line_count == 0) {
        cudaFree(buf);
        cudaFree(line_count);
        cudaFree(item_count);
        pf.db       = nullptr;
        pf.max_item = 0;
        return pf;
    }

    uint32_t* line_positions;
    cudaMallocManaged(&line_positions, (*line_count + 1) * sizeof(uint32_t));

    line_positions[0] = 0;

    find_newline_positions<<<gridSize, blockSize>>>(buf, file_size, line_positions);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t num_newlines_found = line_positions[0];

    line_positions[0] = 0;

    thrust::device_ptr<uint32_t> th_line_pos(line_positions);
    thrust::sort(thrust::device, th_line_pos + 1, th_line_pos + 1 + num_newlines_found);

    line_positions[*line_count] = static_cast<uint32_t>(file_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ... The rest of the function continues ...

    uint32_t* max_diff;
    cudaMallocManaged(&max_diff, sizeof(uint32_t));
    *max_diff = 0;

    uint32_t* items_per_line;
    cudaMallocManaged(&items_per_line, *line_count * sizeof(uint32_t));
    cudaMemset(items_per_line, 0, *line_count * sizeof(uint32_t));

    gridSize = (*line_count);

    get_items_per_line<<<gridSize, blockSize>>>(buf, line_positions, items_per_line, separator, max_diff);
    CUDA_CHECK(cudaDeviceSynchronize());
    *max_diff += 1;

    Database* db;
    cudaMallocManaged(&db, sizeof(Database));
    cudaMallocManaged(&db->transactions, (*line_count) * sizeof(Transaction));
    cudaMallocManaged(&db->data, (*item_count) * sizeof(Item));
    db->num_transactions = *line_count;
    db->num_items        = 0;

    uint32_t* max_item;
    cudaMallocManaged(&max_item, sizeof(uint32_t));
    *max_item = 0;

    parse_file<<<gridSize, 32, *max_diff>>>(buf, line_positions, items_per_line, db, line_count, item_count, max_item,
                                            separator);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(buf);
    cudaFree(line_positions);
    cudaFree(items_per_line);
    cudaFree(line_count);
    cudaFree(item_count);
    cudaFree(max_diff);

    pf.db       = db;
    pf.max_item = *max_item + 2;

    return pf;
}