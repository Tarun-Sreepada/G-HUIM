#include "file/parse.cuh"

#define CUDA_CHECK(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

__global__ void count_lines_items(const char *buf,
                                  size_t n,
                                  char delim,
                                  uint32_t *lineCount,
                                  uint32_t *itemCount)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // 1) per-thread local counters
    unsigned int local_line = 0;
    unsigned int local_item = 0;

    // 2) scan your chunk
    for (size_t i = idx; i < n; i += stride)
    {
        char c = __ldg(&buf[i]);
        local_item += (c == delim);
        local_line += (c == '\n');
    }

    // 3) warp-level reduction via shuffle
    unsigned int lane = threadIdx.x & 31;
    for (int offset = 16; offset > 0; offset /= 2)
    {
        local_item += __shfl_down_sync(0xffffffff, local_item, offset);
        local_line += __shfl_down_sync(0xffffffff, local_line, offset);
    }

    // 4) warp-leader does the atomic
    if (lane == 0)
    {
        atomicAdd(itemCount, local_item);
        atomicAdd(lineCount, local_line);
    }
}

__global__ void find_newline_positions(const char *buf, size_t n,
                                       uint32_t *positions)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && buf[idx] == '\n')
    {
        uint32_t pos = atomicAdd(positions, 1u);
        positions[pos + 1] = static_cast<uint32_t>(idx);
    }
}

__global__ void get_items_per_line(const char *buf,
                                   const uint32_t *positions,
                                   uint32_t *itemsPerLine,
                                   char delim, uint32_t *max_diff)
{
    uint32_t line_id = blockIdx.x;
    __shared__ uint32_t start;
    uint32_t end = positions[line_id + 1];

    __shared__ uint32_t colon_pos;
    __shared__ uint32_t count_items;

    if (threadIdx.x == 0)
    {
        start = positions[line_id];
        atomicMax(max_diff, end - start);
        if (line_id != 0) // skip \n
            start += 1;
        colon_pos = end;
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

    if (threadIdx.x == 0)
    {
        itemsPerLine[line_id] = count_items + 1;
    }
}

__device__ uint32_t my_atou(const char *str, int len)
{
    uint32_t result = 0;
    int i = 0;

    // Skip leading spaces
    while (i < len && str[i] == ' ')
    {
        i++;
    }

    // Parse digits
    for (; i < len; ++i)
    {
        if (str[i] < '0' || str[i] > '9')
        {
            break; // stop if non-digit
        }
        result = result * 10 + (str[i] - '0');
    }

    return result;
}

__global__ void parse_file(const char *buf,
                           const uint32_t *line_positions,
                           const uint32_t *items_per_line,
                           Database *db,
                           uint32_t *line_count,
                           uint32_t *item_count,
                           uint32_t *max_item,
                           char delim)
{
    int line_id = blockIdx.x;
    int thread_id = threadIdx.x;

    __shared__ uint32_t start;
    __shared__ uint32_t end;
    __shared__ uint32_t shared_start;
    __shared__ uint32_t shared_end;

    __shared__ Transaction trans;

    extern __shared__ char shared_buf[];

    __shared__ uint32_t colon_pos[3];

    if (threadIdx.x == 0)
    {
        start = line_positions[line_id];
        if (line_id != 0) // skip \n
            start += 1;
        end = line_positions[line_id + 1]; // skip \n
        if (line_id != (*line_count))
            end -= 1;
        colon_pos[0] = 0;
        colon_pos[1] = 0;
        colon_pos[2] = 0;
        shared_start = 0;
        shared_end = end - start;
    }
    __syncthreads();

    // load line into shared memory
    for (int i = thread_id + start; i <= end; i += blockDim.x)
    {
        shared_buf[i - start] = buf[i];
    }
    __syncthreads();

    // find colon positions
    for (int i = shared_start + thread_id; i < shared_end; i += blockDim.x)
    {
        if (shared_buf[i] == ':')
        {
            colon_pos[atomicAdd(&colon_pos[0], 1) + 1] = i;
        }
    }
    __syncthreads();

    if (thread_id == 0)
    {
        // swap if colon_pos[1] > colon_pos[2]
        if (colon_pos[1] > colon_pos[2])
        {
            uint32_t temp = colon_pos[1];
            colon_pos[1] = colon_pos[2];
            colon_pos[2] = temp;
        }

        colon_pos[1] += 1; // skip :

        // get transaction id and acquire items
        // trans = db->transactions[line_id];
        trans.data = &db->data[atomicAdd(&db->num_items, items_per_line[line_id])];
        trans.length = items_per_line[line_id];
        trans.utility = my_atou(shared_buf + colon_pos[1], colon_pos[2] - colon_pos[1]);

        uint32_t item_idx = 0;
        uint32_t prev = 0;
        uint32_t local_max_item = 0;
        for (int i = shared_start; i < colon_pos[1]; i++)
        {
            if (shared_buf[i] == delim || shared_buf[i] == ':')
            {
                uint32_t item = my_atou(shared_buf + prev, i - prev);
                local_max_item = max(local_max_item, item);
                prev = i + 1;
                trans.data[item_idx].key = item;
                item_idx++;
            }
        }

        atomicMax(max_item, local_max_item);

        item_idx = 0;
        prev = colon_pos[2] + 1;
        for (int i = colon_pos[2] + 1; i < shared_end + 1; i++)
        {
            if (shared_buf[i] == delim)
            {
                uint32_t item = my_atou(shared_buf + prev, i - prev);
                prev = i + 1;
                trans.data[item_idx].util = item;
                item_idx++;
            }
        }
        // parse the last item
        if (prev < shared_end + 1)
        {
            uint32_t item = my_atou(shared_buf + prev, shared_end - prev + 1);
            trans.data[item_idx].util = item;
            item_idx++;
        }

        db->transactions[line_id] = trans;
    }
    __syncthreads();
}

uint64_t get_file_size(const std::string &filename)
{
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

Parsed gpuParse(const std::string &filename, const char separator)
{
    Parsed pf;

    // get file size
    uint64_t file_size = get_file_size(filename);

    std::cout << "File size: " << file_size << " bytes\n";

    char *buf;
    cudaMallocManaged(&buf, file_size);
    std::cout << "Buffer allocated: " << file_size << " bytes\n";

    // open file
    int fd = open(filename.c_str(), O_RDONLY);

    if (fd < 0)
    {
        std::cerr << "Error opening file: " << strerror(errno) << "\n";
        return pf;
    }

    // read file into buffer
    ssize_t bytes_read = read(fd, buf, file_size);
    if (bytes_read < 0)
    {
        std::cerr << "Error reading file: " << strerror(errno) << "\n";
        close(fd);
        return pf;
    }

    // close file
    close(fd);

    // send the buffer to the GPU
    cudaMemAdvise(buf, file_size, cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(buf, file_size, cudaMemAdviseSetPreferredLocation, 0);
    cudaMemAdvise(buf, file_size, cudaMemAdviseSetAccessedBy, 0);
    cudaMemPrefetchAsync(buf, file_size, 0);
    cudaDeviceSynchronize();

    uint32_t *line_count;
    uint32_t *item_count;
    CUDA_CHECK(cudaMallocManaged(&line_count, sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocManaged(&item_count, sizeof(uint32_t)));

    line_count[0] = 1; // because we dont count the last line
    item_count[0] = 0;

    // int blockSize = 1024;
    dim3 blockSize(1024);
    dim3 gridSize((file_size + blockSize.x - 1) / blockSize.x);

    count_lines_items<<<gridSize, blockSize>>>(buf, file_size,
                                               separator,
                                               line_count,
                                               item_count);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Lines: " << *line_count
              << ", Items: " << *item_count
              << "\n";

    uint32_t *line_positions;
    cudaMallocManaged(&line_positions, (*line_count + 1) * sizeof(uint32_t));
    line_positions[0] = 0;
    line_positions[*line_count] = static_cast<uint32_t>(file_size);

    find_newline_positions<<<gridSize, blockSize>>>(buf, file_size,
                                                    line_positions);
    CUDA_CHECK(cudaDeviceSynchronize());
    line_positions[0] = 0;

    thrust::device_ptr<uint32_t> th_line_pos(line_positions);
    thrust::sort(thrust::device, th_line_pos, th_line_pos + *line_count + 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    // // print positions
    // std::cout << "Positions: ";
    // for (int i = 0; i < *line_count + 1; ++i)
    // {
    //     std::cout << line_positions[i] << " ";
    // }
    // std::cout << "\n";

    uint32_t *max_diff;
    cudaMallocManaged(&max_diff, sizeof(uint32_t));
    *max_diff = 0;

    uint32_t *items_per_line;
    cudaMallocManaged(&items_per_line, *line_count * sizeof(uint32_t));
    cudaMemset(items_per_line, 0, *line_count * sizeof(uint32_t));

    gridSize = (*line_count);

    // get_items_per_line<<<gridSize, blockSize>>>(d_buf, line_positions,
    //                                             items_per_line,
    //                                             delimiter, max_diff);
    get_items_per_line<<<gridSize, blockSize, *max_diff>>>(buf, line_positions,
                                                           items_per_line,
                                                           separator, max_diff);
    CUDA_CHECK(cudaDeviceSynchronize());
    *max_diff += 1;

    std::cout << "Max diff: " << *max_diff << "\n";

    // // print items per line
    // std::cout << "Items per line: ";
    // for (int i = 0; i < *line_count; ++i)
    // {
    //     std::cout << items_per_line[i] << " ";
    // }
    // std::cout << "\n";

    Database *db;
    // cudaMalloc(&db, sizeof(Database));
    // cudaMemset(db, 0, sizeof(Database));
    cudaMallocManaged(&db, sizeof(Database));
    cudaMallocManaged(&db->transactions, (*line_count) * sizeof(Transaction));
    cudaMallocManaged(&db->data, (*item_count) * sizeof(Item));
    db->num_transactions = *line_count;
    db->num_items = 0;

    // cudaMemset(&db, 0, sizeof(Database));
    // memset(&db, 0, sizeof(Database));

    uint32_t *max_item;
    cudaMallocManaged(&max_item, sizeof(uint32_t));
    *max_item = 0;

    // parse_file<<<gridSize, 32, *max_diff>>>(d_buf, line_positions, items_per_line,
    //                                         db, line_count, item_count,
    //                                         max_item, delimiter);

    parse_file<<<gridSize, 32, *max_diff>>>(buf, line_positions, items_per_line,
                                            db, line_count, item_count,
                                            max_item, separator);
    CUDA_CHECK(cudaDeviceSynchronize());

    // printDatabase<<<1,1>>>(db);
    // CUDA_CHECK(cudaDeviceSynchronize());
    // timer.recordPoint("Print database");

    std::cout << "Max item: " << *max_item << "\n";
    std::cout << "Item count: " << db->num_items << "\n";

    cudaFree(line_positions);
    cudaFree(items_per_line);
    cudaFree(line_count);
    cudaFree(item_count);

    // cuFileHandleDeregister(handle);
    close(fd);
    // cuFileBufDeregister(d_buf);

    // delete[] h_positions;
    // delete[] h_itemsPerLine;

    pf.db = db;
    pf.max_item = *max_item + 2;

    return pf;
}