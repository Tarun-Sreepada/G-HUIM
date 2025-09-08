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

    __shared__ uint32_t start, end, shared_end;
    __shared__ Transaction trans;
    extern __shared__ char shared_buf[];
    __shared__ uint32_t colon_pos[3];

    if (threadIdx.x == 0)
    {
        // --- SIMPLIFIED BOUNDARY LOGIC ---
        // The logic is now universal for every line, including the last one.
        // `start` is the beginning of the line's content.
        start = line_positions[line_id];
        if (line_id != 0) // Skip the preceding '\n' for all lines but the first
            start += 1;

        // `end` is the position of the line's trailing '\n' or the EOF.
        // It marks the end of the content.
        end = line_positions[line_id + 1];

        shared_end = end - start; // This is the exact length of the line's content.
        
        colon_pos[0] = 0; colon_pos[1] = 0; colon_pos[2] = 0;
    }
    __syncthreads();

    // Load the exact line content into shared memory. The loop correctly
    // stops before `end`, so no newline character is copied.
    for (int i = thread_id + start; i < end; i += blockDim.x)
    {
        shared_buf[i - start] = buf[i];
    }
    __syncthreads();

    // Find colon positions within the line's content.
    for (int i = thread_id; i < shared_end; i += blockDim.x)
    {
        if (shared_buf[i] == ':')
        {
            // This atomic is not guaranteed to find colons in order, hence the sort below.
            uint32_t pos_idx = atomicAdd(&colon_pos[0], 1);
            if (pos_idx < 2) { // To avoid writing out of bounds if a line has >2 colons
                colon_pos[pos_idx + 1] = i;
            }
        }
    }
    __syncthreads();

    if (thread_id == 0)
    {
        // Sort the two colon positions found.
        if (colon_pos[1] > colon_pos[2])
        {
            uint32_t temp = colon_pos[1];
            colon_pos[1] = colon_pos[2];
            colon_pos[2] = temp;
        }

        uint32_t items_start = 0;
        uint32_t utility_start = colon_pos[1] + 1;
        uint32_t item_utils_start = colon_pos[2] + 1;

        trans.data = &db->data[atomicAdd(&db->num_items, items_per_line[line_id])];
        trans.length = items_per_line[line_id];
        trans.utility = my_atou(shared_buf + utility_start, item_utils_start - utility_start - 1);
        
        uint32_t item_idx = 0;
        uint32_t prev = items_start;
        uint32_t local_max_item = 0;

        // Parse item keys
        for (int i = items_start; i < utility_start; i++)
        {
            if (shared_buf[i] == delim || shared_buf[i] == ':')
            {
                uint32_t item = my_atou(shared_buf + prev, i - prev);
                local_max_item = max(local_max_item, item);
                trans.data[item_idx].key = item;
                item_idx++;
                prev = i + 1;
            }
        }
        atomicMax(max_item, local_max_item);
        
        item_idx = 0;
        prev = item_utils_start;

        // Parse item utilities
        for (int i = item_utils_start; i < shared_end; i++)
        {
            if (shared_buf[i] == delim)
            {
                trans.data[item_idx].util = my_atou(shared_buf + prev, i - prev);
                item_idx++;
                prev = i + 1;
            }
        }
        // Parse the final utility value in the line
        if (prev < shared_end)
        {
            trans.data[item_idx].util = my_atou(shared_buf + prev, shared_end - prev);
        }

        db->transactions[line_id] = trans;
    }
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

    if (file_size <= 0) {
        std::cerr << "File is empty or does not exist.\n";
        pf.db = nullptr;
        pf.max_item = 0;
        return pf;
    }
    std::cout << "File size: " << file_size << " bytes\n";

    char *buf;
    cudaMallocManaged(&buf, file_size);
    std::cout << "Buffer allocated: " << file_size << " bytes\n";

    int fd = open(filename.c_str(), O_RDONLY);
    if (fd < 0) { /* ... error handling ... */ cudaFree(buf); return pf; }
    read(fd, buf, file_size);
    close(fd);

    cudaMemAdvise(buf, file_size, cudaMemAdviseSetReadMostly, 0);
    cudaMemPrefetchAsync(buf, file_size, 0);
    cudaDeviceSynchronize();

    uint32_t *line_count;
    uint32_t *item_count;
    CUDA_CHECK(cudaMallocManaged(&line_count, sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocManaged(&item_count, sizeof(uint32_t)));

    *line_count = 0; // Start with 0 lines.
    *item_count = 0;

    dim3 blockSize(1024);
    dim3 gridSize((file_size + blockSize.x - 1) / blockSize.x);

    // --- DETECTION STEP 1: Count all newline characters ---
    // This gives us a baseline for the number of lines.
    count_lines_items<<<gridSize, blockSize>>>(buf, file_size, separator, line_count, item_count);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- ADJUSTMENT STEP 1: Determine the final line count ---
    // We explicitly DETECT if the file ends with a newline.
    // If it doesn't, the last line wasn't counted, so we ADJUST the count.
    if (file_size > 0 && buf[file_size - 1] != '\n')
    {
        (*line_count)++;
    }

    std::cout << "Lines: " << *line_count << ", Est. Items: " << *item_count << "\n";
    if (*line_count == 0) { /* ... handle empty file, cleanup and return ... */ }

    // --- DETECTION STEP 2: Find the exact position of every newline ---
    uint32_t *line_positions;
    cudaMallocManaged(&line_positions, (*line_count + 1) * sizeof(uint32_t));
    
    // The counter for find_newline_positions will start at index 1.
    line_positions[0] = 0; // This will hold the count of newlines found.

    find_newline_positions<<<gridSize, blockSize>>>(buf, file_size, line_positions);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- ADJUSTMENT STEP 2: Build the definitive line boundary map ---
    // The kernel used line_positions[0] as an atomic counter. Let's retrieve that value.
    uint32_t num_newlines_found = line_positions[0];

    // The first line always starts at position 0.
    line_positions[0] = 0;

    // Sort the newline positions found by the kernel, as they are discovered out of order.
    // We sort the `num_newlines_found` elements that were written starting at index 1.
    thrust::device_ptr<uint32_t> th_line_pos(line_positions);
    thrust::sort(thrust::device, th_line_pos + 1, th_line_pos + 1 + num_newlines_found);

    // The final boundary is the end of the file. This acts as a sentinel for the last line,
    // making the logic in the parsing kernel identical for all lines.
    line_positions[*line_count] = static_cast<uint32_t>(file_size);
    CUDA_CHECK(cudaDeviceSynchronize());


    // ... The rest of the function continues ...

    uint32_t *max_diff;
    cudaMallocManaged(&max_diff, sizeof(uint32_t));
    *max_diff = 0;

    uint32_t *items_per_line;
    cudaMallocManaged(&items_per_line, *line_count * sizeof(uint32_t));
    cudaMemset(items_per_line, 0, *line_count * sizeof(uint32_t));

    gridSize = (*line_count);

    // FIX: This kernel does not use dynamic shared memory. The third launch
    // parameter should be 0 or omitted.
    get_items_per_line<<<gridSize, blockSize>>>(buf, line_positions,
                                               items_per_line,
                                               separator, max_diff);
    CUDA_CHECK(cudaDeviceSynchronize());
    *max_diff += 1; // For buffer allocation, needs to be size, not max index.

    std::cout << "Max line length: " << *max_diff << " bytes\n";

    Database *db;
    cudaMallocManaged(&db, sizeof(Database));
    cudaMallocManaged(&db->transactions, (*line_count) * sizeof(Transaction));
    cudaMallocManaged(&db->data, (*item_count) * sizeof(Item));
    db->num_transactions = *line_count;
    db->num_items = 0;

    uint32_t *max_item;
    cudaMallocManaged(&max_item, sizeof(uint32_t));
    *max_item = 0;
    
    // --- PARSING PHASE ---
    // This kernel now receives the perfect map (`line_positions`) and can parse
    // each line without needing special case logic.
    parse_file<<<gridSize, 32, *max_diff>>>(buf, line_positions, items_per_line,
                                            db, line_count, item_count,
                                            max_item, separator);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Max item: " << *max_item << "\n";
    std::cout << "Total items parsed: " << db->num_items << "\n";
    
    // --- Cleanup ---
    cudaFree(buf);
    cudaFree(line_positions);
    cudaFree(items_per_line);
    cudaFree(line_count);
    cudaFree(item_count);
    cudaFree(max_diff);
    // Note: max_item is part of pf, don't free it if you return it.

    pf.db = db;
    pf.max_item = *max_item + 2;

    return pf;
}