#include "file.hpp"

#include <liburing.h> // For io_uring / file io

static inline ssize_t get_file_size(const std::string &filename)
{
    struct stat st;
    if (stat(filename.c_str(), &st) != 0)
    {
        return -1;
    }
    return static_cast<ssize_t>(st.st_size);
}

static inline size_t validate_file(const std::string &filename)
{
    if (access(filename.c_str(), F_OK) == -1)
    {
        throw std::runtime_error("Error: File does not exist: " + filename);
    }
    ssize_t sz = get_file_size(filename);
    if (sz < 0)
    {
        throw std::runtime_error("Error: Unable to get file size: " + filename);
    }
    else if (sz == 0)
    {
        throw std::runtime_error("Error: File is empty: " + filename);
    }
    return static_cast<size_t>(sz);
}

static inline int open_file(const std::string &filename)
{
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1)
    {
        throw std::runtime_error("Error: Unable to open file: " + filename);
    }
    return fd;
}

static inline void init_io_uring(io_uring &ring, unsigned int queue_depth)
{
    if (io_uring_queue_init(queue_depth, &ring, 0) < 0)
    {
        throw std::runtime_error("Error: Unable to initialize io_uring");
    }
}

static inline void set_read_ahead_hint(int fd)
{
#ifdef __linux__
    if (posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED) != 0)
    {
        throw std::runtime_error("Error: Unable to set POSIX_FADV_WILLNEED");
    }
#else
    (void)fd; // No-op on non-Linux systems
#endif
}

static inline char *allocate_memory(size_t size_bytes)
{
    char *data = static_cast<char *>(std::malloc(size_bytes));
    if (!data)
    {
        throw std::runtime_error("Error: Unable to allocate memory for file");
    }
    return data;
}

static void read_chunk_with_io_uring(
    int fd,
    char *buffer,
    size_t chunk_offset,
    size_t chunk_size,
    unsigned int queue_depth,
    size_t page_size)
{
    // Initialize a local ring for this thread
    io_uring ring;
    init_io_uring(ring, queue_depth);

    // We'll track how many bytes we've submitted vs. how many we've completed
    size_t submitted_bytes = 0;
    size_t completed_bytes = 0;

    // We'll fetch completions in batches
    // struct io_uring_cqe *cqes[queue_depth];
    std::vector<io_uring_cqe *> cqes(queue_depth);

    while (completed_bytes < chunk_size)
    {
        // 1) Fill submission queue with read requests
        unsigned int free_slots = io_uring_sq_space_left(&ring);

        for (unsigned int i = 0; i < free_slots && submitted_bytes < chunk_size; i++)
        {
            size_t remaining = chunk_size - submitted_bytes;
            size_t to_read = (remaining < page_size) ? remaining : page_size;

            // Acquire an SQE
            io_uring_sqe *sqe = io_uring_get_sqe(&ring);
            if (!sqe)
            {
                // No more SQEs available
                break;
            }

            size_t file_offset = chunk_offset + submitted_bytes;
            // Prepare the read request
            io_uring_prep_read(
                sqe,
                fd,
                buffer + file_offset, // location in memory
                to_read,
                file_offset // offset in the file
            );

            sqe->user_data = file_offset; // store offset (optional for debugging)

            submitted_bytes += to_read;
        }

        // 2) Submit them
        if (io_uring_submit(&ring) < 0)
        {
            std::cerr << "[OpenMP Thread] Error: io_uring_submit failed\n";
            break;
        }

        // 3) Check completions
        // int num_cqes = io_uring_peek_batch_cqe(&ring, cqes, queue_depth);
        int num_cqes = io_uring_peek_batch_cqe(&ring, cqes.data(), queue_depth);
        while (num_cqes > 0)
        {
            for (int i = 0; i < num_cqes; i++)
            {
                io_uring_cqe *cqe = cqes[i];
                if (cqe->res < 0)
                {
                    std::cerr << "[OpenMP Thread] Error: read failed at offset "
                              << cqe->user_data << "\n";
                }
                else
                {
                    // cqe->res is the number of bytes actually read
                    completed_bytes += cqe->res;
                }
                // Mark this CQE as seen
                io_uring_cqe_seen(&ring, cqe);
            }
            // Peek again
            // num_cqes = io_uring_peek_batch_cqe(&ring, cqes, queue_depth);
            num_cqes = io_uring_peek_batch_cqe(&ring, cqes.data(), queue_depth);
        }
    }

    // Clean up
    io_uring_queue_exit(&ring);
}

raw_file load_file(std::string filename, int threads, Timer &timer)
{
    const unsigned int queue_depth = 1024;
    const size_t page_size = 1024 * 128; // 128KB

    // 1) Validate file, get size
    size_t file_size_bytes = validate_file(filename);

    // 2) Open file
    int fd = open_file(filename);

    // 3) Set read-ahead hint (optional)
    set_read_ahead_hint(fd);

    // 4) Allocate one big buffer for the entire file
    char *data_ptr = allocate_memory(file_size_bytes);

    // 5) Fill raw_file struct for the return
    raw_file rf;
    rf.fd = fd;
    rf.size_bytes = file_size_bytes;
    rf.data = data_ptr;

    // 6) Divide the file into `threads` chunks
    if (threads < 1)
        threads = 1;
    size_t chunk_size = file_size_bytes / threads;
    size_t remainder = file_size_bytes % threads;

#pragma omp parallel for schedule(static, 1) num_threads(threads)
    for (int i = 0; i < threads; i++)
    {
        // figure out each thread's chunk offset and size
        size_t my_offset = i * chunk_size;
        size_t my_chunk_size = (i == (threads - 1))
                                   ? (chunk_size + remainder)
                                   : chunk_size;

        read_chunk_with_io_uring(
            fd,
            data_ptr,
            my_offset,
            my_chunk_size,
            queue_depth,
            page_size);
    }

    // 7) Close file
    close(fd);

    // 8) Record memory usage, etc.
    timer.recordPoint("File Loaded");

    rf.retrieved_bytes = file_size_bytes;

    return std::move(rf);
}

// Helper to parse one transaction line
std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, uint32_t>
parse_line(const char *data, size_t start, size_t end, const char separator)
{
    ParseState state = PARSE_ITEMS;
    std::vector<uint32_t> items, utilities;
    uint32_t total_utility = 0;
    size_t pos = start;

    while (pos < end)
    {
        while (pos < end && (data[pos] == separator || data[pos] == '\n' || data[pos] == ':'))
        {
            if (data[pos] == ':')
            {
                if (state == PARSE_ITEMS)
                    state = PARSE_TOTAL_UTILITY;
                else if (state == PARSE_TOTAL_UTILITY)
                    state = PARSE_UTILITIES;
            }
            ++pos;
        }
        if (pos >= end)
            break;

        size_t token_end = pos;
        while (token_end < end && data[token_end] != separator && data[token_end] != ':' && data[token_end] != '\n')
            ++token_end;

        uint32_t value = 0;
        auto res = std::from_chars(data + pos, data + token_end, value);
        if (res.ec != std::errc())
        {
            std::cerr << "Error parsing number at position " << pos << std::endl;
            continue;
        }

        switch (state)
        {
        case PARSE_ITEMS:
            items.push_back(value);
            break;
        case PARSE_TOTAL_UTILITY:
            total_utility = value;
            break;
        case PARSE_UTILITIES:
            utilities.push_back(value);
            break;
        }

        pos = token_end;
    }

    if (items.size() != utilities.size())
    {
        std::cerr << "Mismatch: " << items.size() << " items vs " << utilities.size() << " utils\n";
    }

    return {items, utilities, total_utility};
}

parsed_file parse_file(raw_file &file, int threads, const char separator, Timer &timer)
{
    parsed_file pf;

    const char *data = file.data;
    size_t size = file.retrieved_bytes;


    // ---- STEP 1: PARALLEL FIND NEWLINE POSITIONS ----
    std::vector<size_t> newline_positions;

    std::vector<std::vector<size_t>> thread_newlines(threads);

#pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        size_t chunk_size = (size + threads - 1) / threads;
        size_t start = tid * chunk_size;
        size_t end = std::min(start + chunk_size, size);

        auto &local_newlines = thread_newlines[tid];
        for (size_t i = start; i < end; ++i)
        {
            if (data[i] == '\n')
            {
                local_newlines.push_back(i);
            }
        }
    }

    #ifdef DEBUG
    timer.recordPoint("Newline Positions");
    #endif

    for (const auto &vec : thread_newlines)
        newline_positions.insert(newline_positions.end(), vec.begin(), vec.end());

    std::sort(newline_positions.begin(), newline_positions.end());

    #ifdef DEBUG
    timer.recordPoint("Newline Positions Sorted");
    #endif


    // ---- STEP 2: BUILD LINE START INDICES ----
    std::vector<size_t> line_starts;
    line_starts.reserve(newline_positions.size() + 2);
    line_starts.push_back(0);
    for (size_t pos : newline_positions)
    {
        if (pos + 1 < size)
            line_starts.push_back(pos + 1);
    }
    line_starts.push_back(size); // final end-of-buffer

    size_t line_count = line_starts.size() - 1;

    #ifdef DEBUG
    timer.recordPoint("Line Starts");
    #endif

    // ---- STEP 3: PARSE LINES IN PARALLEL ----

    std::vector<std::unordered_map<uint32_t, uint64_t>> local_twu(threads);
    pf.key_value_pairs.resize(threads);

#pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        auto &twu = local_twu[tid];
        auto &kvps = pf.key_value_pairs[tid];

#pragma omp for schedule(static)
        for (long i = 0; i < static_cast<long>(line_count); ++i)
        {
            size_t start = line_starts[i];
            size_t end = line_starts[i + 1];

            auto [items, utilities, total_utility] = parse_line(data, start, end, separator);

            for (const auto &item : items)
                twu[item] += total_utility;

            kvps.emplace_back(std::move(items), std::move(utilities));
        }
    }

    #ifdef DEBUG
    timer.recordPoint("Line Parse");
    #endif

    // ---- STEP 4: MERGE LOCAL TWU MAPS ----
    for (const auto &local_map : local_twu)
    {
        for (const auto &[item, utility] : local_map)
        {
            pf.twu[item] += utility;
        }
    }

    timer.recordPoint("File Parse");

#ifdef DEBUG
    // std::cout << "Total Weighted Utility:\n";
    // for (const auto &[item, utility] : pf.twu)
    // {
    //     std::cout << item << ": " << utility << "\n";
    // }

    // std::cout << "Key-Value Pairs:\n";
    // for (const auto &[items, utilities] : pf.key_value_pairs)
    // {
    //     for (uint32_t item : items)
    //     {
    //         std::cout << item << " ";
    //     }
    //     std::cout << ": ";
    //     for (uint32_t utility : utilities)
    //     {
    //         std::cout << utility << " ";
    //     }
    //     std::cout << "\n";
    // }
#endif

    return std::move(pf);
}