#include "file/load.hpp"

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

raw_file load_file(std::string filename, int threads)
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

    rf.retrieved_bytes = file_size_bytes;

    return rf;
}