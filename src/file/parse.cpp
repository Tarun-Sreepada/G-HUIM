#include "file/parse.hpp"

// fast in-place parser: fills `items` and sets txn_util
static inline void parse_line_fast(
    const char *buf, size_t len, char sep,
    std::vector<Item> &items, uint32_t &txn_util)
{
    enum State
    {
        ITEMS,
        TOTAL,
        UTILS
    } st = ITEMS;
    txn_util = 0;
    size_t util_idx = 0;
    items.clear();

    const char *ptr = buf;
    const char *end = buf + len;

    while (ptr < end)
    {
        char c = *ptr;
        if (c == sep || c == '\n' || c == '\r')
        {
            ++ptr;
            continue;
        }
        if (c == ':')
        {
            st = (st == ITEMS ? TOTAL : (st == TOTAL ? UTILS : st)); // if items -> total if total -> utils
            ++ptr;
            continue;
        }

        // parse a number—stop as soon as you hit ANY non-digit
        uint32_t v = 0;
        while (ptr < end && *ptr >= '0' && *ptr <= '9')
        {
            v = v * 10 + (*ptr - '0');
            ++ptr;
        }

        switch (st)
        {
        case ITEMS:
            items.emplace_back(Item{v, 0});
            break;
        case TOTAL:
            txn_util = v;
            break;
        case UTILS:
            if (util_idx < items.size())
                items[util_idx++].util = v;
            break;
        }
    }
}

// optimized parse_file
parsed_file parse(const std::string &filename, int threads, char sep)
{
    parsed_file pf;

    // 1) open & mmap whole file
    int fd = open(filename.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0)
    {
        perror("open");
        return pf;
    }
    struct stat sb;
    fstat(fd, &sb);
    size_t size = sb.st_size;

    char *buf = static_cast<char *>(
        mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (buf == MAP_FAILED)
    {
        perror("mmap");
        close(fd);
        return pf;
    }
    close(fd);

    // 2) compute chunk boundaries
    std::vector<size_t> pos(threads + 1);
    for (int t = 0; t <= threads; ++t)
        pos[t] = t * (size / threads);
    pos[threads] = size;

    // 3) per-thread storage
    std::vector<std::vector<std::vector<Item>>> items_per_thread(threads);
    std::vector<std::vector<uint32_t>> utils_per_thread(threads);

// 4) parallel parse with pre-scan for reservation
#pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        size_t start_offset = pos[tid];
        size_t end_offset = pos[tid + 1];

        // Adjust start boundary: snap to just after the first newline (unless tid == 0)
        if (tid > 0 && start_offset < size)
        {
            char *potential_start = buf + start_offset;
            char *nl = static_cast<char *>(memchr(potential_start, '\n', size - start_offset));
            if (nl)
            {
                start_offset = (nl - buf) + 1;
                // Ensure we don't go past the original end_offset
                if (start_offset > pos[tid + 1])
                {
                    start_offset = pos[tid + 1];
                }
            }
            else
            {
                // No newline found after start_offset in the rest of the file,
                // this thread has nothing to process.
                start_offset = end_offset;
            }
        }

        // Adjust end boundary: snap end *to* the first newline at or after the boundary
        // (unless tid == threads - 1).
        if (tid < threads - 1 && end_offset < size)
        {
            char *potential_end = buf + end_offset;
            // Search for the next newline *starting from* the calculated end_offset
            char *nl = static_cast<char *>(memchr(potential_end, '\n', size - end_offset));
            if (nl)
            {
                end_offset = (nl - buf) + 1; // End *after* the newline
            }
            else
            {
                // No more newlines in the file, extend chunk to the end of the file
                end_offset = size;
            }
        }

        // Pointers for the adjusted chunk for this thread
        char *start_ptr = buf + start_offset;
        char *end_ptr = buf + end_offset;

        // --- Pre-scan phase ---
        size_t line_count = 0;
        if (start_ptr < end_ptr)
        { // Only scan if there's data in the chunk
            char *current = start_ptr;
            while (true)
            {
                char *next_nl = static_cast<char *>(memchr(current, '\n', end_ptr - current));
                if (next_nl)
                {
                    line_count++;
                    current = next_nl + 1;
                }
                else
                {
                    break;
                }
            }
        }

        // --- Reserve exact space ---
        if (line_count > 0)
        {
            items_per_thread[tid].reserve(line_count);
            utils_per_thread[tid].reserve(line_count);
        }

        // --- Parsing phase ---
        char *line_start = start_ptr;
        while (line_start < end_ptr)
        {
            char *eol = static_cast<char *>(memchr(line_start, '\n', end_ptr - line_start));

            if (!eol)
            {
                break; // Stop if no more complete lines found in the chunk
            }

            size_t len = eol - line_start;
            parse_line_fast(
                line_start, len, sep,
                items_per_thread[tid].emplace_back(),
                utils_per_thread[tid].emplace_back());

            line_start = eol + 1; // Move to the beginning of the next line
        }
    } // End of parallel region

    // 5) merge in single thread
    size_t total_txns = 0;
    for (int t = 0; t < threads; ++t)
    {
        total_txns += items_per_thread[t].size();
    }

    pf.database.reserve(total_txns);

    for (int t = 0; t < threads; ++t)
    {
        pf.database.insert(
            pf.database.end(),
            std::make_move_iterator(items_per_thread[t].begin()),
            std::make_move_iterator(items_per_thread[t].end()));

        pf.total_util.insert(
            pf.total_util.end(),
            utils_per_thread[t].begin(),
            utils_per_thread[t].end());
    }

    // 7) Cleanup mmap
    munmap(buf, size);

    return std::move(pf);
}

bool keysAreContiguousFromOne(
    const std::unordered_map<uint32_t, uint32_t> &twu)
{
    if (twu.empty())
        return false;

    // 1) find max key and collect them
    uint32_t max_key = 0;
    std::vector<uint32_t> keys;
    for (auto const &p : twu)
    {
        max_key = std::max(max_key, p.first);
        keys.push_back(p.first);
    }

    // 2) quick size check: if there are holes, size < max_key
    if (keys.size() != max_key)
    {
        std::cout << "keys.size() = " << keys.size() << ", max_key = " << max_key << std::endl;
        return false;
    }

    // 3) mark presence
    std::vector<char> seen(max_key + 1, 0);
    for (auto k : keys)
        seen[k] = 1;

    // 4) ensure every i in [1..max_key] was seen
    for (uint32_t i = 1; i <= max_key; ++i)
    {
        if (!seen[i])
        {
            std::cout << "key " << i << " not seen" << std::endl;
            return false;
        }
    }
    return true;
}