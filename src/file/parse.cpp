#include "file/parse.hpp"


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

parsed_file parse_file(raw_file &file, int threads, const char separator)
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

    for (const auto &vec : thread_newlines)
        newline_positions.insert(newline_positions.end(), vec.begin(), vec.end());

    std::sort(newline_positions.begin(), newline_positions.end());

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

    // ---- STEP 3: PARSE LINES IN PARALLEL ----

    std::vector<std::unordered_map<uint32_t, uint64_t>> local_twu(threads);
    pf.key_value_pairs.resize(threads);
 
#pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        auto &twu = local_twu[tid];
        auto &kvps = pf.key_value_pairs[tid];

        #pragma omp for schedule(guided) nowait
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

    // ---- STEP 4: MERGE LOCAL TWU MAPS ----
    for (const auto &local_map : local_twu)
    {
        for (const auto &[item, utility] : local_map)
        {
            pf.twu[item] += utility;
        }
    }


    return pf;
}