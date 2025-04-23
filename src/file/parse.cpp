#include "file/parse.hpp"

// fast in-place parser: fills `items` and sets txn_util
static inline void parse_line_fast(
    const char *buf, size_t len, char sep,
    std::vector<Item>& items, uint32_t &txn_util)
{
    enum State { ITEMS, TOTAL, UTILS } st = ITEMS;
    txn_util = 0;
    size_t util_idx = 0;

    items.clear();
    items.reserve(16);

    for (size_t i = 0; i < len; /*++i in body*/) {
        char c = buf[i];
        // skip separators & handle ':' flip
        if (c == sep || c == '\n' || c == '\r' || c == ':') {
            if (c == ':') {
                if (st == ITEMS)  st = TOTAL;
                else if (st == TOTAL) st = UTILS;
            }
            ++i;
            continue;
        }

        // parse unsigned integer
        uint32_t v = 0;
        while (i < len && buf[i] >= '0' && buf[i] <= '9') {
            v = v * 10 + (buf[i] - '0');
            ++i;
        }

        switch (st) {
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

bool keysAreContiguousFromOne(
    const std::unordered_map<uint32_t,uint32_t>& twu)
{
    if (twu.empty()) 
        return false;           // or true, depending on how you want to treat empty

    // 1) find max key and collect them
    uint32_t max_key = 0;
    std::vector<uint32_t> keys;
    keys.reserve(twu.size());
    for (auto const& p : twu) {
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
    for (uint32_t i = 1; i <= max_key; ++i) {
        if (!seen[i])
        {
            std::cout << "key " << i << " not seen" << std::endl;
            return false;
        }
    }
    return true;
}

// optimized parse_file
parsed_file parse(raw_file &file, int threads, const char separator)
{
    parsed_file pf;
    const char *data = file.data;
    size_t size = file.retrieved_bytes;

    //── STEP 1: BUILD LINE STARTS (single pass) ──
    std::vector<size_t> line_starts;
    line_starts.reserve(size / 20);  // hint: avg line ~20 bytes
    line_starts.push_back(0);
    for (size_t i = 0; i < size; ++i) {
        if (data[i] == '\n' && i + 1 < size)
            line_starts.push_back(i + 1);
    }
    line_starts.push_back(size);

    size_t line_count = line_starts.size() - 1;
    pf.database.resize(line_count);

    //── STEP 2: PARSE IN PARALLEL ──
    std::vector<std::unordered_map<uint32_t,uint64_t>> local_twu(threads);

#pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        auto &twu = local_twu[tid];

        // thread-local scratch
        std::vector<Item> items;
        items.reserve(16);

        #pragma omp for schedule(guided)
        for (long i = 0; i < (long)line_count; ++i) {
            size_t start = line_starts[i];
            size_t end   = line_starts[i+1];
            uint32_t txn_util;

            // parse directly into items[]
            parse_line_fast(data + start, end - start, separator, items, txn_util);

            // accumulate TWU
            for (auto &it : items)
                twu[it.key] += txn_util;

            // store into final database
            pf.database[i] = items;  // vector<Item> has just the right size
        }
    }

    //── STEP 3: MERGE TWU MAPS ──
    for (auto &lm : local_twu)
        for (auto &kv : lm)
            pf.twu[kv.first] += kv.second;

    //── STEP 4: CHECK KEYS ──
    if (!keysAreContiguousFromOne(pf.twu)) {
        fprintf(stderr, "Error: keys are not contiguous from 1 to max_key.\n");
        exit(1);
    }

    return pf;
}