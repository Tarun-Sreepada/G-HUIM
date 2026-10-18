#include "data/parse.hpp"

#include "data/parallel.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ghuim {
namespace {

// Parsed contents of one contiguous block of lines; transaction offsets are local to the chunk.
struct Chunk
{
    std::vector<uint32_t>       items, utils;
    std::vector<RawTransaction> transactions;
};

// Appends every unsigned 32-bit integer in `field` (separated by `separator` or whitespace) to `out`.
void parse_numbers(std::string_view field, char separator, std::vector<uint32_t>& out)
{
    const char* cursor = field.data();
    const char* end    = cursor + field.size();
    while (cursor < end) {
        const char c = *cursor;
        if (c == separator || c == ' ' || c == '\t' || c == '\r') {
            ++cursor;
            continue;
        }
        const char* digits = cursor;
        uint64_t    value  = 0;
        while (cursor < end && static_cast<unsigned>(*cursor - '0') < 10u && cursor - digits < 10)
            value = value * 10 + static_cast<unsigned>(*cursor++ - '0');
        // Empty, non-numeric, or wider than 32 bits.
        if (cursor == digits || value > std::numeric_limits<uint32_t>::max() ||
            (cursor < end && static_cast<unsigned>(*cursor - '0') < 10u))
            throw std::runtime_error("malformed number in input: " + std::string(field));
        out.push_back(static_cast<uint32_t>(value));
    }
}

// An item listed twice in one transaction (kosarak has 27 such lines) becomes one entry with the summed
// utility; the miners assume each item occurs at most once per transaction. Lines whose items already
// ascend are left alone.
void merge_repeated_items(Chunk& chunk, size_t begin)
{
    const size_t end = chunk.items.size();
    bool         ascending = true;
    for (size_t i = begin + 1; i < end && ascending; ++i)
        ascending = chunk.items[i - 1] < chunk.items[i];
    if (ascending)
        return;
    std::vector<std::pair<uint32_t, uint64_t>> entries;
    for (size_t i = begin; i < end; ++i)
        entries.emplace_back(chunk.items[i], chunk.utils[i]);
    std::sort(entries.begin(), entries.end());
    size_t out = begin;
    for (size_t i = 0; i < entries.size(); ++i) {
        if (out > begin && chunk.items[out - 1] == entries[i].first) {
            const uint64_t sum = chunk.utils[out - 1] + entries[i].second;
            if (sum > std::numeric_limits<uint32_t>::max())
                throw std::runtime_error("utility of a repeated item exceeds 32 bits");
            chunk.utils[out - 1] = static_cast<uint32_t>(sum);
            continue;
        }
        chunk.items[out]   = entries[i].first;
        chunk.utils[out++] = static_cast<uint32_t>(entries[i].second);
    }
    chunk.items.resize(out);
    chunk.utils.resize(out);
}

// Parses "items:transaction utility:item utilities" lines; `text` starts and ends on line boundaries.
void parse_lines(std::string_view text, char separator, Chunk& chunk)
{
    size_t line_start = 0;
    while (line_start < text.size()) {
        const char*  newline  = static_cast<const char*>(std::memchr(text.data() + line_start, '\n', text.size() - line_start));
        const size_t line_end = newline == nullptr ? text.size() : static_cast<size_t>(newline - text.data());
        const std::string_view line = text.substr(line_start, line_end - line_start);
        line_start                  = line_end + 1;
        if (line.empty() || line.front() == '#' || line.front() == '%' || line.front() == '@' ||
            line.find_first_not_of(" \t\r") == std::string_view::npos)
            continue;

        const size_t first  = line.find(':');
        const size_t second = first == std::string_view::npos ? first : line.find(':', first + 1);
        if (second == std::string_view::npos)
            throw std::runtime_error("expected 'items:utility:utilities' line, got: " + std::string(line));

        const size_t begin = chunk.items.size();
        parse_numbers(line.substr(0, first), separator, chunk.items);
        parse_numbers(line.substr(second + 1), separator, chunk.utils);
        if (chunk.items.size() != chunk.utils.size())
            throw std::runtime_error("item and utility counts differ in line: " + std::string(line));
        merge_repeated_items(chunk, begin);
        // Use the sum of item utilities rather than the stated total so the bounds are exact.
        uint64_t utility = 0;
        for (size_t i = begin; i < chunk.utils.size(); ++i)
            utility += chunk.utils[i];
        chunk.transactions.push_back({begin, chunk.items.size(), utility});
    }
}

} // namespace

RawInput parse_input(std::string_view text, char separator)
{
    // Cut the file at line boundaries and parse the pieces concurrently.
    const size_t        pieces = worker_count(text.size(), size_t{1} << 20);
    std::vector<size_t> cut(pieces + 1, text.size());
    cut[0] = 0;
    for (size_t p = 1; p < pieces; ++p) {
        const size_t guess = std::max(text.size() / pieces * p, cut[p - 1]);
        const char*  nl    = static_cast<const char*>(std::memchr(text.data() + guess, '\n', text.size() - guess));
        cut[p]             = nl == nullptr ? text.size() : static_cast<size_t>(nl - text.data()) + 1;
    }
    std::vector<Chunk> chunks(pieces);
    parallel_for(pieces, pieces, [&](size_t, size_t begin, size_t end) {
        for (size_t p = begin; p < end; ++p)
            parse_lines(text.substr(cut[p], cut[p + 1] - cut[p]), separator, chunks[p]);
    });

    // Concatenate the chunks into flat arrays.
    std::vector<size_t> item_base(pieces + 1, 0), tx_base(pieces + 1, 0);
    for (size_t p = 0; p < pieces; ++p) {
        item_base[p + 1] = item_base[p] + chunks[p].items.size();
        tx_base[p + 1]   = tx_base[p] + chunks[p].transactions.size();
    }
    if (tx_base[pieces] == 0)
        throw std::runtime_error("input contains no transactions");
    RawInput input;
    input.items.resize(item_base[pieces]);
    input.utilities.resize(item_base[pieces]);
    input.transactions.resize(tx_base[pieces]);
    parallel_for(pieces, pieces, [&](size_t, size_t begin, size_t end) {
        for (size_t p = begin; p < end; ++p) {
            Chunk& c = chunks[p];
            std::copy(c.items.begin(), c.items.end(), input.items.begin() + item_base[p]);
            std::copy(c.utils.begin(), c.utils.end(), input.utilities.begin() + item_base[p]);
            for (size_t t = 0; t < c.transactions.size(); ++t) {
                const RawTransaction& r        = c.transactions[t];
                input.transactions[tx_base[p] + t] = {r.begin + item_base[p], r.end + item_base[p], r.utility};
            }
            c = Chunk{};
        }
    });
    return input;
}

} // namespace ghuim
