#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace ghuim {

struct RawTransaction
{
    size_t   begin; // into RawInput::items / utilities
    size_t   end;
    uint64_t utility;
};

// The input as written: parallel item/utility arrays and each transaction's range in them.
struct RawInput
{
    std::vector<uint32_t>       items, utilities;
    std::vector<RawTransaction> transactions;
};

// Parses "items:transaction utility:item utilities" lines, using all cores.
RawInput parse_input(std::string_view text, char separator);

} // namespace ghuim
