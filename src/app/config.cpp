#include "app/config.hpp"

#include <charconv>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace ghuim {

uint64_t parse_size(const std::string& text)
{
    const std::string_view value(text);
    uint64_t               number = 0;
    const auto [end, error]       = std::from_chars(value.data(), value.data() + value.size(), number);
    if (error != std::errc{} || end == value.data())
        throw std::invalid_argument("expected a positive integer with optional K, M, G, KiB, MiB, or GiB suffix");

    const std::string_view suffix(end, value.data() + value.size() - end);
    uint64_t               multiplier = 1;
    if (suffix == "K" || suffix == "KiB")
        multiplier = 1024;
    else if (suffix == "M" || suffix == "MiB")
        multiplier = 1024ULL * 1024;
    else if (suffix == "G" || suffix == "GiB")
        multiplier = 1024ULL * 1024 * 1024;
    else if (!suffix.empty() && suffix != "B")
        throw std::invalid_argument("unknown size suffix");

    if (number == 0 || number > std::numeric_limits<uint64_t>::max() / multiplier)
        throw std::invalid_argument("size must be positive and fit in 64 bits");
    return number * multiplier;
}

} // namespace ghuim
