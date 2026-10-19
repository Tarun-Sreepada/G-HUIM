#include "io/reader/text.hpp"

#include <charconv>
#include <stdexcept>

std::string_view trim(std::string_view s) {
	const auto b = s.find_first_not_of(" \t\r\n");
	if (b == std::string_view::npos) return {};
	return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

Utility parse_int(std::string_view token) {
	token = trim(token);
	Utility value = 0;
	const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
	if (ec != std::errc() || ptr != token.data() + token.size() || token.empty())
		throw std::runtime_error("invalid number '" + std::string(token) + "'");
	return value;
}
