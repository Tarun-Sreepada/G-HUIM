#pragma once
// Text helpers for the dataset parser.

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

std::string_view trim(std::string_view s);

// Calls f(token) for each non-empty token of s split at sep.
template <class F>
void for_each_token(const std::string& sep, std::string_view s, F&& f) {
	std::size_t start = 0;
	while (true) {
		std::size_t end;
		if (sep.size() == 1) {
			const void* hit = start < s.size() ? std::memchr(s.data() + start, sep[0], s.size() - start) : nullptr;
			end = hit ? static_cast<std::size_t>(static_cast<const char*>(hit) - s.data()) : std::string_view::npos;
		} else {
			end = s.find(sep, start);
		}
		const std::string_view token = s.substr(start, end == std::string_view::npos ? end : end - start);
		if (!token.empty()) f(token);
		if (end == std::string_view::npos) break;
		start = end + sep.size();
	}
}

// Parses a whole integer token (surrounding whitespace allowed); throws on anything else.
std::int64_t parse_int(std::string_view token);
