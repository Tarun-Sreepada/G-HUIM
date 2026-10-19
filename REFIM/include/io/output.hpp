#pragma once
// Parallel pattern writer: sizes every worker's patterns, then each thread formats its
// share with std::to_chars and pwrite()s it at its own offset of a preallocated file.

#include "mine/scheduler.hpp"

#include <string>
#include <vector>

// Line format: items joined by '\t', then ':' and the utility. Returns bytes written.
std::size_t write_patterns(const std::string& path, const std::vector<Patterns>& results,
                           const std::vector<std::string>& names);
