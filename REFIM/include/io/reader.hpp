#pragma once

#include "mine/level.hpp"

#include <string>
#include <vector>

// Reads the input, removes items with TWU < min_util, renames items so a larger id means a larger
// TWU, sorts and merges the transactions into `root`, and sets the root's primary items (subtree
// utility >= min_util) and their occurrence index. Parsing and sorting use up to `threads` threads. rename[id] = original item name (index 0 unused).
void read_database(const std::string& path, const std::string& sep, Utility min_util, Level& root,
                   std::vector<std::string>& rename, int threads = 1);
