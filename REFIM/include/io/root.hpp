#pragma once

#include "io/reader.hpp"
#include "mine/level.hpp"

#include <cstdint>

inline namespace REFIM_WIDTHS {

// Sorts and merges the transactions of `input` into `root` (freeing its chunks), and sets the root's primary
// items (subtree utility >= min_util) and their occurrence index. Uses up to `threads` threads.
void make_root(Transactions& input, std::uint64_t min_util, int threads, Level& root);

}  // namespace REFIM_WIDTHS
