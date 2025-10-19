#pragma once

#include "build.cuh"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

void save_patterns(const std::string& output_path, const std::pair<uint32_t *, uint64_t>& all_patterns,
                   Item_mapping* item_map, uint32_t max_item);