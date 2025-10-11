#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <iostream>
#include <fstream>
#include <set>
#include <algorithm>
#include <cstdint>
#include "build.cuh"

void save_patterns(const std::string& output_path, const std::vector<std::vector<uint32_t>>& all_patterns, Item_mapping *item_map, uint32_t max_item);