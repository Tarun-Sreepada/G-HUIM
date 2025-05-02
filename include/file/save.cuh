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

void save_patterns(const std::string& output_path, const uint32_t *data, const uint32_t offset, Item_mapping *item_map, uint32_t max_item);