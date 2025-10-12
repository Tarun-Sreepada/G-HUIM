#include "file/save.cuh"

// void save_patterns(const std::string &output_path, const uint32_t *data, const uint32_t offset, Item_mapping
// *item_map, uint32_t max_item)
void save_patterns(const std::string& output_path, const std::vector<std::vector<uint32_t>>& all_patterns,
                   Item_mapping* item_map, uint32_t max_item)
{
    // if /dev/null is used, do not write to files
    if (output_path == "/dev/null") {
        std::cout << "Output is directed to /dev/null, no file will be created.\n";
        return;
    }

    std::unordered_map<uint32_t, std::string> item_map_copy;
    for (uint32_t i = 0; i < max_item; ++i) {
        // item_map_copy[item_map->items[i].id] = item_map->items[i].item;
        item_map_copy[item_map[i].id] = std::to_string(item_map[i].item);
    }

    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open output file: " << output_path << "\n";
        return;
    }

    uint64_t pattern_count = 0;
    for (const auto& pattern_data : all_patterns) {
        std::cout << "Written patterns so far: " << pattern_count << "\r" << std::flush;
        if (!pattern_data.empty()) {
            for (size_t j = 0; j < pattern_data.size(); ++j) {
                // output_file << item_map_copy[pattern_data[j]] << " ";
                if (pattern_data[j] == 0) {
                    output_file << "\n";
                    pattern_count++;
                }
                else {
                    output_file << pattern_data[j] << " ";
                }
            }
        }
    }

    std::cout << "\nTotal patterns written: " << pattern_count << "\n";

    // // create a set to store unique patterns
    // std::set<std::string> unique_patterns;
    // std::vector<uint32_t> pattern_data;
    // std::string pattern;

    // bool first = false;

    // uint64_t i = 0;
    // while(true)
    // {
    //     if (data[i] == 0)
    //     {
    //         if (first)
    //         {
    //             break;
    //         }

    //         output_file << "\n";
    //         first = true;

    //     }
    //     else
    //     {
    //         first = false;
    //         output_file << data[i] << " ";
    //     }
    //     i++;
    // }
    //     if (data[i] != 0)
    //     {
    //         // pattern +=
    //         pattern_data.push_back(data[i]);
    //         // pattern += std::to_string(item_map_copy[data[i]]) + " ";
    //         // output_file << item_map_copy[data[i]] << " ";
    //     }
    //     else
    //     {
    //         if (!pattern_data.empty())
    //         {
    //             // output all items except the last entry, which encodes the utility
    //             const size_t last_index = pattern_data.size() - 1;
    //             for (size_t j = 0; j < last_index; ++j)
    //             {
    //                 // pattern += item_map_copy[pattern_data[j]] + " ";
    //                 pattern += std::to_string(pattern_data[j]) + " ";
    //             }
    //             pattern += ":" + std::to_string(pattern_data.back()) + "\n";

    //             output_file << pattern;
    //             pattern_count++;
    //         }
    //         pattern.clear();
    //         pattern_data.clear();
    //     }
    // }

    // // Handle case where buffer doesn't end with sentinel but contains data
    // if (!pattern_data.empty())
    // {
    //     const size_t last_index = pattern_data.size() - 1;
    //     for (size_t j = 0; j < last_index; ++j)
    //     {
    //         pattern += item_map_copy[pattern_data[j]] + " ";
    //     }
    //     pattern += ":" + std::to_string(pattern_data.back()) + "\n";
    //     output_file << pattern;
    //     pattern_count++;
    // }

    output_file.close();
}