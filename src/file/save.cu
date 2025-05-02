#include "file/save.cuh"


void save_patterns(const std::string& output_path, const uint32_t *data, const uint32_t offset, Item_mapping *item_map, uint32_t max_item)
{

    // if /dev/null is used, do not write to file
    if (output_path == "/dev/null")
    {
        std::cout << "Output is directed to /dev/null, no file will be created.\n";
        return;
    }

    std::unordered_map<uint32_t, std::string> item_map_copy;
    for (uint32_t i = 0; i < max_item; ++i)
    {
        // item_map_copy[item_map->items[i].id] = item_map->items[i].item;
        item_map_copy[item_map[i].id] = std::to_string(item_map[i].item);
    }

    std::ofstream output_file(output_path);
    if (!output_file.is_open())
    {
        std::cerr << "Error: Could not open output file: " << output_path << "\n";
        return;
    }

    // create a set to store unique patterns
    std::set<std::string> unique_patterns;
    std::vector<uint32_t> pattern_data; 
    std::string pattern;

    int32_t pattern_count = 0;
    for (uint32_t i = 0; i < offset; ++i)
    {
        if (data[i] != 0)
        {
            // output_file << data[i] << " ";
            // pattern += 
            pattern_data.push_back(data[i]);    
            // pattern += std::to_string(item_map_copy[data[i]]) + " ";
            // output_file << item_map_copy[data[i]] << " ";
        }
        else
        {
            // output
            // except last one convert all
            for (uint32_t j = 0; j < pattern_data.size() - 1; ++j)
            {
                pattern += item_map_copy[pattern_data[j]] + " ";
            }
            pattern += std::to_string(pattern_data.back()) + "\n";

            output_file << pattern;
            pattern.clear();
            pattern_data.clear();
            pattern_count++;
        }
    }

    std::cout << "Number of patterns: " << pattern_count << "\n";

    output_file.close();
}