#include "file/save.cuh"

// void save_patterns(const std::string &output_path, const uint32_t *data, const uint32_t offset, Item_mapping
// *item_map, uint32_t max_item)
void save_patterns(const std::string& output_path, const std::pair<uint32_t *, uint64_t>& all_patterns,
                   Item_mapping* item_map, uint32_t max_item)
{
    // if /dev/null is used, do not write to files
    if (output_path == "/dev/null") {
        std::cout << "Output is directed to /dev/null, no file will be created.\n";
        return;
    }

    uint32_t max_assigned_id = 0;
    for (uint32_t i = 0; i < max_item; ++i)
        max_assigned_id = std::max(max_assigned_id, item_map[i].id);

    std::vector<std::string> id_to_item(static_cast<size_t>(max_assigned_id) + 1);
    for (uint32_t i = 0; i < max_item; ++i)
        id_to_item[item_map[i].id] = std::to_string(item_map[i].item);

    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open output file: " << output_path << "\n";
        return;
    }

    std::vector<uint32_t> current_pattern;

    auto write_pattern = [&](const std::vector<uint32_t>& pattern_buffer) {
        if (pattern_buffer.empty())
            return;

        const uint32_t utility = pattern_buffer.back();
        std::string     line;

        if (pattern_buffer.size() > 1) {
            line.reserve(pattern_buffer.size() * 8);
            for (size_t idx = 0; idx + 1 < pattern_buffer.size(); ++idx) {
                const uint32_t item_id = pattern_buffer[idx];
                const std::string& mapped = (item_id < id_to_item.size() && !id_to_item[item_id].empty())
                                                    ? id_to_item[item_id]
                                                    : std::to_string(item_id);

                if (!line.empty())
                    line.push_back(' ');
                line += mapped;
            }
            line.push_back(' ');
        }

        line += "UTIL ";
        line += std::to_string(utility);
        output_file << line << '\n';
    };

    // for (const auto& pattern_data : all_patterns) {
    //     for (uint32_t value : pattern_data) {
    //         if (value == 0) {
    //             write_pattern(current_pattern);
    //             current_pattern.clear();
    //         }
    //         else {
    //             current_pattern.push_back(value);
    //         }
    //     }
    // }

    bool _first_zero_skipped = false;

    for (uint64_t i = 0; i < all_patterns.second; ++i) {
        uint32_t value = all_patterns.first[i];
        if (value == 0) {
            if (!_first_zero_skipped) {
                _first_zero_skipped = true;
            }
            else
            {
                break;
            }

            write_pattern(current_pattern);
            current_pattern.clear();
        }
        else {
            _first_zero_skipped = false;
            current_pattern.push_back(value);
        }
    }


    write_pattern(current_pattern);

    output_file.close();
}