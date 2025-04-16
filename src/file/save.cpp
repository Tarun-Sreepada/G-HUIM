#include "file/save.hpp"


void save_patterns(const std::string& output_path, const uint32_t *data, const uint32_t offset)
{
    std::ofstream output_file(output_path);
    if (!output_file.is_open())
    {
        std::cerr << "Error: Could not open output file: " << output_path << "\n";
        return;
    }

    // create a set to store unique patterns
    std::set<std::string> unique_patterns;
    std::string pattern;

    int32_t pattern_count = 0;
    for (uint32_t i = 0; i < offset; ++i)
    {
        if (data[i] != 0)
        {
            output_file << data[i] << " ";
        }
        else
        {
            output_file << "\n";
            pattern_count++;
        }
    }

    std::cout << "Number of patterns: " << pattern_count << "\n";

    output_file.close();
}