#pragma once
#include <string>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>

struct Args
{
    std::string file_path;
    int min_util;
    std::string delimiter;
    int block_size;
    int num_blocks;
};

// Helper function to map separator string to actual character
static char delimiter_byte(const std::string &sepStr)
{
    if (sepStr == "\\s")
        return ' ';
    if (sepStr == "\\t")
        return '\t';
    if (sepStr == "\\n")
        return '\n';
    if (sepStr == ",")
        return ',';
    if (sepStr == ";")
        return ';';
    // Add more separators as needed
    // Default separator
    std::cerr << "Unknown separator: " << sepStr << ". Using space as default.\n";
    return ' ';
}

Args parse_args(int argc, char *argv[])
{
    Args result;
    // We require at least 3 arguments: filename, min_util, delimiter.
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " file_path min_util delimiter [block_size] [num_blocks]" << std::endl;
        std::exit(1);
    }

    result.file_path = argv[1];
    result.min_util = std::atoi(argv[2]);
    result.delimiter = delimiter_byte(argv[3]);

    // Default values for block_size and num_blocks are 512.
    result.block_size = 512;
    result.num_blocks = 512;

    // If provided, override the defaults.
    if (argc >= 5)
    {
        result.block_size = std::atoi(argv[4]);
    }
    if (argc >= 6)
    {
        result.num_blocks = std::atoi(argv[5]);
    }

    return result;
}

