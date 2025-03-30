/**
 * @file file.hpp
 * @brief Provides structures and functions for handling raw and parsed file data, 
 *        including file loading, parsing, and utility calculations.
 *
 * This header defines the following:
 * - `raw_file`: A structure to manage raw file data and its associated metadata.
 * - `parsed_file`: A structure to store parsed key-value pairs and utility data.
 * - `load_file`: A function to load a file into memory using io_uring for asynchronous I/O.
 * - `parse_file`: A function to parse raw file data into structured key-value pairs.
 * - `ParseState`: An enumeration to represent the state of parsing.
 */

 /**
 * @struct raw_file
 * @brief Represents a raw file loaded into memory.
 *
 * This structure contains:
 * - The file's data in memory.
 * - Metadata about the file, including size, submitted, retrieved, and processed bytes.
 * - A vector of retrieved indices for tracking processed data.
 * - The file descriptor for the opened file.
 *
 * @note The `close` method should be called to free allocated memory.
 */

 /**
 * @struct parsed_file
 * @brief Represents a parsed file with structured key-value pairs and utility data.
 *
 * This structure contains:
 * - A nested vector of key-value pairs, where each key and value is a vector of uint32_t.
 * - A map of total weighted utility (TWU) values for items.
 */

 /**
 * @enum ParseState
 * @brief Represents the state of parsing a file.
 *
 * Possible states:
 * - `PARSE_ITEMS`: Parsing items from the file.
 * - `PARSE_TOTAL_UTILITY`: Parsing the total utility value.
 * - `PARSE_UTILITIES`: Parsing individual utilities.
 */

 /**
 * @fn raw_file load_file(std::string filename, int threads, Timer &timer)
 * @brief Loads a file into memory using asynchronous I/O.
 *
 * @param filename The path to the file to be loaded.
 * @param threads The number of threads to use for loading.
 * @param timer A reference to a Timer object for measuring performance.
 * @return A `raw_file` structure containing the loaded file data and metadata.
 *
 * @throws std::runtime_error If the file cannot be opened or read.
 */

 /**
 * @fn parsed_file parse_file(raw_file &file, int threads, const char separator, Timer &timer)
 * @brief Parses a raw file into structured key-value pairs and calculates utility data.
 *
 * @param file A reference to the `raw_file` structure containing the file data.
 * @param threads The number of threads to use for parsing.
 * @param separator The character used to separate items in the file.
 * @param timer A reference to a Timer object for measuring performance.
 * @return A `parsed_file` structure containing the parsed data and utility information.
 *
 * @throws std::invalid_argument If the file data is invalid or cannot be parsed.
 */
#pragma once

#include <iostream>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include <memory>
#include "timer.hpp"
#include <vector>
#include <string>
#include <omp.h>
#include <liburing.h>
#include <unordered_map>
#include <optional>
#include <stdexcept>
#include <algorithm>
#include <utility>
#include <charconv>


struct raw_file
{
    char *data;
    size_t size_bytes;
    size_t submitted_bytes; // bytes submitted to io_uring
    size_t retrieved_bytes; // bytes retrieved from io_uring
    size_t processed_bytes; // bytes processed from data
    std::vector<std::pair<size_t, size_t>> retrieved_indices;

    int fd;

    raw_file() : data(nullptr), size_bytes(0), submitted_bytes(0), retrieved_bytes(0), processed_bytes(0), fd(-1) {}

    void close()
    {
        if (data != nullptr)
            free(data);
    }

};

raw_file load_file(std::string filename, int threads, Timer &timer);

struct parsed_file
{
    std::vector<std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>>> key_value_pairs;
    std::unordered_map<uint32_t, uint32_t> twu;
};

parsed_file parse_file(raw_file &file, int threads, const char separator, Timer &timer);

enum ParseState { PARSE_ITEMS, PARSE_TOTAL_UTILITY, PARSE_UTILITIES };
