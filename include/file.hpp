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

/**
 * @brief Holds raw file data and associated metadata.
 */
struct raw_file
{
    /** @brief Pointer to the file's data loaded into memory. */
    char *data;
    
    /** @brief Total size of the file in bytes. */
    size_t size_bytes;

    /** @brief Number of bytes submitted to io_uring (if using async I/O). */
    size_t submitted_bytes;

    /** @brief Number of bytes retrieved from io_uring (if using async I/O). */
    size_t retrieved_bytes;

    /** @brief Number of bytes already processed from the data buffer. */
    size_t processed_bytes;

    /** @brief Offsets or indices of chunks retrieved via io_uring. */
    std::vector<std::pair<size_t, size_t>> retrieved_indices;

    /** @brief File descriptor for the open file (could be valid or -1 if invalid). */
    int fd;

    /**
     * @brief Constructor initializes defaults.
     */
    raw_file() : data(nullptr), size_bytes(0), submitted_bytes(0),
                 retrieved_bytes(0), processed_bytes(0), fd(-1) {}

    /**
     * @brief Frees the file data buffer, if allocated.
     */
    void close()
    {
        if (data != nullptr)
            free(data);
    }
};

/**
 * @brief Loads a file into a raw_file structure, optionally using multiple threads.
 * @param filename Path to the file to be loaded.
 * @param threads Number of threads to use for loading (if applicable).
 * @param timer Reference to a timing object used for measuring performance.
 * @return A fully populated raw_file struct.
 */
raw_file load_file(std::string filename, int threads, Timer &timer);

/**
 * @brief Holds parsed file contents and associated structures.
 */
struct parsed_file
{
    /**
     * @brief Nested structure of parsed key/value pairs.
     * 
     * Each element represents:
     *   vector< pair< vector<keys>, vector<values> > >
     */
    std::vector<std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>>> key_value_pairs;

    /**
     * @brief A map of item IDs to their total weighted utility (TWU).
     */
    std::unordered_map<uint32_t, uint32_t> twu;
};

/**
 * @brief Enum representing different parsing states for parse_file().
 */
enum ParseState { 
    PARSE_ITEMS,      /**< Parsing item keys. */
    PARSE_TOTAL_UTILITY, /**< Parsing total utility field. */
    PARSE_UTILITIES   /**< Parsing per-item utilities. */
};

/**
 * @brief Parses a raw_file into a structured format (parsed_file).
 * @param file The raw_file to be parsed.
 * @param threads Number of threads to be used during parsing.
 * @param separator The character used to separate fields.
 * @param timer Reference to a timing object for performance measurement.
 * @return A parsed_file containing structured data.
 */
parsed_file parse_file(raw_file &file, int threads, const char separator, Timer &timer);


#include <fstream>
#include <iostream>
#include <string>

void write_patterns_to_file(const std::string& output_path, const uint32_t* patterns);