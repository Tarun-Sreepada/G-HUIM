#pragma once
#include "args.hpp"

/*
 * @file validate.cuh
 * @brief Validation functions for command line arguments.
 *
 * This file contains functions to validate and set default values for command line arguments
 * related to the number of threads, block size, and number of blocks per GPU.
 */

/**
 * @brief Validates the number of threads.
 */
void validate_threads(ArgParser &parser)
{
// print maximum CPU threads
#ifdef DEBUG
    std::cout << "Maximum CPU threads: " << std::thread::hardware_concurrency() << "\n";
#endif
    // validate the number of threads else set to maximum
    if (parser.get_uint32("--threads") > std::thread::hardware_concurrency())
    {
        std::cout << "Number of threads is greater than maximum CPU threads. Setting threads to maximum CPU threads.\n";
        parser.set("--threads", std::to_string(std::thread::hardware_concurrency()));
    }
}

/**
 * @brief Validates the block size.
 */
void validate_block_size(ArgParser &parser)
{
    // validate the number of threads per block else set to 32
    if (parser.get_uint32("--block_size") == 0 || parser.get_uint32("--block_size") > 1024)
    {
        std::cout << "Number of threads per block is invalid. Setting threads per block to 512.\n";
        parser.set("--block_size", "1024");
    }
}


