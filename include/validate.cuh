#pragma once
#include "args.hpp"

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

void validate_block_size(ArgParser &parser)
{
    // validate the number of threads per block else set to 32
    if (parser.get_uint32("--block_size") < 1 || parser.get_uint32("--block_size") > 8192)
    {
        std::cout << "Number of threads per block is invalid. Setting threads per block to 512.\n";
        parser.set("--block_size", "512");
    }
}

void validate_num_blocks(ArgParser &parser)
{
    // Get the number of SMs on the GPU
    int sm_count = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    sm_count += prop.multiProcessorCount;

    // validate the number of blocks per GPU else set to 0
    if (parser.get_uint32("--num_blocks") == 0 || parser.get_uint32("--num_blocks") > sm_count)
    {
        std::cout << "Number of blocks per GPU is invalid. Setting blocks per GPU to SM count.\n";

        parser.set("--num_blocks", std::to_string(sm_count));

        std::cout << "Number of blocks per GPU set to: " << sm_count << "\n";
    }
}
