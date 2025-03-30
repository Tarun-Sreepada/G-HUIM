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


void validate_tb_size(ArgParser &parser)
{
    // validate the number of threads per block else set to 32
    if (parser.get_uint32("--tb_size") < 1 || parser.get_uint32("--tb_size") > 1024)
    {
        std::cout << "Number of threads per block is invalid. Setting threads per block to 512.\n";
        parser.set("--tb_size", "512");
    }
}

void validate_block_size(ArgParser &parser)
{
    // validate the number of blocks per GPU else set to 0
    if (parser.get_uint32("--block_size") < 1 || parser.get_uint32("--block_size") > 65535)
    {
        std::cout << "Number of blocks per GPU is invalid. Setting blocks per GPU to SM count.\n";

        // Get the number of SMs on the GPU
        int device_count;
        cudaGetDeviceCount(&device_count);
        int sm_count = 0;
        for (int i = 0; i < device_count; i++)
        {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            sm_count += prop.multiProcessorCount;
        }
        parser.set("--block_size", std::to_string(sm_count));
        std::cout << "Number of blocks per GPU set to: " << sm_count << "\n";
    }
}
 