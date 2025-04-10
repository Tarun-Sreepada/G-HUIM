#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <queue>
#include <functional>
#include "args.hpp"
#include "file.hpp"
#include "build.hpp"
#include "timer.hpp"
#include "memory.cuh"
#include "stack.cuh"
#include "copy.cuh"
#include "validate.cuh"
#include "mine.cuh"

#define KILO 1024ULL
#define MEGA (KILO * KILO)
#define GIGA (KILO * MEGA)

#define pattern_memory (256 * MEGA)

uint32_t next_pow2(uint32_t n)
{
    if (n == 0)
        return 1;
    uint32_t power = 1;
    while (power < n)
        power <<= 1;
    return power;
}

int main(int argc, char *argv[])
{
#ifdef DEBUG
    std::cout << "DEBUG MODE" << std::endl;
#endif

    ArgParser parser("cuEFIM");
    Timer timer;

    try
    {

        parser.add_argument("--input", "Input file path", true);
        parser.add_argument("--minutil", "Minimum utility", true);
        parser.add_argument("--output", "Output file path", false, "cuDPEFIM.txt");
        parser.add_argument("--managed", "Use managed memory", false, "false");
        parser.add_argument("--threads", "Number of threads to use to read the file", false, "1");
        parser.add_argument("--separator", "Field separator (e.g., \\s, \\t, \\n, , ;)", false, "\\s");
        parser.add_argument("--block_size", "Number of threads per GPU block", false, "1024");
        parser.add_argument("--total_memory", "Total memory to allocate for the global allocator", false, "36G");

        parser.parse(argc, argv);

        std::cout << "Input File: " << parser.get("--input") << "\n";
        std::cout << "Minimum utility: " << parser.get("--minutil") << "\n";
        std::cout << "Separator: '" << parser.get_separator("--separator") << "'\n";
        std::cout << "Output File: " << parser.get("--output") << "\n";
        std::cout << "Threads: " << parser.get_uint32("--threads") << "\n";
        std::cout << "Threads per block: " << parser.get_uint32("--block_size") << "\n";
        std::cout << "Managed Memory: " << (parser.get_bool("--managed") ? "true" : "false") << "\n";

        validate_threads(parser);
        validate_block_size(parser);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    timer.recordPoint("Start");

    // Load file into memory
    raw_file rf = load_file(parser.get("--input"), parser.get_uint32("--threads"), timer);

    // Parse the input file
    parsed_file pf = parse_file(rf, parser.get_uint32("--threads"), parser.get_separator("--separator"), timer);

    // free the raw file
    free(rf.data);

    // build the transaction database
    built_file bf = build_file(pf, parser.get_uint32("--threads"), timer, parser.get_uint64("--minutil"));

    // initialize the global allocator
    init_global_allocator(parser.get_memory("--total_memory"), parser.get_bool("--managed"));
    timer.recordPoint("Allocator Initialized");

    // allocate stack
    stack *curr_stack;
    gpuErrchk(cudaMallocManaged(&curr_stack, sizeof(stack)));
    curr_stack->init();
    timer.recordPoint("Stack Initialized");

    // allocate the stack
    stack *new_stack;
    gpuErrchk(cudaMallocManaged(&new_stack, sizeof(stack)));
    new_stack->init();
    timer.recordPoint("New Stack Initialized");

    // Send data to GPU
    auto [db, primary] = initialize_copy(bf, curr_stack, timer);
    timer.recordPoint("Data Copied to GPU");

    // allocate memory for patterns
    uint32_t *patterns;
    gpuErrchk(cudaMallocManaged(&patterns, pattern_memory));
    gpuErrchk(cudaMemset(patterns, 0, pattern_memory));
    patterns[1] = 2; // first 2 are reserved for the pattern length and the number of patterns
    timer.recordPoint("Patterns Memory Allocated");

// Run the algorithm
#ifdef DEBUG

    std::cout << "Number of workitems: " << s->may_pop() << "\n";
    std::cout << "Threads per block: " << parser.get_uint32("--block_size") << "\n";
    std::cout << "Blocks per GPU: " << parser.get_uint32("--num_blocks") << "\n";
    std::cout << "Max Item ID: " << bf.itemID_to_item.size() << "\n";

#endif

    // mine<<<parser.get_uint32("--num_blocks"), parser.get_uint32("--block_size"), shared_memory_usage>>>(s, patterns, (uint32_t)parser.get_uint32("--minutil"), bf.itemID_to_item.size() + 1, bf.transactions.size() + 1);

    while (curr_stack->top != 0)
    {
        printf("Mining %d\n", curr_stack->top);
        uint32_t thread_count = parser.get_uint32("--block_size");
        // its either block size or next pow2 if average is too small
        uint32_t avg = curr_stack->total_transactions / curr_stack->top;
        // printf("Avg: %d\n", avg);
        // printf("Unique Items: %d\n", curr_stack->unique_items);
        if (avg / 2 < thread_count)
        {
            thread_count = next_pow2(avg);
        }
        if (thread_count < 32)
        {
            thread_count = 32;
        }

        mine<<<curr_stack->top, thread_count>>>(curr_stack, new_stack, patterns, (uint32_t)parser.get_uint32("--minutil"));

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // swap the stacks
        stack *temp = curr_stack;
        curr_stack = new_stack;
        new_stack = temp;
        new_stack->init();
    }

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    timer.recordPoint("Mining Completed");


    // Write the results to a file
    std::ofstream output_file(parser.get("--output"));
    if (!output_file.is_open())
    {
        std::cerr << "Error: Could not open output file: " << parser.get("--output") << "\n";
        return 1;
    }

    int32_t pattern_count = -2;
    for (uint32_t i = 0; i < patterns[1]; ++i)
    {
        // output_file << patterns[i + 2] << " ";
        if (patterns[i + 2] != 0)
        {
            output_file << patterns[i + 2] << " ";
        }
        else
        {
            output_file << "\n";
            pattern_count++;
        }
    }
    std::cout << "Number of patterns: " << pattern_count << "\n";

    // free allocator
    free_global_allocator();

    timer.recordPoint("End");
    timer.printRecords();


    return 0;
}