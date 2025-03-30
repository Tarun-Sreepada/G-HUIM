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

#define total_memory (24 * GIGA)
#define pattern_memory (256 * MEGA)


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
        parser.add_argument("--output", "Output file path", false, "output.txt");
        parser.add_argument("--managed", "Use managed memory", false, "false");
        parser.add_argument("--threads", "Number of threads to use to read the file", false, "1");
        parser.add_argument("--separator", "Field separator (e.g., \\s, \\t, \\n, , ;)", false, "\\s");
        parser.add_argument("--tb_size", "Number of threads per GPU block", false, "32");
        parser.add_argument("--block_size", "Number of blocks per GPU (Default: SM count)", false, "0");

        parser.parse(argc, argv);

        std::cout << "Input File: " << parser.get("--input") << "\n";
        std::cout << "Minimum utility: " << parser.get("--minutil") << "\n";
        std::cout << "Separator: '" << parser.get_separator("--separator") << "'\n";
        std::cout << "Output File: " << parser.get("--output") << "\n";
        std::cout << "Threads: " << parser.get_uint32("--threads") << "\n";

        validate_threads(parser);
        validate_tb_size(parser);
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
    init_global_allocator(total_memory, parser.get_bool("--managed"));
    timer.recordPoint("Allocator Initialized");

    // allocate stack
    stack *s;
    gpuErrchk(cudaMallocManaged(&s, sizeof(stack)));
    s->init();
    timer.recordPoint("Stack Initialized");

    // Send data to GPU
    auto [db, primary] = initialize_copy(bf, s);
    timer.recordPoint("Prepared Data for GPU");

    copy_work<<<1, 32>>>(s, db, primary, bf.primary.size(), bf.itemID_to_item.size());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    timer.recordPoint("Data Copied to GPU");

    // allocate memory for patterns
    uint32_t *patterns;
    gpuErrchk(cudaMallocManaged(&patterns, pattern_memory));
    gpuErrchk(cudaMemset(patterns, 0, pattern_memory));

// Run the algorithm
#ifdef DEBUG
    std::cout << "Number of workitems: " << s->may_pop() << "\n";
    std::cout << "Threads per block: " << parser.get_uint32("--tb_size") << "\n";
    std::cout << "Blocks per GPU: " << parser.get_uint32("--block_size") << "\n";
#endif

    mine<<<parser.get_uint32("--block_size"), parser.get_uint32("--tb_size")>>>(s, patterns, (uint32_t)parser.get_uint32("--minutil"));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Get the results

    // free allocator
    free_global_allocator();

    timer.recordPoint("End");
    timer.printRecords();

    return 0;
}