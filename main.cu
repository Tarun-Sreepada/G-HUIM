
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

#define KILO 1024ULL
#define MEGA (KILO * KILO)
#define GIGA (KILO * MEGA)

#define total_memory (16 * GIGA)

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

        parser.parse(argc, argv);

        std::cout << "Input File: " << parser.get("--input") << "\n";
        std::cout << "Minimum utility: " << parser.get("--minutil") << "\n";
        std::cout << "Separator: '" << parser.get_separator("--separator") << "'\n";
        std::cout << "Output File: " << parser.get("--output") << "\n";
        std::cout << "Threads: " << parser.get_uint32("--threads") << "\n";

        validate_threads(parser);
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

    // Send data to GPU

    // Run the algorithm

    // Get the results

    // free allocator
    free_global_allocator();

    timer.recordPoint("End");
    timer.printRecords();

    return 0;
}