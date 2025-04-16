#include "args.hpp"
#include "timer.hpp"
#include "file/load.hpp"
#include "file/parse.hpp"
#include "file/build.hpp"
#include "file/save.hpp"
#include "memory/memory.cuh"
#include "patterns/patterns.cuh"
#include "mine/dynamic_parallelism.cuh"
#include "mine/static_parallelism.cuh"

int main(int argc, char *argv[])
{
#ifdef DEBUG
    std::cout << "DEBUG" << std::endl;
#endif

    ArgParser parser("cuDPEFIM");
    Timer timer;

    try
    {

        parser.add_argument("--input", "Input file path", true);
        parser.add_argument("--minutil", "Minimum utility", true);
        parser.add_argument("--output", "Output file path", false, "/dev/null");
        parser.add_argument("--managed", "Use managed memory", false, "false");
        parser.add_argument("--threads", "Number of threads used for file operations", false, "1");
        parser.add_argument("--separator", "Field separator (e.g., \\s, \\t, \\n, , ;)", false, "\\s");
        parser.add_argument("--block_size", "Number of threads per GPU block", false, "1024");
        parser.add_argument("--num_blocks", "Number of GPU blocks", false, "0");
        parser.add_argument("--total_memory", "Total memory to allocate for the global allocator", false, "36G");
        parser.add_argument("--patern_memory", "Memory to allocate for the patterns", false, "256M");
        parser.add_argument("--method", "Method to use for mining ({DP}: Dynamic Pattern, {SP}: Static Parallelization)", false, "SP");
        parser.parse(argc, argv);

        std::cout << "Input File: " << parser.get("--input") << "\n";
        std::cout << "Minimum utility: " << parser.get("--minutil") << "\n";
        std::cout << "Separator: '" << parser.get_separator("--separator") << "'\n";
        std::cout << "Output File: " << parser.get("--output") << "\n";
        std::cout << "Threads: " << parser.get_uint32("--threads") << "\n";
        std::cout << "Managed Memory: " << (parser.get_bool("--managed") ? "true" : "false") << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    timer.recordPoint("Start");

    cudaFree(0);
    timer.recordPoint("CUDA Warmup");

    // Load File
    raw_file rf = load_file(parser.get("--input"), parser.get_uint32("--threads"));
    timer.recordPoint("Read");

    // Parse File
    parsed_file pf = parse_file(rf, parser.get_uint32("--threads"), parser.get_separator("--separator"));
    timer.recordPoint("Parse");

    // Build file
    built_file bf = build_file(pf, parser.get_uint32("--threads"), parser.get_uint64("--minutil"));
    timer.recordPoint("Build");

    // Initialize memory for GPU mining
    init_global_allocator(parser.get_memory("--total_memory"), parser.get_bool("--managed"));
    timer.recordPoint("Allocator Initialized");

    // Allocate memory for patterns
    patterns *p;
    gpuErrchk(cudaMallocManaged(&p, sizeof(patterns)));
    p->init(parser.get_memory("--patern_memory"));
    timer.recordPoint("Patterns Allocated");

    if (parser.get("--method") == "SP")
    {
        static_parallel_mining(bf, p, parser);
    }
    else if (parser.get("--method") == "DP")
    {
        dynamic_parallel_mining(bf, p, parser);
    }
    else
    {
        std::cerr << "Invalid method: " << parser.get("--method") << "\n";
        return 1;
    }

    timer.recordPoint("Mining Completed");

#ifdef DEBUG
    std::cout << "Saving to file: " << parser.get("--output") << "\n";
#endif

    save_patterns(parser.get("--output"), p->data, p->offset);
    timer.recordPoint("Save");

    // free allocator
    free_global_allocator();
    p->free();

    timer.recordPoint("End");
    timer.printRecords();

    return 0;
}