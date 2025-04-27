#include "args.hpp"
#include "timer.hpp"
#include "file/load.hpp"
#include "file/parse.cuh"
// #include "file/parse.hpp"
// #include "file/build.hpp"
#include "file/build.cuh"
#include "file/save.hpp"
#include "memory/memory.cuh"
#include "patterns/patterns.cuh"
#include "mine/dynamic_parallelism.cuh"
#include "mine/static_parallelism.cuh"



__global__ void __warmup() { }

// calcTWU<<<pf.db->num_transactions, 1024>>>(pf.db, item_map);
// cudaDeviceSynchronize();

__global__ void calcTWU(Database *db, Item_mapping *item_map)
{
    uint32_t transacion_id = blockIdx.x;

    __shared__ Transaction trans;
    __shared__ uint32_t sum;

    uint32_t thread_id = threadIdx.x;

    if (threadIdx.x == 0)
    {
        trans = db->transactions[transacion_id];
        sum = trans.utility;
    }
    __syncthreads();


    for (int i = thread_id; i < trans.length; i += blockDim.x)
    {
        Item item = trans.data[i];
        atomicAdd(&item_map[item.key].local_utility, sum);
    }
}



int main(int argc, char *argv[])
{
#ifdef DEBUG
    std::cout << "DEBUG" << std::endl;
#endif

    ArgParser parser("cuDPEFIM");
    Timer timer;

    try
    {

        parser.add_argument("--input", "string", "Input file path", true);
        parser.add_argument("--minutil", "int", "Minimum utility", true);
        parser.add_argument("--output", "string", "Output file path", false, "/dev/null");
        parser.add_argument("--managed", "bool", "Use managed memory", false, "false");
        parser.add_argument("--separator", "string", "Field separator (e.g., \\s, \\t, \\n, , ;)", false, "\\s");
        parser.add_argument("--block_size", "int", "Number of threads per GPU block", false, "1024");
        parser.add_argument("--num_blocks", "int", "Number of GPU blocks", false, "0");
        parser.add_argument("--total_memory", "int+str", "Total memory to allocate for the global allocator", false, "36G");
        parser.add_argument("--patern_memory", "int+str", "Memory to allocate for the patterns", false, "256M");
        parser.add_argument("--adaptive", "bool", "Use adaptive thread block sizes", false, "true");
        parser.add_argument("--method", "string", "Method to use for mining ({DP}: Dynamic Pattern, {SP}: Static Parallelization)", false, "SP");
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
    // Warmup GPU
    __warmup<<<1, 1>>>();
    cudaDeviceSynchronize();
    timer.recordPoint("CUDA Warmup");

    // Parse File
    // Parsed pf = gpuParse(parser.get("--input"), parser.get_uint32("--threads"), parser.get_separator("--separator"));
    Parsed pf = gpuParse(parser.get("--input"), parser.get_separator("--separator"));
    timer.recordPoint("Parse");


    // calculate TWU
    Item_mapping *item_map;
    cudaMallocManaged(&item_map, pf.max_item * sizeof(Item_mapping));
    cudaMemset(item_map, 0, pf.max_item * sizeof(Item_mapping));
    // set the ids
    for (uint32_t i = 0; i < pf.max_item; i++)
    {
        item_map[i].item = i;
    }


    uint32_t block_size = pf.db->num_items / pf.db->num_transactions;
    // max is 1024 else nextpow2    
    block_size = std::min(block_size, 1024u);
    block_size = next_pow2(block_size);


    calcTWU<<<pf.db->num_transactions, block_size>>>(pf.db, item_map);
    cudaDeviceSynchronize();
    timer.recordPoint("CalcTWU");


    // Build file
    built_file bf = build(pf, parser.get_uint64("--minutil"), parser, item_map);
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

    std::cout << "Number of patterns: " << p->pattern_count << "\n";

    // Free resources

    return 0;
}