#include "args.hpp"
#include "file/build.cuh"
#include "file/parse.cuh"
#include "file/save.cuh"
#include "memory/memory.cuh"
#include "mine/static_parallelism.cuh"
#include "patterns/patterns.cuh"
#include "timer.hpp"

#include <limits>
#include <stdexcept>

// Triggers a minimal kernel launch to initialize the CUDA context.
__global__ void __warmup() {}

// Queries the device for maximum shared memory per block.
size_t get_shared_memory_per_block()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Assuming device 0
    return prop.sharedMemPerBlock;
}

// Computes transaction-weighted utility per item across the database.
__global__ void calcTWU(Database* db, Item_mapping* item_map)
{
    uint32_t transacion_id = blockIdx.x;

    __shared__ Transaction trans;
    __shared__ uint32_t    sum;

    uint32_t thread_id = threadIdx.x;

    if (threadIdx.x == 0) {
        trans = db->transactions[transacion_id];
        sum   = trans.utility;
    }
    __syncthreads();

    for (int i = thread_id; i < trans.length; i += blockDim.x) {
        Item item = trans.data[i];
        atomicAdd(&item_map[item.key].local_utility, sum);
    }
}

// Prints the first portion of the pattern buffer for debugging.
__global__ void print_pattern(patterns* p)
{
    for (uint32_t i = 0; i < 100; ++i) {
        printf("%u ", p->data[i]);
    }
    printf("\n");
}

// Entry point configuring the mining pipeline and launching kernels.
int main(int argc, char* argv[])
{
#ifdef DEBUG
    std::cout << "DEBUG" << std::endl;
#endif

    cudaFree(0);
    __warmup<<<1, 1>>>();
    cudaDeviceSynchronize();


    ArgParser parser("cuDPEFIM");
    Timer     timer;
    uint32_t  stack_capacity = kDefaultStackCapacity;

    try {
        parser.add_argument("--input", "string", "Input file path", true);
        parser.add_argument("--minutil", "int", "Minimum utility", true);
        parser.add_argument("--output", "string", "Output file path", false, "/dev/null");
        parser.add_argument("--managed", "bool", "Use managed memory", false, "false");
        parser.add_argument("--separator", "string", "Field separator (e.g., \\s, \\t, \\n, , ;)", false, "\\s");
        parser.add_argument("--block_size", "int", "Number of threads per GPU block", false, "1024");
        parser.add_argument("--total_memory", "int+str", "Total memory to allocate for the global allocator", false,
                            "36G");
        parser.add_argument("--pattern_memory", "int+str", "Memory to allocate for the patterns", false, "2G");
        parser.add_argument("--stack_capacity", "int+str",
                            "Maximum number of tasks per stack (e.g., 128M)", false, "128M");
        parser.add_argument("--adaptive", "bool", "Use adaptive thread block sizes", false, "true");
        parser.parse(argc, argv);

        const uint64_t stack_capacity_value = parser.get_scaled_uint("--stack_capacity");
        if (stack_capacity_value == 0 || stack_capacity_value > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("--stack_capacity must fit in 32 bits and be greater than zero");
        stack_capacity = static_cast<uint32_t>(stack_capacity_value);

        std::cout << "Input File: " << parser.get("--input") << "\n";
        std::cout << "Minimum utility: " << parser.get("--minutil") << "\n";
        std::cout << "Separator: '" << parser.get_separator("--separator") << "'\n";
        std::cout << "Output File: " << parser.get("--output") << "\n";
        std::cout << "Managed Memory: " << (parser.get_bool("--managed") ? "true" : "false") << "\n";
        std::cout << "Adaptive: " << (parser.get_bool("--adaptive") ? "true" : "false") << "\n";
        std::cout << "Pattern Memory: " << parser.get("--pattern_memory") << "\n";
        std::cout << "Total Memory: " << parser.get("--total_memory") << "\n";
        std::cout << "Block Size: " << parser.get("--block_size") << "\n";
        std::cout << "Task Stack Capacity: " << parser.get("--stack_capacity") << " (" << stack_capacity
                  << " tasks)\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    timer.recordPoint("Start");

    std::ifstream infile(parser.get("--input"));
    if (!infile.good()) {
        std::cerr << "Error: Input file does not exist or is not a file: " << parser.get("--input") << "\n";
        return 1;
    }
    infile.close();

    Parsed pf = gpuParse(parser.get("--input"), parser.get_separator("--separator"));
    timer.recordPoint("Parse");

    Item_mapping* item_map;
    cudaMallocManaged(&item_map, pf.max_item * sizeof(Item_mapping));
    cudaMemset(item_map, 0, pf.max_item * sizeof(Item_mapping));
    for (uint32_t i = 0; i < pf.max_item; i++) {
        item_map[i].item = i;
    }

    uint32_t block_size = pf.db->num_items / pf.db->num_transactions;
    block_size = std::min(block_size, 1024u);
    block_size = next_pow2(block_size);

    calcTWU<<<pf.db->num_transactions, block_size>>>(pf.db, item_map);
    cudaDeviceSynchronize();
    timer.recordPoint("CalcTWU");

    built_file bf = build(pf, parser.get_uint64("--minutil"), parser, item_map);
    timer.recordPoint("Build");

    init_global_allocator(parser.get_memory("--total_memory"), parser.get_bool("--managed"));
    timer.recordPoint("Allocator Initialized");

    patterns* p;
    gpuErrchk(cudaMallocManaged(&p, sizeof(patterns)));
    std::cout << "Allocating memory for patterns: " << parser.get_memory("--pattern_memory") << " B\n";
    p->init(parser.get_memory("--pattern_memory"));

    std::pair<uint32_t *, uint64_t> all_patterns;
    cudaMallocManaged(&all_patterns.first, sizeof(uint32_t) * 32 * GIGA);
    all_patterns.second = 0;

    timer.recordPoint("Patterns Allocated");

    size_t shared_memory = get_shared_memory_per_block();

    static_parallel_mining(bf, p, parser, timer, shared_memory, all_patterns, stack_capacity);

    timer.recordPoint("End");

    timer.printRecords();
    std::cout << "Number of patterns: " << p->pattern_count << "\n";

#ifdef DEBUG
    std::cout << "Saving to file: " << parser.get("--output") << "\n";
#endif

    save_patterns(parser.get("--output"), all_patterns, item_map, pf.max_item);

    p->free();
    cudaFree(p);

    free_global_allocator();
    cudaFree(all_patterns.first);

    cudaFree(item_map);

    return 0;
}