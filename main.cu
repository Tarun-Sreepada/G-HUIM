
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include "args.h"   // Include the args parser
#include "parser.h" // Include the file reader
#include "work.cuh" // Include the work queue
#include "memory.cuh"
#include "timer.h"
#include "patterns.cuh"
#include "mine.cuh"

#define KILO 1024ULL
#define MEGA (1024 * KILO)
#define GIGA (1024 * MEGA)

#define total_memory (4 * GIGA)

// make && ./cuDPHIM '/home/tarun/testing/test.txt' 5 \\s
// make && time ./cuDPHIM '/home/tarun/cuEFIM/datasets/accidents_utility_spmf.txt' 15000000 \\s

void flattenTransactions(const std::map<std::vector<int>, std::vector<int>> &filteredTransactions,
                         std::vector<Item> &items,
                         std::vector<int> &start,
                         std::vector<int> &end)
{
    // Precompute total items and reserve memory.
    size_t totalItems = 0;

    for (const auto &[key, val] : filteredTransactions)
    {
        totalItems += key.size();
    }
    items.reserve(totalItems);
    start.reserve(filteredTransactions.size());
    end.reserve(filteredTransactions.size());

    // Flatten the transactions.
    for (const auto &[key, val] : filteredTransactions)
    {
        start.push_back(items.size());
        size_t oldSize = items.size();
        items.resize(oldSize + key.size());
        // Instead of push_back, fill in the new range.
        for (size_t i = 0; i < key.size(); i++)
        {
            items[oldSize + i] = {key[i], val[i]};
        }
        end.push_back(items.size());
    }
}

int main(int argc, char *argv[])
{
    // Parse the command line arguments
    Args args = parse_args(argc, argv);

    Timer timer;
    timer.recordPoint("start");

    // Read the input file
    ReadFileResult read_result = read_file(args.file_path, args.delimiter, args.min_util, timer);


    // Flatten the transactions
    std::vector<Item> items;
    std::vector<int> start;
    std::vector<int> end;
    flattenTransactions(read_result.filteredTransactions, items, start, end);
    timer.recordPoint("flatten_transactions");

    // Stack
    AtomicWorkStack *stack;
    cudaMallocManaged(&stack, sizeof(AtomicWorkStack));
    stack->init_host();

    // Allocate memory for the database
    init_global_allocator(total_memory, 0);

    //Copy work
    WorkItem *work_item;
    cudaMallocManaged(&work_item, sizeof(WorkItem));
    work_item->pattern = nullptr;
    work_item->pattern_length = 0;
    work_item->work_count = read_result.primary.size();
    work_item->max_item = read_result.max_item;
    work_item->work_done = nullptr;

    Database *db;
    cudaMallocManaged(&db, sizeof(Database));
    db->numItems = items.size();
    db->numTransactions = start.size();
    db->d_data = nullptr;
    db->d_transactions = nullptr;

    // Allocate and copy item data.
    cudaMallocManaged(&db->d_data, items.size() * sizeof(Item));
    cudaMemcpy(db->d_data, items.data(), items.size() * sizeof(Item), cudaMemcpyHostToDevice);

    // Allocate and set up transactions.
    cudaMallocManaged(&db->d_transactions, start.size() * sizeof(Transaction));
    for (int i = 0; i < start.size(); i++)
    {
        db->d_transactions[i].data = db->d_data + start[i];
        db->d_transactions[i].utility = 0;
        db->d_transactions[i].length = end[i] - start[i];
    }
    work_item->db = db;

    // Allocate primary array in unified memory and copy host data.
    int *d_primary;
    cudaMallocManaged(&d_primary, read_result.primary.size() * sizeof(int));
    cudaMemcpy(d_primary, read_result.primary.data(), read_result.primary.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel.
    copy_work<<<1, args.block_size>>>(stack, work_item, d_primary, read_result.primary.size());
    cudaDeviceSynchronize();

    timer.recordPoint("copy_work");



    // Allocate memory for the patterns
    HighUtilityPatterns *patterns;
    cudaMallocManaged(&patterns, sizeof(HighUtilityPatterns));
    patterns->capacity = 128 * 1024 * 1024;
    patterns->d_patterns = nullptr;
    cudaMallocManaged(&patterns->d_patterns, patterns->capacity * sizeof(int32_t));
    memset(patterns->d_patterns, 0, patterns->capacity * sizeof(int32_t));
    patterns->d_patterns[0] = 0;
    patterns->d_patterns[1] = HEADER_SIZE;

    std::cout << "Active tasks: " << stack->active << std::endl;



    // Mine the patterns
    mine<<<1, 1>>>(stack, patterns, args.min_util);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "Error: " << cudaGetErrorString(error) << std::endl;
    }
    cudaDeviceSynchronize();
    std::cout << "Active tasks: " << stack->active << std::endl;


    // patterns->parse_patterns();
    std::map<std::string, int> pattern_map = patterns->parse_patterns();
    std::cout << "Number of patterns: " << pattern_map.size() << std::endl;

    timer.recordPoint("mine_patterns");




    timer.printRecords();

    return 0;
}