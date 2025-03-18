
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include "args.h"
#include "file.cuh"
#include "timer.h"

#define KILO 1024ULL
#define MEGA KILO *KILO
#define GIGA KILO *MEGA

#define total_memory (5 * GIGA)

// make && ./cuEFIM '/home/tarun/testing/test.txt' 5 \\s
// make && ./cuDPEFIM --input ../accidents_utility_spmf.txt --minutil 1500000

// make && ./cuDPEFIM --input '/home/tarun/cuDPEFIM/test.txt' --minutil 5


int main(int argc, char *argv[])
{
    #ifdef DEBUG
    std::cout << "DEBUG MODE" << std::endl;
    #endif

    ArgParser parser("cuEFIM");
    Timer timer;

    try {

        parser.add_argument("--input", "Input file path", true);
        parser.add_argument("--minutil", "Minimum utility", true);
        parser.add_argument("--output", "Output file path", false, "output.txt");
        parser.add_argument("--separator", "Field separator (e.g., \\s, \\t, \\n, , ;)", false, "\\s");

        parser.parse(argc, argv);

        std::cout << "Input File: " << parser.get("--input") << "\n";
        std::cout << "Minimum utility: " << parser.get("--minutil") << "\n";
        std::cout << "Separator: '" << parser.get_separator("--separator") << "'\n";

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    timer.recordPoint("Start");


    // load file
    FileBuffer *fd = load_file(parser.get("--input"), timer);

    // get number of transactions and total item count
    file_parse(fd, parser.get_separator("--separator"), parser.get_uint32("--minutil"), timer);


    timer.recordPoint("End");
    timer.printRecords();

    return 0;
}