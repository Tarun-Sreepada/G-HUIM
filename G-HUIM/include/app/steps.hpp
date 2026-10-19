#pragma once
// The stages of main(), in the order it runs them. Each is implemented in src/app/steps/.

#include "app/config.hpp"
#include "data/dataset.hpp"
#include "mine/result.hpp"

#include <chrono>
#include <future>

namespace ghuim {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start);

// Parses argv into `config`. Returns false when the process should exit with `exit_code` (--help, bad usage).
bool parse_command_line(int argc, char* argv[], Config& config, int& exit_code);
// Creates the CUDA context on another thread; the future holds the seconds it took.
std::future<double> start_cuda_context();
// Loads the dataset and logs what it holds. `start` is when the process began its work.
Dataset load_data(const Config& config, Clock::time_point start);
// Waits for the context and returns the seconds this thread waited.
double wait_for_context(std::future<double>& context);
// Mines `data` and logs the search summary.
MiningResult mine_data(const Dataset& data, const Config& config);
// Logs the pattern count and timings. The wait for the CUDA context is left out of "Total time".
void report_totals(const MiningResult& result, Clock::time_point start, double context_wait_s);

} // namespace ghuim
