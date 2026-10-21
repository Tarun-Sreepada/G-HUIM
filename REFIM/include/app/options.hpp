#pragma once
// Command-line options of refim.

#include <string>

struct Options {
	std::string input, output, sep = "space", log_level = "info", log_file, numa = "on", simd = "auto", thp = "auto";
	std::string compact = "on", leaf = "on";
	long long min_util = 0;
	int threads = 1, refine = -1;
	int key_bits = 0, util_bits = 0, occ_bits = 0;  // 0 = the narrowest that holds the dataset
	bool stats = false;
	double refine_min_drop = 0;
};

// Parses argv (both the option form and the legacy `<input> <output> <minUtil> [sep] [threads]`).
// Returns false when the process should exit with `exit_code` (--help, bad usage).
bool parse_command_line(int argc, char** argv, Options& options, int& exit_code);
