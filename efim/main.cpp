#include "efim.hpp"

#include <iostream>
#include <string>

// "tab" and "space" are names for separators that are awkward to type in a shell.
static std::string parse_sep(const std::string& arg) {
	if (arg == "tab") return "\t";
	if (arg == "space") return " ";
	return arg;
}

int main(int argc, char** argv) {
	if (argc < 4 || argc > 6) {
		std::cerr << "Usage: " << argv[0] << " <inputFile> <outputFile> <minUtil> [sep] [threads]\n"
				  << "  sep: tab (default), space, or any literal separator like ; or ,\n";
		return 1;
	}
	try {
		const std::string sep = parse_sep(argc >= 5 ? argv[4] : "tab");
		const int threads = argc >= 6 ? std::stoi(argv[5]) : 1;
		Efim miner(argv[1], std::stoll(argv[3]), sep, threads);
		if (miner.threads() != threads)
			std::cerr << "Note: threads " << threads << " capped to " << miner.threads()
					  << " (allowed range 1-" << Efim::max_threads() << ")\n";
		miner.mine();
		miner.save(argv[2]);
		miner.print_results();
	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << '\n';
		return 1;
	}
}
