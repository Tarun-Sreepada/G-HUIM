#include "app/logging.hpp"
#include "app/options.hpp"
#include "app/run.hpp"

#include <spdlog/spdlog.h>

#include <exception>

int main(int argc, char** argv) {
	Options options;
	int exit_code = 0;
	if (!parse_command_line(argc, argv, options, exit_code)) return exit_code;

	init_logging(options.log_level, options.log_file);
	try {
		run(options);
	} catch (const std::exception& e) {
		spdlog::error("{}", e.what());
		return 1;
	}
}
