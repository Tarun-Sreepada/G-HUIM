#include "app/logging.hpp"
#include "app/steps.hpp"

#include <spdlog/spdlog.h>

int main(int argc, char* argv[])
{
    ghuim::Config config;
    int           exit_code = 0;
    if (!ghuim::parse_command_line(argc, argv, config, exit_code))
        return exit_code;

    try {
        ghuim::initialize_logging(config);

        const auto start   = ghuim::Clock::now();
        auto       context = ghuim::start_cuda_context();
        const auto data    = ghuim::load_data(config, start);
        const auto wait_s  = ghuim::wait_for_context(context);
        const auto result  = ghuim::mine_data(data, config);
        ghuim::report_totals(result, start, wait_s);
        return 0;
    }
    catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return 1;
    }
}
