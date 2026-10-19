#include "app/steps.hpp"

#include "app/cli.hpp"

namespace ghuim {

bool parse_command_line(int argc, char* argv[], Config& config, int& exit_code)
{
    CLI::App app{"GPU-accelerated high-utility itemset mining (EFIM)", "G-HUIM"};
    configure_cli(app, config);
    try {
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError& error) {
        exit_code = app.exit(error);
        return false;
    }
    return true;
}

} // namespace ghuim
