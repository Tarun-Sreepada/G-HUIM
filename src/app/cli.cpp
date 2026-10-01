#include "app/cli.hpp"

#include <stdexcept>

namespace ghuim {
namespace {

// A byte/count option with K/M/G suffixes.
void add_size_option(CLI::App& app, const std::string& name, uint64_t& target, const std::string& description)
{
    app.add_option_function<std::string>(
            name,
            [&target, name](const std::string& value) {
                try {
                    target = parse_size(value);
                }
                catch (const std::invalid_argument& error) {
                    throw CLI::ValidationError(name, error.what());
                }
            },
            description);
}

} // namespace

void configure_cli(CLI::App& app, Config& config)
{
    app.add_option("--input", config.input, "SPMF utility dataset")->required()->check(CLI::ExistingFile);
    app.add_option("--minutil", config.mining.minimum_utility, "Minimum itemset utility")
            ->required()
            ->check(CLI::PositiveNumber);
    add_size_option(app, "--total-memory,--total_memory", config.mining.memory_bytes,
                    "Device memory for projected databases (K/M/G suffixes); default uses most free memory");
    add_size_option(app, "--stack-capacity,--stack_capacity", config.mining.stack_capacity,
                    "Most tasks one level may hold (K/M/G suffixes)");
    app.add_option("--output", config.mining.output, "Write the itemsets (\"items... UTIL u\") to this file");
    app.add_option_function<std::string>(
               "--separator",
               [&config](const std::string& value) {
                   if (value == "\\s" || value == "space")
                       config.separator = ' ';
                   else if (value == "\\t" || value == "tab")
                       config.separator = '\t';
                   else if (value.size() == 1 && value[0] != '\n' && value[0] != '\r' && value[0] != ':')
                       config.separator = value[0];
                   else
                       throw CLI::ValidationError("--separator", "use one character, space, or tab");
               },
               "Separator between items within each SPMF field")
            ->default_str("space");
    app.add_option("--log-level", config.log_level, "Logging verbosity")
            ->check(CLI::IsMember({"trace", "debug", "info", "warn", "error", "critical", "off"}))
            ->capture_default_str();
    app.add_option("--log-file", config.log_file, "Also write diagnostics to a log file");
}

} // namespace ghuim
