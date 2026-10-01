#include "app/cli.hpp"

#include <map>
#include <stdexcept>

namespace ghuim {

void configure_cli(CLI::App& app, Config& config)
{
    app.add_option("--input", config.input, "SPMF utility dataset")->required()->check(CLI::ExistingFile);
    app.add_option("--minutil", config.mining.minimum_utility, "Minimum itemset utility")
            ->required()
            ->check(CLI::PositiveNumber);
    app.add_option("--method", config.mining.method, "Work partitioning")
            ->transform(CLI::CheckedTransformer(
                    std::map<std::string, Method>{{"static", Method::Static},
                                                  {"adaptive", Method::Adaptive},
                                                  {"adaptive-optimized", Method::AdaptiveOptimized}},
                    CLI::ignore_case))
            ->default_str("adaptive");
    app.add_option("--block-size,--block_size", config.mining.block_size, "Threads per block-sized task")
            ->check(CLI::Range(32, 1024))
            ->capture_default_str();
    app.add_option_function<std::string>(
               "--memory",
               [&config](const std::string& value) {
                   try {
                       config.mining.memory_bytes = parse_size(value);
                   }
                   catch (const std::invalid_argument& error) {
                       throw CLI::ValidationError("--memory", error.what());
                   }
               },
               "Device memory for projected databases (K/M/G suffixes); default uses most free memory");
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

    app.callback([&config] {
        const uint32_t size = config.mining.block_size;
        if ((size & (size - 1)) != 0)
            throw CLI::ValidationError("--block-size", "must be a power of two (32-1024)");
    });
}

} // namespace ghuim
