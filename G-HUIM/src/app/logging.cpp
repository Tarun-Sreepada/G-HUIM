#include "app/logging.hpp"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace ghuim {

void initialize_logging(const Config& config)
{
    std::vector<spdlog::sink_ptr> sinks{std::make_shared<spdlog::sinks::stderr_color_sink_mt>()};
    if (!config.log_file.empty())
        sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(config.log_file, false));
    auto logger = std::make_shared<spdlog::logger>("ghuim", sinks.begin(), sinks.end());
    logger->set_pattern("[%l] %v");
    logger->set_level(spdlog::level::from_str(config.log_level));
    logger->flush_on(spdlog::level::err);
    spdlog::set_default_logger(std::move(logger));
}

} // namespace ghuim
