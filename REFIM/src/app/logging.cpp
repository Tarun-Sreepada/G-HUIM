#include "app/logging.hpp"

#include <memory>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <vector>

void init_logging(const std::string& level, const std::string& file) {
	std::vector<spdlog::sink_ptr> sinks{std::make_shared<spdlog::sinks::stderr_color_sink_mt>()};
	if (!file.empty()) sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(file, false));
	auto logger = std::make_shared<spdlog::logger>("refim", sinks.begin(), sinks.end());
	logger->set_pattern("[%l] %v");
	logger->set_level(spdlog::level::from_str(level));
	logger->flush_on(spdlog::level::err);
	spdlog::set_default_logger(std::move(logger));
}
