#pragma once

#include "app/config.hpp"

#include <CLI/CLI.hpp>

namespace ghuim {
void configure_cli(CLI::App& app, Config& config);
} // namespace ghuim
