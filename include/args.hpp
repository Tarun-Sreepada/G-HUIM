#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <optional>
#include <iomanip>

class ArgParser
{
public:
    struct Argument
    {
        std::string name;
        std::string description;
        bool required;
        bool is_flag;
        std::optional<std::string> default_value;
        std::optional<std::string> value;
    };

    explicit ArgParser(std::string prog_name);

    void add_argument(const std::string &name,
                      const std::string &desc,
                      bool required = false,
                      std::optional<std::string> default_val = std::nullopt);

    void add_flag(const std::string &name, const std::string &desc);

    void parse(int argc, char *argv[]);

    std::string get(const std::string &name);

    bool get_bool(const std::string &name);

    uint32_t get_uint32(const std::string &name);

    uint64_t get_uint64(const std::string &name);

    uint64_t get_memory(const std::string &name);

    char get_separator(const std::string &name);

    bool flag_set(const std::string &name);

    void print_help();

    void set(const std::string &name, const std::string &value);

private:
    std::unordered_map<std::string, Argument> args;
    std::vector<std::string> positional_args;
    std::string program_name;
};
