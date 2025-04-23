#pragma once

#include <cstdint>
#include <iomanip>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

class ArgParser
{
public:
    struct Argument
    {
        std::string name;                         // e.g. "--threads"
        std::string type;                         // e.g. "int", "bool"
        std::string description;                  // human-readable help text
        bool required;                            // must be provided?
        bool is_flag;                             // true for --help style flags
        std::optional<std::string> default_value; // default if not supplied
        std::optional<std::string> value;         // actual user-supplied value
    };

    explicit ArgParser(std::string prog_name);

    void add_argument(const std::string &name,
                      const std::string &type,
                      const std::string &desc,
                      bool required = false,
                      std::optional<std::string> default_val = std::nullopt);

    void add_flag(const std::string &name, const std::string &desc);

    void parse(int argc, char *argv[]);

    std::string get(const std::string &name);
    bool get_bool(const std::string &name);
    uint32_t get_uint32(const std::string &name);
    uint64_t get_uint64(const std::string &name);
    uint64_t get_memory(const std::string &name); // 10G / 512M / 128K
    char get_separator(const std::string &name);  // '\n', ',', …

    bool flag_set(const std::string &name);
    std::string get_type(const std::string &name) const;

    // ---------- misc ----------
    void print_help();
    void set(const std::string &name, const std::string &value);

private:
    std::unordered_map<std::string, Argument> args;
    std::vector<std::string> positional_args;
    std::string program_name;
};
