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
        std::string                name;          // e.g. "--threads"
        std::string                type;          // e.g. "int", "bool"
        std::string                description;   // human-readable help text
        bool                       required;      // must be provided?
        bool                       is_flag;       // true for --help style flags
        std::optional<std::string> default_value; // default if not supplied
        std::optional<std::string> value;         // actual user-supplied value
    };

  // Creates an argument parser scoped to the given program name.
  explicit ArgParser(std::string prog_name);

  // Registers a positional or value-taking argument with optional default.
    void add_argument(const std::string& name, const std::string& type, const std::string& desc, bool required = false,
                      std::optional<std::string> default_val = std::nullopt);

  // Registers a boolean flag that acts as a presence toggle.
    void add_flag(const std::string& name, const std::string& desc);

  // Parses the provided argc/argv array and stores argument values.
    void parse(int argc, char* argv[]);

  // Retrieves a value argument as a string.
    std::string get(const std::string& name);
  // Retrieves an argument interpreted as a boolean.
    bool        get_bool(const std::string& name);
  // Retrieves an argument interpreted as a 32-bit unsigned integer.
    uint32_t    get_uint32(const std::string& name);
  // Retrieves an argument interpreted as a 64-bit unsigned integer.
    uint64_t    get_uint64(const std::string& name);
  // Parses a human-readable memory size (e.g. 512M) into bytes.
    uint64_t    get_memory(const std::string& name);    // 10G / 512M / 128K
  // Retrieves a separator character from a one-character argument.
    char        get_separator(const std::string& name); // '\n', ',', …

  // Returns true when a flag was provided on the command line.
    bool        flag_set(const std::string& name);
  // Exposes the declared type for an argument.
    std::string get_type(const std::string& name) const;

  // Prints the usage summary with details for each argument.
    void print_help();
  // Overrides or injects an argument value programmatically.
    void set(const std::string& name, const std::string& value);

  private:
    std::unordered_map<std::string, Argument> args;
    std::vector<std::string>                  positional_args;
    std::string                               program_name;
};
