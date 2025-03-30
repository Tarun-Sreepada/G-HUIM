/**
 * @file args.hpp
 * @brief Lightweight C++ command-line argument parser.
 *
 * This file declares the ArgParser class, which supports defining and parsing
 * flags, required arguments, and optional arguments with default values. It also
 * provides type-safe getters, validation, and an auto-generated help message.
 *
 * The file also includes a helper function, getSeparator, to convert string
 * representations of separators (e.g., "\\s", "\\t") into their corresponding
 * character values.
 *
 * @note The parser exits with an error if arguments are invalid or required ones
 *       are missing.
 */
#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <optional>
#include <stdexcept>
#include <thread>

/**
 * @brief Converts a string representing a separator (e.g., "\\s", "\\t") into
 *        the corresponding character.
 *
 * @param sepStr The string representation of the separator (e.g., "\\s", "\\t").
 * @return The corresponding character (e.g., ' ', '\t').
 *
 * @note Exits the program with an error if the separator string is unknown.
 */
static char getSeparator(const std::string &sepStr)
{
    if (sepStr == "\\s")
        return ' ';
    if (sepStr == "\\t")
        return '\t';
    if (sepStr == "\\n")
        return '\n';
    if (sepStr == ",")
        return ',';
    if (sepStr == ";")
        return ';';
    if (sepStr == "|")
        return '|';

    // Default separator (fallback to space)
    std::cerr << "Unknown separator: " << sepStr << ". Exiting.\n";
    std::exit(1);
}

/**
 * @class ArgParser
 * @brief Simple command-line argument parser.
 *
 * The ArgParser class allows you to define and parse both required and optional
 * arguments (with optional default values) and flags. It also automatically
 * generates a help message for display. 
 */
class ArgParser
{
public:
    /**
     * @struct Argument
     * @brief Represents a single command-line argument or flag.
     *
     * @var Argument::name
     * The name of the argument (e.g., "--input").
     *
     * @var Argument::description
     * A short description of the argument’s purpose.
     *
     * @var Argument::required
     * Indicates whether this argument is required.
     *
     * @var Argument::is_flag
     * Indicates whether this argument is a flag (no value needed).
     *
     * @var Argument::default_value
     * Optional default value for this argument if not supplied.
     *
     * @var Argument::value
     * The actual value assigned to this argument after parsing.
     */
    struct Argument
    {
        std::string name;
        std::string description;
        bool required;
        bool is_flag;
        std::optional<std::string> default_value;
        std::optional<std::string> value;
    };

private:
    std::unordered_map<std::string, Argument> args; ///< Stores arguments by their name
    std::vector<std::string> positional_args;        ///< Captures any positional arguments
    std::string program_name;                       ///< Name of the program (used in help messages)

public:
    /**
     * @brief Constructs an ArgParser with the specified program name.
     * @param prog_name The program name (used in help messages).
     *
     * @note Set the program name to something meaningful for better help output.
     */
    explicit ArgParser(std::string prog_name) : program_name(std::move(prog_name)) {}

    /**
     * @brief Adds a new argument (not a flag) to the parser.
     *
     * @param name The name of the argument (e.g., "--input").
     * @param desc A description of the argument.
     * @param required Whether the argument is required (default: false).
     * @param default_val Default value if the argument is not provided (optional).
     *
     * @note If required is true and the user does not supply this argument,
     *       parsing will fail and the program will exit with an error.
     */
    void add_argument(const std::string &name,
                      const std::string &desc,
                      bool required = false,
                      std::optional<std::string> default_val = std::nullopt)
    {
        args[name] = {name, desc, required, false, default_val, std::nullopt};
    }

    /**
     * @brief Adds a new flag (boolean switch) to the parser.
     *
     * @param name The name of the flag (e.g., "--verbose").
     * @param desc A description of the flag.
     *
     * @note Flags do not require a value. If present, the value is "true";
     *       otherwise, it remains unset (which you can interpret as false).
     */
    void add_flag(const std::string &name, const std::string &desc)
    {
        args[name] = {name, desc, false, true, std::nullopt, std::nullopt};
    }

    /**
     * @brief Parses the command-line arguments.
     *
     * @param argc The number of command-line arguments.
     * @param argv The array of command-line argument strings.
     *
     * @throw std::runtime_error If a required argument is missing or if a non-flag
     *        argument lacks an associated value.
     * @note If the user passes "--help" or "-h", the help message is printed
     *       and the program terminates.
     */
    void parse(int argc, char *argv[])
    {
        try
        {
            if (argc > 0)
                program_name = argv[0];

            for (int i = 1; i < argc; ++i)
            {
                std::string arg = argv[i];

                // Check if the argument matches a known flag or argument
                if (args.find(arg) != args.end())
                {
                    // It's a flag
                    if (args[arg].is_flag)
                    {
                        args[arg].value = "true";
                    }
                    // It's a regular argument
                    else if (i + 1 < argc)
                    {
                        args[arg].value = argv[++i];
                    }
                    // Value missing
                    else
                    {
                        throw std::runtime_error("Missing value for argument: " + arg);
                    }
                }
                else if (arg == "--help" || arg == "-h")
                {
                    print_help();
                    std::exit(0);
                }
                else
                {
                    // Anything that doesn't match a known argument/flag is treated as positional
                    positional_args.push_back(arg);
                }
            }

            // Validate required arguments
            for (const auto &[key, arg] : args)
            {
                if (arg.required && !arg.value.has_value())
                {
                    throw std::runtime_error("Missing required argument: " + key);
                }
            }

            // Apply default values where needed
            for (auto &[key, arg] : args)
            {
                if (!arg.value.has_value() && arg.default_value.has_value())
                {
                    arg.value = arg.default_value;
                }
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << "\n\n";
            print_help();
            std::exit(1);
        }
    }

    /**
     * @brief Retrieves the value of a named argument.
     *
     * @param name The name of the argument to retrieve.
     * @return The value of the argument as a std::string.
     *
     * @throw std::runtime_error If the argument name is unknown.
     *
     * @note If the argument was not provided and has no default value, returns an empty string.
     */
    std::string get(const std::string &name)
    {
        if (args.find(name) == args.end())
            throw std::runtime_error("Unknown argument: " + name);
        return args[name].value.value_or("");
    }

    /**
     * @brief Retrieves the boolean state of a flag.
     *
     * @param name The name of the flag to retrieve.
     * @return True if the flag is present, otherwise false.
     *
     * @throw std::runtime_error If the flag name is unknown.
     */
    bool get_bool(const std::string &name)
    {
        if (args.find(name) == args.end())
            throw std::runtime_error("Unknown argument: " + name);
        return args[name].value.has_value() && args[name].value.value() == "true";
    }

    /**
     * @brief Retrieves the value of a named argument as a 32-bit unsigned integer.
     *
     * @param name The name of the argument to retrieve.
     * @return The argument’s value as a uint32_t.
     *
     * @throw std::runtime_error If the argument name is unknown or the conversion fails.
     */
    uint32_t get_uint32(const std::string &name)
    {
        return std::stoul(get(name));
    }

    /**
     * @brief Retrieves the value of a named argument as a 64-bit unsigned integer.
     *
     * @param name The name of the argument to retrieve.
     * @return The argument’s value as a uint64_t.
     *
     * @throw std::runtime_error If the argument name is unknown or the conversion fails.
     */
    uint64_t get_uint64(const std::string &name)
    {
        return std::stoull(get(name));
    }

    /**
     * @brief Retrieves the separator character corresponding to a named argument’s value.
     *
     * @param name The name of the argument that holds the separator string.
     * @return The separator character (e.g., ' ', '\t', ',').
     *
     * @throw std::runtime_error If the argument name is unknown, or if the separator
     *        string is invalid (which will cause the program to exit).
     */
    char get_separator(const std::string &name)
    {
        return getSeparator(get(name));
    }

    /**
     * @brief Checks whether a flag is set.
     *
     * @param name The name of the flag to check.
     * @return True if the flag has been set, false otherwise.
     *
     * @throw std::runtime_error If the flag name is unknown.
     */
    bool flag_set(const std::string &name)
    {
        if (args.find(name) == args.end())
            throw std::runtime_error("Unknown flag: " + name);
        return args[name].value.has_value();
    }

    /**
     * @brief Prints the help message to stdout.
     *
     * Displays usage information based on the defined arguments and flags,
     * including any default values if provided.
     */
    void print_help()
    {
        std::cout << "Usage: " << program_name << " [options]\n\n"
                  << "Options:\n";
        for (const auto &[key, arg] : args)
        {
            std::cout << "  " << key;
            if (!arg.is_flag)
                std::cout << " <value>";
            std::cout << "\t" << arg.description;
            if (arg.default_value.has_value())
                std::cout << " (default: " << arg.default_value.value() << ")";
            std::cout << "\n";
        }
        std::cout << "  --help, -h\tShow this help message\n";
    }

    /**
     * @brief Manually sets the value of an argument.
     *
     * @param name The name of the argument to set.
     * @param value The value to assign to the argument.
     *
     * @throw std::runtime_error If the argument name is unknown.
     *
     * @note This can be useful for programmatically overriding argument values
     *       after parsing.
     */
    void set(const std::string &name, const std::string &value)
    {
        if (args.find(name) == args.end())
            throw std::runtime_error("Unknown argument: " + name);
        args[name].value = value;
    }
};
