#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <optional>
#include <stdexcept>
#include <thread>

// Helper function to map separator string to actual character
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
    exit(1);
}

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

private:
    std::unordered_map<std::string, Argument> args;
    std::vector<std::string> positional_args;
    std::string program_name;

public:
    explicit ArgParser(std::string prog_name) : program_name(std::move(prog_name)) {}

    // Adds a required or optional argument
    void add_argument(const std::string &name, const std::string &desc, bool required = false, std::optional<std::string> default_val = std::nullopt)
    {
        args[name] = {name, desc, required, false, default_val, std::nullopt};
    }

    // Adds a flag (boolean switch)
    void add_flag(const std::string &name, const std::string &desc)
    {
        args[name] = {name, desc, false, true, std::nullopt, std::nullopt};
    }
    // Parses command-line arguments
    void parse(int argc, char *argv[])
    {
        try
        {
            if (argc > 0)
                program_name = argv[0];

            for (int i = 1; i < argc; ++i)
            {
                std::string arg = argv[i];

                if (args.find(arg) != args.end())
                {
                    if (args[arg].is_flag)
                    {
                        args[arg].value = "true";
                    }
                    else if (i + 1 < argc)
                    {
                        args[arg].value = argv[++i];
                    }
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

            // Apply default values
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

    // Retrieves argument value
    std::string get(const std::string &name)
    {
        if (args.find(name) == args.end())
            throw std::runtime_error("Unknown argument: " + name);
        return args[name].value.value_or("");
    }

    bool get_bool(const std::string &name)
    {
        if (args.find(name) == args.end())
            throw std::runtime_error("Unknown argument: " + name);
        return args[name].value.has_value() && args[name].value.value() == "true";
    }

    uint32_t get_uint32(const std::string &name)
    {
        return std::stoul(get(name));
    }

    uint64_t get_uint64(const std::string &name)
    {
        return std::stoull(get(name));
    }

    // Retrieves separator as a character
    char get_separator(const std::string &name)
    {
        return getSeparator(get(name));
    }

    // Checks if a flag is set
    bool flag_set(const std::string &name)
    {
        if (args.find(name) == args.end())
            throw std::runtime_error("Unknown flag: " + name);
        return args[name].value.has_value();
    }

    // Prints help documentation
    void print_help()
    {
        std::cout << "Usage: " << program_name << " [options]\n\nOptions:\n";
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

    void set(const std::string &name, const std::string &value)
    {
        if (args.find(name) == args.end())
            throw std::runtime_error("Unknown argument: " + name);
        args[name].value = value;
    }
};



void validate_threads(ArgParser &parser)
{
    // print maximum CPU threads
    #ifdef DEBUG
    std::cout << "Maximum CPU threads: " << std::thread::hardware_concurrency() << "\n";
    #endif
    // validate the number of threads else set to maximum
    if (parser.get_uint32("--threads") > std::thread::hardware_concurrency())
    {
        std::cout << "Number of threads is greater than maximum CPU threads. Setting threads to maximum CPU threads.\n";
        parser.set("--threads", std::to_string(std::thread::hardware_concurrency()));
    }
}
