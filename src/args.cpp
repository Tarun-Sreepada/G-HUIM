#include "args.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>
#include <utility>

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

    std::cerr << "Unknown separator: " << sepStr << ". Exiting.\n";
    std::exit(1);
}

ArgParser::ArgParser(std::string prog_name)
    : program_name(std::move(prog_name))
{
}

void ArgParser::add_argument(const std::string &name,
                             const std::string &desc,
                             bool required,
                             std::optional<std::string> default_val)
{
    args[name] = {name, desc, required, false, default_val, std::nullopt};
}

void ArgParser::add_flag(const std::string &name, const std::string &desc)
{
    args[name] = {name, desc, false, true, std::nullopt, std::nullopt};
}

void ArgParser::parse(int argc, char *argv[])
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
                else
                {
                    if (i + 1 < argc)
                    {
                        args[arg].value = argv[++i];
                    }
                    else
                    {
                        throw std::runtime_error("Missing value for argument: " + arg);
                    }
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

        for (const auto &[key, arg] : args)
        {
            if (arg.required && !arg.value.has_value())
            {
                throw std::runtime_error("Missing required argument: " + key);
            }
        }

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

std::string ArgParser::get(const std::string &name)
{
    if (args.find(name) == args.end())
        throw std::runtime_error("Unknown argument: " + name);
    return args[name].value.value_or("");
}

bool ArgParser::get_bool(const std::string &name)
{
    if (args.find(name) == args.end())
        throw std::runtime_error("Unknown argument: " + name);
    return args[name].value.has_value() && args[name].value.value() == "true";
}

uint32_t ArgParser::get_uint32(const std::string &name)
{
    return static_cast<uint32_t>(std::stoul(get(name)));
}

uint64_t ArgParser::get_uint64(const std::string &name)
{
    return std::stoull(get(name));
}

uint64_t ArgParser::get_memory(const std::string &name)
{
    std::string value = get(name);
    if (value.empty())
        throw std::runtime_error("Memory argument is empty or missing.");

    char suffix = value.back();
    value.pop_back();

    uint64_t multiplier = 1;
    if (suffix == 'G')
        multiplier = 1024ULL * 1024ULL * 1024ULL;
    else if (suffix == 'M')
        multiplier = 1024ULL * 1024ULL;
    else if (suffix == 'K')
        multiplier = 1024ULL;
    else
        throw std::runtime_error("Invalid or missing memory suffix. Use G, M, or K.");

    return std::stoull(value) * multiplier;
}

char ArgParser::get_separator(const std::string &name)
{
    return getSeparator(get(name));
}

bool ArgParser::flag_set(const std::string &name)
{
    if (args.find(name) == args.end())
        throw std::runtime_error("Unknown flag: " + name);
    return args[name].value.has_value();
}

void ArgParser::print_help()
{
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";

    int longest_arg = 0;
    for (const auto &[key, arg] : args)
    {
        longest_arg = std::max(longest_arg, static_cast<int>(key.length()));
    }
    longest_arg += 2; // For padding

    for (const auto &[key, arg] : args)
    {
        std::cout << "  " << std::left << std::setw(longest_arg) << key;
        if (!arg.is_flag)
            std::cout << " <value>";
        std::cout << "\t" << arg.description;
        if (arg.default_value.has_value())
            std::cout << " (default: " << arg.default_value.value() << ")";
        std::cout << "\n";
    }
    std::cout << "  --help, -h\tShow this help message\n";

}

void ArgParser::set(const std::string &name, const std::string &value)
{
    if (args.find(name) == args.end())
        throw std::runtime_error("Unknown argument: " + name);
    args[name].value = value;
}
