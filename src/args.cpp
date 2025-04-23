#include "args.hpp"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

static char decode_separator(const std::string& s)
{
    if (s == "\\s") return ' ';
    if (s == "\\t") return '\t';
    if (s == "\\n") return '\n';
    if (s == ",")   return ',';
    if (s == ";")   return ';';
    if (s == "|")   return '|';

    std::cerr << "Unknown separator: " << s << ". Exiting.\n";
    std::exit(1);
}

ArgParser::ArgParser(std::string prog_name)
    : program_name(std::move(prog_name))
{}

void ArgParser::add_argument(const std::string& name,
                             const std::string& type,
                             const std::string& desc,
                             bool               required,
                             std::optional<std::string> default_val)
{
    args[name] = Argument{
        name,
        type,
        desc,
        required,
        /*is_flag*/ false,
        std::move(default_val),
        /*value*/   std::nullopt
    };
}

void ArgParser::add_flag(const std::string& name, const std::string& desc)
{
    args[name] = Argument{
        name,
        /*type*/     "flag",
        desc,
        /*required*/ false,
        /*is_flag*/  true,
        /*default*/  std::nullopt,
        /*value*/    std::nullopt
    };
}

void ArgParser::parse(int argc, char* argv[])
{
    try
    {
        if (argc > 0) program_name = argv[0];

        for (int i = 1; i < argc; ++i)
        {
            std::string token = argv[i];

            if (auto it = args.find(token); it != args.end())
            {
                Argument& a = it->second;
                if (a.is_flag)
                {
                    a.value = "true";
                }
                else
                {
                    if (i + 1 >= argc)
                        throw std::runtime_error("Missing value for argument: " + token);
                    a.value = argv[++i];
                }
            }
            else if (token == "--help" || token == "-h")
            {
                print_help();
                std::exit(0);
            }
            else
            {
                positional_args.push_back(token);
            }
        }

        // check required
        for (auto const& [key, a] : args)
            if (a.required && !a.value.has_value())
                throw std::runtime_error("Missing required argument: " + key);

        // fill defaults
        for (auto& [_, a] : args)
            if (!a.value.has_value() && a.default_value.has_value())
                a.value = a.default_value;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n\n";
        print_help();
        std::exit(1);
    }
}

std::string ArgParser::get(const std::string& name)
{
    auto it = args.find(name);
    if (it == args.end())
        throw std::runtime_error("Unknown argument: " + name);
    return it->second.value.value_or("");
}

bool ArgParser::get_bool(const std::string& name)
{
    auto it = args.find(name);
    if (it == args.end())
        throw std::runtime_error("Unknown argument: " + name);
    return it->second.value.has_value() && it->second.value.value() == "true";
}

uint32_t ArgParser::get_uint32(const std::string& name)
{
    return static_cast<uint32_t>(std::stoul(get(name)));
}

uint64_t ArgParser::get_uint64(const std::string& name)
{
    return std::stoull(get(name));
}

uint64_t ArgParser::get_memory(const std::string& name)
{
    std::string v = get(name);
    if (v.empty())
        throw std::runtime_error("Memory argument empty or missing: " + name);

    char suffix = v.back();
    v.pop_back();

    uint64_t mul = 1;
    if (suffix == 'G') mul = 1024ULL * 1024ULL * 1024ULL;
    else if (suffix == 'M') mul = 1024ULL * 1024ULL;
    else if (suffix == 'K') mul = 1024ULL;
    else
        throw std::runtime_error("Invalid memory suffix for " + name + "; use G, M, or K");

    return std::stoull(v) * mul;
}

char ArgParser::get_separator(const std::string& name)
{
    return decode_separator(get(name));
}

bool ArgParser::flag_set(const std::string& name)
{
    auto it = args.find(name);
    if (it == args.end())
        throw std::runtime_error("Unknown flag: " + name);
    return it->second.value.has_value();
}

std::string ArgParser::get_type(const std::string& name) const
{
    auto it = args.find(name);
    if (it == args.end())
        throw std::runtime_error("Unknown argument: " + name);
    return it->second.type;
}

void ArgParser::print_help()
{
    std::cout << "Usage: " << program_name << " [options]\n\n"
              << "Options:\n";

    int widest = 0;
    for (auto const& [key, _] : args)
        widest = std::max(widest, static_cast<int>(key.length()));
    widest += 2; // padding

    int widest_type = 0;
    for (auto const& [_, arg] : args)
        widest_type = std::max(widest_type, static_cast<int>(arg.type.length()));
    widest_type += 2; // padding

    for (auto const& [key, arg] : args)
    {
        std::cout << "  " << std::left << std::setw(widest) << key;
        std::cout << " <" << arg.type << ">" << std::setw(widest_type - arg.type.length());
        std::cout << "  " << arg.description;
        std::cout << '\n';
    }
    std::cout << "  --help, -h  Show this help message\n";
}

void ArgParser::set(const std::string& name, const std::string& value)
{
    auto it = args.find(name);
    if (it == args.end())
        throw std::runtime_error("Unknown argument: " + name);
    it->second.value = value;
}
