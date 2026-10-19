#include "app/cli.hpp"

#include <map>
#include <stdexcept>

namespace ghuim {
namespace {

// A byte/count option with K/M/G suffixes.
void add_size_option(CLI::App& app, const std::string& name, uint64_t& target, const std::string& description)
{
    app.add_option_function<std::string>(
            name,
            [&target, name](const std::string& value) {
                try {
                    target = parse_size(value);
                }
                catch (const std::invalid_argument& error) {
                    throw CLI::ValidationError(name, error.what());
                }
            },
            description);
}

} // namespace

void configure_cli(CLI::App& app, Config& config)
{
    app.add_option("-i,--input", config.input, "SPMF utility dataset")->required()->check(CLI::ExistingFile);
    app.add_option("-m,--minutil", config.mining.minimum_utility, "Minimum itemset utility")
            ->required()
            ->check(CLI::PositiveNumber);
    app.add_option("-e,--search", config.mining.search, "Search order")
            ->transform(CLI::CheckedTransformer(
                    std::map<std::string, Search>{{"bfs", Search::Bfs}, {"dfs", Search::Dfs}}, CLI::ignore_case))
            ->default_str("bfs");
    app.add_option("-M,--method", config.mining.method, "Work partitioning")
            ->transform(CLI::CheckedTransformer(
                    std::map<std::string, Method>{{"static", Method::Static}, {"adaptive", Method::Adaptive},
                                                  {"staged", Method::Staged},
                                                  {"tile", Method::Tile}},
                    CLI::ignore_case))
            ->default_str("adaptive");
    app.add_option("-b,--block-size,--block_size", config.mining.block_size, "Threads per task (static method)")
            ->check(CLI::Range(32, 1024))
            ->capture_default_str();
    app.add_option("-w,--io-warps,--io_warps", config.mining.io_warps, "IO warps per block (staged method)")
            ->check(CLI::Range(1, 4))
            ->capture_default_str();
    app.add_option("-R,--ring-kib", config.mining.ring_kib, "Shared memory for the staged tile ring, KiB")
            ->check(CLI::Range(1, 200))
            ->group("");
    app.add_option("-O,--oversubscribe", config.mining.oversubscribe,
                   "BFS: size tasks for this many times the resident warps (more warps per task balances the load)")
            ->check(CLI::Range(1, 256))
            ->group("");
    app.add_flag("-B,--bin-times", config.mining.bin_times, "BFS: run the bins one after another and log each bin's time per level")
            ->group("");
    CLI::Option* index =
            app.add_option("-I,--index-min-trans", config.mining.index_min_trans,
                           "BFS adaptive/static: build occurrence indexes for databases with at least this many transactions "
                           "(0 = off; by default only sparse ones, see --index-sparsity)")
                    ->capture_default_str();
    CLI::Option* sparsity =
            app.add_option("-D,--index-sparsity", config.mining.index_sparsity,
                           "Index only databases at most 1/this filled (0 = any; default 0 with an explicit --index-min-trans)")
                    ->capture_default_str();
    CLI::Option* local_tail =
            app.add_option("-k,--local-tail-items", config.mining.local_tail_items,
                           "BFS adaptive count-only: finish small subtrees with up to this many remaining items locally "
                           "(0 = off; on by default where it applies)")
                    ->check(CLI::Range(0, 31))
                    ->capture_default_str();
    app.add_option("-V,--vert-items", config.mining.vert_items,
                   "BFS adaptive count-only: mine subtrees with at most this many remaining items vertically, from "
                   "transaction bitmaps (0 = off)")
            ->check(CLI::Range(0, 32))
            ->capture_default_str();
    app.add_option("-X,--vert-trans", config.mining.vert_trans, "Most transactions of a vertical subtree")
            ->check(CLI::Range(1, 1024))
            ->capture_default_str();
    app.add_option("-j,--local-tail-trans", config.mining.local_tail_trans,
                   "Maximum supporting transactions for local subtree processing")
            ->check(CLI::Range(1, 32))->capture_default_str();
    app.add_option("-r,--refine", config.mining.refine_rounds,
                   "Bound refinement: recompute local utilities without the items each task drops, this many rounds "
                   "(-1 = until none drops, 0 = off; huge tasks run at most 8)")
            ->check(CLI::Range(-1, 1000))->capture_default_str();
    add_size_option(app, "-g,--total-memory,--total_memory,--memory", config.mining.memory_bytes,
                    "Device memory for projected databases (K/M/G suffixes); default uses most free memory");
    add_size_option(app, "-C,--stack-capacity,--stack_capacity", config.mining.stack_capacity,
                    "Most tasks one level may hold (K/M/G suffixes); default is derived from free device memory");
    app.add_option("-o,--output", config.mining.output, "Write the itemsets (\"items... UTIL u\") to this file");
    app.add_option_function<std::string>(
               "-s,--separator",
               [&config](const std::string& value) {
                   if (value == "\\s" || value == "space")
                       config.separator = ' ';
                   else if (value == "\\t" || value == "tab")
                       config.separator = '\t';
                   else if (value.size() == 1 && value[0] != '\n' && value[0] != '\r' && value[0] != ':')
                       config.separator = value[0];
                   else
                       throw CLI::ValidationError("--separator", "use one character, space, or tab");
               },
               "Field separator: space, tab or one character")
            ->default_str("space");
    app.add_option("-l,--log-level", config.log_level, "Logging verbosity")
            ->check(CLI::IsMember({"trace", "debug", "info", "warn", "error", "critical", "off"}))
            ->capture_default_str();
    app.add_option("-L,--log-file", config.log_file, "Also write diagnostics to a log file");

    app.callback([&config, local_tail, index, sparsity] {
        MiningOptions& mining = config.mining;
        // The default local subtrees only apply to count-only BFS adaptive lane mining; elsewhere they are
        // turned off, and only an explicit --local-tail-items is an error.
        const bool counting = mining.output.empty() || mining.output == "/dev/null";
        if (local_tail->count() == 0 && (mining.search != Search::Bfs || mining.method != Method::Adaptive || !counting))
            mining.local_tail_items = 0;
        // Default indexes likewise apply only to BFS adaptive/static. An explicit --index-min-trans indexes every
        // database that large unless --index-sparsity is also given.
        if (index->count() == 0 && (mining.search != Search::Bfs ||
                                    (mining.method != Method::Adaptive && mining.method != Method::Static)))
            mining.index_min_trans = 0;
        if (index->count() != 0 && sparsity->count() == 0)
            mining.index_sparsity = 0;
        if (mining.index_min_trans != 0 || mining.local_tail_items != 0) {
            if (mining.search != Search::Bfs || (mining.method != Method::Adaptive && mining.method != Method::Static))
                throw CLI::ValidationError("--index-min-trans/--local-tail-items", "needs BFS adaptive or static");
            if (mining.local_tail_items != 0 && !mining.output.empty() && mining.output != "/dev/null")
                throw CLI::ValidationError("--local-tail-items", "only available for count-only mining (omit --output)");
            if (mining.local_tail_items != 0 && mining.method != Method::Adaptive)
                throw CLI::ValidationError("--local-tail-items", "needs --method adaptive");
        }
        if (mining.vert_items != 0 && (mining.search != Search::Bfs || mining.method != Method::Adaptive || !counting))
            throw CLI::ValidationError("--vert-items", "needs BFS adaptive count-only mining");
        if (mining.method == Method::Static && (mining.block_size & (mining.block_size - 1)) != 0)
            throw CLI::ValidationError("--block-size", "must be a power of two (32-1024)");
        if (mining.search == Search::Dfs && (mining.method == Method::Staged || mining.method == Method::Tile))
            throw CLI::ValidationError("--method", "staged and tile are only available with --search bfs");
        if (mining.search == Search::Dfs && !mining.output.empty())
            throw CLI::ValidationError("--output", "itemsets can only be written by --search bfs");
    });
}

} // namespace ghuim
