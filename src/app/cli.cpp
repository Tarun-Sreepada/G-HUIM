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
    app.add_option("--input", config.input, "SPMF utility dataset")->required()->check(CLI::ExistingFile);
    app.add_option("--minutil", config.mining.minimum_utility, "Minimum itemset utility")
            ->required()
            ->check(CLI::PositiveNumber);
    app.add_option("--search", config.mining.search, "Search order")
            ->transform(CLI::CheckedTransformer(
                    std::map<std::string, Search>{{"bfs", Search::Bfs}, {"dfs", Search::Dfs}}, CLI::ignore_case))
            ->default_str("bfs");
    app.add_option("--method", config.mining.method, "Work partitioning")
            ->transform(CLI::CheckedTransformer(
                    std::map<std::string, Method>{{"static", Method::Static}, {"adaptive", Method::Adaptive},
                                                  {"warp", Method::Warp},
                                                  {"staged", Method::Staged},
                                                  {"stagedwarp", Method::StagedWarp},
                                                  {"tile", Method::Tile}},
                    CLI::ignore_case))
            ->default_str("adaptive");
    app.add_option("--block-size,--block_size", config.mining.block_size, "Threads per task (static method)")
            ->check(CLI::Range(32, 1024))
            ->capture_default_str();
    app.add_option("--io-warps,--io_warps", config.mining.io_warps, "IO warps per block (staged method)")
            ->check(CLI::Range(1, 4))
            ->capture_default_str();
    app.add_option("--ring-kib", config.mining.ring_kib, "Shared memory for the staged tile ring, KiB")
            ->check(CLI::Range(1, 200))
            ->group("");
    app.add_flag("--smem-tables", config.mining.smem_tables,
                 "Keep per-item tables in shared memory for block-sized tasks (adaptive and staged methods, BFS)");
    app.add_option("--shape-modes", config.mining.shape_modes,
                   "BFS adaptive: comma-separated mode per task shape, longest transaction <=4,<=8,<=16,longer "
                   "(lane, group, g4, g8, g16, g32), e.g. lane,lane,lane,g32")
            ->group("");
    app.add_option("--min-entries-per-warp", config.mining.min_entries_per_warp,
                   "BFS: size tasks by entries instead of transactions; fewest entries worth a warp (0 = by transactions)")
            ->group("");
    app.add_option("--oversubscribe", config.mining.oversubscribe,
                   "BFS: size tasks for this many times the resident warps (more warps per task balances the load)")
            ->check(CLI::Range(1, 256))
            ->group("");
    app.add_flag("--bin-times", config.mining.bin_times, "BFS: run the bins one after another and log each bin's time per level")
            ->group("");
    app.add_option("--index-min-trans", config.mining.index_min_trans,
                   "BFS adaptive/static lanes: build occurrence indexes for databases with at least this many transactions (0 = off)");
    app.add_option("--local-tail-items", config.mining.local_tail_items,
                   "BFS adaptive count-only: finish small subtrees with up to this many remaining items locally (0 = off)")
            ->check(CLI::Range(0, 20));
    app.add_option("--local-tail-trans", config.mining.local_tail_trans,
                   "Maximum supporting transactions for local subtree processing")
            ->check(CLI::Range(1, 32))->capture_default_str();
    app.add_option("--table-kib", config.mining.table_kib, "Shared memory for the per-item tables, KiB")
            ->check(CLI::Range(1, 48))
            ->group("");
    add_size_option(app, "--total-memory,--total_memory,--memory", config.mining.memory_bytes,
                    "Device memory for projected databases (K/M/G suffixes); default uses most free memory");
    add_size_option(app, "--stack-capacity,--stack_capacity", config.mining.stack_capacity,
                    "Most tasks one level may hold (K/M/G suffixes)");
    app.add_option("--output", config.mining.output, "Write the itemsets (\"items... UTIL u\") to this file");
    app.add_option_function<std::string>(
               "--separator",
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
               "Separator between items within each SPMF field")
            ->default_str("space");
    app.add_option("--log-level", config.log_level, "Logging verbosity")
            ->check(CLI::IsMember({"trace", "debug", "info", "warn", "error", "critical", "off"}))
            ->capture_default_str();
    app.add_option("--log-file", config.log_file, "Also write diagnostics to a log file");

    app.callback([&config] {
        const MiningOptions& mining = config.mining;
        if (mining.index_min_trans != 0 || mining.local_tail_items != 0) {
            if (mining.search != Search::Bfs || (mining.method != Method::Adaptive && mining.method != Method::Static) ||
                (!mining.shape_modes.empty() && mining.shape_modes != "lane,lane,lane,lane"))
                throw CLI::ValidationError("--index-min-trans/--local-tail-items", "needs BFS adaptive/static with lane modes");
            if (mining.local_tail_items != 0 && !mining.output.empty() && mining.output != "/dev/null")
                throw CLI::ValidationError("--local-tail-items", "only available for count-only mining (omit --output)");
            if (mining.local_tail_items != 0 && mining.method != Method::Adaptive)
                throw CLI::ValidationError("--local-tail-items", "needs --method adaptive");
        }
        if (mining.method == Method::Static && (mining.block_size & (mining.block_size - 1)) != 0)
            throw CLI::ValidationError("--block-size", "must be a power of two (32-1024)");
        if (mining.search == Search::Dfs && (mining.method == Method::Warp || mining.method == Method::Staged ||
                                              mining.method == Method::StagedWarp || mining.method == Method::Tile))
            throw CLI::ValidationError("--method", "warp, staged, stagedwarp and tile are only available with --search bfs");
        if (!mining.shape_modes.empty() && (mining.search != Search::Bfs || mining.method != Method::Adaptive))
            throw CLI::ValidationError("--shape-modes", "needs --search bfs with --method adaptive");
        if (mining.smem_tables && (mining.search != Search::Bfs ||
                                   (mining.method != Method::Adaptive && mining.method != Method::Staged &&
                                    mining.method != Method::StagedWarp)))
            throw CLI::ValidationError("--smem-tables", "needs --search bfs with --method adaptive, staged or stagedwarp");
        if (mining.search == Search::Dfs && !mining.output.empty())
            throw CLI::ValidationError("--output", "itemsets can only be written by --search bfs");
    });
}

} // namespace ghuim
