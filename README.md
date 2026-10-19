# G-HUIM


For 2026 DASFAA paper please refer to the following commit : 
[Commit](https://github.com/Tarun-Sreepada/G-HUIM/commit/b08df0c2de23f2949935fc2522f66ff672a396b7)


GPU-accelerated high-utility itemset mining (EFIM) on NVIDIA hardware. This branch searches the
itemset tree either level by level (`--search bfs`: every task of a level runs before any task of
the next) or depth first from a task stack (`--search dfs`), with the work partitioned statically
or adaptively (`--method`; see [Methods and speeds](#methods-and-speeds)).

## Build

Requires CMake 3.24+, a C++20 compiler, and CUDA 12.x. The first configuration downloads pinned
CLI11 2.6.2 and spdlog 1.16.0.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
```

The GPU architecture is auto-detected (`native`, sm_80 or newer); pass
`-DCMAKE_CUDA_ARCHITECTURES=...` to override it. The build produces `build/G-HUIM` (GPU miner) and `build/refim` (CPU miner, REFIM).

REFIM (CPU) usage: `build/refim -i data.txt -m <minutil> [-o out.txt] [-t 48]`. Workers are pinned and steal work
nearest-first; `-l debug` prints reader phase times and per-worker steal counts. Both miners share `-i/--input`,
`-m/--minutil`, `-o/--output`, `-s/--separator` (default space), `-r/--refine`, `-l/--log-level` and `-L/--log-file`.
CPU-only options:

| Option | Meaning |
| --- | --- |
| `-t`, `--threads` | Worker threads (default 1, `0` = all) |
| `-n`, `--numa` | `on` (default) pins workers and places memory per NUMA node, or `off` |
| `-v`, `--simd` | Vector kernels: `auto` (default), `avx512` or `scalar` |
| `-p`, `--thp` | Transparent huge pages: `auto` (default; off above 3/4 of the hardware threads), `on` or `off` |
| `-r`, `--refine` | Rounds of recomputing each child's bounds without the items it rules out (default `-1` = until none is; `0` = off) |
| `-d`, `--refine-min-drop` | Refine only while a pass drops at least this fraction of the candidates (default 0) |
| `-f`, `--leaf` | `on` (default): sum leaf extensions from the occurrence index instead of projecting them |
| `-c`, `--compact` | `on` (default): rewrite each child without the items it rules out |
| `-S`, `--stats` | Log search counters (nodes, refinement, compaction, leaves) |

Both miners only count patterns when output is omitted or set to `/dev/null`;
a real output path enables itemset collection and writing.
Both miners store items in 32 bits and utilities in 64 bits (`refim` stops with an error if any item's total
utility exceeds 2^63-1, G-HUIM if the database total exceeds 2^64-1). `build/refim` is LTO-built.

## Run

```bash
./build/G-HUIM --help
./build/G-HUIM --input datasets/accidents_utility_spmf.txt --minutil 3000000
./build/G-HUIM --input datasets/accidents_utility_spmf.txt --minutil 3000000 --search dfs --method static --block-size 128
./build/G-HUIM --input datasets/accidents_utility_spmf.txt --minutil 3000000 --log-level debug
```

| Option | Meaning |
| --- | --- |
| `-i`, `--input` | SPMF utility dataset (required) |
| `-m`, `--minutil` | Minimum itemset utility (required) |
| `-e`, `--search` | `bfs` (default, level by level) or `dfs` (task stack) |
| `-M`, `--method` | `adaptive` (default, size-binned kernels: lanes, warp, block or several blocks per task), `static` (one block of `--block-size` threads per task), `staged` (BFS only: adaptive kernels with IO warps staging transactions in shared memory) or `tile` (BFS only: adaptive scheduling over databases stored in warp tiles). See [Methods and speeds](#methods-and-speeds) |
| `-b`, `--block-size` | Threads per task for `--method static`: a power of two, 32-1024 (default 256) |
| `-w`, `--io-warps` | `staged`: IO warps added to each block, 1-4 (default 1; 4 was fastest) |
| `-R`, `--ring-kib` | `staged`: shared memory for the tile ring, KiB (default 64) |
| `-O`, `--oversubscribe` | BFS: size tasks for this many times the resident warps (default 12; more, smaller blocks balance the load) |
| `-B`, `--bin-times` | BFS: run the cells one after another and log the time of each per level |
| `-I`, `--index-min-trans` | BFS adaptive/static: build shared occurrence indexes for databases with at least this many transactions (default 4096; `0` = off). By default only sparse databases are indexed (`-D`); an explicit `-I` alone indexes any density |
| `-D`, `--index-sparsity` | Index only databases with at most 1/this of their transaction-item pairs present (default 32; `0` = any) |
| `-k`, `--local-tail-items` | BFS adaptive, count-only: process small subtrees locally when at most this many items remain (default 8, on wherever it applies; `0` = off; maximum 31) |
| `-j`, `--local-tail-trans` | Maximum supporting transactions for local subtree processing, 1-32 (default 32) |
| `-V`, `--vert-items`, `-X`, `--vert-trans` | BFS adaptive, count-only: a warp mines subtrees with at most `-V` remaining items (<= 32) and `-X` transactions (<= 1024, default 256) from per-item transaction bitmaps and utility columns, without projecting (default off; turned off with a warning when the total utility exceeds 2^32-1, since its columns are 32-bit). `-V 16 -X 64`: retail 0.31 -> 0.19 s, accidents -6%; dense data 2-5x slower (one warp walks a whole dense subtree) |
| `-r`, `--refine` | Bound refinement: recompute each task's local utilities without the items it drops, this many rounds (default 1; `-1` = until none drops, `0` = off; huge tasks run at most 8). Same itemsets; fewer extensions on sparse data |
| `-g`, `--total-memory` (`--memory`) | Device memory for projected databases (`K`/`M`/`G`); default uses the free memory |
| `-C`, `--stack-capacity` | Most tasks one level may hold (BFS; default: as many as device memory allows) |
| `-o`, `--output` | Write itemsets as `items... UTIL u` (BFS only); without it they are only counted |
| `-s`, `--separator` | Separator between items within each SPMF field (default space) |
| `-l`, `--log-level`, `-L`, `--log-file` | Logging verbosity and optional log file |

Help and argument validation run before CUDA initialization. `--log-level debug` prints stage
timings and per-iteration statistics.

## Methods and speeds

How to run each method (BFS search; swap `--minutil` and `--input` for your data):

```bash
IN="--input datasets/accidents_utility_spmf.txt --minutil 6000000"
./build/G-HUIM $IN                                                      # adaptive (default, fastest)
./build/G-HUIM $IN --method static --block-size 256                     # one block per task
./build/G-HUIM $IN --method staged --io-warps 4 --ring-kib 64           # IO warps stage transactions in shared memory
./build/G-HUIM $IN --bin-times                                          # time per scheduling cell and level
```

### How the methods work

Every task extends a prefix itemset with one item. A task reads the transactions of its database
(each is a sorted list of item keys with utilities) in four passes:

1. **Local utilities.** Each transaction is searched for the extension item. The entries behind it add
   to the local utility of every later item, and the transaction's prefix utility adds to the itemset's
   utility. The position found is stored, so later passes do not search again.
2. **Rename and subtree utilities.** Items whose local utility reaches `--minutil` are kept and
   renamed 0, 1, ...; for each kept item the subtree utility is added up. Each projected transaction is
   hashed so identical ones are merged, and the exact size of the child database is counted. The kept
   count of each transaction is stored.
3. **Write the child database.** One representative per group of identical projected transactions writes
   its kept entries; the others add their utilities into the representative's.
4. **Emit child tasks** for every item whose subtree utility reaches `--minutil`.

The methods differ in how a task's transactions are mapped onto GPU threads.

**`adaptive` (default).** Tasks are sorted into size bins from their share of the chunk's work (every
task of a chunk is sized for `--oversubscribe` times the resident warps), and every bin runs its own
kernel, concurrently on separate streams:

- up to 32 transactions: a group of lanes (a power of two, up to a warp) owns the whole task;
- a few transactions: one warp per task, many tasks packed into one block;
- more: one block of 64 to 1024 threads per task, the warps splitting the transactions;
- very large: the task is sliced over many blocks, with one launch per pass.

Lane-group tasks that the local-subtree walk finishes on its own (`--local-tail-items`, decided from the
task's sizes) get a cell of their own, so its larger kernel runs only on them. The cell counts are updated on the GPU as tasks are created,
and every non-empty cell is launched. In the kernel a lane owns one transaction: a warp covers 32
consecutive transactions per step, each lane searches its own transaction and walks the entries behind the
extension item. This is simple and has little overhead, but neighbouring lanes read far-apart memory,
which makes the loads uncoalesced.

**`static`.** Every task gets one block of `--block-size` threads, whatever its size. The warps split the
transactions as in `adaptive`. It is the simple baseline: small tasks waste most of the block and huge
tasks run on a single block.

**`staged`.** `adaptive`, with some warps added to each block as IO warps. An IO warp copies tiles of 32
transactions from global memory into a ring of shared-memory slots with `cp.async`, keeping up to two
tiles in flight. The other (compute) warps run the lane-per-transaction passes on the tiles, so their
loads come from shared memory, laid out so the 32 lanes fall on different banks. Per-slot counters
tell the compute warps when a tile has landed and the IO warps when a slot is free. All waits are bounded,
so a protocol bug ends the kernel with an error and does not hang the GPU. It applies to block-sized and
huge tasks; smaller tasks run unstaged. From pass 2 on, only transactions that contain the extension
item are copied. `--io-warps` sets how many IO warps a block gets and `--ring-kib` how much shared
memory the ring has, which is also the prefetch depth. Keys and utilities are copied with `cp.async`.

**`tile`.** `adaptive` scheduling, but every database is stored in tiles of 32 transactions with the entries
column-major: entry *k* of the 32 transactions of a tile is one contiguous row, so lanes that read the same
row read neighbouring words (a database of fewer than 32 transactions is one narrower tile, the next power
of two). A child is built in tile order: pass 2 counts the representatives of each length class (0..30 exact,
31 = longer), a scan gives every class its first slot, each representative takes a slot of its class
(a counting sort by length, so a tile's transactions have similar lengths and padding is small), the longest
transaction of each tile sets the tile's length, and a scan over the tiles gives their offsets. Only then is
the child allocated at its exact size. The passes are still a lane per transaction, so the instruction
count is the same as `adaptive`; only the loads change. Status: correct (same counts as `adaptive` at 8M, 6M
and 3M, `compute-sanitizer` clean on a 2,000-transaction subset), not faster yet: see the table below (n/m = total time not measured; tile builds its root image on one host thread, about 0.2 s more than `adaptive` at 8M). Peak arena use matches `adaptive` at 6M (664 vs 661 MiB) and is 35% higher at 3M (2590 vs 1914 MiB).

`--search dfs` processes tasks from a stack instead of level by level and has its own `static` and
`adaptive` kernels; `staged` and `tile` are BFS only.

Historical measurements, before the changes in [Recent optimizations](#recent-optimizations); `warp`,
`stagedwarp`, `--shape-modes`, `--smem-tables` and `--min-entries-per-warp` have since been removed. Each cell is **kernel time / total time**, the `kernels` and `Total time` log lines, best of 3, on
`accidents_utility_spmf`, NVIDIA H100 PCIe, BFS search, run with `--total-memory 8G`. The kernel time is
the mining itself; the total time adds loading the dataset (about 0.16 s), building and uploading the root
database, allocating the arenas and reading back, about 0.5 s in these runs. With the default memory
setting (all free memory) the arena allocation is larger and the overhead grows (about 1 s in one 3M run
with 36 GB regions). Every method returns the same itemset count at a given `--minutil` (3,030,087 at 8M;
10,079,776 at 6M; 104,518,894 at 3M).

| Method | minutil 8M | minutil 6M | minutil 3M |
| --- | --- | --- | --- |
| `adaptive` (default) | **0.109 / 0.631 s** | **0.237 / 0.753 s** | **1.492 / 2.023 s** |
| `adaptive`, `--oversubscribe 1` (previous sizing) | 0.152 / 0.667 s | 0.342 / 0.858 s | 1.928 / 2.455 s |
| `adaptive`, before the 2D-cell and oversubscription changes | 0.158 / 0.672 s | 0.368 / 0.896 s | 2.125 / 2.663 s |
| `staged` (`--io-warps 4 --ring-kib 64`) | 0.169 / 0.688 s | 0.326 / 0.843 s | 1.752 / 2.281 s |
| `tile` (first version) | 0.126 / n/m | 0.264 / n/m | 1.671 / n/m |
| `static` (`--block-size 256`) | 0.225 / 0.738 s | 0.415 / 0.937 s | 2.544 / 3.073 s |
| `warp` | 0.197 / 0.712 s | 0.516 / 1.040 s | 3.715 / 4.259 s |
| `stagedwarp` (`--io-warps 4 --ring-kib 32`) | 0.367 / 0.880 s | 0.782 / 1.300 s | 4.592 / 5.136 s |

What the measurements say:

- Use `adaptive`. Sizing tasks for 12 times the resident warps (`--oversubscribe`, the default) gave
  most of the gain; the curve is flat from about 8 to 24.
- IO warps help the staged methods (1 IO warp 0.37 s, 2 IO warps 0.25 s, 4 IO warps 0.22 s for `staged`
  before the oversubscription change) but do not catch up with `adaptive`. A ring above 64 KiB is slower
  because it lowers occupancy.
- Lane groups (`warp`, `stagedwarp`, `--shape-modes` with a group mode) lose to a lane per transaction in
  every task shape tried: the coalesced loads cost more instructions than they save. Compile-time group
  sizes (`g4` to `g32`) are about 30% faster than `group`, which picks the size at run time.
- `--smem-tables` was neutral at 4 KiB and slower above it. `--min-entries-per-warp` made no difference.
- DFS (`--search dfs`) was not part of these measurements.

Only one dataset (accidents) was measured; the `--oversubscribe` default and the ranking may differ on
others.

### Occurrence indexes and local subtree processing

Two opt-in paths reduce repeated work. `--index-min-trans 256` builds one index
per sufficiently large database and shares transaction IDs and item positions
among its extensions. Indexed tasks visit only matching transactions and use
that count for scratch allocation and scheduling. Index construction and its
extra storage can outweigh these savings.

For count-only mining, `--local-tail-items 8 --local-tail-trans 32` processes
small subtrees within a lane group. It reads suffixes from the existing parent
database and maintains a bounded local DFS stack instead of allocating child
databases or scheduling each descendant. Larger subtrees use the ordinary
projection path. This does not retain ancestor pointers across arena reuse.

```bash
./build/G-HUIM --input datasets/accidents_utility_spmf.txt --minutil 3000000 \
  --total-memory 8G --local-tail-items 8 --local-tail-trans 32
```

Local processing requires BFS adaptive lane kernels with no output file (or
`--output /dev/null`). Indexing also supports BFS static lane kernels and full
itemset output. Both options default to off. With local processing, reported
task counts are globally scheduled tasks, and a level's pattern count includes
descendants completed locally. See [measurements and validation](experiments/gpu_occurrence_index.md).

## Recent optimizations

Kernel time (BFS adaptive defaults, `-g 16G`) before and after the October 19 changes, best of one
run each; every count matches REFIM:

| Dataset (minutil) | Before | After |
| --- | ---: | ---: |
| kosarak (720K) | 0.163 s | 0.028 s |
| BMS (1,999K) | 0.227 s | 0.186 s |
| retail (1,440) | 0.582 s | 0.309 s |
| pumsb (9,680K) | 0.276 s | 0.221 s |
| accidents (3,900K) | 0.591 s | 0.283 s |
| connect (7,400K) | fails (level capacity) | 0.114 s |
| chess (170K) | fails (level capacity) | 0.214 s |
| mushroom (5,500) | fails (level capacity) | 0.077 s |

In order of effect: local subtrees on by default in their own scheduling cell; occurrence indexes
by default for large sparse databases (kosarak); the last child item counted without a task;
packed-warp kernels on the smallest full-occupancy block; 16-bit entry keys (since widened back to 32-bit
keys with 64-bit utilities); one backward sweep in
pass 2; the root database uploaded without a host image and indexed on the GPU (setup 72 -> 15 ms on
accidents); sparse input item ids compacted (ecommerce loading 16 s -> 0.05 s). CPU against GPU
across all datasets: [cpu_gpu_sweep.md](experiments/cpu_gpu_sweep.md).

## Checking results against the CPU baseline

```bash
python3 experiments/compare_counts.py datasets/accidents_utility_spmf.txt 3000000
```

Runs `refim` and `G-HUIM` for all four search/method pairs (`--configs bfs-adaptive ...` to pick), and
fails if any pattern count differs from REFIM's.

## Source layout

Two self-contained projects, each with its own `CMakeLists.txt`, `include/` (headers) and `src/`
(sources). The root `CMakeLists.txt` only adds both; either also builds alone
(`cmake -S REFIM -B build-refim`, `cmake -S G-HUIM -B build-ghuim`). Each program's `main` is a
short list of named steps, and every step has its own small file.

```
G-HUIM/                   GPU miner -> build/G-HUIM
  src/main.cpp              parse_command_line, start_cuda_context, load_data, mine_data, report_totals
  src/app/                  configuration, CLI11 options, logging; steps/ holds the main() steps
  src/data/                 dataset loading: parallel parsing, TWU pruning, renaming, rebuild
  src/mine/                 miner entry, itemset output, scan primitives
    bfs/                    level-synchronous miner
      bfs_miner.cu            mine_bfs: build the miner, run it, write the itemsets
      host/                   auto_tune, budget, itemsets; level_miner/ has one file per step
                              (allocate, seed_first_level, plan_chunks, mine_chunk, mine_level, run)
      small_kernels.cu        lane-group, warp and block kernels
      huge_kernels.cu         multi-block kernels for very large tasks
      schedule.cu             scratch sizes and binning
      occupancy.cu            block-size selection from kernel occupancy
      root_image.cu           root database in arena format
    dfs/                    depth-first miner (own database layout, in namespace ghuim::dfs)
      dfs_miner.cu            host driver (task stack, batches, deferrals)
      schedule.cu, bins.cu    batch selection and binning
      arena_gc.cu             arena compaction
      baseline/               static method: one block per task
      optimized/              adaptive method: size-binned kernels
      root_image.cu           root database in arena format
  include/                  headers mirroring src/
    mine/bfs/, mine/dfs/    per-engine layout, placement/scheduling, device code
    mine/bfs/device/        staging/ (cp.async ring) and tile/ (tile method) are split by stage
    cuda/                   checked buffers, libcu++ atomics
REFIM/                    CPU miner -> build/refim
  src/main.cpp              parse_command_line, init_logging, run
  src/app/                  options, logging, run (read, configure, mine, save)
  src/io/                   reader.cpp calls the steps in io/reader/ (read file, parse, rank items,
                            rename, concatenate, sort, build root); itemset output
  src/mine/                 search: refim.cpp, simd.cpp; worker/ has one file per step of expand
  src/sys/                  NUMA, allocator
  include/                  headers mirroring src/
experiments/              benchmark, comparison and plotting scripts
```

The two searches keep separate database layouts, so `bfs/` and `dfs/` do not share device code; they
share the data layer, scan primitives, atomics and buffers.

`experiments/timeline.py` and `iteration.py` parse the old per-iteration timer table, which this
version no longer prints; `compare_counts.py` and `run_experiments.py` use the new log lines.



## Datasets
```html
https://www.philippe-fournier-viger.com/spmf/index.php?link=datasets.php
```

Use the SPMF utility datasets (e.g., accidents_utility_spmf.txt) for testing. The datasets are in a specific format where each line represents a transaction, and items are followed by their utility values.
