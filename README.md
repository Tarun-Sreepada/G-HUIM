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
`-DCMAKE_CUDA_ARCHITECTURES=...` to override it. The build produces `build/G-HUIM` (GPU miner) and `build/efim` (CPU baseline).

CPU baseline usage: `build/efim -i data.txt -m <minutil> [-o out.txt] [-t 48]`. Workers are pinned and steal work
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
Items are stored in 16 bits and utilities in 32 bits, so `efim` stops with an error if more than 65535 items
survive TWU pruning (raise minutil) or if any item's total utility exceeds 2^31-1. `build/efim` is LTO-built.

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
| `-M`, `--method` | `adaptive` (default, size-binned kernels: lanes, warp, block or several blocks per task), `static` (one block of `--block-size` threads per task), `warp` (BFS only: a lane group of 4, 8, 16 or 32 lanes per transaction), `staged` (BFS only: adaptive kernels with IO warps staging transactions in shared memory), `stagedwarp` (BFS only: `staged` with lane groups) or `tile` (BFS only: adaptive scheduling over databases stored in warp tiles). See [Methods and speeds](#methods-and-speeds) |
| `-b`, `--block-size` | Threads per task for `--method static`: a power of two, 32-1024 (default 256) |
| `-w`, `--io-warps` | `staged`/`stagedwarp`: IO warps added to each block, 1-4 (default 1; 4 was fastest) |
| `-R`, `--ring-kib` | `staged`/`stagedwarp`: shared memory for the tile ring, KiB (default 64) |
| `-T`, `--smem-tables`, `-K`, `--table-kib` | `adaptive`/`staged`: keep per-item tables of block-sized tasks in shared memory (default off; no gain measured) |
| `-O`, `--oversubscribe` | BFS: size tasks for this many times the resident warps (default 12; more, smaller blocks balance the load) |
| `-P`, `--shape-modes` | BFS `adaptive`: mode per task shape (longest transaction <=4, <=8, <=16, longer), e.g. `lane,lane,lane,g32`; modes are `lane`, `group`, `g4`, `g8`, `g16`, `g32` (default all `lane`) |
| `-E`, `--min-entries-per-warp` | BFS: size tasks by entries instead of transactions (default 0 = by transactions; neutral so far) |
| `-B`, `--bin-times` | BFS: run the cells one after another and log the time of each per level |
| `-I`, `--index-min-trans` | BFS adaptive/static lane kernels: build shared occurrence indexes for databases with at least this many transactions; `0` disables them (default). Indexes add memory and construction work; measure before enabling |
| `-k`, `--local-tail-items` | BFS adaptive, count-only: process small subtrees locally when at most this many items remain; `0` disables it (default), maximum 20 |
| `-j`, `--local-tail-trans` | Maximum supporting transactions for local subtree processing, 1-32 (default 4); try `--local-tail-items 8 --local-tail-trans 32` |
| `-r`, `--refine` | Bound refinement: recompute each task's local utilities without the items it drops, this many rounds (default 1; `-1` = until none drops, `0` = off; huge tasks run at most 8). Same itemsets; fewer extensions on sparse data |
| `-g`, `--total-memory` (`--memory`) | Device memory for projected databases (`K`/`M`/`G`); default uses the free memory |
| `-C`, `--stack-capacity` | Most tasks one level may hold (BFS; default 128M) |
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
./build/G-HUIM $IN --method warp                                        # lane group per transaction
./build/G-HUIM $IN --method staged --io-warps 4 --ring-kib 64           # IO warps stage transactions in shared memory
./build/G-HUIM $IN --method stagedwarp --io-warps 4 --ring-kib 32       # IO warps + lane groups
./build/G-HUIM $IN --shape-modes lane,lane,lane,g32                     # per-shape mode (adaptive); all lane is the default
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

A bin is further split by the task's shape, the length class of its longest transaction (<=4, <=8,
<=16, longer), so each (size, shape) cell can run its own kernel (`--shape-modes`; every shape uses the
same lane-per-transaction mode by default). The cell counts are updated on the GPU as tasks are created,
and every non-empty cell is launched. In the kernel a lane owns one transaction: a warp covers 32
consecutive transactions per step, each lane searches its own transaction and walks the entries behind the
extension item. This is simple and has little overhead, but neighbouring lanes read far-apart memory,
which makes the loads uncoalesced.

**`static`.** Every task gets one block of `--block-size` threads, whatever its size. The warps split the
transactions as in `adaptive`. It is the simple baseline: small tasks waste most of the block and huge
tasks run on a single block.

**`warp`.** A group of 4, 8, 16 or 32 lanes owns one transaction at a time, and its lanes sit on
consecutive entries, so the key and utility arrays are read coalesced. The group size comes from the
task's longest transaction. Positions inside the transaction come from ballots, popcounts and group scans
instead of per-lane loops. The loads get much better (about 1.7 sectors per request instead of about 10), but the same work
takes several times the instructions, so it ends up limited by instruction issue.

**`staged`.** `adaptive`, with some warps added to each block as IO warps. An IO warp copies tiles of 32
transactions from global memory into a ring of shared-memory slots with `cp.async`, keeping up to two
tiles in flight. The other (compute) warps run the lane-per-transaction passes on the tiles, so their
loads come from shared memory, laid out so the 32 lanes fall on different banks. Per-slot counters
tell the compute warps when a tile has landed and the IO warps when a slot is free. All waits are bounded,
so a protocol bug ends the kernel with an error and does not hang the GPU. It applies to block-sized and
huge tasks; smaller tasks run unstaged. From pass 2 on, only transactions that contain the extension
item are copied. `--io-warps` sets how many IO warps a block gets and `--ring-kib` how much shared
memory the ring has, which is also the prefetch depth.

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

**`stagedwarp`.** The same IO-warp ring, but each compute warp takes a whole tile and its lane groups
split the tile's transactions, as in `warp`.

`--search dfs` processes tasks from a stack instead of level by level and has its own `static` and
`adaptive` kernels; `warp`, `staged` and `stagedwarp` are BFS only.

Each cell is **kernel time / total time**, the `kernels` and `Total time` log lines, best of 3, on
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

## Checking results against the CPU baseline

```bash
python3 experiments/compare_counts.py datasets/accidents_utility_spmf.txt 3000000
```

Runs `efim` and `G-HUIM` for all four search/method pairs (`--configs bfs-adaptive ...` to pick), and
fails if any pattern count differs from EFIM's.

## Source layout

```
src/main.cpp              entry point
src/app/                  configuration, CLI11 options, logging
src/data/                 dataset loading: parallel parsing, TWU pruning, renaming, rebuild
src/mine/                 miner entry, itemset output, scan primitives
  bfs/                    level-synchronous miner
    bfs_miner.cu            host driver (levels, chunks, memory budget)
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
  cuda/                   checked buffers, libcu++ atomics
efim/                     CPU EFIM baseline (main.cpp)
  io/                       dataset reader, itemset output
  mine/                     search: levels, scheduler, workers, SIMD projection
  sys/                      NUMA, allocator, threading helpers
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
