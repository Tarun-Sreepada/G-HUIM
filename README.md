# G-HUIM

GPU-accelerated high-utility itemset mining (EFIM) on NVIDIA hardware. This branch searches the
itemset tree level by level: every task of a level runs before any task of the next. (The
depth-first miner lives on the separate dfs-refactor branch.)

## Build

Requires CMake 3.20+, a C++20 compiler, and CUDA 12.x. The first configuration downloads pinned
CLI11 2.6.2 and spdlog 1.16.0.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j 4
```

The build produces `build/G-HUIM` (GPU miner) and `build/efim` (CPU baseline).

## Run

```bash
./build/G-HUIM --help
./build/G-HUIM --input datasets/accidents_utility_spmf.txt --minutil 3000000
./build/G-HUIM --input datasets/accidents_utility_spmf.txt --minutil 3000000 --log-level debug
```

| Option | Meaning |
| --- | --- |
| `--input` | SPMF utility dataset (required) |
| `--minutil` | Minimum itemset utility (required) |
| `--total-memory` | Device memory for projected databases (`K`/`M`/`G`); default uses the free memory |
| `--stack-capacity` | Most tasks one level may hold (default 128M) |
| `--output` | Write itemsets as `items... UTIL u`; without it they are only counted |
| `--separator` | Separator between items within each SPMF field (default space) |
| `--log-level`, `--log-file` | Logging verbosity and optional log file |

Help and argument validation run before CUDA initialization. `--log-level debug` prints stage
timings and per-iteration statistics.

### Running out of memory

Allocations never write past their buffers: a full arena region or level stops the run with a
message naming the option to raise. Memory is plain device memory (no managed-memory
oversubscription) and 512 MiB of the free memory stays unused for the runtime.

## Checking results against the CPU baseline

```bash
python3 experiments/compare_counts.py datasets/accidents_utility_spmf.txt 3000000
```

Runs `efim` and `G-HUIM`, and fails if the pattern counts differ.

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
include/                  headers mirroring src/
  mine/bfs/               layout, placement, device/ (groups, passes, child database)
  cuda/                   checked buffers, libcu++ atomics
efim/                     CPU EFIM baseline
experiments/              benchmark, comparison and plotting scripts
```

`experiments/timeline.py` and `iteration.py` parse the old per-iteration timer table, which this
version no longer prints; `compare_counts.py` and `run_experiments.py` use the new log lines.
