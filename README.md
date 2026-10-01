# G-HUIM

GPU-accelerated high-utility itemset mining (EFIM) on NVIDIA hardware.

## Build

Requires CMake 3.20+, a C++20 compiler, and CUDA 12.x or 13.x. The first
configuration downloads pinned CLI11 2.6.2 and spdlog 1.16.0 dependencies.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j 4
```

Architecture 86 (RTX 3000 series) is the minimum; supply the architecture for
your GPU when configuring. Use `-DCMAKE_BUILD_TYPE=Debug` for a debug build.
The build produces `build/G-HUIM` (GPU miner) and `build/efim` (CPU baseline).

## Run

```bash
./build/G-HUIM --help
./build/G-HUIM --input datasets/accidents_utility_spmf.txt --minutil 4000000
./build/G-HUIM --input datasets/accidents_utility_spmf.txt --minutil 4000000 \
  --method adaptive-optimized --log-level debug
```

| Option | Meaning |
| --- | --- |
| `--input` | SPMF utility dataset (required) |
| `--minutil` | Minimum itemset utility (required) |
| `--method` | `static`, `adaptive` (default) or `adaptive-optimized`: how tasks are mapped to threads |
| `--block-size` | Threads per block-sized task, 32-1024 (default 256) |
| `--memory` | Device memory for projected databases (`K`/`M`/`G`); default uses most free memory |
| `--separator` | Separator between items within each SPMF field (default space) |
| `--log-level`, `--log-file` | Logging verbosity and optional log file |

Help and argument validation run before CUDA initialization. `--log-level debug`
prints stage timings, per-batch statistics and bin distributions.

## Checking results against the CPU baseline

`efim` is a multithreaded C++ EFIM used as the reference:

```bash
./build/efim <input> <output> <minUtil> [sep] [threads]
uv run experiments/compare_counts.py datasets/accidents_utility_spmf.txt 4000000
```

`compare_counts.py` runs `efim` once and `G-HUIM` with every method, then fails
if any pattern count differs. Other scripts in `experiments/` run benchmarks and
produce plots; see `experiments/adaptive_baseline.md`.

## Source layout

```
src/main.cpp              entry point
src/app/                  configuration, CLI11 options, logging
src/data/                 dataset loading: parse.cpp (parallel text parsing),
                          dataset.cpp (TWU pruning, renaming, rebuild)
src/mine/                 host side of the search
  miner.cu                batch loop (select, bin, launch, reclaim)
  bins.cu                 method -> kernel mapping, device auto-tune
  schedule.cu             batch selection, binning, arena compaction kernels
  arena_gc.cu             arena compaction driver
  root_image.cu           root database in arena format
  primitives.cu           scans and other device primitives
  baseline/kernels.cu     static / adaptive kernels and launchers
  optimized/              adaptive-optimized kernels: small.cu (warp, block,
                          lane-group tasks), huge.cu (multi-block tasks),
                          occupancy.cu (block-size selection)
include/                  headers mirroring src/
  mine/layout.cuh         database and task memory layout
  mine/device/            device helpers shared by all kernels
  mine/baseline/          thread groups and passes of the baseline kernels
  mine/optimized/         thread groups, projection and passes of the optimized kernels
  mine/host/              host/device copy and timing helpers
  cuda/buffer.cuh         checked CUDA allocations
efim/                     CPU EFIM baseline (level, scheduler, worker, reader, main)
experiments/              benchmark, comparison and plotting scripts
datasets/                 SPMF utility datasets; download_datasets.sh fetches more
```

Formatting follows the repository `.clang-format`.
