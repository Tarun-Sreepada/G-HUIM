# G-HUIM

For 2026 DASFAA paper please refer to the following commit:
[b08df0c](https://github.com/Tarun-Sreepada/G-HUIM/commit/b08df0c2de23f2949935fc2522f66ff672a396b7)

GPU-accelerated High-Utility Itemset Mining on NVIDIA hardware (`G-HUIM/`), plus a CPU miner, REFIM (`REFIM/`).

---

## Requirements

- CUDA toolkit 12.x, an sm_80 or newer GPU (G-HUIM only)
- CMake ≥ 3.24, a C++20 host compiler and NVCC
- Network on first configure: pinned CLI11 and spdlog are fetched

---

## Configure & Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j"$(nproc)"
```

Produces `build/G-HUIM` and `build/refim`. Each project also builds alone
(`cmake -S G-HUIM -B build-ghuim`, `cmake -S REFIM -B build-refim`). The GPU architecture is
auto-detected; override with `-DCMAKE_CUDA_ARCHITECTURES=...`.

---

## Run

```bash
./build/G-HUIM -h
./build/G-HUIM --input datasets/accidents_utility_spmf.txt --minutil 3000000 --output results.txt
./build/refim  --input datasets/accidents_utility_spmf.txt --minutil 3000000 --threads 0
```

| G-HUIM option | Meaning |
| --- | --- |
| `-i`, `--input`, `-m`, `--minutil` | SPMF dataset and minimum utility (required) |
| `-e`, `--search` | `bfs` (default) or `dfs` |
| `-M`, `--method` | `adaptive` (default), `static`, `staged` or `tile` (last two BFS only) |
| `-b`, `--block-size` | Threads per task for `static` (32-1024, default 256) |
| `-g`, `--total-memory` | Device memory for projected databases (`K`/`M`/`G`); default uses free memory |
| `-C`, `--stack-capacity` | Most tasks one level may hold (BFS); default derived from device memory |
| `-o`, `--output` | Write itemsets (BFS only); without it they are only counted |

Both miners also take `-s/--separator`, `-r/--refine`, `-l/--log-level` and `-L/--log-file`. REFIM adds
`-t/--threads` (`0` = all), `-n/--numa`, `-v/--simd` and `-p/--thp`. See `--help` for the rest.
Method details, tuning options and measurements: [experiments/methods_and_speeds.md](experiments/methods_and_speeds.md).

Check results against the CPU baseline (all search/method pairs must match REFIM's count):

```bash
python3 experiments/compare_counts.py datasets/accidents_utility_spmf.txt 3000000
```

---

## Layout

Two self-contained projects, each with its own `CMakeLists.txt`, `include/` and `src/`. Drivers are short
lists of named steps, one small file per step.

```
G-HUIM/   GPU miner: src/app (CLI, main steps), src/data (loading), src/mine/{bfs,dfs} (engines)
REFIM/    CPU miner: src/app (options, run), src/io (reader steps), src/mine (search, workers), src/sys
experiments/   benchmark, comparison and plotting scripts
```

---

## Formatting

```bash
find G-HUIM REFIM -type f \( -name '*.hpp' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.cu' \) -print0 | \
  xargs -0 clang-format -i
```

---

## Datasets

SPMF utility datasets (e.g. `accidents_utility_spmf.txt`) go in `datasets/`; `datasets/download_datasets.sh`
fetches them. Each line is a transaction: items, total utility, then item utilities.
https://www.philippe-fournier-viger.com/spmf/index.php?link=datasets.php
