# G-HUIM

GPU-accelerated High-Utility Itemset Mining on NVIDIA hardware.

---

## Requirements

- CUDA toolkit 12.x (tested with 12.2)
- CMake ≥ 3.20
- A C++20-compatible host compiler and NVCC

---

## Configure & Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j"$(nproc)"
```

For Debug builds swap `-DCMAKE_BUILD_TYPE=Release` with `-DCMAKE_BUILD_TYPE=Debug`.

---

## Run

Show the runtime options:

```bash
./build/G-HUIM -h
```

Typical execution (update paths and thresholds to match your dataset):

```bash
./build/G-HUIM \
  --input datasets/accidents_utility_spmf.txt \
  --minutil 8000000 \
  --method SP \
  --adaptive true \
  --total_memory 20G \
  --output results.txt
```

Use `--block_size <N>` when running in fixed-block mode and `--pattern_memory` to cap pattern buffer size.

---

## Formatting

Apply the repository `.clang-format` style to all headers and sources:

```bash
find include src -type f \( -name '*.hpp' -o -name '*.h' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.cu' \) -print0 | \
  xargs -0 clang-format -i
```

---

## Datasets

Sample SPMF-formatted datasets live in `datasets/`. Use `datasets/download_datasets.sh` to fetch additional inputs or generate synthetic workloads from `experiments/synthetic.py`.
