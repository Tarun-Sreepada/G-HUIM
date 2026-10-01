# Adaptive unified-memory baseline

Measured on 2026-10-01 using an NVIDIA RTX 3070 (8 GiB VRAM), CUDA 13.4.92,
GNU 16.2.1, and a Release build targeting SM 86.

## Original implementation configuration

All runs use `--managed true --adaptive true --total_memory 8G
--pattern_memory 16M --stack_capacity 256K`. The initial block size is 1024.
The 8 GiB setting applies to the custom unified-memory allocator, not total
process memory. Task stacks mix managed and device allocations. The original
pattern implementation interprets 16M as a count of uint32 elements, allocating
64 MiB despite printing 16 MiB. Its separate output buffer reserves 128 GiB of
managed virtual address space; it does not touch all of that memory in these runs.

## Results

Elapsed seconds below include process startup, mining, and output writing.
Each workload ran twice. All eight runs exited 0 with no CUDA error or stack
overflow diagnostics. Sorted output lines match between repetitions, line counts
match reported pattern counts, and output utilities meet the requested threshold.
This checks repeatability and execution stability. Independent Python EFIM
comparisons are recorded below.

| Dataset | Minimum utility | Patterns | Elapsed seconds (two runs) |
| --- | ---: | ---: | --- |
| retail_utility_spmf.txt | 100,000 | 22 | 0.515, 0.415 |
| retail_utility_spmf.txt | 30,000 | 114 | 0.465, 0.465 |
| chainstore.txt | 10,000,000 | 10 | 0.415, 0.415 |
| chainstore.txt | 1,000,000 | 382 | 12.037, 12.036 |

Retail at 1,000,000 and chainstore at 100,000,000 also exited 0 but returned
zero patterns, so they are not the recommended workloads for exercising mining.
Chainstore at 1,000,000 reaches approximately 6,626 MiB of allocator offset.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
./build/G-HUIM --input datasets/retail_utility_spmf.txt --minutil 30000 \
  --managed true --adaptive true --total_memory 8G \
  --pattern_memory 16M --stack_capacity 256K --output build/retail-patterns.txt
./build/G-HUIM --input datasets/chainstore.txt --minutil 1000000 \
  --managed true --adaptive true --total_memory 8G \
  --pattern_memory 16M --stack_capacity 256K --output build/chainstore-patterns.txt
```

## Fixes required before establishing the baseline

- Support CUDA 13 memory advice/prefetch signatures while retaining CUDA 12 calls.
- Allocate subtree utility metadata for the inclusive maximum remapped ID and
  zero its counters before atomic accumulation. The original allocation produced
  out-of-bounds atomic accesses under Compute Sanitizer on retail.

Raw logs, outputs, and timing records are in `build/workload-tests/` (ignored by Git).

## CUDA memory checks

Retail at minimum utility 30,000 completed Compute Sanitizer memcheck with
`ERROR SUMMARY: 0 errors` after the metadata fix. Chainstore at 1,000,000 did
not complete its instrumented run within a 120-second timeout; its ordinary
runs completed twice in approximately 12 seconds. That timeout does not establish
whether the full chainstore workload is memory-error-free.

Chainstore at minimum utility 10,000,000 also completed memcheck with
`ERROR SUMMARY: 0 errors`.

## Independent Python EFIM comparison

The comparison imports the supplied repository-root `efimparallel.py`, runs
`efimParallel` once per workload with two joblib workers and the `threading`
backend, and reads `getPatterns()`. The threading backend shares the read-only
database rather than serializing it for each task. It preserves the reference
mining algorithm but its timings are not multiprocessing speed benchmarks.
The supplied `save()` reads `_finalPatterns` instead of `Patterns`, so the
comparison harness writes the returned patterns directly.

Both outputs are normalized to sorted numeric itemsets and compared for missing
itemsets, extra itemsets, duplicate itemsets, and exact integer utility values.

Reproduce a comparison using the isolated environment:

```bash
uv venv build/efim-venv --python 3.12
uv pip install --python build/efim-venv/bin/python pami joblib psutil deprecated
build/efim-venv/bin/python experiments/compare_efim.py \
  --input datasets/retail_utility_spmf.txt --minutil 30000 \
  --cuda-output build/workload-tests/final-retail_utility_spmf-30000-1.txt \
  --threads 2 --backend threading \
  --report build/workload-tests/compare-retail_utility_spmf-30000.json
```

The installed dependency versions are captured in
`build/workload-tests/efim-requirements.txt`.

### Completed exact comparisons

| Dataset | Minimum utility | Python / CUDA patterns | Python seconds | Result |
| --- | ---: | ---: | ---: | --- |
| retail_utility_spmf.txt | 100,000 | 22 / 22 | 1.257 | Exact match |
| retail_utility_spmf.txt | 30,000 | 114 / 114 | 11.106 | Exact match |
| chainstore.txt | 10,000,000 | 10 / 10 | 13.953 | Exact match |

Chainstore at 1,000,000 was stopped at the user's request after approximately
eight minutes of Python mining. No reference result was produced for that
threshold; its CUDA output has not been independently verified. Use retail at
100,000 and chainstore at 10,000,000 for quick, nonempty correctness baselines.

## First application refactor validation

The refactored build uses CLI11 2.6.2 and spdlog 1.16.0, with a separate CLI
library, CUDA core library, and C++ entry point. Options are typed, CUDA allocations
have local owners, parser item storage is sized from actual transaction counts,
and collected results grow in a host vector instead of a 128 GiB managed reservation.
Pattern memory now means bytes; 64M preserves the original baseline's 16M allocation.

Validation used `--managed true --adaptive true --total-memory 8G
--pattern-memory 64M --stack-capacity 256K`. Fresh runs of the supplied Python
EFIM implementation matched every CUDA itemset and utility:

| Dataset | Minimum utility | Python / CUDA patterns | Result |
| --- | ---: | ---: | --- |
| retail_utility_spmf.txt | 100,000 | 22 / 22 | Exact match |
| chainstore.txt | 10,000,000 | 10 / 10 | Exact match |

Both baseline workloads completed Compute Sanitizer memcheck with zero errors.
All CTest groups passed: configuration validation, help without a GPU, and GPU
regressions against exhaustive answers in fixed/adaptive modes. GPU regressions
also cover singleton transactions, CRLF/missing final newline, empty results,
and nonzero exits for exhausted pattern buffers, task stacks, and allocator pools.
Experiment metric extraction was checked against both new debug logs.

Refactor logs, timing records, and exact comparison reports are in
`build/workload-tests/refactor-*`.
