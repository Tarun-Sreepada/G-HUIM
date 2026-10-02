# efim versus DPHIM (DPEFIM)

DPHIM: `~/dphim` ("Dynamic Parallelization for High-utility Itemset Mining"), EFIM search
on a C++20 coroutine task runtime (`nova`) with NUMA-aware scheduling and allocation.
Measured 2026-10-02 on accidents, two EPYC 9224 sockets (48 cores), one H100 PCIe for G-HUIM.

## Setup

- DPHIM built from the working tree of `~/dphim` (Release, g++-13, `-Ofast`, jemalloc) into a
  separate build directory. The checked-in `build/run` links `libpapi.so.6.0`, which this
  machine no longer has (7.1).
- `build/efim -t 48`, count-only. DPHIM `-a efim -t 48 -o /dev/null`.
- G-HUIM: `--total-memory 8G --local-tail-items 8 --local-tail-trans 32`, after the startup
  changes in [gpu_performance.md](gpu_performance.md).
- "Reported" is each program's own total (efim `Total ExecutionTime`, DPHIM `Total time ~`,
  G-HUIM `Total time`). Wall is `/usr/bin/time`. DPHIM sleeps 1.5 s after starting its
  scheduler (`main.cpp`, `exec_dp`), so its wall time includes that; compare reported times.
- Medians of three runs. All 36 runs matched: 3,030,087 / 10,079,776 / 104,518,894 itemsets.

## Results

| minutil | efim | G-HUIM | DPHIM `dphim` | DPHIM `sp` |
| --- | ---: | ---: | ---: | ---: |
| 8M reported / wall (s) | 0.399 / 0.46 | 0.322 / 0.70 | 1.972 / 3.61 | 3.113 / 3.20 |
| 6M reported / wall (s) | 0.491 / 0.56 | 0.387 / 0.76 | 3.932 / 5.61 | 4.628 / 4.71 |
| 3M reported / wall (s) | 1.161 / 1.25 | 0.838 / 1.22 | 21.50 / 23.41 | 18.30 / 18.39 |
| 3M peak host RSS | 3.4 GB | 0.35 GB (+1.9 GB device arena) | 20-27 GB | 0.5 GB |

Single run each, 8M, reported seconds:

| threads | efim | DPHIM `dphim` | DPHIM `sp` |
| ---: | ---: | ---: | ---: |
| 1 | 6.94 | 23.39 | 19.82 |
| 8 | 1.04 | 3.42 | |
| 24 | 0.49 | 1.44 | |
| 48 | 0.39 | 1.97 | 3.11 |

efim is 2.9x faster on one core and 18x faster at 3M with 48 threads. DPHIM's best time at 8M
is at 24 threads, so it gets slower when the second socket is added.

The `global`, `local` and `local-numa` schedulers could not be measured. On the committed
DPHIM they exit because PAPI cannot open perf counters (`perf_event_paranoid` = 4). With the
uncommitted PAPI change they abort with `bad alloc`: the 34 MB input exceeds `alpha1` (20 MB),
so the database is split per NUMA node and `cloneTransaction` calls `nova::malloc_on_thread`,
whose per-thread jemalloc arenas only the `dphim` scheduler sets up. The thresholds can only be
changed with `-s dphim`.

## Why efim is faster per core

| | efim | DPHIM |
| --- | --- | --- |
| Entry storage | SoA, `uint16` item + `int32` utility (6 B) | AoS `pair<uint32, uint64>` (16 B) |
| Transaction storage | Contiguous per level; projected views point into the parent | One heap block per owned transaction |
| Finding item x | Occurrence index per database: visits only transactions containing x | Binary search in every transaction |
| Upper bounds | Item-indexed scratch arrays, SIMD (AVX-512) filtering | `lower_bound` into `itemsToKeep` for each entry |
| Allocation in the hot path | Per-depth buffers reused; none once grown | New `Database` and cloned transactions per extension |
| Merging identical transactions | Adjacent comparison (reverse-lex order) | Same idea, adjacent comparison |

## Why DPHIM scales and uses memory as it does

- Every extension becomes a coroutine and `when_all` starts all siblings, so many projected
  databases are alive at the same time. That is the 20-27 GB RSS at 3M (`sp` stays at 0.5 GB).
  efim searches depth first and splits work only near the top or when a worker is idle.
- Its NUMA design moves *data*: transactions scatter across node-local jemalloc arenas, and
  tasks migrate to the node that holds a large partition. On a 340k-transaction database
  that fits in L3/DRAM easily this is overhead; remote frees and node-pinned arenas also cost.
  efim pins workers, lets each allocate its own data (first touch), and steals nearest first.
- `sp` (static partitioning) balances poorly: 48 threads give it only a 6.4x speedup at 8M
  (19.8 s to 3.1 s). At 3M it beats `dphim` (18.3 s vs 21.5 s), so dynamic scheduling does not
  recover its overhead on this workload.

## Advantages of DPHIM

- 32-bit items and 64-bit utilities: no 65,535-item limit and no `int32` utility limit,
  both of which efim rejects at load time.
- A general task runtime (coroutines, several schedulers) shared by EFIM and FHM.
- Designed for inputs far larger than these: NUMA partitioning and persistent memory
  (`--pmem`) target databases that are bandwidth- or capacity-bound, which accidents is not.
- Tunable speculation thresholds (alpha/beta).

## Disadvantages of DPHIM

- 3x slower per core and negative scaling past one socket on this workload.
- High peak memory from breadth-first task creation.
- Fixed 1.5 s sleep before mining; requires PAPI with perf access; non-`dphim` schedulers
  crash on inputs larger than 20 MB.
- Utility values are `uint64`, so negative utilities are not representable (neither tool
  supports them).

## Limits of this comparison

Only accidents (dense, 340k transactions, 148 promising items) was measured. DPHIM's paper
targets large sparse datasets (Kosarak, Chainstore, BMS). The ranking may differ there,
especially where efim's 16-bit item limit applies or the database exceeds per-node memory.

## Reproduce

```bash
cmake -S ~/dphim -B <dir> -DCMAKE_BUILD_TYPE=Release && cmake --build <dir> -j
<dir>/run -a efim -s dphim -t 48 -i datasets/accidents_utility_spmf.txt -o /dev/null -m 3000000
build/efim -i datasets/accidents_utility_spmf.txt -m 3000000 --separator space -t 48
```
