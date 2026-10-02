# GPU versus CPU: diagnosis and first improvement

Subsequent occurrence-index and local-subtree implementations are described in
[gpu_occurrence_index.md](gpu_occurrence_index.md). The measurements below are
the historical chunk-planning comparison.

Measured 2026-10-02 on the current working tree, one H100 PCIe versus 48 CPU
workers on two EPYC 9224 sockets. Dataset: accidents. GPU: BFS adaptive,
`--total-memory 8G`, no output file. Medians of three runs, seconds:

| Minimum utility | CPU reported total | GPU before | GPU after chunk planning change |
| --- | ---: | ---: | ---: |
| 8M | 0.405 | 0.322 | 0.320 |
| 6M | 0.507 | 0.455 | 0.424 |
| 3M | 1.194 | 1.630 | 1.318 |

CPU measurements preceded alternating before/after GPU runs. GPU totals exclude
CUDA context creation; these are not command wall times. At 3M, median command
wall time was 2.389 s before and 2.089 s after, versus 1.314 s for the CPU.
A separate, single run of the PGO CPU binary reported 1.119 s.
The GPU still loses at 3M; this change is a 19% reduction in its reported total,
not evidence of an overall CPU win.

The CPU measurements above predate the count-only fix: that version stored
itemsets even when no output file was requested. Both executables now only count
when output is omitted or set to `/dev/null`; a real output path enables itemset
collection. After the fix, three fresh 48-worker CPU runs at 3M gave a median
of 1.167 s and 3,403 MiB peak RSS, all with 104,518,894 patterns. These use
`build/efim`; the earlier PGO result is from the old binary and needs retraining
before comparison. For enumeration, use the same output requirements.

## Startup (measured later the same day)

Context creation was about 0.6 s because the machine has two H100s and the driver
initializes every visible GPU (`cuInit` about 475 ms with two, 190-280 ms with
one; the primary context adds about 135 ms). Module loading is already lazy in
CUDA 12.8 (`CUDA_MODULE_LOADING=EAGER` made no difference) and an 8 GiB
`cudaMalloc` takes about 1 ms. Two changes:

- `initialize_device` sets `CUDA_VISIBLE_DEVICES=0` when it is unset (only one
  GPU is used; set the variable yourself to choose another).
- The context is created on a separate thread while the dataset loads. The log
  prints the time spent waiting for it; `Total time` excludes that wait and
  `Wall time` includes everything after argument parsing.

At 3M with `--local-tail-items 8 --local-tail-trans 32`, three runs each, both
GPUs visible, command wall time: before 1.75/1.57/1.58 s, after 1.11/1.13/1.12 s,
CPU (`build/efim -t 48`) 1.25 s each run. All runs: 104,518,894 patterns. About
0.17 s is still spent waiting for the context after loading, plus about 0.1 s
of process start and teardown outside `Wall time`.

## What changed

`LevelMiner::plan_chunks` used to download every task's 64-bit scratch prefix
and scan it on the CPU to choose chunks. The largest 3M level has 21,585,429
tasks: that is about 173 MB downloaded for just that level.

The new scheduling kernel searches the monotonic prefix array on the GPU and
returns pages of at most 256 boundaries and their scratch bases (about 3 KB per
page). It chooses the same maximal fitting chunks and retains the oversized-task
error. The single-chunk fast path is unchanged. Paging handles arbitrarily many
chunks without allocating a boundary array proportional to the level size.

At 3M, reported mining-loop time fell from 1.415 s to 1.111 s. Despite the log
label `kernels`, this timer includes chunk planning, transfers, scheduling and
synchronization. It is not a sum of CUDA event times.

Files: `include/mine/bfs/kernels.cuh`, `src/mine/bfs/schedule.cu`,
`src/mine/bfs/bfs_miner.cu`.

## Why the other approaches have struggled

1. **The CPU avoids work the GPU still does.** The CPU uses an occurrence index
   containing transaction IDs and item positions, so it visits only matching
   transactions. The GPU searches every transaction for each extension item.
   The CPU can retain suffix views; the GPU materializes child databases.
   The CPU accumulates local/subtree utilities together with private counters;
   the GPU uses separate passes and atomic updates. These are different work
   and memory costs, even though both implement EFIM.
2. **Coalescing is only one cost.** Existing profiles and README experiments show
   that lane groups improve load coalescing but add scans, shuffles and other
   instructions. Staging adds copying, synchronization and shared-memory usage.
   Tile layout adds child-layout construction. Their net costs exceeded the
   savings on the measured dataset. The stored Nsight profiles predate this
   change and should not be presented as fresh profiles of today's kernels.
3. **Small tasks dominate the task population.** A separate serialized-bin
   diagnostic at 3M assigns about 96% of tasks to groups of at most 32 lanes.
   Millions of tiny projections still pay allocation, hashing, task emission
   and scheduling costs. Task frequency alone does not establish time dominance:
   the huge-task bin also accounts for substantial device time.
4. **Startup is a separate problem.** Context initialization was around 0.6 s
   in the repeated runs. Smaller workloads can win in reported processing time
   and lose in command wall time. Faster mining cannot remove process startup.

NVIDIA also cautions that higher occupancy need not improve performance:
[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).
The objective is lower elapsed time for equivalent results, not a better single
hardware metric.

## Next experiments, in order

1. Measure current scheduling and projection costs separately. In particular,
   inspect bin-count/scatter contention and the bytes written per useful task.
   Measure transaction hit rates using the existing `GHUIM_WORK_STATS` hooks
   in a separate instrumented build.
2. Prototype occurrence indexes for large projected databases. Build one index
   per parent and share it across sibling extensions. Include index construction
   and storage in the timing; bypass it for small databases when it cannot pay
   for itself. This removes unsuccessful searches and repeated position lookup.
3. Prototype a bounded local DFS for small subtrees: keep descendants within a
   warp or block, reuse local storage, and send overflow/large work to the
   existing scheduler. Merely selecting today's `--search dfs` does not provide
   this: a separate 3M run took 2.182 s total and performed four arena compactions.
   Choose the cutoff by measurements, with an explicit resource bound and fallback.
4. For repeated queries, retain the CUDA context and reusable allocations in a
   long-lived process. Report cold-command time and steady-state query time
   separately; threshold-dependent pruning still needs correct rebuilding.

Do not combine all these into one rewrite. Require a repeatable elapsed-time win
and correct outputs from each experiment. A large speedup is plausible only if
substantial work or materialization is removed; no measured result here supports
a promised 10x improvement or a win across datasets.

## Validation and reproduction

- Release build and existing CTest passed.
- All 18 successful before/after benchmark runs matched the CPU counts:
  3,030,087 at 8M; 10,079,776 at 6M; 104,518,894 at 3M.
- All six BFS methods matched 255 exhaustively enumerated itemsets and exact
  utilities on a deterministic small dataset.
- The chunk planner passed 106 cases, including exact capacity, oversized tasks,
  zero sizes, task-count limits, randomized inputs and multiple result pages.
  Compute Sanitizer memcheck reported zero errors for this focused test.

Run the focused planner check with:

```bash
nvcc -std=c++20 -arch=sm_90 -Iinclude profiling/check_chunk_plan.cu \
  src/mine/bfs/schedule.cu -o build/check_chunk_plan
compute-sanitizer --tool memcheck --error-exitcode 1 build/check_chunk_plan
```

Benchmark commands:

```bash
build/efim -i datasets/accidents_utility_spmf.txt -m 3000000 --separator space -t 48
build/G-HUIM --input datasets/accidents_utility_spmf.txt --minutil 3000000 --total-memory 8G
```

Raw logs, benchmark JSON, the saved pre-change GPU executable, exhaustive test
data, and sanitizer output are in `build/gpu-diagnosis/` (local build artifacts).
One initial A/B sequence stopped on a CUDA context-creation out-of-memory error
before mining began. A GPU/memory check showed ample free memory; the repeated
3M sequence succeeded. Failed runs were not included in the medians.
