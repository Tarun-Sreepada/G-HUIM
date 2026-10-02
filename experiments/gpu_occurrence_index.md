# Occurrence indexes and local subtree processing

Both paths are opt-in. They implement different ways to avoid work:

- `--index-min-trans N`: databases with at least N transactions carry item offsets
  and `(transaction ID, position)` occurrence lists. All sibling tasks share one
  index. Their scratch allocation and scheduling use occurrence counts, and their
  projection passes read only those transactions. Zero disables indexing.
- `--local-tail-items K --local-tail-trans T`: small subtrees are completed within
  a lane group, directly reading immutable parent suffixes. The local stack stores
  per-lane utility and position, plus pending-item bitsets. It avoids writing child
  databases, merging them, and globally scheduling their descendants.

The index supports BFS adaptive/static lane kernels, including output. Local
subtrees support BFS adaptive lanes, count-only. K is bounded to 20 and T to 32;
larger tasks fall back to the ordinary projection path. Default K is zero (off).
The local kernel is compiled separately so its stack storage does not penalize
runs with the option disabled.

## Correctness of local counting

Each lane carries the utility and matching position of the current prefix in
one supporting transaction. Extending a prefix drops lanes without that item.
Remaining suffix utility supplies an upper bound for pruning. A lower bound
uses only transactions containing every remaining candidate item: when their
prefix utilities already reach the threshold, every subset qualifies and the
kernel can add `2^k` directly. Otherwise it traverses descendants normally.
The input parser accepts nonnegative utilities, which these bounds require.

A single supporting transaction uses a specialized bounded subset counter.
Merged transaction utilities remain additive, so both paths count exact utility
itemsets. Counting does not require preserving individual descendant nodes;
writing itemsets therefore uses the ordinary path and rejects local-subtree flags.

This is suffix reuse within a bounded subtree, not arbitrary references to older
BFS arena regions. Those regions are still recycled at level boundaries. Logged
task counts count globally scheduled work; local descendants are included in the
pattern total at the level where their ancestor was scheduled.

## Reproduction

Recommended local-path trial on accidents:

```bash
build/G-HUIM --input datasets/accidents_utility_spmf.txt --minutil 3000000 \
  --total-memory 8G --local-tail-items 8 --local-tail-trans 32
```

Index experiment (the extra storage needs more arena memory at 3M):

```bash
build/G-HUIM --input datasets/accidents_utility_spmf.txt --minutil 3000000 \
  --total-memory 16G --index-min-trans 256
```

Both flags can be combined. An index is not automatically a win: the initial
3M index-only experiment increased peak arena use from 1,914 to 3,617 MiB and
was slower than the original projection path. Building the root index, building
child indexes and moving their extra bytes all count toward processing time.

## Measured results

Measured 2026-10-02, medians of three runs with rotated execution order. All GPU
runs in this table use `--total-memory 8G`. Local uses `--local-tail-items 8
--local-tail-trans 32` without an occurrence index. Before is the saved executable
from immediately before this work (it already includes GPU chunk planning).

| Minimum utility | CPU count-only total | GPU before total | New GPU, options off | GPU local total | GPU local mining loop |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8M | 0.402 s | 0.323 s | 0.321 s | 0.304 s | 0.091 s |
| 6M | 0.506 s | 0.419 s | 0.420 s | 0.371 s | 0.159 s |
| 3M | 1.170 s | 1.320 s | 1.335 s | **0.818 s** | 0.610 s |

At 3M this is 38% less reported total time than the previous GPU and 30% less
than the CPU. Globally scheduled tasks fall from 126,296,640 to 15,093,037
(88% fewer), with the same 104,518,894 patterns. At 8M and 6M the counts also
match: 3,030,087 and 10,079,776. These 36 comparison runs all passed count checks.

The cold-command comparison still favors the CPU: at 3M, median command wall
time is **1.255 s CPU versus 1.610 s GPU local** (2.098 s for the previous GPU).
This implementation improves mining and scheduling; it does not amortize CUDA
context startup across queries.

## Validation

```bash
python3 experiments/check_gpu_index.py
```

The deterministic exhaustive check covers single transactions, merged suffixes,
sparse item occurrences, zero utilities, huge-task kernels, pruning, empty results,
adaptive/static index construction, exact output utilities and three local group
limits. It compares 36 configurations against independent exhaustive answers.

CUDA memcheck, synccheck and racecheck on a 400-transaction fixture with indexing
and local processing enabled reported zero errors and zero race hazards. Raw
logs, the pre-change executable and benchmark JSON are in `build/gpu-index/`.
Measurements are on accidents only, one H100 PCIe versus 48 CPU workers on two
EPYC 9224 sockets. CPU runs omit output and use the corrected count-only path.
GPU reported total time excludes CUDA context initialization; command wall time
must be compared separately.
