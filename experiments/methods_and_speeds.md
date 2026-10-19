# G-HUIM methods, measurements and optimizations

(Moved out of the README; section links below refer to this file.)

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
