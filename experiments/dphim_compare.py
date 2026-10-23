"""REFIM vs DPHIM (Kimura et al., TKDE 2026; ~/dphim): runtime and peak RSS, with every pattern output.

Usage: python3 experiments/dphim_compare.py [--reps 3] [--only chess,mushroom] [--threads 1,24,48]
                                           [--mode output|count] [--out experiments/results/dphim.csv]

  output   : (default) both miners keep every pattern and write it to a symlink to /dev/null. Both treat the
             literal path /dev/null as count-only, so the symlink makes them do the real work: REFIM then
             enumerates every pattern too (its closed-form subtree counts are off when patterns are kept).
  count    : -o /dev/null; REFIM counts subtrees without enumerating them, DPHIM still visits each pattern.
  1 thread : every (dataset, minutil, miner, rep) at once, one pinned core each, half on each NUMA node;
             DPHIM runs its sequential EFIM (-s sp), since its scheduler pins itself to core 0 and fails
             under taskset.
  >1 thread: serial; DPHIM with its own scheduler (-s dphim, default thresholds), which fills node 0's cores
             before node 1's, so REFIM runs on node 0 alone (-N 0) whenever the threads fit there (24 here).
             Minutils run easiest first; a miner that fails or times out skips the harder ones.
  runtime  : time_s = each miner's own total, mining plus keeping the patterns in memory (REFIM "Total
             ExecutionTime", DPHIM json total_time); wall_s adds start-up and writing the patterns.
  memory   : peak RSS from /usr/bin/time. Each run is capped by a cgroup (MemoryMax) so kept patterns
             cannot exhaust the shared machine; a run over the cap is killed and recorded as failed.
"""

import argparse
import csv
import json
import os
import queue
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sweep import ROOT

DPHIM = Path.home() / "dphim/build/run"
CORES = list(range(0, 8)) + list(range(24, 32))  # 1-thread runs at once, 8 per node
NODE_CPUS = 24  # cores per NUMA node
FIELDS = ["dataset", "minutil", "miner", "threads", "rep", "cpu", "status", "patterns", "time_s", "wall_s", "rss_mb"]

# Per dataset, about 1e8, 1e7 and 1e6 patterns (bisected with REFIM count mode); kosarak, pumsb and BMS keep
# sweep.py's hardest, middle and easiest points, which are below 1e8 patterns already.
THRESHOLDS = {
    "chess_utility_spmf.txt": [196_972, 278_319, 367_187],
    "connect_utility_spmf.txt": [8_291_015, 10_587_890, 12_474_609],
    "accidents_utility_spmf.txt": [3_048_925, 6_015_234, 10_102_148],
    "mushroom_utility_spmf.txt": [15_848, 44_806, 101_763],
    "ecommerce_utility_no_timestamps.txt": [403_026, 529_003, 738_476],
    "chainstore.txt": [89_581, 90_968, 92_077],
    "kosarak_utility_spmf.txt": [640_000, 642_000, 644_000],
    "pumsb_utility_spmf.txt": [8_800_000, 9_000_000, 9_200_000],
    "BMS_utility_spmf.txt": [1_989_000, 1_991_000, 1_993_000],
}


def command(miner, data, minutil, threads, sink):
    if miner == "REFIM":
        nodes = ["-N", "0"] if 1 < threads <= NODE_CPUS else []
        return [str(ROOT / "build/REFIM"), "-i", data, "-m", str(minutil), "-t", str(threads), "-o", sink] + nodes
    sched = "sp" if threads == 1 else "dphim"
    return [str(DPHIM), "-a", "efim", "-s", sched, "-t", str(threads), "-i", data, "-o", sink,
            "-m", str(minutil), "--json"]


def run(miner, data, minutil, threads, timeout, cpu=None, sink="/dev/null", memory_max="200G"):
    cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={memory_max}", "-p", "MemorySwapMax=0",
           "timeout", "-k", "5", str(timeout), "/usr/bin/time", "-f", "WALL %e RSS %M"]
    if cpu is not None:
        cmd = ["taskset", "-c", str(cpu)] + cmd
    p = subprocess.run(cmd + command(miner, data, minutil, threads, sink), capture_output=True, text=True)
    wall = re.search(r"WALL ([\d.]+) RSS (\d+)", p.stderr)
    row = {"status": f"failed rc={p.returncode}"}
    if p.returncode != 0 or not wall:
        return row
    if miner == "REFIM":
        count = re.search(r"High Utility Patterns: (\d+)", p.stdout)
        total = re.search(r"Total ExecutionTime in seconds: ([\d.]+)", p.stdout)
        if not count or not total:
            return row
        patterns, time_s = int(count.group(1)), float(total.group(1))
    else:
        try:
            result = json.loads(p.stdout[p.stdout.index("{"):])["result"]
        except (ValueError, KeyError):
            return row
        patterns, time_s = result["hui_count"], result["total_time"] / 1000
    return {"status": "ok", "patterns": patterns, "time_s": time_s, "wall_s": float(wall.group(1)),
            "rss_mb": int(wall.group(2)) / 1024}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", default="1,24,48")
    ap.add_argument("--mode", default="output", choices=["output", "count"])
    ap.add_argument("--timeout-single", type=int, default=900)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--memory-single", default="20G", help="cgroup MemoryMax of each 1-thread run")
    ap.add_argument("--memory", default="200G", help="cgroup MemoryMax of each multi-thread run")
    ap.add_argument("--only", default="", help="comma-separated dataset name prefixes")
    ap.add_argument("--out", default=str(ROOT / "experiments/results/dphim.csv"))
    args = ap.parse_args()
    only = [s for s in args.only.split(",") if s]
    datasets = {d: sorted(ms, reverse=True) for d, ms in THRESHOLDS.items()
                if not only or any(d.startswith(o) for o in only)}
    thread_counts = [int(t) for t in args.threads.split(",")]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sink_dir = tempfile.TemporaryDirectory()
    sink = "/dev/null"
    if args.mode == "output":
        sink = os.path.join(sink_dir.name, "patterns.txt")
        os.symlink("/dev/null", sink)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()

        def record(data, minutil, miner, threads, rep, cpu, r):
            row = {"dataset": data.split("_")[0].removesuffix(".txt"), "minutil": minutil, "miner": miner,
                   "threads": threads, "rep": rep, "cpu": cpu, **r}
            w.writerow(row)
            f.flush()
            print(row, flush=True)

        if 1 in thread_counts:
            free = queue.Queue()
            for c in CORES:
                free.put(c)

            def job(data, minutil, miner, rep):
                cpu = free.get()
                try:
                    r = run(miner, str(ROOT / "datasets" / data), minutil, 1, args.timeout_single, cpu, sink,
                            args.memory_single)
                finally:
                    free.put(cpu)
                return data, minutil, miner, 1, rep, cpu, r

            jobs = [(d, m, miner, rep) for d, ms in datasets.items() for m in reversed(ms)  # hardest first
                    for rep in range(args.reps) for miner in ("REFIM", "DPHIM")]
            with ThreadPoolExecutor(len(CORES)) as pool:
                for result in pool.map(lambda j: job(*j), jobs):
                    record(*result)

        for threads in (t for t in thread_counts if t > 1):
            for data, ms in datasets.items():
                for miner in ("REFIM", "DPHIM"):
                    for minutil in ms:
                        ok = True
                        for rep in range(args.reps):
                            r = run(miner, str(ROOT / "datasets" / data), minutil, threads, args.timeout, None,
                                    sink, args.memory)
                            record(data, minutil, miner, threads, rep, None, r)
                            if r["status"] != "ok":
                                ok = False
                                break
                        if not ok:
                            break


if __name__ == "__main__":
    main()
