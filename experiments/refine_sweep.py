"""REFIM refinement rounds (-r 0..4 and -1 = until nothing drops): runtime, memory and search counters.

Usage: python3 experiments/refine_sweep.py [--reps 3] [--only chess] [--out experiments/results/refine.csv]

One thread per run, every (dataset, minutil, rounds, rep) at once, one pinned core each, half on each NUMA node,
so all configs share the same batches. Count mode (-o omitted), --stats for the counters. The minutils are
dphim_compare.THRESHOLDS. The binary is copied first, so rebuilding mid-run is safe.
"""

import argparse
import csv
import os
import queue
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

from dphim_compare import THRESHOLDS
from sweep import ROOT

CORES = list(range(0, 12)) + list(range(24, 36))
ROUNDS = [0, 1, 2, 3, 4, -1]
STATS = re.compile(r"stats: nodes (\d+) primary (\d+) secondary (\d+) \| refine rounds (\d+) secondary -(\d+) "
                   r"primary -(\d+) \| compacted (\d+) txs (\d+) -> (\d+) items (\d+) -> (\d+)")
STAT_FIELDS = ["nodes", "primary", "secondary", "refine_rounds", "refine_secondary", "refine_primary",
               "compacted", "compact_tx_in", "compact_tx_out", "compact_items_in", "compact_items_out"]
FIELDS = ["dataset", "minutil", "rounds", "rep", "cpu", "status", "patterns", "time_s", "rss_mb"] + STAT_FIELDS


def run(binary, data, minutil, rounds, timeout, cpu):
    cmd = ["taskset", "-c", str(cpu), "timeout", "-k", "5", str(timeout), "/usr/bin/time", "-f", "RSS %M",
           binary, "-i", data, "-m", str(minutil), "-t", "1", "-r", str(rounds), "-S"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = p.stdout + p.stderr
    count = re.search(r"High Utility Patterns: (\d+)", out)
    total = re.search(r"Total ExecutionTime in seconds: ([\d.]+)", out)
    rss, stats = re.search(r"RSS (\d+)", out), STATS.search(out)
    if p.returncode != 0 or not (count and total and rss and stats):
        return {"status": f"failed rc={p.returncode}"}
    return {"status": "ok", "patterns": int(count.group(1)), "time_s": float(total.group(1)),
            "rss_mb": int(rss.group(1)) / 1024, **{k: int(v) for k, v in zip(STAT_FIELDS, stats.groups())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--only", default="", help="comma-separated dataset name prefixes")
    ap.add_argument("--out", default=str(ROOT / "experiments/results/refine.csv"))
    args = ap.parse_args()
    only = [s for s in args.only.split(",") if s]
    bin_dir = tempfile.TemporaryDirectory()
    binary = shutil.copy(ROOT / "build/REFIM", bin_dir.name)
    free = queue.Queue()
    for c in CORES:
        free.put(c)

    def job(data, minutil, rounds, rep):
        cpu = free.get()
        try:
            return data, minutil, rounds, rep, cpu, run(binary, str(ROOT / "datasets" / data), minutil, rounds,
                                                         args.timeout, cpu)
        finally:
            free.put(cpu)

    # Hardest minutils first, so the long runs start early.
    jobs = [(d, m, r, rep) for d, ms in THRESHOLDS.items() if not only or any(d.startswith(o) for o in only)
            for m in sorted(ms) for rep in range(args.reps) for r in ROUNDS]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f, ThreadPoolExecutor(len(CORES)) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for data, minutil, rounds, rep, cpu, r in pool.map(lambda j: job(*j), jobs):
            row = {"dataset": data.split("_")[0].removesuffix(".txt"), "minutil": minutil, "rounds": rounds,
                   "rep": rep, "cpu": cpu, **r}
            w.writerow(row)
            f.flush()
            print(row, flush=True)


if __name__ == "__main__":
    main()
