"""REFIM (CPU) vs G-HUIM (GPU): runtime, patterns, memory and energy across minimum utility.

Usage: python3 experiments/sweep.py [--reps 2] [--only chess,mushroom] [--out experiments/results/sweep.csv]

Runs are serial (energy is read system-wide), each under `timeout`, G-HUIM on a single device.
  runtime  : the miner's own "Total time" (REFIM and G-HUIM both exclude process start-up).
  memory   : host peak RSS (/usr/bin/time) for both; G-HUIM also reports device peak arena + scratch.
  energy   : CPU package energy via `perf stat -a -e power/energy-pkg/` (both sockets), plus, for G-HUIM,
             GPU board energy integrated from 100 ms nvidia-smi power samples. REFIM's GPU is idle and not counted.
"""

import argparse
import csv
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GPU = "0"


def ladder(start, step, points=5):
    return [start + step * i for i in range(points)]


# Each ladder starts at a low minutil (long runs) and steps up.
SWEEP = {
    "chess_utility_spmf.txt": ladder(50_000, 10_000, 6),
    "connect_utility_spmf.txt": ladder(3_000_000, 500_000),
    "accidents_utility_spmf.txt": ladder(900_000, 100_000),
    "mushroom_utility_spmf.txt": ladder(10, 10, 6),
    "ecommerce_utility_no_timestamps.txt": ladder(50_000, 10_000),
    "pumsb_utility_spmf.txt": ladder(8_800_000, 100_000),
    "kosarak_utility_spmf.txt": ladder(640_000, 1_000),
    "chainstore.txt": ladder(71_000, 1_000),
    "BMS_utility_spmf.txt": ladder(1_989_000, 1_000),
}

PATTERNS = {
    "REFIM": (r"High Utility Patterns: (\d+)", r"Total ExecutionTime in seconds: ([\d.]+)"),
    "G-HUIM": (r"Number of patterns: (\d+)", r"Total time: ([\d.]+) s"),
}
FIELDS = ["dataset", "minutil", "miner", "rep", "status", "patterns", "time_s", "wall_s", "rss_mb", "device_mb",
          "cpu_j", "gpu_j", "energy_j"]


class GpuSampler:
    """Samples board power of one device every 100 ms and integrates it to joules."""

    def __init__(self):
        self.proc = subprocess.Popen(
            ["nvidia-smi", "-i", GPU, "--query-gpu=timestamp,power.draw", "--format=csv,noheader,nounits", "-lms", "100"],
            stdout=subprocess.PIPE, text=True)
        self.t0 = time.time()
        self.marks = []

    def stop_joules(self, start, end):
        self.proc.terminate()
        lines = self.proc.stdout.read().splitlines()
        watts = [float(l.split(",")[1]) for l in lines if "," in l]
        dt = (time.time() - self.t0) / max(len(watts), 1)
        i0, i1 = int((start - self.t0) / dt), int((end - self.t0) / dt) + 1
        window = watts[max(i0, 0):i1]
        return sum(window) * dt if window else 0.0


def command(miner, data, minutil):
    if miner == "REFIM":
        return [str(ROOT / "build/REFIM"), "-i", data, "-m", str(minutil), "-t", "0", "-v", "scalar", "-p", "on"]
    return [str(ROOT / "build/G-HUIM"), "-i", data, "-m", str(minutil)]


def run(miner, data, minutil, timeout):
    with tempfile.NamedTemporaryFile("r") as perf_out:
        cmd = ["perf", "stat", "-a", "-x,", "-o", perf_out.name, "-e", "power/energy-pkg/",
               "timeout", "-k", "5", str(timeout), "/usr/bin/time", "-f", "WALL %e RSS %M"] + command(miner, data, minutil)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU)
        sampler = GpuSampler() if miner == "G-HUIM" else None
        if sampler:
            time.sleep(0.5)
        start = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        end = time.time()
        gpu_j = sampler.stop_joules(start, end) if sampler else 0.0
        perf = perf_out.read()
    out = p.stdout + p.stderr
    count_re, time_re = PATTERNS[miner]
    count, reported, wall = re.search(count_re, out), re.search(time_re, out), re.search(r"WALL ([\d.]+) RSS (\d+)", out)
    if p.returncode != 0 or not count or not reported or not wall:
        return {"status": f"failed rc={p.returncode}"}
    cpu_j = sum(float(l.split(",")[0]) for l in perf.splitlines() if "energy-pkg" in l and l.split(",")[0].replace(".", "").isdigit())
    row = {"status": "ok", "patterns": int(count.group(1)), "time_s": float(reported.group(1)), "wall_s": float(wall.group(1)),
           "rss_mb": int(wall.group(2)) / 1024, "cpu_j": cpu_j, "gpu_j": gpu_j, "energy_j": cpu_j + gpu_j}
    if miner == "G-HUIM":
        arena = re.search(r"peak arena region use (\d+) MiB, peak scratch (\d+) MiB", out)
        if arena:
            row["device_mb"] = 2 * int(arena.group(1)) + int(arena.group(2))  # two arena regions (double buffer) + scratch
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--only", default="", help="comma-separated dataset name prefixes")
    ap.add_argument("--out", default=str(ROOT / "experiments/results/sweep.csv"))
    args = ap.parse_args()
    only = [s for s in args.only.split(",") if s]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for data, minutils in SWEEP.items():
            if only and not any(data.startswith(o) for o in only):
                continue
            for minutil in minutils:
                for miner in ("REFIM", "G-HUIM"):
                    for rep in range(args.reps):
                        if miner == "G-HUIM":
                            time.sleep(3)  # a GPU run started right after another one's exit waits ~0.3 s
                        r = run(miner, str(ROOT / "datasets" / data), minutil, args.timeout)
                        w.writerow({"dataset": data.split("_")[0].removesuffix(".txt"), "minutil": minutil, "miner": miner,
                                    "rep": rep, **r})
                        f.flush()
                        print(data, minutil, miner, rep, r, flush=True)
                        if r["status"] != "ok":
                            break


if __name__ == "__main__":
    main()
