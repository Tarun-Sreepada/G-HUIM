"""On/off ablation of REFIM and G-HUIM options over the sweep.py minutil ladders.

Usage: python3 experiments/ablation.py [--reps 2] [--timeout 60] [--only chess] [--out experiments/results/ablation.csv]

Each config is the default with one option flipped (REFIM also gets a numa x leaf x compact grid). Runs are serial:
REFIM uses all 48 threads and energy is read system-wide. The binaries are copied first, so rebuilding mid-run is safe.
Before the sweep every G-HUIM config passes compute-sanitizer memcheck on a small input; a config that fails is
skipped. The CSV is appended to and finished rows are skipped, so rerunning resumes.
"""

import argparse
import csv
import itertools
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import sweep
from sweep import GPU, ROOT, SWEEP

BIN = ROOT / "experiments/results/ablation_bin"
PREFLIGHT = ("chess_utility_spmf.txt", 400_000)

# name -> (args, env)
REFIM_BASE = ["-t", "48"]
REFIM = {"refim/default": ([], {})}
for flag, values in {"-n": ["off"], "-f": ["off"], "-c": ["off"], "-p": ["on", "off"], "-r": ["0", "1"],
                     "-v": ["scalar", "avx2", "avx512"]}.items():
    for v in values:
        REFIM[f"refim/{flag.lstrip('-')}={v}"] = ([flag, v], {})
for n, f, c in itertools.product(["on", "off"], repeat=3):
    REFIM.setdefault(f"refim/grid n={n} f={f} c={c}", (["-n", n, "-f", f, "-c", c], {}))

GHUIM = {
    "bfs/adaptive": ([], {}),
    "bfs/static": (["-M", "static"], {}),
    "bfs/staged": (["-M", "staged"], {}),
    "bfs/tile": (["-M", "tile"], {}),
    "bfs/local-tail=off": (["-k", "0"], {}),
    "bfs/vert=on": (["-V", "16"], {}),
    "bfs/index=off": (["-I", "0"], {}),
    "bfs/refine=off": (["-r", "0"], {}),
    "bfs/refine=-1": (["-r", "-1"], {}),
    "dfs": (["-e", "dfs"], {}),
    "persistent": (["-e", "persistent"], {}),
    "persistent/lane=off": (["-e", "persistent"], {"GHUIM_LANE": "0"}),
    "persistent/warp-mask=on": (["-e", "persistent"], {"GHUIM_WARP_MASK": "1"}),
    "persistent/vert=on": (["-e", "persistent"], {"GHUIM_VERT": "1"}),
    "persistent/l2-root=off": (["-e", "persistent"], {"GHUIM_L2_ROOT": "0"}),
    "persistent/refine=off": (["-e", "persistent"], {"GHUIM_REFINE": "0"}),
}
FIELDS = ["dataset", "minutil", "miner", "config"] + sweep.FIELDS[3:]


def install(cfg_args, cfg_env):
    """Points sweep.command/run at the copied binaries and this config's flags."""
    def command(miner, data, minutil):
        if miner == "REFIM":
            return [str(BIN / "REFIM"), "-i", data, "-m", str(minutil)] + REFIM_BASE + cfg_args
        return [str(BIN / "G-HUIM"), "-i", data, "-m", str(minutil)] + cfg_args
    sweep.command = command
    os.environ.update(cfg_env)


def clear_env():
    for k in [k for k in os.environ if k.startswith("GHUIM_")]:
        del os.environ[k]


def preflight(name, args, env, log):
    data, minutil = PREFLIGHT
    cmd = ["timeout", "-k", "10", "900", "compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "99",
           str(BIN / "G-HUIM"), "-i", str(ROOT / "datasets" / data), "-m", str(minutil), "-g", "16G"] + args
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, **env, CUDA_VISIBLE_DEVICES=GPU))
    out = p.stdout + p.stderr
    ok = p.returncode == 0 and "ERROR SUMMARY: 0 errors" in out
    log.write(f"=== {name} rc={p.returncode} ok={ok}\n{out[-4000:]}\n")
    log.flush()
    print("preflight", name, "ok" if ok else f"FAILED rc={p.returncode}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--only", default="", help="comma-separated dataset name prefixes")
    ap.add_argument("--out", default=str(ROOT / "experiments/results/ablation.csv"))
    args = ap.parse_args()
    only = [s for s in args.only.split(",") if s]
    BIN.mkdir(parents=True, exist_ok=True)
    for b in ("REFIM", "G-HUIM"):
        shutil.copy2(ROOT / "build" / b, BIN / b)

    clear_env()
    with open(BIN / "preflight.log", "a") as log:
        ghuim = {n: c for n, c in GHUIM.items() if preflight(n, *c, log)}
    configs = [("REFIM", n, c) for n, c in REFIM.items()] + [("G-HUIM", n, c) for n, c in ghuim.items()]

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(r["dataset"], r["minutil"], r["config"], r["rep"]) for r in csv.DictReader(f)}
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for data, minutils in SWEEP.items():
            if only and not any(data.startswith(o) for o in only):
                continue
            ds = data.split("_")[0].removesuffix(".txt")
            for minutil in minutils:
                for miner, name, (cfg_args, cfg_env) in configs:
                    for rep in range(args.reps):
                        if (ds, str(minutil), name, str(rep)) in done:
                            continue
                        clear_env()
                        install(cfg_args, cfg_env)
                        if miner == "G-HUIM":
                            time.sleep(3)
                        r = sweep.run(miner, str(ROOT / "datasets" / data), minutil, args.timeout)
                        w.writerow({"dataset": ds, "minutil": minutil, "miner": miner, "config": name, "rep": rep, **r})
                        f.flush()
                        print(ds, minutil, name, rep, r, flush=True)
                        if r["status"] != "ok":
                            break
    print("ABLATION DONE", flush=True)


if __name__ == "__main__":
    main()
