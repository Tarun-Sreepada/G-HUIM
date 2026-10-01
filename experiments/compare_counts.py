"""Check G-HUIM against EFIM by number of high-utility patterns.

    python3 experiments/compare_counts.py datasets/accidents_utility_spmf.txt 3900000
Runs build/efim once and build/G-HUIM, prints counts and wall times,
and exits non-zero if any count differs from EFIM's.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd, pattern):
    start = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - start
    match = re.search(pattern, proc.stdout + proc.stderr)
    if proc.returncode != 0 or not match:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
        return None, elapsed, " | ".join(tail)
    return int(match.group(1)), elapsed, ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("minutil", type=int)
    parser.add_argument("--efim-threads", type=int, default=8)
    parser.add_argument("--efim-count", type=int, help="skip running EFIM and use this count")
    args = parser.parse_args()

    if args.efim_count is None:
        reference, seconds, error = run(
            [ROOT / "build/efim", args.dataset, "/dev/null", str(args.minutil), "space", str(args.efim_threads)],
            r"Total number of High Utility Patterns: (\d+)",
        )
        if reference is None:
            sys.exit(f"efim failed: {error}")
        print(f"{'efim':<20} {reference:>12} {seconds:8.2f} s")
    else:
        reference = args.efim_count

    count, seconds, error = run(
        [ROOT / "build/G-HUIM", "--input", args.dataset, "--minutil", str(args.minutil)],
        r"Number of patterns: (\d+)",
    )
    ok = count == reference
    status = "ok" if ok else f"MISMATCH {error}"
    print(f"{'G-HUIM':<20} {count if count is not None else '-':>12} {seconds:8.2f} s  {status}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
