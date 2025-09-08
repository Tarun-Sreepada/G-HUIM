#!/usr/bin/env python3
# hu_stats.py
import subprocess, re, csv, sys, os
from pathlib import Path
from typing import Dict, Optional, List

# --- CONFIG ---
SPMF_JAR = os.environ.get("SPMF_JAR", "spmf.jar")
JAVA_BIN = os.environ.get("JAVA_BIN", "java")  # set this if you keep Java 21 in a custom path
TIMEOUT_SEC = 30
OUT_CSV = "hu_stats.csv"

# --- helpers: manual transactions + file size ---
def fast_count_lines(path: Path, chunk_size: int = 8 * 1024 * 1024) -> int:
    """
    Count lines fast in binary mode by counting b'\\n'.
    If the file doesn't end with a newline but is non-empty, add 1 for the last line.
    Interprets 'transactions = number of lines'.
    """
    size = path.stat().st_size
    if size == 0:
        return 0
    count = 0
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            count += buf.count(b"\n")
        # if last byte isn't '\n', we still have one last line
        f.seek(size - 1)
        if f.read(1) != b"\n":
            count += 1
    return count

def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 3)

# --- helpers: SPMF stats parsing (kept for other fields) ---
STAT_PATTERNS = {
    "transactions": re.compile(r"Number of transa?ctions\s*:\s*(\d+)", re.I),
    "distinct_items": re.compile(r"Number of distinct items\s*:\s*(\d+)", re.I),
    "avg_len": re.compile(r"Average length of transaction\s*:\s*([0-9]*\.?[0-9]+)", re.I),
    "max_len": re.compile(r"Maximum length of transaction\s*:\s*(\d+)", re.I),
    "max_item_id": re.compile(r"Maximum Id of item\s*:\s*(\d+)", re.I),
    "total_utility": re.compile(r"Total utility\s*:\s*([0-9]*\.?[0-9]+)", re.I),
}

def run_spmf_stats(file_path: str) -> Optional[str]:
    """Run SPMF 'Calculate_stats_for_a_transaction_database_with_utility' and return stdout or None on failure."""
    cmd = [
        JAVA_BIN, "-Xmx1G", "-jar", SPMF_JAR, "run",
        "Calculate_stats_for_a_transaction_database_with_utility",
        file_path, "no_output_file"
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SEC)
        if proc.returncode != 0:
            sys.stderr.write(f"[ERR] SPMF failed for {file_path}\nSTDERR:\n{proc.stderr}\n")
            return None
        return proc.stdout
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"[TIMEOUT] {file_path} (> {TIMEOUT_SEC}s)\n")
        return None

def parse_stats(stdout: str) -> Dict[str, float]:
    vals: Dict[str, float] = {}
    for key, pat in STAT_PATTERNS.items():
        m = pat.search(stdout)
        if m:
            vals[key] = float(m.group(1)) if key in ("avg_len","total_utility") else int(m.group(1))
    # derived density
    if "avg_len" in vals and "distinct_items" in vals and vals["distinct_items"] > 0:
        vals["density"] = round(float(vals["avg_len"]) / float(vals["distinct_items"]), 4)
    else:
        vals["density"] = None
    return vals

def analyze_files(files: List[str], out_csv: str = OUT_CSV):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "file", "file_size_mb", "transactions", "distinct_items", "avg_len",
            "density", "max_len", "max_item_id", "total_utility",
            "completed", "timed_out_or_error"
        ])
        for fp in files:
            path = Path(fp)
            base = path.stem
            # if base.contains _utility_spmf, remove that part
            if "_utility_spmf" in base:
                base = base.replace("_utility_spmf", "")
            tx = fast_count_lines(path)              # <-- manual transactions
            size_mb = file_size_mb(path)             # <-- file size in MB

            out = run_spmf_stats(fp)
            if out is None:
                # We still store name, size, transactions; leave others blank
                w.writerow([base, size_mb, tx, "", "", "", "", "", "", 0, 1])
                continue

            stats = parse_stats(out)
            w.writerow([
                base,
                size_mb,
                tx,                                  # <-- override SPMF's transactions
                stats.get("distinct_items",""),
                stats.get("avg_len",""),
                stats.get("density",""),
                stats.get("max_len",""),
                stats.get("max_item_id",""),
                stats.get("total_utility",""),
                1, 0
            ])

if __name__ == "__main__":
    # Build the file list:
    files: List[str] = []

    # synthetic_hu but skip *_plain.txt
    data_dir = Path("/export/home1/ltarun/cuDPEFIM/experiments/synthetic_hu")
    for f in data_dir.glob("*.txt"):
        if "_plain" not in f.name:
            files.append(str(f))

    # add shipped datasets
    data_dir2 = Path("/export/home1/ltarun/cuDPEFIM/datasets")
    for f in data_dir2.glob("*.txt"):
        files.append(str(f))

    analyze_files(files)
    print(f"Wrote {OUT_CSV}")
