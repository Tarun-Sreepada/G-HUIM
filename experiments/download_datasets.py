"""Download the SPMF utility datasets used by the experiments into datasets/.

Usage: python3 experiments/download_datasets.py [--only chess,connect] [--out datasets] [--all]
Files that already exist are skipped. --all also fetches retail, foodmart and fruithut
(REFIM cannot read fruithut_utility.txt: it has @ITEM headers).
"""

import argparse
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://www.philippe-fournier-viger.com/spmf/publicdatasets/"
SWEEP = [
    "chess_utility_spmf.txt",
    "connect_utility_spmf.txt",
    "accidents_utility_spmf.txt",
    "mushroom_utility_spmf.txt",
    "ecommerce_utility_no_timestamps.txt",
    "pumsb_utility_spmf.txt",
    "kosarak_utility_spmf.txt",
    "chainstore.txt",
    "BMS_utility_spmf.txt",
]
EXTRA = ["retail_utility_spmf.txt", "foodmart.txt", "fruithut_utility.txt"]


def fetch(name, out):
    dest = out / name
    if dest.exists():
        print(f"skip  {name} (exists)")
        return True
    tmp = dest.with_suffix(".part")
    try:
        req = urllib.request.Request(BASE + name, headers={"User-Agent": "Mozilla/5.0"})  # site 403s the default UA
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"FAIL  {name}: {e}")
        return False
    tmp.rename(dest)
    print(f"got   {name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "datasets"))
    ap.add_argument("--only", help="comma-separated dataset name prefixes, e.g. chess,connect")
    ap.add_argument("--all", action="store_true", help="also fetch retail, foodmart, fruithut")
    args = ap.parse_args()
    names = SWEEP + (EXTRA if args.all else [])
    if args.only:
        keys = args.only.split(",")
        names = [n for n in names if any(n.startswith(k) for k in keys)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ok = [fetch(n, out) for n in names]
    raise SystemExit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
