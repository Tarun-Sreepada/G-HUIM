import os, random, subprocess, math, shutil
from pathlib import Path
from typing import Iterable, List, Tuple
import requests
import time
import csv

# ---------- Config ----------
SPMF_JAR = Path("spmf.jar")
OUT_DIR  = Path("/export/home1/ltarun/cuDPEFIM/datasets/synthetic")         # all outputs here
JAVA_XMX = "64G"                         # bump if you like
SEED     = 12345

# scale grids (tweak as needed)
NUM_TRANSACTIONS = [10**3, 10**6,10**7]
NUM_ITEMS        = [50, 100, 250]  # “num distinct items” in SPMF generator
TX_LENGTHS       = [10, 25, 90]  # average transaction length

# ---------- SPMF bootstrap ----------
def ensure_spmf(jar_path: Path = SPMF_JAR):
    if jar_path.exists():
        return
    jar_url = "https://www.philippe-fournier-viger.com/spmf/spmf.jar"
    print(f"Downloading SPMF from {jar_url} ...")
    r = requests.get(jar_url, timeout=60)
    r.raise_for_status()
    jar_path.write_bytes(r.content)
    print("SPMF ready.")

# ---------- Naming helpers ----------
def abbr(n: int) -> str:
    # 1234 -> "1.2k"; 1_000_000 -> "1M"
    if n >= 10**12: return f"{n/10**12:.0f}T"
    if n >= 10**9:  return f"{n/10**9:.0f}B"
    if n >= 10**6:  return f"{n/10**6:.0f}M"
    if n >= 10**3:  return f"{n/10**3:.0f}k"
    return str(n)

def synth_name(nT: int, nI: int, L: int) -> str:
    # huT1M_ni1000_avg20.txt
    return f"huT{abbr(nT)}_ni{nI}_avg{L}.txt"

# ---------- Safety checks ----------
def approx_bytes(nT: int, L: int) -> int:
    # Very rough line size estimate for utility format:
    # items (L*2 chars incl. spaces) + TU (~6) + per-item utils (L*2) + separators
    # multiply by ~6 to be safer, then by nT
    per_line = max(20, int((L*4 + 12) * 6))
    return per_line * nT

def enough_disk_free(target_bytes: int, keep_free_ratio: float = 0.2, path: Path = Path(".")) -> bool:
    total, used, free = shutil.disk_usage(path)
    return free - target_bytes > total * keep_free_ratio

# ---------- SPMF wrappers ----------
def run_spmf(args: List[str], xmx: str = JAVA_XMX, timeout: int = 600):
    cmd = ["java", f"-Xmx{xmx}", "-jar", str(SPMF_JAR)] + args
    # print("RUN:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"SPMF error\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    return res.stdout

def generate_plain_tx(out_txt: Path, nT: int, nI: int, L: int):
    # SPMF doc: java -jar spmf.jar run Generate_a_transaction_database output.txt nT nI L
    run_spmf(["run", "Generate_a_transaction_database", str(out_txt), str(nT), str(nI), str(L)])  # :contentReference[oaicite:1]{index=1}

def add_utilities(in_txt: Path, out_txt: Path, max_qty: int = 10, ext_mult: int = 1):
    # SPMF doc: java -jar spmf.jar run Generate_utility_values_for_transaction_database input output maxQty mult
    run_spmf(["run", "Generate_utility_values_for_transaction_database",
              str(in_txt), str(out_txt), str(max_qty), str(ext_mult)])  # :contentReference[oaicite:2]{index=2}

# ---------- Main generator ----------
def generate_hu_corpus(
    nT_list: Iterable[int],
    nI_list: Iterable[int],
    L_list: Iterable[int],
    out_dir: Path = OUT_DIR,
    keep_free_ratio: float = 0.2,
    max_qty: int = 10,
    ext_mult: int = 1
) -> List[Tuple[str, int, int, int, str]]:
    """
    Returns a manifest list of (file_path, nT, nI, L, status).
    """
    ensure_spmf()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    meta_csv = out_dir / "manifest.csv"

    for nT in nT_list:
        for nI in nI_list:
            for L in L_list:
                # sanitize: cannot have more items per tx than #items
                if L > nI:
                    L_adj = max(1, nI // 2)
                    print(f"[WARN] avg {L} > num_items {nI}. Using {L_adj} instead.")
                    L = L_adj

                # crude size guard
                est = approx_bytes(nT, L)
                if not enough_disk_free(est, keep_free_ratio, out_dir):
                    print(f"[SKIP] {synth_name(nT,nI,L)} ~{est/1e9:.1f} GB exceeds safe disk policy.")
                    manifest.append((str(out_dir / synth_name(nT,nI,L)), nT, nI, L, "skipped_disk"))
                    continue

                # build names
                plain_tx = out_dir / (synth_name(nT, nI, L).replace(".txt", "_plain.txt"))
                hu_file  = out_dir / synth_name(nT, nI, L)

                try:
                    if not plain_tx.exists():
                        t0 = time.time()
                        generate_plain_tx(plain_tx, nT, nI, L)      # transactions only
                        print(f"[OK] Plain TX generated: {plain_tx} in {time.time()-t0:.1f}s")

                    if not hu_file.exists():
                        t1 = time.time()
                        add_utilities(plain_tx, hu_file, max_qty=max_qty, ext_mult=ext_mult)  # add utilities
                        print(f"[OK] HU DB generated:  {hu_file} in {time.time()-t1:.1f}s")

                    manifest.append((str(hu_file), nT, nI, L, "ok"))
                except Exception as e:
                    print(f"[ERR] {synth_name(nT,nI,L)} -> {e}")
                    manifest.append((str(hu_file), nT, nI, L, "error"))

    # write manifest CSV
    with meta_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "transactions", "num_items", "tx_len", "status"])
        w.writerows(manifest)
    print(f"[DONE] Wrote manifest: {meta_csv}")

    return manifest

if __name__ == "__main__":
    # you proposed 10**5..10**11; be cautious with disk/IO. Start smaller and scale after you confirm throughput.
    manifest = generate_hu_corpus(
        nT_list=NUM_TRANSACTIONS,
        nI_list=NUM_ITEMS,
        L_list=TX_LENGTHS,
        out_dir=OUT_DIR,
        keep_free_ratio=0.2,   # keep at least 20% disk free
        max_qty=10,            # internal qty in [1..10]
        ext_mult=1             # external utility ~ N(0,1)*mult (SPMF doc)
    )

    # Print a few paths you can feed into your runner
    for row in manifest[:5]:
        print(row)
