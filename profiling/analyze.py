#!/usr/bin/env python3
"""Digest nsys sqlite exports into per-config JSON: CPU phases, GPU kernels, busy/idle, copies, API/OS time."""
import json, re, sqlite3, sys
from collections import defaultdict
from pathlib import Path

RES = Path(__file__).parent / "results"
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

# H100 PCIe limits (from TARGET_INFO_GPU): per SM
REGS_SM, SMEM_SM, THREADS_SM, BLOCKS_SM, WARPS_SM = 65536, 233472, 2048, 32, 64


def merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def occupancy(regs, block, smem):
    warps = max(1, (block + 31) // 32)
    by_threads = THREADS_SM // block if block else 0
    by_regs = REGS_SM // max(1, ((regs * 32 + 255) // 256 * 256) * warps) if regs else BLOCKS_SM
    by_smem = SMEM_SM // smem if smem else BLOCKS_SM
    blocks = max(0, min(by_threads, by_regs, by_smem, BLOCKS_SM))
    return blocks * warps / WARPS_SM, ("threads" if blocks == by_threads else "regs" if blocks == by_regs else "smem" if blocks == by_smem else "blocks")


def analyze(path):
    c = sqlite3.connect(path)
    sid = dict(c.execute("select id, value from StringIds"))
    nv = [(s, e, t) for s, e, t in c.execute("select start, end, text from NVTX_EVENTS where end is not null and text is not null")]
    mine = next(((s, e) for s, e, t in nv if t == "mine"), None)
    load = next(((s, e) for s, e, t in nv if t == "load dataset"), None)
    r = {"name": Path(path).stem}
    # ---- CPU phases (NVTX)
    ph = defaultdict(lambda: [0, 0])
    for s, e, t in nv:
        ph[t][0] += e - s
        ph[t][1] += 1
    r["cpu_phases"] = [{"range": t, "total_ms": v[0] / 1e6, "count": v[1]} for t, v in sorted(ph.items(), key=lambda kv: -kv[1][0])]
    r["mine_ms"] = (mine[1] - mine[0]) / 1e6
    r["load_ms"] = (load[1] - load[0]) / 1e6
    # ---- kernels
    ks = c.execute("select start,end,streamId,demangledName,registersPerThread,gridX*gridY*gridZ,blockX*blockY*blockZ,"
                   "staticSharedMemory,dynamicSharedMemory,localMemoryPerThread from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    kern = [dict(s=a, e=b, stream=st, name=sid[n], regs=rg, grid=g, block=bl, smem=ss + ds, local=lm) for a, b, st, n, rg, g, bl, ss, ds, lm in ks]
    first, last = min(k["s"] for k in kern), max(k["e"] for k in kern)
    busy = merge([(k["s"], k["e"]) for k in kern])
    busy_t = sum(e - s for s, e in busy)
    ksum = sum(k["e"] - k["s"] for k in kern)
    r.update(kernel_launches=len(kern), kernel_sum_ms=ksum / 1e6, gpu_busy_ms=busy_t / 1e6,
             first_kernel_offset_ms=(first - mine[0]) / 1e6, kernel_span_ms=(last - first) / 1e6,
             busy_frac_of_mine=busy_t / (mine[1] - mine[0]), busy_frac_of_span=busy_t / (last - first),
             tail_after_last_kernel_ms=(mine[1] - last) / 1e6)
    # ---- idle gaps inside the kernel span, attributed to the innermost NVTX range covering the gap start
    inner = sorted([x for x in nv if x[2] not in ("mine", "load dataset") and mine[0] <= x[0] <= mine[1]], key=lambda x: x[1] - x[0])
    gap = defaultdict(float)
    for (s0, e0), (s1, e1) in zip(busy, busy[1:]):
        mid = (e0 + s1) // 2
        lab = next((t for s, e, t in inner if s <= mid <= e), "(outside any range)")
        gap[lab] += s1 - e0
    r["idle_gaps_ms"] = {k: v / 1e6 for k, v in sorted(gap.items(), key=lambda kv: -kv[1])}
    r["idle_gap_count"] = len(busy) - 1
    # ---- per-kernel table
    by = defaultdict(list)
    for k in kern:
        by[k["name"]].append(k)
    tab = []
    for n, v in by.items():
        d = [k["e"] - k["s"] for k in v]
        blk = max(set(k["block"] for k in v), key=[k["block"] for k in v].count)
        occ, lim = occupancy(v[0]["regs"], blk, v[0]["smem"])
        short = re.sub(r"\(.*", "", n).replace("ghuim::", "").replace("<unnamed>::", "")
        tab.append(dict(kernel=short, launches=len(v), total_ms=sum(d) / 1e6, avg_us=sum(d) / len(d) / 1e3, max_us=max(d) / 1e3,
                        pct_of_kernel_time=100 * sum(d) / ksum, avg_grid=sum(k["grid"] for k in v) / len(v), block=blk,
                        regs=v[0]["regs"], smem_bytes=v[0]["smem"], local_bytes_per_thread=v[0]["local"],
                        theoretical_occupancy=occ, occupancy_limiter=lim))
    r["kernels"] = sorted(tab, key=lambda x: -x["total_ms"])
    # ---- launch fill: share of the GPU's resident-block slots one launch can occupy at once (1.0 = a full wave or more)
    sms = c.execute("select smCount from TARGET_INFO_GPU limit 1").fetchone()[0]
    occ_of = {t["kernel"]: t for t in tab}
    num = den = 0.0
    fills = []
    for k in kern:
        short = re.sub(r"\(.*", "", k["name"]).replace("ghuim::", "").replace("<unnamed>::", "")
        t = occ_of[short]
        wpb = max(1, (k["block"] + 31) // 32)
        bps = max(1, round(t["theoretical_occupancy"] * WARPS_SM / wpb))
        fill = min(1.0, k["grid"] / (sms * bps))
        d = k["e"] - k["s"]
        num += fill * d; den += d
        fills.append((short, k["grid"], k["block"], fill, d / 1e6))
    r["time_weighted_launch_fill"] = num / den
    big = sorted(fills, key=lambda x: -x[4])[:6]
    r["longest_launches"] = [dict(kernel=a, grid=b, block=cc, fill=d, ms=e) for a, b, cc, d, e in big]
    r["time_ms_in_launches_below_50pct_fill"] = sum(f[4] for f in fills if f[3] < 0.5)
    # ---- copies
    names = {1: "H2D", 2: "D2H", 8: "D2D", 4: "H2H"}
    cp = defaultdict(lambda: [0, 0, 0])
    for s, e, b, k in c.execute("select start,end,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"):
        x = cp[names.get(k, str(k))]
        x[0] += 1; x[1] += b; x[2] += e - s
    r["memcpy"] = {k: dict(count=v[0], MiB=v[1] / 2**20, ms=v[2] / 1e6, GBps=(v[1] / max(v[2], 1))) for k, v in cp.items()}
    # ---- CUDA API time inside mining window and before first kernel
    api = defaultdict(lambda: [0, 0])
    setup = defaultdict(lambda: [0, 0])
    for s, e, n in c.execute("select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
        if mine[0] <= s <= mine[1]:
            nm = re.sub(r"_v\d+$", "", sid[n])
            api[nm][0] += e - s; api[nm][1] += 1
            if e <= first:
                setup[nm][0] += e - s; setup[nm][1] += 1
    r["cuda_api_in_mine"] = [dict(api=k, total_ms=v[0] / 1e6, calls=v[1]) for k, v in sorted(api.items(), key=lambda kv: -kv[1][0])[:12]]
    r["cuda_api_before_first_kernel"] = [dict(api=k, total_ms=v[0] / 1e6, calls=v[1]) for k, v in sorted(setup.items(), key=lambda kv: -kv[1][0])[:8]]
    # ---- OS runtime calls in load and mine windows (all threads)
    for label, w in (("load", load), ("mine", mine)):
        osr = defaultdict(lambda: [0, 0])
        for s, e, n in c.execute("select start,end,nameId from OSRT_API"):
            if w[0] <= s <= w[1]:
                osr[sid[n]][0] += e - s; osr[sid[n]][1] += 1
        r[f"osrt_{label}"] = [dict(call=k, total_ms=v[0] / 1e6, calls=v[1]) for k, v in sorted(osr.items(), key=lambda kv: -kv[1][0])[:8]]
    # ---- device memory
    ev = c.execute("select start,bytes,memoryOperationType,memKind from CUDA_GPU_MEMORY_USAGE_EVENTS order by start").fetchall()
    cur = peak = 0
    for _, b, op, _k in ev:
        cur += b if op == 0 else -b
        peak = max(peak, cur)
    r["device_alloc_peak_MiB"] = peak / 2**20
    r["device_allocs_MiB"] = sorted((b / 2**20 for _, b, op, _k in ev if op == 0), reverse=True)[:8]
    # ---- sync waits (host blocked on the GPU)
    sy = c.execute("select sum(end-start), count(*) from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where start>=? and end<=?", mine).fetchone()
    r["sync_ms_in_mine"], r["sync_calls_in_mine"] = (sy[0] or 0) / 1e6, sy[1]
    return r


def main():
    allr = {}
    for p in sorted(RES.glob("*.sqlite")):
        r = analyze(p)
        allr[r["name"]] = r
        # keep the log-derived numbers next to it
        log = (RES / (r["name"] + ".log")).read_text()
        m = re.search(r"kernels ([\d.]+) s, peak live arena (\d+) MiB \(([\d.]+) s\)", log)
        r["program_kernel_s"], r["peak_live_arena_MiB"], r["program_mine_s"] = float(m[1]), int(m[2]), float(m[3])
        r["patterns"] = int(re.search(r"Number of patterns: (\d+)", log)[1])
        r["tasks"] = int(re.search(r"Mined with .*?: (\d+) tasks", log)[1])
    (OUT / "nsys_summary.json").write_text(json.dumps(allr, indent=1))
    # compact console table
    print(f"{'config':24} {'mine ms':>8} {'1st kern':>8} {'span':>7} {'busy':>7} {'busy%mine':>9} {'busy%span':>9} {'ksum':>7} {'launch':>6} {'sync ms':>7}")
    for n, r in allr.items():
        print(f"{n:24} {r['mine_ms']:8.1f} {r['first_kernel_offset_ms']:8.1f} {r['kernel_span_ms']:7.1f} {r['gpu_busy_ms']:7.1f} "
              f"{100*r['busy_frac_of_mine']:8.1f}% {100*r['busy_frac_of_span']:8.1f}% {r['kernel_sum_ms']:7.1f} {r['kernel_launches']:6d} {r['sync_ms_in_mine']:7.1f}")


if __name__ == "__main__":
    main()
