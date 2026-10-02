#!/usr/bin/env python3
"""Digest ncu reports: per-kernel speed-of-light, memory and stall metrics, plus the hottest source lines."""
import csv, glob, io, json, re, subprocess, sys
from collections import defaultdict
from pathlib import Path

NCU = "/usr/local/cuda-12.8/bin/ncu"
RES = Path(__file__).parent / "results"
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)


def run(args):
    return subprocess.run([NCU, *args], capture_output=True, text=True).stdout


def num(x):
    try:
        return float(x.replace(",", ""))
    except Exception:
        return None


def raw(rep):
    rows = list(csv.reader(io.StringIO(run(["-i", str(rep), "--page", "raw", "--csv"]))))
    h, u = rows[0], rows[1]
    return [dict(zip(h, r)) | {"__units": dict(zip(h, u))} for r in rows[2:]]


STALLS = ["long_scoreboard", "short_scoreboard", "barrier", "wait", "branch_resolving", "mio_throttle", "lg_throttle", "membar",
          "math_pipe_throttle", "no_instruction", "not_selected", "selected", "dispatch_stall", "drain", "imc_miss", "sleeping", "tex_throttle", "misc"]


def kernel_record(d, rep, idx):
    g = lambda k: num(d.get(k, ""))
    u = d["__units"]
    scale = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}.get(u.get("gpu__time_duration.sum", "ns"), 1e-6)
    ld_r, ld_s = g("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"), g("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
    st_r, st_s = g("l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"), g("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum")
    dur_ms = g("gpu__time_duration.sum") * scale
    br = g("dram__bytes_read.sum"); bw = g("dram__bytes_write.sum")
    bu = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "KB": 1e3, "MB": 1e6, "GB": 1e9}
    br *= bu.get(u.get("dram__bytes_read.sum", "byte"), 1); bw *= bu.get(u.get("dram__bytes_write.sum", "byte"), 1)
    stalls = {s: g(f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio") or 0 for s in STALLS}
    cyc = sum(stalls.values()) or 1
    return dict(
        report=rep.stem, idx=idx, kernel=re.sub(r"\(.*", "", d["Kernel Name"]).replace("ghuim::", "").replace("<unnamed>::", "").replace("unnamed>::", ""),
        grid=d["Grid Size"], block=d["Block Size"], duration_ms=dur_ms,
        sm_pct=g("sm__throughput.avg.pct_of_peak_sustained_elapsed"), mem_pct=g("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
        l1_pct=g("l1tex__throughput.avg.pct_of_peak_sustained_elapsed"), l2_pct=g("lts__throughput.avg.pct_of_peak_sustained_elapsed"),
        dram_pct=g("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        dram_read_gb=br / 1e9, dram_write_gb=bw / 1e9, dram_gbps=(br + bw) / (dur_ms * 1e-3) / 1e9 if dur_ms else 0,
        l1_hit=g("l1tex__t_sector_hit_rate.pct"), l2_hit=g("lts__t_sector_hit_rate.pct"),
        ld_sectors_per_req=ld_s / ld_r if ld_r else None, st_sectors_per_req=st_s / st_r if st_r else None,
        ld_sectors_g=(ld_s or 0) / 1e9, red_sectors_m=(g("lts__t_sectors_op_red.sum") or 0) / 1e6, atom_sectors_m=(g("lts__t_sectors_op_atom.sum") or 0) / 1e6,
        l1_traffic_gb=((ld_s or 0) + (st_s or 0)) * 32 / 1e9, l2_excess_pct=None,
        warps_active_pct=g("sm__warps_active.avg.pct_of_peak_sustained_active"), theo_occ_pct=g("sm__maximum_warps_per_active_cycle_pct"),
        issue_active_pct=g("smsp__issue_active.avg.pct_of_peak_sustained_active"), ipc=g("sm__inst_executed.avg.per_cycle_active"),
        active_threads_per_warp=g("smsp__thread_inst_executed_per_inst_executed.ratio"), waves=g("launch__waves_per_multiprocessor"),
        regs=g("launch__registers_per_thread"), instr_m=(g("smsp__inst_executed.sum") or 0) / 1e6,
        stall_cycles_per_issue=cyc, stall_share={s: 100 * v / cyc for s, v in stalls.items() if v / cyc > 0.01})


ROOT = Path(__file__).resolve().parents[1]
_defs = {}


def enclosing(path, line):
    """Name of the __device__/__global__ function whose definition most recently precedes `line`."""
    if path not in _defs:
        out = []
        try:
            for n, t in enumerate((ROOT / path).read_text().splitlines(), 1):
                m = re.match(r"\s*(?:template\s*<[^>]*>\s*)?(?:__host__\s+)?(?:__device__|__global__)\s+[^(]*?\b(\w+)\s*\(", t)
                if m:
                    out.append((n, m.group(1)))
        except OSError:
            pass
        _defs[path] = out
    best = None
    for n, name in _defs[path]:
        if n <= line:
            best = name
    return best or "(outside a device function)"


def hot_lines(rep, idx, top=8):
    out = run(["-i", str(rep), "--page", "source", "--csv", "-s", str(idx), "-c", "1", "--print-source", "cuda,sass"])
    files = out.split('"File Path"')
    agg = defaultdict(lambda: [0, 0, 0, 0, ""])  # samples, inst, l1 req, l2 sectors excess
    total = 0
    for blk in files[1:]:
        rows = list(csv.reader(io.StringIO('"File Path"' + blk)))
        path = rows[0][1]
        hi = next(i for i, r in enumerate(rows) if r and r[0] == "Line No")
        h = rows[hi]
        col = lambda n: h.index(n) if n in h else None
        cs, ci = col("Warp Stall Sampling (All Samples)"), col("Instructions Executed")
        cr, ce = col("L1 Tag Requests Global"), col("L2 Theoretical Sectors Global Excessive")
        for r in rows[hi + 1:]:
            if len(r) < 5 or not r[0].isdigit():
                continue
            a = agg[(path, int(r[0]))]
            a[0] += int(num(r[cs]) or 0); a[1] += int(num(r[ci]) or 0)
            a[2] += int(num(r[cr]) or 0) if cr else 0; a[3] += int(num(r[ce]) or 0) if ce else 0
            a[4] = r[1].strip()
            total += int(num(r[cs]) or 0)
    top_l = sorted(agg.items(), key=lambda kv: -kv[1][0])[:top]
    byfn = defaultdict(lambda: [0, 0])
    for (path, line), v in agg.items():
        rel = path.split("G-HUIM-merged/")[-1]
        f = byfn[enclosing(rel, line) if rel.startswith(("include", "src")) else "(library)"]
        f[0] += v[0]; f[1] += v[1]
    fns = sorted(((k, 100 * v[0] / total if total else 0, v[1] / 1e6) for k, v in byfn.items()), key=lambda x: -x[1])[:8]
    return dict(by_function=[dict(fn=a, stall_pct=b, inst_m=c) for a, b, c in fns], lines=[dict(file=k[0].split("G-HUIM-merged/")[-1], line=k[1], text=v[4][:110], stall_pct=100 * v[0] / total if total else 0,
                 inst_m=v[1] / 1e6, l1_req_m=v[2] / 1e6, l2_excess_m=v[3] / 1e6) for k, v in top_l])


def main():
    allr = []
    for rep in sorted(RES.glob("ncu_*.ncu-rep")):
        for i, d in enumerate(raw(rep)):
            k = kernel_record(d, rep, i)
            hl = hot_lines(rep, i)
            k["hot_lines"], k["by_function"] = hl["lines"], hl["by_function"]
            allr.append(k)
            print(rep.stem, i, k["kernel"], k["grid"], f'{k["duration_ms"]:.2f}ms sm{k["sm_pct"]:.0f} mem{k["mem_pct"]:.0f} dram{k["dram_pct"]:.0f}', file=sys.stderr)
    (OUT / "ncu_summary.json").write_text(json.dumps(allr, indent=1))


if __name__ == "__main__":
    main()
