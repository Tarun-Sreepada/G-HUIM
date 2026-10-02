#!/usr/bin/env python3
"""Build report.html from data/nsys_summary.json and the debug logs."""
import json, re, html
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
D = json.loads((HERE / "data/nsys_summary.json").read_text())
LOG = HERE / "results"
CFG = ["bfs_static", "bfs_adaptive", "dfs_static", "dfs_adaptive"]
LABEL = {"bfs_static": "BFS static", "bfs_adaptive": "BFS adaptive", "dfs_static": "DFS static", "dfs_adaptive": "DFS adaptive"}
CV = {"bfs_static": "var(--c1)", "bfs_adaptive": "var(--c2)", "dfs_static": "var(--c3)", "dfs_adaptive": "var(--c4)"}
esc = html.escape


def levels(cfg, M):
    t = (LOG / f"debug_{cfg}_{M}.log").read_text()
    if cfg.startswith("bfs"):
        return [(int(a), int(b), float(c)) for a, b, c in re.findall(r"Level\s+(\d+):\s+(\d+) tasks,.*?, ([\d.]+) ms", t)]
    return [(int(a), int(b), float(c)) for a, b, c in re.findall(r"Batch\s+(\d+):\s+(\d+) tasks.*?kernels ([\d.]+) ms", t)]


def segs(r):
    root = sum(p["total_ms"] for p in r["cpu_phases"] if "root database" in p["range"])
    init = r["first_kernel_offset_ms"] - root
    return [("CUDA startup + allocation", init, "var(--p1)"), ("Dataset load (CPU)", r["load_ms"], "var(--p2)"),
            ("Root database build + upload", root, "var(--p3)"), ("Mining kernels + host scheduling", r["kernel_span_ms"] + r["tail_after_last_kernel_ms"], "var(--p4)")]


def stack_chart(M):
    rows = []
    mx = max(sum(s[1] for s in segs(D[f"{c}_{M}"])) for c in CFG)
    for c in CFG:
        r = D[f"{c}_{M}"]
        sg = segs(r)
        tot = sum(s[1] for s in sg)
        bars = "".join(f'<span class="seg" style="width:{100*s[1]/mx:.3f}%;background:{s[2]}" title="{esc(s[0])}: {s[1]:.0f} ms"></span>' for s in sg)
        gpu = 100 * r["gpu_busy_ms"] / tot
        rows.append(f'<div class="srow"><div class="slabel">{LABEL[c]}</div><div class="sbar">{bars}</div><div class="sval">{tot:.0f} ms<small>GPU busy {gpu:.0f}%</small></div></div>')
    leg = "".join(f'<span><i style="background:{s[2]}"></i>{esc(s[0])}</span>' for s in segs(D[f"{CFG[0]}_{M}"]))
    return f'<div class="stack">{"".join(rows)}</div><div class="legend">{leg}</div>'


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def kernel_table(c, M, n=7):
    r = D[f"{c}_{M}"]
    rows = []
    for k in r["kernels"][:n]:
        rows.append([f'<code>{esc(k["kernel"].replace("opt::", "").replace("bfs::", "").replace("dfs::", ""))}</code>', k["launches"], f'{k["total_ms"]:.1f}',
                     f'<span class="inl"><i style="width:{k["pct_of_kernel_time"]:.1f}%"></i></span>{k["pct_of_kernel_time"]:.0f}%',
                     f'{k["avg_us"]:.0f}', k["block"], k["regs"], f'{100*k["theoretical_occupancy"]:.0f}% ({k["occupancy_limiter"]})'])
    return table(["Kernel", "Launches", "Total ms", "Share of kernel time", "Avg µs", "Block", "Regs", "Theoretical occupancy"], rows, "num")


def level_chart():
    A = levels("bfs_adaptive", 8000000)
    S = levels("bfs_static", 8000000)
    n = max(len(A), len(S))
    W, H, L, B, T = 760, 250, 44, 34, 18
    top = 90.0
    pw = (W - L - 8) / n
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Kernel time per BFS level, static versus adaptive">']
    for v in (0, 20, 40, 60, 80):
        y = H - B - (H - B - T) * v / top
        out.append(f'<line x1="{L}" x2="{W-8}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/><text x="{L-6}" y="{y+4:.1f}" class="ax" text-anchor="end">{v}</text>')
    for i in range(n):
        x0 = L + i * pw
        for j, (series, col) in enumerate(((S, "var(--c1)"), (A, "var(--c2)"))):
            if i < len(series):
                h = (H - B - T) * series[i][2] / top
                bw = pw * 0.38
                x = x0 + pw * 0.1 + j * (bw + 2)
                out.append(f'<rect x="{x:.1f}" y="{H-B-h:.1f}" width="{bw:.1f}" height="{max(h,0.6):.1f}" rx="2" fill="{col}"><title>level {i+1} {"adaptive" if j else "static"}: {series[i][2]:.1f} ms, {series[i][1]} tasks</title></rect>')
        out.append(f'<text x="{x0+pw/2:.1f}" y="{H-B+14}" class="ax" text-anchor="middle">{i+1}</text>')
        if i < len(A) and i < 8:
            t = A[i][1]
            lab = f"{t/1000:.0f}k" if t >= 10000 else (f"{t/1000:.1f}k" if t >= 1000 else str(t))
            out.append(f'<text x="{x0+pw/2:.1f}" y="{H-B+27}" class="ax2" text-anchor="middle">{lab}</text>')
    out.append(f'<text x="{L}" y="{T-6}" class="ax">kernel ms per level</text><text x="{W-8}" y="{H-B+27}" class="ax2" text-anchor="end">tasks →</text>')
    # direct labels
    out.append(f'<text x="{L+pw*3.2:.1f}" y="{T+26}" class="lab" fill="var(--c1t)">static: 106 tasks → 80 ms</text>')
    out.append(f'<text x="{L+pw*3.2:.1f}" y="{T+44}" class="lab" fill="var(--c2t)">adaptive: 106 tasks → 28 ms</text>')
    out.append("</svg>")
    return "".join(out)


def share_early(c, M, k=3):
    L = levels(c, M)
    return 100 * sum(x[2] for x in L[:k]) / sum(x[2] for x in L)


def fill_rows(M):
    rows = []
    for c in CFG:
        r = D[f"{c}_{M}"]
        low = r["time_ms_in_launches_below_50pct_fill"]
        rows.append([LABEL[c], f'{r["kernel_sum_ms"]:.0f}', f'{r["gpu_busy_ms"]:.0f}', f'{100*r["busy_frac_of_span"]:.0f}%', f'{100*r["time_weighted_launch_fill"]:.0f}%',
                     f'{low:.0f} ms ({100*low/r["kernel_sum_ms"]:.0f}%)', f'{r["idle_gap_count"]}', f'{sum(r["idle_gaps_ms"].values()):.1f}'])
    return table(["Config", "Σ kernel ms", "GPU busy ms", "Busy inside kernel span", "Time-weighted launch fill", "Time in launches under 50% fill", "Idle gaps", "Idle ms"], rows, "num")


def summary_rows():
    rows = []
    for M in (8000000, 6000000):
        for c in CFG:
            r = D[f"{c}_{M}"]
            tot = r["load_ms"] + r["mine_ms"]
            rows.append([f"{M/1e6:.0f}M", LABEL[c], f'{r["patterns"]:,}', f'{r["tasks"]:,}', f'{tot:.0f}', f'{r["first_kernel_offset_ms"]:.0f}', f'{r["kernel_span_ms"]:.0f}',
                         f'{100*r["gpu_busy_ms"]/tot:.0f}%', f'{r["peak_live_arena_MiB"]:,}', f'{r["device_alloc_peak_MiB"]/1024:.1f}'])
    return table(["minutil", "Config", "Patterns", "Tasks", "Process ms", "ms before 1st kernel", "Kernel span ms", "GPU busy / process", "Peak live arena MiB", "Device alloc GiB"], rows, "num")



CSS = r"""<style>
:root {
  /* Layout: single reading column; findings first, then evidence sections each ending in a table. */
  --bg:#f7f8f8; --surface:#ffffff; --fg:#111418; --fg2:#4a5159; --fg3:#6d747c; --line:#dcdfe2; --line2:#eceef0;
  --c1:#2a78d6; --c2:#eb6834; --c3:#1baf7a; --c4:#eda100; --c1t:#1f5fae; --c2t:#b8480f;
  --p1:#9aa1a8; --p2:#c1c6cb; --p3:#dde0e3; --p4:#4a3aa7; --warn:#b8480f; --ok:#0a7a52;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace; --sans: "IBM Plex Sans", system-ui, sans-serif; --disp:"Fraunces", Georgia, serif;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#101214; --surface:#181b1e; --fg:#f2f4f5; --fg2:#b5bcc2; --fg3:#8b939a; --line:#2c3136; --line2:#23272b;
  --c1:#3987e5; --c2:#d95926; --c3:#199e70; --c4:#c98500; --c1t:#7db3f2; --c2t:#f0956d;
  --p1:#6d747c; --p2:#454b51; --p3:#2f343a; --p4:#9085e9; --warn:#f0956d; --ok:#4cc79a; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg:#101214; --surface:#181b1e; --fg:#f2f4f5; --fg2:#b5bcc2; --fg3:#8b939a; --line:#2c3136; --line2:#23272b;
  --c1:#3987e5; --c2:#d95926; --c3:#199e70; --c4:#c98500; --c1t:#7db3f2; --c2t:#f0956d;
  --p1:#6d747c; --p2:#454b51; --p3:#2f343a; --p4:#9085e9; --warn:#f0956d; --ok:#4cc79a; color-scheme: dark; }
* { box-sizing: border-box; }
body { background:var(--bg); color:var(--fg); font:15px/1.6 var(--sans); padding-inline:max(16px, 4vw); padding-block:32px 72px; }
main { max-width: 960px; margin: 0 auto; display:flex; flex-direction:column; gap:44px; }
header { display:flex; flex-direction:column; gap:10px; }
h1 { font:600 clamp(30px,5vw,44px)/1.08 var(--disp); margin:0; letter-spacing:-0.01em; text-wrap:balance; }
h2 { font:600 22px/1.25 var(--disp); margin:0 0 4px; text-wrap:balance; }
h3 { font:600 13px/1.3 var(--sans); letter-spacing:.06em; text-transform:uppercase; color:var(--fg3); margin:20px 0 8px; }
p { margin:0 0 12px; max-width:68ch; color:var(--fg2); }
p.lead { font-size:17px; color:var(--fg); max-width:66ch; }
section { display:flex; flex-direction:column; gap:6px; min-width:0; }
code { font:12.5px var(--mono); }
.meta { font:12.5px var(--mono); color:var(--fg3); }
.verdicts { display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:14px; }
.v { background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:16px; min-width:0; }
.v b { display:block; font:600 12px var(--sans); letter-spacing:.06em; text-transform:uppercase; margin-bottom:6px; }
.v.ok b { color:var(--ok); } .v.no b { color:var(--warn); } .v.unk b { color:var(--fg3); }
.v p { margin:0; font-size:14px; }
.stack { display:flex; flex-direction:column; gap:10px; }
.srow { display:grid; grid-template-columns:104px 1fr 112px; gap:12px; align-items:center; }
.slabel { font-weight:600; font-size:13px; }
.sbar { display:flex; gap:2px; height:26px; }
.seg { display:block; height:100%; border-radius:2px; min-width:1px; }
.sval { font:13px var(--mono); text-align:right; font-variant-numeric:tabular-nums; }
.sval small { display:block; font:11px var(--sans); color:var(--fg3); }
.legend { display:flex; flex-wrap:wrap; gap:6px 18px; font-size:12.5px; color:var(--fg2); margin-top:8px; }
.legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px; }
.tw { overflow-x:auto; border:1px solid var(--line); border-radius:6px; background:var(--surface); }
table { border-collapse:collapse; width:100%; font-size:13px; }
th { text-align:left; font:600 11.5px var(--sans); color:var(--fg3); letter-spacing:.04em; padding:9px 12px; border-bottom:1px solid var(--line); white-space:nowrap; background:var(--surface); }
td { padding:8px 12px; border-bottom:1px solid var(--line2); white-space:nowrap; }
tr:last-child td { border-bottom:0; }
table.num td:not(:first-child), table.num th:not(:first-child) { font-variant-numeric:tabular-nums; }
.inl { display:inline-block; width:64px; height:7px; background:var(--line2); border-radius:2px; margin-right:8px; vertical-align:1px; }
.inl i { display:block; height:100%; background:var(--p4); border-radius:2px; }
.inl.big { width:100%; margin:0; height:12px; }
.hrow { display:grid; grid-template-columns:200px 1fr 70px; gap:12px; align-items:center; font-size:13px; }
.hrow b { font:12.5px var(--mono); text-align:right; font-weight:400; }
.chart { background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:14px; overflow-x:auto; }
.chart svg { display:block; min-width:600px; width:100%; height:auto; }
svg .grid { stroke:var(--line2); stroke-width:1; }
svg .ax { fill:var(--fg3); font:11px var(--sans); }
svg .ax2 { fill:var(--fg3); font:10px var(--mono); }
svg .lab { font:600 12px var(--sans); }
.note { border-left:3px solid var(--line); padding:2px 0 2px 14px; color:var(--fg2); font-size:14px; max-width:68ch; }
.cmd { font:12.5px var(--mono); background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:12px 14px; overflow-x:auto; white-space:pre; color:var(--fg2); }
ul { margin:0 0 12px; padding-left:20px; color:var(--fg2); max-width:68ch; } li { margin-bottom:6px; }
@media (max-width:640px) { .srow { grid-template-columns:1fr; gap:4px; } .sval { text-align:left; } .hrow { grid-template-columns:1fr; gap:2px; } .hrow b { text-align:left; } }
</style>"""
N = json.loads((HERE / "data/ncu_summary.json").read_text())
SWEEP = [l.split() for l in (HERE / "data/static_blocksize_sweep.txt").read_text().splitlines() if l.strip()]
byrep = {}
for k in N:
    byrep.setdefault(k["report"][4:], []).append(k)
def nk(rep, i): return byrep[rep][i]

ROWS = [  # label, kernel record
    ("BFS static · level 1 · 106 tasks", nk("bfs_static_block", 0)),
    ("BFS static · level 2 · 1,202 tasks", nk("bfs_static_block", 1)),
    ("BFS static · level 3 · 6,201 tasks", nk("bfs_static_block", 2)),
    ("BFS static · late level · 65k blocks", nk("bfs_static_block_late", 0)),
    ("BFS adaptive · block bin · 514 blocks", nk("bfs_adaptive_block", 2)),
    ("BFS adaptive · huge_local", nk("bfs_adaptive_huge", 0)),
    ("BFS adaptive · huge_subtree", nk("bfs_adaptive_huge", 1)),
    ("BFS adaptive · huge_write", nk("bfs_adaptive_huge", 2)),
    ("BFS adaptive · huge_merge", nk("bfs_adaptive_huge", 3)),
    ("BFS adaptive · sub-warp bin · 80 blocks", nk("bfs_adaptive_sub", 2)),
    ("DFS static · level 1 · 106 tasks", nk("dfs_static_block", 0)),
    ("DFS static · level 2 · 1,202 tasks", nk("dfs_static_block", 1)),
    ("DFS static · late level · 734k blocks", nk("dfs_static_block_late", 0)),
]

def bar(v, cls=""):
    v = max(0, min(100, v or 0))
    return f'<span class="mb {cls}"><i style="width:{v:.0f}%"></i></span><b>{v:.0f}</b>'

def sol_table():
    trs = []
    for lab, k in ROWS:
        trs.append(f'<tr><td>{lab}</td><td class="r">{k["duration_ms"]:.1f}</td><td>{bar(k["sm_pct"],"a")}</td><td>{bar(k["issue_active_pct"],"a")}</td>'
                   f'<td>{bar(k["l1_pct"],"b")}</td><td>{bar(k["l2_pct"],"b")}</td><td>{bar(k["dram_pct"],"b")}<span class="gb">{k["dram_gbps"]:.0f} GB/s</span></td>'
                   f'<td>{bar(k["warps_active_pct"],"c")}</td></tr>')
    head = ("<tr><th>Kernel launch</th><th>ms (base clock)</th><th>SM throughput %</th><th>Issue slots busy %</th><th>L1/TEX %</th><th>L2 %</th>"
            "<th>DRAM % (GB/s of 2,040)</th><th>Warps resident %</th></tr>")
    return f'<div class="tw"><table class="sol"><thead>{head}</thead><tbody>{"".join(trs)}</tbody></table></div>'

STALL_ORDER = [("long_scoreboard", "Waiting on a global load", "var(--c1)"), ("wait", "Fixed-latency dependency", "var(--c3)"),
               ("barrier", "Waiting at __syncthreads", "var(--c2)"), ("short_scoreboard", "Shared / special-function", "var(--c4)")]

def stall_chart():
    out = []
    for lab, k in ROWS:
        sh = k["stall_share"]; used = 0; segs = ""
        for key, name, col in STALL_ORDER:
            v = sh.get(key, 0); used += v
            lbl = f"{v:.0f}%" if v >= 9 else ""
            segs += f'<span class="seg" style="width:{v:.2f}%;background:{col}" title="{name}: {v:.1f}%">{lbl}</span>'
        segs += f'<span class="seg oth" style="width:{max(0,100-used):.2f}%" title="Other: {100-used:.1f}%"></span>'
        out.append(f'<div class="srow2"><div class="slabel2">{lab}</div><div class="sbar2">{segs}</div></div>')
    leg = "".join(f'<span><i style="background:{c}"></i>{n}</span>' for _, n, c in STALL_ORDER) + '<span><i class="oth"></i>Other</span>'
    return f'<div class="stack2">{"".join(out)}</div><div class="legend">{leg}</div>'

FN_ROWS = [("find_item", "binary search for the extension item"), ("same_kept_items", "duplicate check (2 more binary searches)"),
           ("scan_local", "pass 1: local utilities"), ("scan_subtree", "pass 2: subtree utilities + hashing"),
           ("write_representatives", "pass 3a: write child transactions"), ("merge_duplicates", "pass 3b: merge duplicates"),
           ("rename_items", "rename kept items"), ("allocate_child", "allocate child database"),
           ("emit_children", "emit next-level tasks"), ("process_task", "task setup and loop overhead")]
FN_COLS = [("S1", nk("bfs_static_block", 0)), ("S2", nk("bfs_static_block", 1)), ("S3", nk("bfs_static_block", 2)), ("late", nk("bfs_static_block_late", 0)),
           ("local", nk("bfs_adaptive_huge", 0)), ("subtree", nk("bfs_adaptive_huge", 1)), ("write", nk("bfs_adaptive_huge", 2)), ("merge", nk("bfs_adaptive_huge", 3))]

def fn_table():
    head = "<tr><th>Function (source)</th>" + "".join(f"<th>{c}</th>" for c, _ in FN_COLS) + "</tr>"
    trs = []
    for fn, desc in FN_ROWS:
        cells = ""
        for _, k in FN_COLS:
            v = sum(f["stall_pct"] for f in k["by_function"] if f["fn"] == fn)
            cells += f'<td class="h" style="--v:{min(v,60):.0f}%">{v:.0f}%</td>' if v >= 0.5 else '<td class="h dim">–</td>'
        trs.append(f"<tr><td><code>{fn}</code><small>{desc}</small></td>{cells}</tr>")
    return f'<div class="tw"><table class="fn"><thead>{head}</thead><tbody>{"".join(trs)}</tbody></table></div>'

def line_table():
    want = [("include/mine/bfs/device/task_view.cuh", 47), ("include/mine/bfs/device/passes.cuh", 82), ("include/mine/bfs/device/passes.cuh", 34),
            ("include/mine/bfs/device/child.cuh", 131), ("include/mine/bfs/device/child.cuh", 90), ("include/mine/bfs/device/child.cuh", 21)]
    cols = [("static L2", nk("bfs_static_block", 1)), ("static L3", nk("bfs_static_block", 2)), ("huge_local", nk("bfs_adaptive_huge", 0)),
            ("huge_merge", nk("bfs_adaptive_huge", 3)), ("static late", nk("bfs_static_block_late", 0))]
    src = {}
    for f, l in want:
        src[(f, l)] = (ROOT / f).read_text().splitlines()[l - 1].strip()
    head = "<tr><th>Line</th>" + "".join(f"<th>{c}</th>" for c, _ in cols) + "</tr>"
    trs = []
    for f, l in want:
        cells = ""
        for _, k in cols:
            v = next((h["stall_pct"] for h in k["hot_lines"] if h["file"] == f and h["line"] == l), None)
            cells += f"<td>{v:.0f}%</td>" if v is not None else '<td class="dim">&lt;3%</td>'
        trs.append(f'<tr><td><code>{f.split("/")[-1]}:{l}</code><small>{esc(src[(f, l)][:84])}</small></td>{cells}</tr>')
    return f'<div class="tw"><table class="fn"><thead>{head}</thead><tbody>{"".join(trs)}</tbody></table></div>'

def sweep_chart():
    rows = [r for r in SWEEP if r[1] == "static"]
    mx = max(float(r[3]) for r in rows)
    out = []
    for s_ in ("bfs", "dfs"):
        for r in [x for x in rows if x[0] == s_]:
            v = float(r[3]); col = "var(--c1)" if s_ == "bfs" else "var(--c3)"
            out.append(f'<div class="hrow2"><span>{s_.upper()} static · {r[2]} threads</span><span class="inl big"><i style="width:{100*v/mx:.0f}%;background:{col}"></i></span><b>{v*1000:.0f} ms</b></div>')
    for r in [x for x in SWEEP if x[1] == "adaptive"]:
        v = float(r[3]); out.append(f'<div class="hrow2"><span>{r[0].upper()} adaptive</span><span class="inl big"><i style="width:{100*v/mx:.0f}%;background:var(--c2)"></i></span><b>{v*1000:.0f} ms</b></div>')
    return "".join(out)

d8 = {c: D[f"{c}_8000000"] for c in CFG}
busy_min = min(100 * d8[c]["busy_frac_of_span"] for c in CFG)
early_s = share_early("bfs_static", 8000000)
early_a = share_early("bfs_adaptive", 8000000)
lmax = 73.1
loadphases = [("parse (parallel)", 73.1), ("rebuild + sort transactions", 54.1), ("TWU histogram", 7.4), ("map file", 4.6), ("subtree utilities", 2.2), ("merge, prune, rename", 0.3)]
loadbars = "".join(f'<div class="hrow"><span>{n}</span><span class="inl big"><i style="width:{100*v/lmax:.0f}%"></i></span><b>{v:.1f} ms</b></div>' for n, v in loadphases)
cpu = d8["dfs_adaptive"]
s2 = nk("bfs_static_block", 1)

PAGE = f"""<title>G-HUIM Profile Report</title>
{CSS}
<style>
.mb {{ display:inline-block; width:54px; height:7px; background:var(--line2); border-radius:2px; margin-right:6px; vertical-align:1px; }}
.mb i {{ display:block; height:100%; border-radius:2px; background:var(--c1); }}
.mb.a i {{ background:var(--c1); }} .mb.b i {{ background:var(--c2); }} .mb.c i {{ background:var(--c3); }}
td b {{ font:12px var(--mono); font-weight:400; }} .gb {{ font:11px var(--mono); color:var(--fg3); margin-left:6px; }}
td.r {{ text-align:right; font-variant-numeric:tabular-nums; }}
table.sol td, table.sol th {{ white-space:nowrap; }}
.stack2 {{ display:flex; flex-direction:column; gap:7px; }}
.srow2 {{ display:grid; grid-template-columns:230px 1fr; gap:12px; align-items:center; }}
.slabel2 {{ font-size:12.5px; }} .sbar2 {{ display:flex; gap:2px; height:22px; }}
.sbar2 .seg {{ font:600 11px/22px var(--sans); color:#fff; text-align:center; overflow:hidden; }}
.sbar2 .seg.oth, .legend i.oth {{ background:var(--p2); }}
.hrow2 {{ display:grid; grid-template-columns:200px 1fr 70px; gap:12px; align-items:center; font-size:13px; }}
.hrow2 b {{ font:12.5px var(--mono); text-align:right; font-weight:400; }}
table.fn td small {{ display:block; color:var(--fg3); font-size:11.5px; max-width:340px; overflow:hidden; text-overflow:ellipsis; }}
table.fn td.h {{ background:color-mix(in srgb, var(--c1) var(--v), transparent); text-align:right; font-variant-numeric:tabular-nums; }}
table.fn td.dim {{ color:var(--fg3); text-align:right; }}
table.fn td:first-child {{ white-space:normal; min-width:230px; }}
.fix {{ display:flex; flex-direction:column; gap:12px; }}
.fix > div {{ background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:14px 16px; min-width:0; }}
.fix h3 {{ margin:0 0 6px; text-transform:none; letter-spacing:0; font:600 15px var(--sans); color:var(--fg); }}
.fix p {{ margin:0 0 6px; font-size:14px; }} .fix p:last-child {{ margin:0; }}
.tag {{ font:600 11px var(--sans); letter-spacing:.05em; text-transform:uppercase; color:var(--fg3); }}
@media (max-width:640px) {{ .srow2 {{ grid-template-columns:1fr; gap:3px; }} .hrow2 {{ grid-template-columns:1fr; gap:2px; }} .hrow2 b {{ text-align:left; }} }}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=IBM+Plex+Mono&family=IBM+Plex+Sans:wght@400;600&display=swap">
<main>
<header>
  <div class="meta">G-HUIM · NVIDIA H100 PCIe (114 SMs, 2,040 GB/s) · accidents_utility_spmf · merge-search · Nsight Systems 2024.6 + Nsight Compute 2025.1</div>
  <h1>What is slowing G-HUIM down, from the host to the warp</h1>
  <p class="lead">The GPU is not waiting on data transfer and it is not short of bandwidth: no unit passes 60% of its peak and DRAM never passes 30%. It is waiting on <em>latency</em>. In the kernels that dominate the run, 60–90% of warp stall cycles are spent waiting for a global load to come back, because each lane walks its own transaction through 8-byte entries, so only 14% of every 32-byte sector fetched is used. Your layout suspicion is right about the cause, but the limit is load latency, not bandwidth.</p>
</header>

<section>
  <h2>Findings</h2>
  <div class="verdicts">
    <div class="v ok"><b>Measured</b><p>While mining, the GPU is busy {busy_min:.0f}–99% of the time between the first and last kernel. Host syncs and readbacks cost 1.5–2 ms per run. The GPU is not starved by the CPU.</p></div>
    <div class="v no"><b>Measured</b><p>Inside the kernels, speed of light is low everywhere: SM throughput 8–38%, issue slots 13–49%, L2 up to 60%, DRAM up to 30% (602 GB/s). Nothing is saturated, so the kernels are latency-bound.</p></div>
    <div class="v no"><b>Measured</b><p>Level 1 to 3 hold {early_s:.0f}% of BFS static kernel time. There, {byrep["bfs_static_block"][1]["stall_share"]["long_scoreboard"]:.0f}% of stall cycles are global-load waits, Nsight reports 71–78% of requested L2 sectors as excess from uncoalesced access, and global atomics move {s2["red_sectors_m"]/1000:.1f} G sectors (35 GB) through L2 in one launch.</p></div>
    <div class="v unk"><b>Measured</b><p>Late levels are a different problem: 65–67% of stall cycles are <code>__syncthreads</code> barriers, because hundreds of thousands of tiny tasks each run a 256-thread block through several synchronised passes.</p></div>
  </div>
  <p class="note">The profiles use the base clock (1.11 GHz, <code>--clock-control base</code>) and flush caches between replays, so absolute times in this section are about 1.4× the nsys times. Ratios and percentages are unaffected. Each kernel family was sampled on 2–3 launches chosen to cover its large launches.</p>
</section>

<section>
  <h2>Speed of light: how much compute and bandwidth is used</h2>
  <p>Each row is one kernel launch, as a percentage of the hardware peak. SM throughput is the busiest compute pipe. Issue slots busy is how often a scheduler issued an instruction (100% would mean one every cycle). L1/TEX, L2 and DRAM are the memory levels. Warps resident is achieved occupancy.</p>
  {sol_table()}
  <p class="note">Highest compute use anywhere: <code>huge_subtree</code> at 38% SM and 49% issue slots. Highest bandwidth use: BFS static level 3 at 60% of L2 and DFS static level 2 at 30% of DRAM (602 GB/s). DRAM reads for BFS static level 2 are {s2["dram_read_gb"]:.0f} GB, while {s2["l1_traffic_gb"]:.0f} GB passes through L1 to deliver them: the same parent database is re-read by 1,202 tasks and 4 passes.</p>
</section>

<section>
  <h2>Why the units are idle: warp stall reasons</h2>
  <p>Each bar splits the average cycles a warp waits between two issued instructions. The blue segment is waiting for a global load result (long scoreboard), which is what a latency-bound memory access pattern looks like.</p>
  {stall_chart()}
</section>

<section>
  <h2>Where in the code</h2>
  <p>Stall samples per source line, grouped by the device function the line belongs to. Columns S1 to S3 are BFS static levels 1 to 3, <em>late</em> is a late level, and the last four are the BFS adaptive huge-task kernels at level 1. All of these are in <code>include/mine/bfs/device/</code>, and the DFS kernels show the same split.</p>
  {fn_table()}
  <p>Binary search (<code>find_item</code>) is the single biggest cost at about 21% of samples, and it runs in every pass, plus twice more inside <code>same_kept_items</code>. The four passes together account for most of the rest. The hottest individual lines:</p>
  {line_table()}
</section>

<section>
  <h2>The layout question</h2>
  <ul>
    <li><b>Access pattern, confirmed.</b> Every pass has each lane take one transaction (<code>i = i0 + lanes.rank</code>) and walk its entries <code>e[k+1+d]</code> alone. A warp-wide load therefore touches up to 32 different transactions. At BFS static level 1 Nsight reports only 4.6 of 32 bytes per sector used by loads (6.8 for stores), and 71–78% of the L2 sectors the big launches request as excess from uncoalesced access (41% in the late levels).</li>
    <li><b>Dependent loads, confirmed.</b> Each kept entry costs two chained global loads: the entry, then <code>rename[entry.item]</code> in scratch. That chain is 46% of the stall samples in <code>huge_merge</code> (<code>child.cuh:131</code>) and 10–19% in <code>scan_subtree</code> (<code>passes.cuh:82</code>).</li>
    <li><b>Contended global atomics, confirmed.</b> <code>add(p.local[item], …)</code> and <code>add(p.subtree[z], …)</code> issue one 64-bit atomic per entry into arrays of only ~100 items. BFS static level 2 sends 1.1 G atomic sectors to L2 (35 GB) in 52 ms, and L2 is the busiest unit there at 54%.</li>
    <li><b>Repeated work, confirmed.</b> Four passes each re-read the parent entries, and each re-runs the binary search. Level 2 pushes {s2["l1_traffic_gb"]:.0f} GB through L1 to move {s2["dram_read_gb"]:.0f} GB from DRAM.</li>
    <li><b>Bandwidth, ruled out.</b> DRAM never exceeds 30% of peak and the largest launches use 0.2–0.6 TB/s of 2.04 TB/s. A better layout would raise useful bytes per sector rather than reduce total bytes.</li>
    <li><b>Host transfer, ruled out.</b> The only host-to-device copy is the 88 MiB root database (4 ms).</li>
  </ul>
</section>

<section>
  <h2>What to change, in order of evidence</h2>
  <p>These are proposals from the measurements above. None is implemented, and the estimates are not promises.</p>
  <div class="fix">
    <div><span class="tag">1 · Largest, least risky</span><h3>Compute each transaction's extension-item position once</h3><p><code>find_item</code> plus <code>same_kept_items</code> are about 20% + 16–28% of stall samples in every kernel. Pass 1 already finds the position <code>k</code> for every transaction. Store it in the per-task scratch (4 bytes per transaction, which the scratch layout already budgets for) and have passes 2, 3a, 3b and the duplicate check read it.</p></div>
    <div><span class="tag">2 · Large</span><h3>Keep the per-item tables in shared memory</h3><p><code>local[]</code>, <code>subtree[]</code> and <code>rename[]</code> are indexed by item and hold 100 or fewer items at the first levels (fewer deeper). Accumulate with shared-memory atomics and flush once per block, and read <code>rename[]</code> from shared memory. That removes the global atomic traffic (35 GB through L2 in one launch) and one of the two dependent loads per entry. Fall back to the current path for tasks with too many items.</p></div>
    <div><span class="tag">3 · Medium, more work</span><h3>Make entry scans coalesced</h3><p>Split <code>Entry</code> into separate item and utility arrays. Passes that only test <code>rename[item]</code> then read 4 bytes per entry instead of 8, and utilities are read only for kept entries. For the large-task kernels also let a warp scan one transaction with lanes on consecutive entries, instead of one transaction per lane.</p></div>
    <div><span class="tag">4 · Late levels</span><h3>Stop using a 256-thread block for tiny tasks</h3><p>Barriers are 65–67% of stall cycles at late levels. Adaptive already routes small tasks to warp and sub-warp kernels. For the static method, the sweep below shows 256 to 512 threads is the best fixed size, so static has no better setting to fall back on.</p></div>
    <div><span class="tag">5 · DFS only</span><h3>Replace the single-block selector</h3><p><code>select_kernel</code> is one block of 1,024 threads. It costs 22 ms of 318 ms of adaptive kernel time at 8M, with barriers about 30% of its stalls. The sampled launches were its first, smallest batches, so its large-batch behaviour is only known from nsys timing.</p></div>
  </div>
</section>

<section>
  <h2>Static block size sweep (kernel time at minutil 8M)</h2>
  <p>Static work is a fixed block per task, so block size is its only knob. Small blocks lose because the lane-per-transaction scans need many lanes in flight to hide load latency. 1,024 loses to barrier and occupancy costs. Adaptive beats the best static size by 1.4× (BFS) and 1.7× (DFS).</p>
  {sweep_chart()}
</section>

<section>
  <h2>Process time by phase</h2>
  <p>Mining kernels are the violet segment. The rest of the process is CUDA startup, loading and the root database build, which the kernels cannot affect.</p>
  {stack_chart(8000000)}
  <h3>Minutil 6M (10.1M patterns)</h3>
  {stack_chart(6000000)}
  <p class="note">At 8M the GPU is busy only 17–24% of the whole process. About 0.3–0.4 s is CUDA context creation (a bare <code>cudaFree(0)</code> takes 300–420 ms here), 0.15 s is loading the dataset and about 0.05 s is the CPU building the root database.</p>
</section>

<section>
  <h2>Early levels carry the kernel time</h2>
  <p>BFS at minutil 8M, kernel time per level. Level 1 has 106 tasks, each over the whole 88 MiB database, so a static launch is 106 blocks on 114 SMs: 0.15 waves and 12% of warp slots filled. Adaptive slices those tasks over many blocks and finishes the level in 28 ms instead of 80 ms.</p>
  <div class="chart">{level_chart()}</div>
</section>

<section>
  <h2>GPU time by kernel (Nsight Systems)</h2>
  <p>Per config at minutil 8M. Theoretical occupancy is from registers, shared memory and block size on the H100 limits.</p>
  {''.join(f'<h3>{LABEL[c]}</h3>' + kernel_table(c, 8000000) for c in CFG)}
  <h3>GPU busy and launch fill</h3>
  {fill_rows(8000000)}
</section>

<section>
  <h2>CPU time</h2>
  <p>Nsight Systems CPU sampling is disabled on this machine (<code>perf_event_paranoid=4</code>), so there are no per-function CPU samples. The breakdown comes from NVTX ranges added around each phase, plus OS-runtime and CUDA API tracing.</p>
  <h3>Dataset loading, 151 ms</h3>
  <div class="stack">{loadbars}</div>
  <p class="note" style="margin-top:10px">Loading spawns 282 threads (<code>pthread_create</code>, 33 ms in total), because each parallel stage creates its own workers.</p>
  <h3>Mining phase, host side (DFS adaptive, 8M)</h3>
  {table(["Phase", "Total ms", "Calls", "What it is"], [
    ["CUDA startup before the first runtime call", "≈510", "1", "Driver and context creation, not under any NVTX range"],
    ["<code>cudaDeviceSynchronize</code>", f'{cpu["cuda_api_in_mine"][0]["total_ms"]:.0f}', cpu["cuda_api_in_mine"][0]["calls"], "Host waiting for the batch's kernels"],
    ["dfs: build root database", "44", "1", "CPU builds the arena-format image (<code>root_image</code>)"],
    ["dfs: select + bin batch", "28", "19", "Includes the single-block select kernel and its sync"],
    ["<code>cudaStreamSynchronize</code>", f'{cpu["cuda_api_in_mine"][1]["total_ms"]:.0f}', cpu["cuda_api_in_mine"][1]["calls"], "Waiting for select and binning"],
    ["<code>cudaFree</code> / <code>cudaMalloc</code>", "4.9 / 2.2", "22 / 11", "Buffer setup and teardown"],
    ["dfs: upload root database", "4.1", "1", "88 MiB host-to-device copy, about 22 GB/s"],
  ], "num")}
</section>

<section>
  <h2>All runs</h2>
  {summary_rows()}
  <p class="note">All 8 configurations return the pattern counts of the CPU EFIM baseline (3,030,087 at 8M and 10,079,776 at 6M).</p>
</section>

<section>
  <h2>How this was measured, and gaps</h2>
  <ul>
    <li>The code is built with <code>-O3 -lineinfo</code>, so Nsight Compute can attribute stall samples to source lines in optimized code, including lines inside inlined device functions. <code>cmake -DGHUIM_PROFILE=ON</code> also prints registers, spills and shared memory per kernel at compile time. <code>-G</code> is not used, because it disables optimization and would profile different code.</li>
    <li>ncu ran 13 targeted jobs through <code>profiling/run_ncu.sh</code> as root (GPU counters are restricted for normal users). Three of the DFS adaptive jobs (block, warp, sub-warp) matched no kernel because of a regex, and the BFS adaptive warp job caught only tiny launches. The script is fixed, and these are the missing data points. DFS adaptive shares its kernels' structure with BFS adaptive, and its huge-task kernels match BFS's within 5% on every metric.</li>
    <li>Late-level and sub-warp numbers come from one or two launches each, so they show the regime, not a precise average.</li>
    <li>Per-function shares are stall-sample shares, not elapsed-time shares, and they are rounded.</li>
  </ul>
  <div class="cmd">sudo bash profiling/run_ncu.sh                                    # all targeted ncu runs
ONLY="dfs_adaptive_block dfs_adaptive_warp dfs_adaptive_sub bfs_adaptive_warp" sudo -E bash profiling/run_ncu.sh   # the missing ones
python3 profiling/analyze.py && python3 profiling/analyze_ncu.py && python3 profiling/make_report.py</div>
</section>
</main>
"""
(HERE / "report.html").write_text(PAGE)
print(len(PAGE))
