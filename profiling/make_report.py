#!/usr/bin/env python3
"""Build report.html from data/nsys_summary.json and the debug logs."""
import json, re, html
from pathlib import Path

HERE = Path(__file__).parent
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


d8 = {c: D[f"{c}_8000000"] for c in CFG}
cpu = d8["dfs_adaptive"]
loadphases = [("parse (parallel)", 73.1), ("rebuild + sort transactions", 54.1), ("TWU histogram", 7.4), ("map file", 4.6), ("subtree utilities", 2.2), ("merge, prune, rename", 0.3)]
lmax = max(v for _, v in loadphases)
loadbars = "".join(f'<div class="hrow"><span>{n}</span><span class="inl big"><i style="width:{100*v/lmax:.0f}%"></i></span><b>{v:.1f} ms</b></div>' for n, v in loadphases)
busy_min = min(100 * d8[c]["busy_frac_of_span"] for c in CFG)
early_s = share_early("bfs_static", 8000000)
early_a = share_early("bfs_adaptive", 8000000)

PAGE = f"""<title>G-HUIM Profile Report</title>
<style>
:root {{
  /* Layout: single reading column; findings first, then evidence sections each ending in a table. */
  --bg:#f7f8f8; --surface:#ffffff; --fg:#111418; --fg2:#4a5159; --fg3:#6d747c; --line:#dcdfe2; --line2:#eceef0;
  --c1:#2a78d6; --c2:#eb6834; --c3:#1baf7a; --c4:#eda100; --c1t:#1f5fae; --c2t:#b8480f;
  --p1:#9aa1a8; --p2:#c1c6cb; --p3:#dde0e3; --p4:#4a3aa7; --warn:#b8480f; --ok:#0a7a52;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace; --sans: "IBM Plex Sans", system-ui, sans-serif; --disp:"Fraunces", Georgia, serif;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#101214; --surface:#181b1e; --fg:#f2f4f5; --fg2:#b5bcc2; --fg3:#8b939a; --line:#2c3136; --line2:#23272b;
  --c1:#3987e5; --c2:#d95926; --c3:#199e70; --c4:#c98500; --c1t:#7db3f2; --c2t:#f0956d;
  --p1:#6d747c; --p2:#454b51; --p3:#2f343a; --p4:#9085e9; --warn:#f0956d; --ok:#4cc79a; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg:#101214; --surface:#181b1e; --fg:#f2f4f5; --fg2:#b5bcc2; --fg3:#8b939a; --line:#2c3136; --line2:#23272b;
  --c1:#3987e5; --c2:#d95926; --c3:#199e70; --c4:#c98500; --c1t:#7db3f2; --c2t:#f0956d;
  --p1:#6d747c; --p2:#454b51; --p3:#2f343a; --p4:#9085e9; --warn:#f0956d; --ok:#4cc79a; color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.6 var(--sans); padding-inline:max(16px, 4vw); padding-block:32px 72px; }}
main {{ max-width: 920px; margin: 0 auto; display:flex; flex-direction:column; gap:44px; }}
header {{ display:flex; flex-direction:column; gap:10px; }}
h1 {{ font:600 clamp(30px,5vw,44px)/1.08 var(--disp); margin:0; letter-spacing:-0.01em; text-wrap:balance; }}
h2 {{ font:600 22px/1.25 var(--disp); margin:0 0 4px; text-wrap:balance; }}
h3 {{ font:600 13px/1.3 var(--sans); letter-spacing:.06em; text-transform:uppercase; color:var(--fg3); margin:20px 0 8px; }}
p {{ margin:0 0 12px; max-width:68ch; color:var(--fg2); }}
p.lead {{ font-size:17px; color:var(--fg); max-width:66ch; }}
section {{ display:flex; flex-direction:column; gap:6px; min-width:0; }}
code {{ font:12.5px var(--mono); }}
.meta {{ font:12.5px var(--mono); color:var(--fg3); }}
.verdicts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:14px; }}
.v {{ background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:16px; min-width:0; }}
.v b {{ display:block; font:600 12px var(--sans); letter-spacing:.06em; text-transform:uppercase; margin-bottom:6px; }}
.v.ok b {{ color:var(--ok); }} .v.no b {{ color:var(--warn); }} .v.unk b {{ color:var(--fg3); }}
.v p {{ margin:0; font-size:14px; }}
.stack {{ display:flex; flex-direction:column; gap:10px; }}
.srow {{ display:grid; grid-template-columns:104px 1fr 112px; gap:12px; align-items:center; }}
.slabel {{ font-weight:600; font-size:13px; }}
.sbar {{ display:flex; gap:2px; height:26px; }}
.seg {{ display:block; height:100%; border-radius:2px; min-width:1px; }}
.sval {{ font:13px var(--mono); text-align:right; font-variant-numeric:tabular-nums; }}
.sval small {{ display:block; font:11px var(--sans); color:var(--fg3); }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 18px; font-size:12.5px; color:var(--fg2); margin-top:8px; }}
.legend i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px; }}
.tw {{ overflow-x:auto; border:1px solid var(--line); border-radius:6px; background:var(--surface); }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th {{ text-align:left; font:600 11.5px var(--sans); color:var(--fg3); letter-spacing:.04em; padding:9px 12px; border-bottom:1px solid var(--line); white-space:nowrap; background:var(--surface); }}
td {{ padding:8px 12px; border-bottom:1px solid var(--line2); white-space:nowrap; }}
tr:last-child td {{ border-bottom:0; }}
table.num td:not(:first-child), table.num th:not(:first-child) {{ font-variant-numeric:tabular-nums; }}
.inl {{ display:inline-block; width:64px; height:7px; background:var(--line2); border-radius:2px; margin-right:8px; vertical-align:1px; }}
.inl i {{ display:block; height:100%; background:var(--p4); border-radius:2px; }}
.inl.big {{ width:100%; margin:0; height:12px; }}
.hrow {{ display:grid; grid-template-columns:200px 1fr 70px; gap:12px; align-items:center; font-size:13px; }}
.hrow b {{ font:12.5px var(--mono); text-align:right; }}
.chart {{ background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:14px; overflow-x:auto; }}
.chart svg {{ display:block; min-width:600px; width:100%; height:auto; }}
svg .grid {{ stroke:var(--line2); stroke-width:1; }}
svg .ax {{ fill:var(--fg3); font:11px var(--sans); }}
svg .ax2 {{ fill:var(--fg3); font:10px var(--mono); }}
svg .lab {{ font:600 12px var(--sans); }}
.note {{ border-left:3px solid var(--line); padding:2px 0 2px 14px; color:var(--fg2); font-size:14px; max-width:68ch; }}
.cmd {{ font:12.5px var(--mono); background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:12px 14px; overflow-x:auto; white-space:pre; color:var(--fg2); }}
ul {{ margin:0 0 12px; padding-left:20px; color:var(--fg2); max-width:68ch; }} li {{ margin-bottom:6px; }}
@media (max-width:640px) {{ .srow {{ grid-template-columns:1fr; gap:4px; }} .sval {{ text-align:left; }} .hrow {{ grid-template-columns:1fr; gap:2px; }} .hrow b {{ text-align:left; }} }}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=IBM+Plex+Mono&family=IBM+Plex+Sans:wght@400;600&display=swap">
<main>
<header>
  <div class="meta">G-HUIM · NVIDIA H100 PCIe (114 SMs) · accidents_utility_spmf · merge-search @ {esc('21dfcd3+')} · Nsight Systems 2024.6.2</div>
  <h1>Where the time goes in G-HUIM, and what is still unmeasured</h1>
  <p class="lead">The GPU is not starved by the host while it mines: it is busy {busy_min:.0f}–99% of the time between the first and last kernel. Most of a run is spent before that, in CUDA startup, file loading and database build. Whether the kernels themselves stall on memory could not be measured, because GPU performance counters are blocked on this machine.</p>
</header>

<section>
  <h2>Findings</h2>
  <div class="verdicts">
    <div class="v ok"><b>Measured</b><p>Between kernels the GPU is almost never idle: {min(d8[c]["idle_gap_count"] for c in CFG)}–{max(d8[c]["idle_gap_count"] for c in CFG)} gaps per run at minutil 8M, 1.5–2.1 ms in total. At 6M the DFS runs lose under 3 ms and the BFS runs about 26 ms (still under 8% of kernel span), from the extra scratch chunks of the large middle levels.</p></div>
    <div class="v no"><b>Measured</b><p>At minutil 8M a run takes about 0.9 s, of which only 17–24% has the GPU busy. About 0.3–0.4 s is CUDA context creation (a bare <code>cudaFree(0)</code> takes 300–420 ms here), 0.15 s is loading the dataset and 0.05 s builds the root database on the CPU.</p></div>
    <div class="v no"><b>Measured</b><p>Inside the kernel span, early levels dominate. In BFS the first three levels hold {early_s:.0f}% (static) and {early_a:.0f}% (adaptive) of kernel time, with only 106 to 6,201 tasks. Launches under 50% fill take {min(100*d8[c]["time_ms_in_launches_below_50pct_fill"]/d8[c]["kernel_sum_ms"] for c in CFG):.0f}–{max(100*d8[c]["time_ms_in_launches_below_50pct_fill"]/d8[c]["kernel_sum_ms"] for c in CFG):.0f}% of summed kernel time.</p></div>
    <div class="v unk"><b>Not measured</b><p>Stall reasons, cache hit rates, coalescing and DRAM throughput. <code>ncu</code> fails with ERR_NVGPUCTRPERM, and nsys GPU metrics are blocked the same way. The data-layout hypothesis therefore stays unproven. The next section lists what the available numbers do and do not say about it.</p></div>
  </div>
</section>

<section>
  <h2>Process time by phase, minutil 8M</h2>
  <p>Every bar is one whole process, from the start of dataset loading to the end of mining. The violet segment is the only part where kernels run. Hover a segment for its duration.</p>
  {stack_chart(8000000)}
  <h3>Minutil 6M (10.1M patterns)</h3>
  {stack_chart(6000000)}
</section>

<section>
  <h2>The GPU is busy while it mines</h2>
  <p>Idle gaps are the stretches inside the first-to-last-kernel span where no kernel ran on any stream. Launch fill is the share of resident-block slots one launch can occupy at once, from its grid size and the kernel's theoretical occupancy, averaged by kernel time. A value of 100% means a full wave or more. It says how much of the GPU a launch could use, not how efficiently the warps run.</p>
  {fill_rows(8000000)}
  <p class="note">Adaptive configs have a summed kernel time well above their GPU busy time because bins run on parallel streams. Static configs run one kernel at a time, so for them the two are equal.</p>
</section>

<section>
  <h2>Early levels carry the kernel time</h2>
  <p>The same search, level by level, in BFS at minutil 8M. Level 1 has 106 tasks, each a large database, so a static launch is 106 blocks of 256 threads on 114 SMs: about 15% of the resident-block slots, for 70–80 ms. Adaptive slices those tasks over many blocks and finishes level 1 in 28 ms. From level 7 on, the levels hold hundreds of thousands of small tasks and take 5 ms or less.</p>
  <div class="chart">{level_chart()}</div>
</section>

<section>
  <h2>What the numbers say about data layout</h2>
  <ul>
    <li><b>Writes are not the limit.</b> Level 1 writes 432 MiB of child databases in 28.5 ms (adaptive), about 16 GB/s against 2,040 GB/s peak. The same pattern holds on the next levels.</li>
    <li><b>Re-reading the parent is plausible but unverified.</b> The root database is 87.9 MiB (the H2D copy), larger than the 50 MB L2. Level 1 has 106 tasks that each scan it, so one pass is at least 9.1 GB of reads. In 28.5 ms that is 327 GB/s per pass (16% of peak), and in 80 ms (static) 114 GB/s. Each task makes several passes, which would put the adaptive level 1 well above that. Pass counts and cache behaviour need counters.</li>
    <li><b>Register-limited occupancy.</b> The task kernels use 31–63 registers per thread and the block, warp and sub-warp kernels are capped at 50–78% theoretical occupancy, by registers. None of the top kernels uses local memory.</li>
    <li><b>The DFS selector.</b> DFS <code>select_kernel</code> is a single block of 1024 threads, 22 ms in total at 8M (7–10% of its kernel time), so it runs one block while the rest of the GPU waits.</li>
  </ul>
  <p class="note">None of this proves memory stalls. The distinguishing counters are long-scoreboard stalls, sectors per request, L1 and L2 hit rates and DRAM throughput. They are collected by <code>profiling/run_ncu.sh</code>, once counter access is enabled.</p>
</section>

<section>
  <h2>GPU time by kernel</h2>
  <p>Per config at minutil 8M, from the CUDA kernel trace. Theoretical occupancy comes from registers, shared memory and block size on the H100 limits, not from a hardware counter.</p>
  {''.join(f'<h3>{LABEL[c]}</h3>' + kernel_table(c, 8000000) for c in CFG)}
</section>

<section>
  <h2>CPU time</h2>
  <p>Nsight Systems CPU sampling is disabled on this machine (<code>perf_event_paranoid=4</code>), so there are no per-function CPU samples. The breakdown below comes from NVTX ranges added around each phase, and from OS-runtime and CUDA API tracing.</p>
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
    ["dfs: batch readback", "0.3", "19", "Control block copy"],
  ], "num")}
</section>

<section>
  <h2>All runs</h2>
  {summary_rows()}
  <p class="note">All 8 configurations return the pattern counts of the CPU EFIM baseline (3,030,087 at 8M and 10,079,776 at 6M). DFS also passed the run gate with default memory and with <code>--total-memory 1G</code>. BFS needs more than 1 GB: at 1G it stops with its designed &ldquo;raise --total-memory&rdquo; error.</p>
</section>

<section>
  <h2>To finish the memory and compute analysis</h2>
  <p>Hardware counters need administrator action on this server. Either run the prepared script once as root, or lift the restriction in the driver (this needs a driver reload, so nobody else may be using the GPUs):</p>
  <div class="cmd">sudo bash profiling/run_ncu.sh          # 13 targeted runs, one GPU, each under timeout

# or, to allow every user:
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-prof.conf
# then reload the nvidia kernel modules or reboot, and run the script without sudo</div>
  <p>The script profiles the heaviest launches of every kernel family at minutil 8M, with the speed-of-light, memory-workload, warp-state, occupancy and source-counter sections plus raw metrics for coalescing, cache hit rates, atomics and DRAM bytes. It uses <code>-c</code> and <code>-s</code> to limit each run to a few launches. Its metric names were checked against this GPU's metric list, but the script itself has not run, since counters are blocked.</p>
  <h3>Commands used for the nsys data</h3>
  <div class="cmd">nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true --sample=none -o &lt;cfg&gt;_&lt;minutil&gt; \\
  build/G-HUIM --input accidents_utility_spmf.txt --minutil &lt;M&gt; --search &lt;bfs|dfs&gt; --method &lt;static|adaptive&gt; --total-memory 8G
python3 profiling/analyze.py      # sqlite exports to profiling/data/nsys_summary.json</div>
</section>
</main>
"""
(HERE / "report.html").write_text(PAGE)
print(len(PAGE))
