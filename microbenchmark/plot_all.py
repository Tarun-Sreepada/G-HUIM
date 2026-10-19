"""Plot every micro-benchmark CSV in results/ to plots/NN_name.png.

Usage: .venv/bin/python microbenchmark/plot_all.py
Colors are the validated categorical order (fixed per series); each series also gets its own marker so identity
never rests on color alone.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
RES, OUT = HERE / "results", HERE / "plots"
OUT.mkdir(exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "font.size": 10, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 6, "lines.markeredgecolor": SURFACE, "lines.markeredgewidth": 1.5,
})


def load(name):
    df = pd.read_csv(RES / f"{name}.csv")
    bad = df[df["ok"] != 1] if "ok" in df else df.iloc[0:0]
    if len(bad):
        print(f"  WARNING {name}: {len(bad)} rows failed validation")
    return df


def panels(n, ncols=None, w=4.6, h=3.6, sharey=False):
    ncols = ncols or n
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows + 1.6), sharey=sharey, squeeze=False)
    flat = list(axes.flat)
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat


def lines(ax, df, x, y, series, order=None, label=lambda s: str(s)):
    keys = order or list(dict.fromkeys(df[series]))
    for i, k in enumerate(keys):
        d = df[df[series] == k].sort_values(x)
        ax.plot(d[x], d[y], color=COLORS[i % 6], marker=MARKERS[i % 6], label=label(k))


def finish(fig, name, title, handles_ax, ncol=6, note=None):
    _, h_in = fig.get_size_inches()
    h, l = handles_ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, 0.05 / h_in))
    fig.text(0.01, 1 - 0.12 / h_in, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    if note:
        fig.text(0.01, 1 - 0.42 / h_in, note, ha="left", va="top", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.95 / h_in + 0.02, 1, 1 - (0.85 if note else 0.55) / h_in))
    fig.savefig(OUT / f"{name}.png", dpi=150)
    plt.close(fig)
    print("  wrote", name)


def fmt_bytes(v, _=None):
    for unit, s in (("MB", 1 << 20), ("KB", 1 << 10)):
        if v >= s:
            return f"{v / s:g} {unit}"
    return f"{v:g} B"


def p01():
    df = load("01_tiled_layout")
    fig, axs = panels(2)
    for ax, mode, ylabel in zip(axs, ("scan", "search"), ("GB/s (higher is better)", "Mqueries/s (higher is better)")):
        lines(ax, df[df["mode"] == mode], "len", "rate", "stride", label=lambda s: f"stride {s}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("transaction length (keys)")
        ax.set_ylabel(ylabel)
        ax.set_title("Sequential scan" if mode == "scan" else "Binary search (8 queries / tx)")
    finish(fig, "01_tiled_layout", "Tiled vs contiguous key layout", axs[0], note="stride 1 = contiguous, 32 = warp-wide tile")


def p02():
    df = load("02_search_methods")
    qs = sorted(df["queries"].unique())
    fig, axs = panels(len(qs), ncols=2, sharey=False)
    names = {"thread_binary": "thread binary", "thread_linear": "thread linear", "warp_ballot": "warp ballot", "warp_two_level": "warp two-level"}
    for ax, q in zip(axs, qs):
        lines(ax, df[df["queries"] == q], "len", "mqueries_per_s", "method", order=list(names), label=names.get)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{q} queries per transaction")
        ax.set_xlabel("transaction length")
        ax.set_ylabel("Mqueries/s")
    finish(fig, "02_search_methods", "Item lookup in a sorted transaction (one warp per transaction)", axs[0], ncol=4)


def p03():
    df = load("03_rename_gather")
    fig, axs = panels(2, sharey=True)
    for ax, pat in zip(axs, ("random", "clustered")):
        lines(ax, df[df["pattern"] == pat], "table_bytes", "glookups_per_s", "variant", order=["global", "shared"])
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_bytes))
        ax.set_title(f"{pat} indices")
        ax.set_xlabel("rename[] table size")
    axs[0].set_ylabel("Glookups/s")
    finish(fig, "03_rename_gather", "rename[] gather vs table size", axs[0], ncol=2)


def p04():
    df = load("04_cp_async_ring")
    fig, axs = panels(2, sharey=False)
    for ax, p in zip(axs, sorted(df["passes"].unique())):
        lines(ax, df[df["passes"] == p], "depth", "gb_per_s", "entry_bytes", label=lambda s: fmt_bytes(s) + " entry")
        ax.set_xticks(range(5))
        ax.set_xticklabels(["direct", "1", "2", "3", "4"])
        ax.set_xlabel("ring depth (slots)")
        ax.set_title(f"{p} compute passes per chunk")
    axs[0].set_ylabel("GB/s")
    finish(fig, "04_cp_async_ring", "cp.async staging ring: depth and entry size", axs[0], ncol=4)


def p05():
    df = load("05_staging_reuse")
    tiles = sorted(df["tile_bytes"].unique())
    fig, axs = panels(len(tiles), sharey=False)
    for ax, t in zip(axs, tiles):
        lines(ax, df[df["tile_bytes"] == t], "reuse", "ms", "mode", order=["direct", "staged", "cp_async"])
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df["reuse"].unique()))
        ax.set_xticklabels([str(v) for v in sorted(df["reuse"].unique())])
        ax.set_title(f"{fmt_bytes(t)} tile")
        ax.set_xlabel("reuse (passes over the tile)")
    axs[0].set_ylabel("ms (lower is better)")
    finish(fig, "05_staging_reuse", "Direct global reads vs shared-memory staging", axs[0], ncol=3)


def p06():
    df = load("06_thread_warp_block")
    fig, axs = panels(1, w=7)
    lines(axs[0], df, "len", "gitems_per_s", "mapping", order=["thread", "warp", "block"])
    axs[0].set_xscale("log", base=2)
    axs[0].set_xlabel("transaction length (items)")
    axs[0].set_ylabel("Gitems/s")
    axs[0].set_title("Work unit per transaction")
    finish(fig, "06_thread_warp_block", "Thread vs warp vs block per transaction", axs[0], ncol=3)


def p07():
    df = load("07_occupancy")
    df = df[df["blocks_per_sm"] > 0].copy()
    df["ms"] = df["ms"].astype(float)
    fig, axs = panels(2)
    lines(axs[0], df, "smem_bytes", "ms", "block", label=lambda b: f"block {b}")
    lines(axs[1], df, "smem_bytes", "occupancy_pct", "block", label=lambda b: f"block {b}")
    for ax, yl, t in zip(axs, ("ms (lower is better)", "theoretical occupancy (%)"), ("Measured time", "Occupancy API")):
        ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_bytes))
        ax.set_xlabel("dynamic shared memory per block")
        ax.set_ylabel(yl)
        ax.set_title(t)
        ax.tick_params(axis="x", rotation=30)
    finish(fig, "07_occupancy", "Latency-bound kernel vs occupancy", axs[0], ncol=5, note="missing points = shared-memory request does not fit")


def p08():
    df = load("08_load_imbalance")
    order = ["block_per_task", "persistent_static", "queue", "queue_lpt", "warp_queue", "warp_queue_lpt"]
    fig, axs = panels(2)
    # x: alpha treated as categories
    alphas = sorted(df["alpha"].unique())
    for ax, col, yl, t in zip(axs, ("ms", "ms"), ("ms (lower is better)", "ms relative to best strategy"), ("Absolute", "Relative")):
        for i, k in enumerate(order):
            d = df[df["strategy"] == k].set_index("alpha").loc[alphas]
            y = d["ms"] if t == "Absolute" else d["ms"] / df.groupby("alpha")["ms"].min().loc[alphas]
            ax.plot(range(len(alphas)), y.values, color=COLORS[i], marker=MARKERS[i], label=k)
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{a:g}" for a in alphas])
        ax.set_xlabel("Zipf alpha (0 = uniform; larger = more skew)")
        ax.set_ylabel(yl)
        ax.set_title(t)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
    finish(fig, "08_load_imbalance", "Scheduling skewed task sizes (constant total work)", axs[0], ncol=3)


def p09():
    df = load("09_utility_accum")
    fig, axs = panels(2, sharey=True)
    for ax, d in zip(axs, ("uniform", "skewed")):
        lines(ax, df[df["dist"] == d], "counters", "gupdates_per_s", "variant", order=["atomic32", "atomic64", "warp_agg", "smem"])
        ax.set_xscale("log", base=4)
        ax.set_yscale("log")
        ax.set_xlabel("number of counters (fewer = more contention)")
        ax.set_title(f"{d} index distribution")
    axs[0].set_ylabel("Gupdates/s")
    finish(fig, "09_utility_accum", "Utility accumulation under contention", axs[0], ncol=4)


def p10():
    df = load("10_output_append")
    fig, axs = panels(1, w=7)
    lines(axs[0], df, "density_pct", "gitems_per_s", "variant", order=["atomic_per_item", "warp_aggregated", "block_scan"])
    axs[0].set_xscale("log")
    axs[0].set_xticks(sorted(df["density_pct"].unique()))
    axs[0].set_xticklabels([f"{v}%" for v in sorted(df["density_pct"].unique())])
    axs[0].set_xlabel("fraction of items selected")
    axs[0].set_ylabel("Gitems/s")
    axs[0].set_title("Compacted append")
    finish(fig, "10_output_append", "Output append strategies", axs[0], ncol=3)


def p11():
    df = load("11_stream_overlap")
    fig, axs = panels(3, sharey=False)
    sel = [("copy", 0, "Copy only"), ("both", 16, "Copy + kernel (16 rounds)"), ("both", 256, "Copy + kernel (256 rounds)")]
    for ax, (mode, rounds, t) in zip(axs, sel):
        d = df[(df["mode"] == mode) & (df["rounds"] == rounds)]
        lines(ax, d, "streams", "ms", "chunk_mb", label=lambda c: f"{c} MB chunks")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("streams")
        ax.set_title(t)
    axs[0].set_ylabel("ms for 256 MB (lower is better)")
    finish(fig, "11_stream_overlap", "Chunked H2D copy + consumer kernel", axs[0], ncol=4)


def p12():
    df = load("12_arena_alloc")
    sizes = sorted(df["bytes"].unique())
    fig, axs = panels(len(sizes), ncols=2)
    for ax, b in zip(axs, sizes):
        lines(ax, df[df["bytes"] == b], "allocs", "ms", "variant", order=["malloc_free", "bump_thread", "bump_warp"])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{b} B per allocation")
        ax.set_xlabel("allocations (one per thread)")
        ax.set_ylabel("ms (lower is better)")
    finish(fig, "12_arena_alloc", "Device malloc/free vs bump-pointer arena", axs[0], ncol=3, note="configs over 512 MB are skipped")


def p13():
    df = load("13_bmma_equality")
    fig, axs = panels(1, w=7)
    lines(axs[0], df, "m_equals_n", "gpairs_per_s", "variant", order=["cuda_core", "tensor_b1"], label=lambda v: {"cuda_core": "CUDA cores (64-bit ==)", "tensor_b1": "tensor cores (b1 AND+popc)"}[v])
    axs[0].set_xscale("log", base=2)
    axs[0].set_xlabel("M = N (keys per side)")
    axs[0].set_ylabel("Gpairs/s")
    axs[0].set_title("All-pairs 64-bit key equality")
    finish(fig, "13_bmma_equality", "Tensor-core vs CUDA-core equality", axs[0], ncol=2)


if __name__ == "__main__":
    for fn in (p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13):
        fn()
