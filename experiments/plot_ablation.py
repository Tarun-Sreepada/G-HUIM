"""Plot ablation.csv: every REFIM / G-HUIM config against each other on runtime, energy, memory and patterns.

Usage: python3 experiments/plot_ablation.py [--csv experiments/results/ablation.csv] [--out experiments/plots/ablation]

Writes to --out:
  ratio_<metric>.png  heatmap, config x dataset: geomean over minutils of metric / refim/default (<1 = better)
  bars_<metric>.png   one panel per dataset, every config at the hardest (lowest) minutil, absolute values
  lines_<metric>.png  one panel per dataset, the four engine defaults vs minutil
  patterns.png        config x dataset: matches the consensus count / mismatch / didn't run
Memory is host RSS + device MB (REFIM has no device part). A failed run is a red cross: "didn't run".
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm

ROOT = Path(__file__).resolve().parent.parent
BASE = "refim/default"
FAMILIES = {"refim": "#2a78d6", "bfs": "#eb6834", "dfs": "#1baf7a", "persistent": "#6250d6"}
ENGINES = {"refim/default": "REFIM", "bfs/adaptive": "G-HUIM bfs", "dfs": "G-HUIM dfs", "persistent": "G-HUIM persistent"}
METRICS = [("time_s", "Runtime (s)", "runtime"), ("energy_j", "Energy (J)", "energy"),
           ("memory", "Peak memory, host RSS + device (MB)", "memory")]
INK, MUTED, GRID, FAIL = "#0b0b0b", "#898781", "#e1e0d9", "#d03b3b"
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#1c5cab", "#86b6ef", "#f0efec", "#ec8a8a", "#b02a2a"])

plt.rcParams.update({"axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def family(config):
    return config.split("/")[0]


def load(path):
    df = pd.read_csv(path)
    df["memory"] = df.rss_mb + df.device_mb.fillna(0)
    ok = df[df.status == "ok"].groupby(["dataset", "minutil", "config"]).mean(numeric_only=True).reset_index()
    configs = list(dict.fromkeys(df.config))
    return df, ok, list(dict.fromkeys(df.dataset)), configs


def ran(df):
    """(dataset, minutil, config) -> True if any rep finished."""
    return df.assign(ok=df.status == "ok").groupby(["dataset", "minutil", "config"]).ok.any()


def ratio_heatmap(df, ok, datasets, configs, metric, label, out):
    piv = ok.pivot_table(index=["dataset", "minutil"], columns="config", values=metric)
    rel = np.log(piv.div(piv[BASE], axis=0))
    geo = np.exp(rel.groupby("dataset").mean()).reindex(index=datasets, columns=configs).T
    did = ran(df).groupby(["dataset", "config"]).all().unstack("dataset").reindex(index=configs, columns=datasets)
    # symmetric log range around 1, 0.5x..2x at least; clipped at 20x so one outlier doesn't wash out the rest
    lim = min(20.0, max(2.0, math.exp(np.nanmax(np.abs(np.log(geo.values))))))
    fig, ax = plt.subplots(figsize=(1.0 * len(datasets) + 3.5, 0.28 * len(configs) + 1.6))
    im = ax.imshow(geo.values, cmap=DIVERGING, norm=LogNorm(1 / lim, lim, clip=True), aspect="auto")
    for i, c in enumerate(configs):
        for j, d in enumerate(datasets):
            v = geo.iat[i, j]
            if np.isnan(v):
                ax.text(j, i, "×", ha="center", va="center", color=FAIL, fontsize=11, fontweight="bold")
                continue
            dark = abs(math.log(v)) > 0.6 * math.log(lim)
            txt = f"{v:.2f}" if v < 10 else f"{v:.0f}"
            ax.text(j, i, txt + ("*" if not did.iat[i, j] else ""), ha="center", va="center", fontsize=7,
                    color="white" if dark else INK)
    ax.set_xticks(range(len(datasets)), datasets, rotation=30, ha="right")
    ax.set_yticks(range(len(configs)), configs)
    for t in ax.get_yticklabels():
        t.set_color(FAMILIES[family(t.get_text())])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(f"{label.split(' (')[0]} / {BASE}  (geomean over minutils)")
    ax.set_title(f"{label.split(' (')[0]} relative to {BASE}: <1 better, >1 worse.  "
                 f"× never ran, * didn't run at some minutils (geomean of the rest)", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def bars(df, ok, datasets, configs, metric, label, out):
    cols = 3
    rows = -(-len(datasets) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 0.17 * len(configs) * rows + 1.2 * rows), squeeze=False)
    for ax, ds in zip(axes.flat, datasets):
        mu = df[df.dataset == ds].minutil.min()
        vals = ok[(ok.dataset == ds) & (ok.minutil == mu)].set_index("config")[metric].reindex(configs)
        y = np.arange(len(configs))
        ax.barh(y, vals.values, height=0.72, color=[FAMILIES[family(c)] for c in configs])
        if metric == "memory":
            ax.set_xscale("log")
        xmax = ax.get_xlim()[1]
        for yi in y[vals.isna().values]:
            ax.plot(ax.get_xlim()[0], yi, "x", color=FAIL, ms=6, mew=1.8, clip_on=False)
        base = vals.get(BASE)
        if base is not None and not np.isnan(base):
            ax.axvline(base, color=MUTED, lw=1, ls="--")
        ax.set_xlim(right=xmax)
        ax.set_yticks(y, configs, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_title(f"{ds}  (minutil {mu})", loc="left")
        ax.set_xlabel(label)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    for ax in list(axes.flat)[len(datasets):]:
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILIES.values()]
    handles += [plt.Line2D([], [], color=MUTED, ls="--"), plt.Line2D([], [], color=FAIL, marker="x", ls="", mew=1.8)]
    fig.legend(handles, list(FAMILIES) + [BASE, "didn't run"], loc="upper center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def lines(df, ok, datasets, metric, label, out):
    cols = 3
    rows = -(-len(datasets) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.4 * rows), squeeze=False)
    for ax, ds in zip(axes.flat, datasets):
        for cfg, name in ENGINES.items():
            g = ok[(ok.dataset == ds) & (ok.config == cfg)].sort_values("minutil")
            ax.plot(g.minutil, g[metric], "o-", lw=2, ms=5, color=FAMILIES[family(cfg)], label=name)
        if metric == "time_s":
            ax.set_ylim(bottom=0)
        else:
            ax.set_yscale("log")
        failed = df[(df.dataset == ds) & df.config.isin(ENGINES)]
        failed = failed.groupby(["config", "minutil"]).status.apply(lambda s: (s != "ok").all())
        xs = sorted({mu for (_, mu), f in failed.items() if f})
        if xs:
            ax.plot(xs, [ax.get_ylim()[0]] * len(xs), "x", color=FAIL, ms=8, mew=2, clip_on=False, label="didn't run")
        ax.set_title(ds, loc="left")
        ax.set_xlabel("minimum utility")
        ax.set_ylabel(label)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(color=GRID, lw=0.6)
    for ax in list(axes.flat)[len(datasets):]:
        ax.axis("off")
    seen = {}
    for a in axes.flat:
        for h, l in zip(*a.get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), loc="upper center", ncol=len(seen), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def patterns(df, datasets, configs, out):
    """0 = every finished rep matches the consensus count at every minutil, 1 = some mismatch, 2 = never ran."""
    okr = df[df.status == "ok"]
    consensus = okr.groupby(["dataset", "minutil"]).patterns.agg(lambda s: s.mode().iat[0])
    wrong = okr.patterns.values != consensus.reindex(pd.MultiIndex.from_frame(okr[["dataset", "minutil"]])).values
    bad = okr.assign(wrong=wrong).groupby(["config", "dataset"]).wrong.agg(["sum", "count"])
    state = np.full((len(configs), len(datasets)), 2)
    text = [["" for _ in datasets] for _ in configs]
    for (c, d), r in bad.iterrows():
        i, j = configs.index(c), datasets.index(d)
        state[i, j] = 1 if r["sum"] else 0
        text[i][j] = f"{int(r['sum'])}/{int(r['count'])}" if r["sum"] else "✓"
    fig, ax = plt.subplots(figsize=(1.0 * len(datasets) + 3.5, 0.28 * len(configs) + 1.6))
    ax.imshow(state, cmap=ListedColormap(["#d7ecd7", "#f4b9b9", "#ecebe6"]), vmin=0, vmax=2, aspect="auto")
    for i in range(len(configs)):
        for j in range(len(datasets)):
            t = text[i][j] or "×"
            ax.text(j, i, t, ha="center", va="center", fontsize=7,
                    color={0: "#006300", 1: "#8a1c1c", 2: FAIL}[state[i, j]], fontweight="bold" if state[i, j] else None)
    ax.set_xticks(range(len(datasets)), datasets, rotation=30, ha="right")
    ax.set_yticks(range(len(configs)), configs)
    for t in ax.get_yticklabels():
        t.set_color(FAMILIES[family(t.get_text())])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("High-utility itemset count vs the consensus across configs:  ✓ all runs match,  "
                 "k/n = k of n finished runs wrong,  × never ran", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "experiments/results/ablation.csv"))
    ap.add_argument("--out", default=str(ROOT / "experiments/plots/ablation"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, ok, datasets, configs = load(args.csv)
    for metric, label, name in METRICS:
        ratio_heatmap(df, ok, datasets, configs, metric, label, out / f"ratio_{name}.png")
        bars(df, ok, datasets, configs, metric, label, out / f"bars_{name}.png")
        lines(df, ok, datasets, metric, label, out / f"lines_{name}.png")
    lines(df, ok, datasets, "patterns", "High-utility itemsets", out / "lines_patterns.png")
    patterns(df, datasets, configs, out / "patterns.png")
    for p in sorted(out.glob("*.png")):
        print("wrote", p)


if __name__ == "__main__":
    main()
