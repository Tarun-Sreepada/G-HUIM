"""Plot sweep.csv: one figure per metric (runtime, patterns, memory, energy), one panel per dataset, x = minutil.

Usage: python3 experiments/plot_sweep.py [--csv experiments/results/sweep.csv] [--out experiments/plots/sweep]
A G-HUIM run that failed (e.g. BFS arena exhausted) is drawn as a red cross on the x axis: "didn't run".
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COLORS = {"REFIM": "#1f77b4", "G-HUIM": "#d95f02"}
METRICS = [
    ("time_s", "Runtime (s)", "runtime"),
    ("patterns", "High-utility itemsets", "patterns"),
    ("memory", "Peak memory (MB)", "memory"),
    ("energy_j", "Energy (J)", "energy"),
]


def panel(ax, data, metric):
    failed = {}
    for miner, g in data.groupby("miner"):
        ok = g[g.status == "ok"].groupby("minutil").mean(numeric_only=True).sort_index()
        if metric == "memory":
            ax.plot(ok.index, ok.rss_mb, "o-", color=COLORS[miner], label=f"{miner} host RSS")
            if miner == "G-HUIM" and ok.device_mb.notna().any():
                ax.plot(ok.index, ok.device_mb, "s--", color=COLORS[miner], label="G-HUIM device (arena+scratch)")
        else:
            ax.plot(ok.index, ok[metric], "o-", color=COLORS[miner], label=miner)
        failed[miner] = g[g.status != "ok"].minutil.unique()
    ax.set_yscale("log")
    for miner, xs in failed.items():
        if len(xs):
            ax.plot(xs, [ax.get_ylim()[0]] * len(xs), "x", color="red", ms=9, mew=2, label=f"{miner} didn't run", clip_on=False)
    ax.set_xlabel("minimum utility")
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "experiments/results/sweep.csv"))
    ap.add_argument("--out", default=str(ROOT / "experiments/plots/sweep"))
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    datasets = list(dict.fromkeys(df.dataset))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    cols = 3
    rows = -(-len(datasets) // cols)
    for metric, label, name in METRICS:
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows), squeeze=False)
        for ax, ds in zip(axes.flat, datasets):
            panel(ax, df[df.dataset == ds], metric)
            ax.set_title(ds)
            ax.set_ylabel(label)
        for ax in list(axes.flat)[len(datasets):]:
            ax.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        seen = {}
        for a in axes.flat:
            for h, l in zip(*a.get_legend_handles_labels()):
                seen.setdefault(l, h)
        fig.legend(seen.values(), seen.keys(), loc="upper center", ncol=len(seen), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(Path(args.out) / f"{name}.png", dpi=130)
        plt.close(fig)
        print("wrote", Path(args.out) / f"{name}.png")


if __name__ == "__main__":
    main()
