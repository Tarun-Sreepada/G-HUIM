import os
import subprocess
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

import itertools
from matplotlib.patches import Patch

import matplotlib

OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

RES_DIR = Path("results")
RES_DIR.mkdir(exist_ok=True)


# Configure Matplotlib for LaTeX-friendly output (PGF/PDF).
# Note: rendering PGF with LaTeX requires a LaTeX installation when compiling
# the generated .pgf files into a document. Saving .pdf provides a ready-to-use
# vector figure as well.
matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "pgf.rcfonts": False,
    }
)


def _check_latex_working() -> bool:
    """Return True if a minimal LaTeX run succeeds (used by matplotlib tex rendering).

    Matplotlib with text.usetex=True will attempt to run 'latex' to measure text.
    Some TeX Live installations may miss small packages (e.g., type1ec) and fail.
    We try a minimal run and fall back to disabling usetex on failure.
    """
    import subprocess

    tex = r"""\documentclass{article}
\begin{document}
lp
\end{document}
"""
    try:
        p = subprocess.run(
            ["latex", "-interaction=nonstopmode", "--halt-on-error", "file.tex"],
            input=tex.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except Exception as e:
        return False


# If LaTeX isn't fully available at runtime, disable text.usetex so matplotlib
# won't call out to latex for text measurement (which causes runtime failures
# such as missing type1ec.sty). We still keep saving PGF/PDF files, but matplotlib
# will use its internal font metrics rather than compiling TeX for labels.
if matplotlib.rcParams.get("text.usetex", False):
    if not _check_latex_working():
        matplotlib.rcParams["text.usetex"] = False




CSV_CACHE = RES_DIR / "iterations.csv"

CACHE_COLUMNS = [
    "dataset",
    "minutil",
    "method",
    "adaptive",
    "block_size",
    "iteration",
    "time_ms",
    "gpu_mem_mb",
]


def _load_cache() -> pd.DataFrame:
    """Load existing cache or create an empty DataFrame with the right schema."""
    if CSV_CACHE.exists():
        return pd.read_csv(CSV_CACHE)
    return pd.DataFrame(columns=CACHE_COLUMNS)


def _append_to_cache(rows: List[Tuple[Union[str, int, float]]]) -> None:
    """Append new rows to the CSV cache in an atomic‑ish way."""
    df_new = pd.DataFrame(rows, columns=CACHE_COLUMNS)
    if CSV_CACHE.exists():
        df_new.to_csv(CSV_CACHE, mode="a", header=False, index=False)
    else:
        df_new.to_csv(CSV_CACHE, mode="w", header=True, index=False)


def run_iteration_times(
    input_file: str,
    minutil: int,
    adaptive: bool = False,
    block_size: int | None = None,
    total_memory: str = "78G",
    method: str = "SP",
) -> Dict[int, Tuple[float, int]]:
    """Run G-HUIM once and return iteration → (time_ms, gpu_mem_mb) mapping."""

    cmd = [
        "../../build/G-HUIM",
        "--input",
        input_file,
        "--minutil",
        str(minutil),
        "--method",
        method,
        "--total_memory",
        total_memory,
        "--output",
        "/dev/null",
    ]

    if adaptive:
        cmd += ["--adaptive", "true"]
    else:
        cmd += ["--adaptive", "false", "--block_size", str(block_size)]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )

    times: Dict[int, Tuple[float, int]] = {}
    for line in result.stdout.splitlines():
        if line.strip().startswith("Iteration"):
            parts = line.split()
            it = int(parts[1])
            t_ms = float(parts[2])
            mem_mb = int(parts[5])
            times[it] = (t_ms, mem_mb)
    return times


def get_or_run_iteration_times(
    input_file: str,
    minutil: int,
    adaptive: bool,
    block_size: int | None,
    total_memory: str,
    method: str,
) -> Dict[int, Tuple[float, int]]:
    """Return iteration times, reusing cached results when available."""
    cache = _load_cache()

    if not cache.empty:
        mask = (
            (cache["dataset"] == input_file)
            & (cache["minutil"] == minutil)
            & (cache["method"] == method)
            & (cache["adaptive"] == adaptive)
            & (cache["block_size"].fillna(-1).astype(int) == (block_size or -1))
        )
        subset = cache.loc[mask]
        if not subset.empty:

            return dict(
                zip(
                    subset["iteration"].astype(int),
                    zip(
                        subset["time_ms"].astype(float),
                        subset["gpu_mem_mb"].astype(int),
                    ),
                )
            )

    times_and_mems = run_iteration_times(
        input_file,
        minutil,
        adaptive=adaptive,
        block_size=block_size,
        total_memory=total_memory,
        method=method,
    )

    rows = [
        (
            input_file,
            minutil,
            method,
            adaptive,
            block_size,
            it,
            t,
            mem,
        )
        for it, (t, mem) in times_and_mems.items()
    ]
    _append_to_cache(rows)
    return times_and_mems


method_nice_names = {
    "DPHIM_DPHIM_dphim": "DPHIM",
    "G-HUIM_SP_fixed_32": "GS-HUIM(32)",
    "G-HUIM_SP_fixed_64": "GS-HUIM(64)",
    "G-HUIM_SP_fixed_128": "GS-HUIM(128)",
    "G-HUIM_SP_fixed_256": "GS-HUIM(256)",
    "G-HUIM_SP_fixed_512": "GS-HUIM(512)",
    "G-HUIM_SP_fixed_1024": "GS-HUIM(1024)",
    "G-HUIM_SP_adaptive": "GA-HUIM",
}


phases = ["Parse", "TWU", "Build", "Search"]
phase_colors = {
    "Parse": "#2ca02c",
    "TWU": "#1f77b4",
    "Build": "#ffdd57",
    "Search": "#ff7f0e",
}

method_colors = {
    "DPHIM_DPHIM_dphim": "#1f77b4",
    "G-HUIM_SP_adaptive": "#d62728",
}
method_markers = {
    "DPHIM_DPHIM_dphim": "^",
    "G-HUIM_SP_adaptive": "o",
}

static_methods = [
    "G-HUIM_SP_fixed_32",
    "G-HUIM_SP_fixed_64",
    "G-HUIM_SP_fixed_128",
    "G-HUIM_SP_fixed_256",
    "G-HUIM_SP_fixed_512",
    "G-HUIM_SP_fixed_1024",
]

green_steps = ["5F", "5F", "5F", "5F", "5F", "5F"]

for method, green_hex in zip(static_methods, green_steps):
    hex_color = f"#00{green_hex}00"
    method_colors[method] = hex_color
    method_markers[method] = ""

line_styles = ["-", "--", ":", "-.", (0, (3, 5, 1, 5)), (0, (1, 1)), (0, (5, 10))]
all_methods = list(method_nice_names.keys())
method_linestyles = {
    m: line_styles[i % len(line_styles)] for i, m in enumerate(all_methods)
}

linestyles = ["-", "--", ":", "-."]
hatches = ["/", "\\", "x", "o", ".", "*", "+", "-"]
method_style = {
    m: {
        "linestyle": linestyles[i % len(linestyles)],
        "hatch": hatches[i % len(hatches)],
    }
    for i, m in enumerate(all_methods)
}


import os
import subprocess
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


CACHE_COLUMNS = [
    "dataset",
    "minutil",
    "method",
    "adaptive",
    "block_size",
    "iteration",
    "time_ms",
    "gpu_mem_mb",
]


def _load_cache() -> pd.DataFrame:
    """Load existing cache or create an empty DataFrame with the right schema."""
    if CSV_CACHE.exists():
        return pd.read_csv(CSV_CACHE)
    return pd.DataFrame(columns=CACHE_COLUMNS)


def _append_to_cache(rows: List[Tuple[Union[str, int, float]]]) -> None:
    """Append new rows to the CSV cache in an atomic‑ish way."""
    df_new = pd.DataFrame(rows, columns=CACHE_COLUMNS)
    if CSV_CACHE.exists():
        df_new.to_csv(CSV_CACHE, mode="a", header=False, index=False)
    else:
        df_new.to_csv(CSV_CACHE, mode="w", header=True, index=False)


def run_iteration_times(
    input_file: str,
    minutil: int,
    adaptive: bool = False,
    block_size: int | None = None,
    total_memory: str = "78G",
    method: str = "SP",
) -> Dict[int, Tuple[float, int]]:
    """Run G-HUIM once and return iteration → (time_ms, gpu_mem_mb) mapping."""

    cmd = [
        "../build/G-HUIM",
        "--input",
        input_file,
        "--minutil",
        str(minutil),
        "--method",
        method,
        "--total_memory",
        total_memory,
        "--output",
        "/dev/null",
    ]

    if adaptive:
        cmd += ["--adaptive", "true"]
    else:
        cmd += ["--adaptive", "false", "--block_size", str(block_size)]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )

    times: Dict[int, Tuple[float, int]] = {}
    for line in result.stdout.splitlines():
        if line.strip().startswith("Iteration"):
            parts = line.split()
            it = int(parts[1])
            t_ms = float(parts[2])
            mem_mb = int(parts[5])
            times[it] = (t_ms, mem_mb)
    return times


def get_or_run_iteration_times(
    input_file: str,
    minutil: int,
    adaptive: bool,
    block_size: int | None,
    total_memory: str,
    method: str,
) -> Dict[int, Tuple[float, int]]:
    """Return iteration times, reusing cached results when available."""
    cache = _load_cache()

    if not cache.empty:
        mask = (
            (cache["dataset"] == input_file)
            & (cache["minutil"] == minutil)
            & (cache["method"] == method)
            & (cache["adaptive"] == adaptive)
            & (cache["block_size"].fillna(-1).astype(int) == (block_size or -1))
        )
        subset = cache.loc[mask]
        if not subset.empty:

            return dict(
                zip(
                    subset["iteration"].astype(int),
                    zip(
                        subset["time_ms"].astype(float),
                        subset["gpu_mem_mb"].astype(int),
                    ),
                )
            )

    times_and_mems = run_iteration_times(
        input_file,
        minutil,
        adaptive=adaptive,
        block_size=block_size,
        total_memory=total_memory,
        method=method,
    )

    rows = [
        (
            input_file,
            minutil,
            method,
            adaptive,
            block_size,
            it,
            t,
            mem,
        )
        for it, (t, mem) in times_and_mems.items()
    ]
    _append_to_cache(rows)
    return times_and_mems


method_nice_names = {
    "DPHIM_DPHIM_dphim": "DPHIM",
    "G-HUIM_SP_fixed_32": "GS-HUIM(32)",
    "G-HUIM_SP_fixed_64": "GS-HUIM(64)",
    "G-HUIM_SP_fixed_128": "GS-HUIM(128)",
    "G-HUIM_SP_fixed_256": "GS-HUIM(256)",
    "G-HUIM_SP_fixed_512": "GS-HUIM(512)",
    "G-HUIM_SP_fixed_1024": "GS-HUIM(1024)",
    "G-HUIM_SP_adaptive": "GA-HUIM",
}


phases = ["Parse", "TWU", "Build", "Search"]
phase_colors = {
    "Parse": "#2ca02c",
    "TWU": "#1f77b4",
    "Build": "#ffdd57",
    "Search": "#ff7f0e",
}

method_colors = {
    "DPHIM_DPHIM_dphim": "#1f77b4",
    "G-HUIM_SP_adaptive": "#d62728",
}
method_markers = {
    "DPHIM_DPHIM_dphim": "^",
    "G-HUIM_SP_adaptive": "o",
}

static_methods = [
    "G-HUIM_SP_fixed_32",
    "G-HUIM_SP_fixed_64",
    "G-HUIM_SP_fixed_128",
    "G-HUIM_SP_fixed_256",
    "G-HUIM_SP_fixed_512",
    "G-HUIM_SP_fixed_1024",
]

green_steps = ["5F", "5F", "5F", "5F", "5F", "5F"]

for method, green_hex in zip(static_methods, green_steps):
    hex_color = f"#00{green_hex}00"
    method_colors[method] = hex_color
    method_markers[method] = ""

line_styles = ["-", "--", ":", "-.", (0, (3, 5, 1, 5)), (0, (1, 1)), (0, (5, 10))]
all_methods = list(method_nice_names.keys())
method_linestyles = {
    m: line_styles[i % len(line_styles)] for i, m in enumerate(all_methods)
}

linestyles = ["-", "--", ":", "-."]
hatches = ["/", "\\", "x", "o", ".", "*", "+", "-"]
method_style = {
    m: {
        "linestyle": linestyles[i % len(linestyles)],
        "hatch": hatches[i % len(hatches)],
    }
    for i, m in enumerate(all_methods)
}


def plot_for_configs(
    input_file: str,
    minutil: int,
    block_sizes: List[int],
    total_memory: str = "60G",
    method: str = "SP",
    include_legend: bool = False,
) -> None:
    """
    Plot iteration-wise times and memory, saving the legend to a separate file.
    """
    configs = [("adaptive", None)] + [(f"static_{bs}", bs) for bs in block_sizes]

    all_iterations: set[int] = set()
    results: Dict[str, Dict[int, Tuple[float, int]]] = {}

    for label, bs in configs:
        adaptive_flag = label == "adaptive"
        times_and_mems = get_or_run_iteration_times(
            input_file,
            minutil,
            adaptive=adaptive_flag,
            block_size=bs,
            total_memory=total_memory,
            method=method,
        )
        results[label] = times_and_mems
        all_iterations.update(times_and_mems.keys())

    if not all_iterations:
        print("No data to plot.")
        return

    iterations = sorted(list(all_iterations))

    fig, ax1 = plt.subplots(figsize=(3, 5))
    ax2 = ax1.twinx()


    for label, bs in configs:
        adaptive_flag = label == "adaptive"
        method_key = "G-HUIM_SP_adaptive" if adaptive_flag else f"G-HUIM_SP_fixed_{bs}"
        display_name = method_nice_names.get(method_key, label)
        y_times = [results[label].get(i, (0, 0))[0] for i in iterations]

        ax1.plot(
            iterations,
            y_times,
            label=display_name,
            linestyle=method_linestyles.get(method_key, "-"),
            marker=method_markers.get(method_key, None),
            color=method_colors.get(method_key, None),
        )

        # add the lines
        ax1.yaxis.grid(True, which="major", linestyle=(0, (1, 3)), linewidth=0.9, color="gray", alpha=0.6)

    # reduce the x-axis ticks to be less crowded
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    raw_adaptive_mems = [results["adaptive"].get(i, (0, 0))[1] for i in iterations]

    plotted_mems = []
    if raw_adaptive_mems:

        plotted_mems.append(raw_adaptive_mems[0])

        for i in range(1, len(raw_adaptive_mems)):
            plotted_mems.append(raw_adaptive_mems[i - 1] + raw_adaptive_mems[i])

    if plotted_mems:
        ax2.fill_between(
            iterations,
            plotted_mems,
            alpha=0.3,
            color="gray",
            label="Active GPU Memory (GA-HUIM)",
        )

        # Make the execution-time axis (ax1) grid visually prominent with large dotted lines
        # Use a dotted dash pattern with thicker linewidth to emulate "large dots".
        ax1.yaxis.grid(
            True,
            which="major",
            linestyle=(0, (1, 5)),
            linewidth=2.0,
            color="gray",
            alpha=0.25,
        )
        # ax1.yaxis.grid(
        #     True,
        #     which="minor",
        #     linestyle=(0, (1, 5)),
        #     linewidth=1.0,
        #     color="gray",
        #     alpha=0.6,
        # )
        # ax1.minorticks_on()

        # For the memory axis (ax2) use thicker solid horizontal lines for easy mapping
        ax2.yaxis.grid(True, which="major", linestyle="-", linewidth=2.0, color="gray", alpha=0.25)
        # ax2.yaxis.grid(True, which="minor", linestyle="-", linewidth=1.0, color="gray", alpha=0.6)
        # ax2.minorticks_on()

    # collect legend entries from ax1 and add the memory patch
    handles1, labels1 = ax1.get_legend_handles_labels()
    mem_patch = Patch(color="gray", alpha=0.3, label="Active GPU Memory (GA-HUIM)")
    all_handles = handles1 + [mem_patch]
    all_labels = labels1 + ["Active GPU Memory"]

    adaptive_times = [results["adaptive"].get(i, (0, 0))[0] for i in iterations]
    max_adaptive_time = max(adaptive_times) if adaptive_times else 1
    height_cutoff = 1.5 * max_adaptive_time
    ax1.set_ylim(0, height_cutoff)
    ax2.set_ylim(bottom=0)

    ax1.set_xlabel("Iteration", fontsize=14)

    if input_file.lower().find('accidents') != -1:
        ax1.set_ylabel("Time (ms)", fontsize=14)
        ax2.set_ylabel("GPU Memory Usage (MB)", fontsize=14, rotation=270, labelpad=20)
    # ax1.set_ylabel("Execution Time (ms)", fontsize=14)
    # ax2.set_ylabel("Active GPU Memory (MB)", fontsize=14, rotation=270, labelpad=20)

    ax1.tick_params(axis="y", labelsize=14)
    ax2.tick_params(axis="y", labelsize=14)
    ax1.tick_params(axis="x", labelsize=14)

    if iterations:
        ax1.set_xlim(min(iterations) - 0.2, max(iterations) + 0.2)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    file_base = Path(input_file).stem
    file_name = f"{file_base}_minutil_{minutil}_iteration.png"

    title = file_base.split("_")[0]
    # plt.title(f"Iteration-wise Execution Time and Memory on {title}", fontsize=16)

    fig.savefig(OUT_DIR / file_name, dpi=500, bbox_inches="tight")
    plt.close(fig)

    if include_legend:
        num_items = len(all_labels)
        ncol = 2
        fig_width = 8
        fig_height = (num_items + ncol - 1) // ncol * 0.5

        fig_legend = plt.figure(figsize=(fig_width, fig_height))
        fig_legend.legend(
            handles=all_handles,
            labels=all_labels,
            loc="center",
            frameon=False,
            ncol=ncol,
            fontsize=14,
        )

        legend_filename = "iteration_times_legend.png"

        fig_legend.savefig(OUT_DIR / legend_filename, dpi=500, bbox_inches="tight")
        plt.close(fig_legend)


if __name__ == "__main__":
    block_sizes = [32, 64, 128, 256, 512, 1024]

    datasets_to_process = [
        [
            "../datasets/pumsb_utility_spmf.txt",
            [9680000, 9690000, 9700000, 9710000, 9720000, 9730000],
        ],
        [
            "../datasets/BMS_utility_spmf.txt",
            [1999000, 2000000, 2001000, 2003000, 2004000, 2005000],
        ],
        [
            "../datasets/kosarak_utility_spmf.txt",
            [720000, 730000, 740000, 750000, 760000, 770000],
        ],
        [
            "../datasets/accidents_utility_spmf.txt",
            [3900000, 4000000, 4100000, 4200000, 4300000, 4400000],
        ],
        [
            "../datasets/connect_utility_spmf.txt",
            [7400000, 7500000, 7600000, 7700000, 7800000, 7900000],
        ],
        [
            "../datasets/chess_utility_spmf.txt",
            [170000, 175000, 180000, 185000, 190000, 195000],
        ],
        [
            "../datasets/retail_utility_spmf.txt",
            [1440, 1445, 1450, 1455, 1460, 1465],
        ],
        [
            "../datasets/mushroom_utility_spmf.txt",
            [5500, 6000, 6500, 7000, 7500, 8000],
        ],
    ]

    for input_path, minutils in datasets_to_process:
        minutil = minutils[0]
        print(f"Processing {input_path} with minutil {minutil}")

        plot_for_configs(input_path, minutil, block_sizes, include_legend=True)
