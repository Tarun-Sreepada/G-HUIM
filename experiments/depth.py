import os
import subprocess
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

###############################################################################
# Persistent cache (CSV) utilities                                            #
###############################################################################

CSV_CACHE = Path("iteration_times_cache.csv")

CACHE_COLUMNS = [
    "dataset",       # absolute path of the dataset file                     
    "minutil",       # minimum utility threshold                             
    "method",        # cuDPEFIM_SP_fixed_32 | cuDPEFIM_SP_adaptive | ...     
    "adaptive",      # bool                                                 
    "block_size",    # int or None                                          
    "iteration",     # iteration index (int)                                 
    "time_ms",       # execution time in ms (float)                          
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


###############################################################################
# Low‑level execution helper                                                  #
###############################################################################

def run_iteration_times(
    input_file: str,
    minutil: int,
    adaptive: bool = False,
    block_size: int | None = None,
    total_memory: str = "78G",
    method: str = "SP",
) -> Dict[int, float]:
    """Run cuDPEFIM once and return iteration→time (ms) mapping."""

    cmd = [
        "/export/home1/ltarun/cuDPEFIM/build/cuDPEFIM",
        "--input", input_file,
        "--minutil", str(minutil),
        "--method", method,
        "--total_memory", total_memory,
        "--output", "/dev/null",
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

    times: Dict[int, float] = {}
    for line in result.stdout.splitlines():
        if line.startswith("Iteration"):
            parts = line.split()
            it = int(parts[1])
            t = float(parts[2])
            times[it] = t
    return times


###############################################################################
# Public helper with caching                                                  #
###############################################################################

def get_or_run_iteration_times(
    input_file: str,
    minutil: int,
    adaptive: bool,
    block_size: int | None,
    total_memory: str,
    method: str,
) -> Dict[int, float]:
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
            # Reconstruct dict and return
            return dict(zip(subset["iteration"].astype(int), subset["time_ms"].astype(float)))

    # Otherwise execute and store
    times = run_iteration_times(
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
        )
        for it, t in times.items()
    ]
    _append_to_cache(rows)
    return times


###############################################################################
# Nice‑name mapping & style dictionaries (DO NOT MODIFY)                      #
###############################################################################

method_nice_names = {
    "DPHIM_DPHIM_dphim": "DPHIM",
    "cuDPEFIM_SP_fixed_32": "GS-HUIM(32)",
    "cuDPEFIM_SP_fixed_64": "GS-HUIM(64)",
    "cuDPEFIM_SP_fixed_128": "GS-HUIM(128)",
    "cuDPEFIM_SP_fixed_256": "GS-HUIM(256)",
    "cuDPEFIM_SP_fixed_512": "GS-HUIM(512)",
    "cuDPEFIM_SP_fixed_1024": "GS-HUIM(1024)",
    "cuDPEFIM_SP_adaptive": "GA-HUIM",
}

# Define styles
phases = ["Parse", "TWU", "Build", "Search"]
phase_colors = {
    "Parse": "#2ca02c", "TWU": "#1f77b4", "Build": "#ffdd57", "Search": "#ff7f0e"
}

method_colors = {
    "DPHIM_DPHIM_dphim": "#1f77b4",         # Blue
    "cuDPEFIM_SP_adaptive": "#d62728",      # Red
}
method_markers = {
    "DPHIM_DPHIM_dphim": "^",
    "cuDPEFIM_SP_adaptive": "o",
}

static_methods = [
    "cuDPEFIM_SP_fixed_32", "cuDPEFIM_SP_fixed_64", "cuDPEFIM_SP_fixed_128",
    "cuDPEFIM_SP_fixed_256", "cuDPEFIM_SP_fixed_512", "cuDPEFIM_SP_fixed_1024",
]

# Hex values for green going from light to dark (FF → E5 → CC → B2 → 99 → 7F, etc.)
# green_steps = ["FF", "DF", "AF", "9F", "7F", "5F"]
green_steps = ["5F", "5F", "5F", "5F", "5F", "5F"]


for method, green_hex in zip(static_methods, green_steps):
    hex_color = f"#00{green_hex}00"
    method_colors[method] = hex_color
    method_markers[method] = ""

line_styles = ["-", "--", ":", "-.", (0, (3, 5, 1, 5)), (0, (1, 1)), (0, (5, 10))]
all_methods = list(method_nice_names.keys())
method_linestyles = {m: line_styles[i % len(line_styles)] for i, m in enumerate(all_methods)}

linestyles = ["-", "--", ":", "-."]
hatches = ["/", "\\", "x", "o", ".", "*", "+", "-"]
method_style = {
    m: {"linestyle": linestyles[i % len(linestyles)], "hatch": hatches[i % len(hatches)]}
    for i, m in enumerate(all_methods)
}

###############################################################################
# Plotting                                                                   #
###############################################################################

def plot_for_configs(
    input_file: str,
    minutil: int,
    block_sizes: List[int],
    total_memory: str = "78G",
    method: str = "SP",
    include_legend: bool = False,
) -> None:
    """Plot iteration‑wise times for adaptive + several fixed block sizes."""

    configs = [("adaptive", None)] + [(f"static_{bs}", bs) for bs in block_sizes]

    all_iterations: set[int] = set()
    results: Dict[str, Dict[int, float]] = {}

    # Collect / compute data (cache aware)
    for label, bs in configs:
        adaptive_flag = label == "adaptive"
        method_key = (
            "cuDPEFIM_SP_adaptive" if adaptive_flag else f"cuDPEFIM_SP_fixed_{bs}"
        )
        times = get_or_run_iteration_times(
            input_file,
            minutil,
            adaptive=adaptive_flag,
            block_size=bs,
            total_memory=total_memory,
            method=method,
        )
        results[label] = times
        all_iterations.update(times.keys())

    iterations = sorted(all_iterations)

    # ──────────────────────────────────────────────────────────── plotting ────
    plt.figure(figsize=(4, 4))

    legend_handles: list[Line2D] = []

    adaptive_times = [results["adaptive"].get(i, 0) for i in iterations]
    max_adaptive_time = max(adaptive_times) if adaptive_times else 0
    height_cutoff = 1.5 * max_adaptive_time

    for label, bs in configs:
        adaptive_flag = label == "adaptive"
        method_key = (
            "cuDPEFIM_SP_adaptive" if adaptive_flag else f"cuDPEFIM_SP_fixed_{bs}"
        )
        display_name = method_nice_names.get(method_key, label)

        y = [results[label].get(i, 0) for i in iterations]

        line, = plt.plot(
            iterations,
            y,
            label=display_name,
            linestyle=method_linestyles[method_key],
            marker=method_markers.get(method_key, None),
            color=method_colors.get(method_key, None),
        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle=method_linestyles[method_key],
                marker=method_markers.get(method_key, None),
                color=line.get_color(),
                label=display_name,
            )
        )

    # plt.ylim(0, height_cutoff)
    # plt.xlabel("Iteration", fontsize=14)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)

    # plt.xlim(1, max(iterations))  # Start from 1, not 0
    # plt.xticks([1])  # Force tick at 1
    # plt.set_xlim(0.8, max(iterations))  # start slightly before 1
    # plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Let matplotlib pick the rest


    ax = plt.gca()
    ax.set_xlim(0.8, max(iterations))
    ax.set_ylim(0, height_cutoff)

    # Ensure integer ticks and include tick 1
    ticks = ax.get_xticks()
    if 1 not in ticks:
        ax.set_xticks([1] + list(ticks))
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Labels
    plt.ylim(0, height_cutoff)
    plt.xticks([1] + list(ax.get_xticks()), fontsize=14)
    plt.xlabel("Iteration", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)




    file_base = Path(input_file).stem
    file_name = f"iteration_times_{file_base}_minutil_{minutil}.png"

    if include_legend:
        plt.ylabel("Execution Time (ms)", fontsize=14)
        plt.legend(handles=legend_handles, fontsize=12, loc="best", frameon=True)

    plt.tight_layout()
    plt.savefig(file_name, dpi=500)
    plt.close()


###############################################################################
# Example usage                                                              #
###############################################################################

if __name__ == "__main__":
    block_sizes = [32, 64, 128, 256, 512, 1024]

    process_data = [
        [
            "/export/home1/ltarun/cuEFIM/datasets/accidents_utility_spmf.txt",
            [8000000, 9000000, 10000000, 11000000, 12000000, 15000000, 20000000],
        ],
        [
            "/export/home1/ltarun/cuEFIM/datasets/connect_utility_spmf.txt",
            [10000000, 10500000, 11000000, 11500000, 12000000, 15000000, 16000000],
        ],
        [
            "/export/home1/ltarun/cuEFIM/datasets/pumsb_utility.txt",
            [9500000, 11300000, 11400000],
        ],
        [
            "/export/home1/ltarun/cuEFIM/datasets/retail.txt",
            [1345, 1346, 1347, 1348, 1349],
        ],
    ]

    for input_path, minutils in process_data:
        minutil = minutils[0]
        print(f"Processing {input_path} with minutil {minutil}")
        if input_path.endswith("accidents_utility_spmf.txt"):
            plot_for_configs(input_path, minutil, block_sizes, include_legend=True)
        else:
            plot_for_configs(input_path, minutil, block_sizes)
