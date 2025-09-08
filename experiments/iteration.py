import os
import subprocess
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import itertools
from matplotlib.patches import Patch

###############################################################################
# Persistent cache (CSV) utilities                                            #
###############################################################################

CSV_CACHE = Path("iteration_times_cache_v2.csv")

CACHE_COLUMNS = [
    "dataset",  # absolute path of the dataset file
    "minutil",  # minimum utility threshold
    "method",  # cuDPEFIM_SP_fixed_32 | cuDPEFIM_SP_adaptive | ...
    "adaptive",  # bool
    "block_size",  # int or None
    "iteration",  # iteration index (int)
    "time_ms",  # execution time in ms (float)
    "gpu_mem_mb",  # GPU memory used in MB (int)
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
) -> Dict[int, Tuple[float, int]]:
    """Run cuDPEFIM once and return iteration → (time_ms, gpu_mem_mb) mapping."""

    cmd = [
        "/export/home1/ltarun/cuDPEFIM/build/cuDPEFIM",
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
            # Reconstruct dict and return
            return dict(
                zip(
                    subset["iteration"].astype(int),
                    zip(
                        subset["time_ms"].astype(float),
                        subset["gpu_mem_mb"].astype(int),
                    ),
                )
            )

    # Otherwise execute and store
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
    "Parse": "#2ca02c",
    "TWU": "#1f77b4",
    "Build": "#ffdd57",
    "Search": "#ff7f0e",
}

method_colors = {
    "DPHIM_DPHIM_dphim": "#1f77b4",  # Blue
    "cuDPEFIM_SP_adaptive": "#d62728",  # Red
}
method_markers = {
    "DPHIM_DPHIM_dphim": "^",
    "cuDPEFIM_SP_adaptive": "o",
}

static_methods = [
    "cuDPEFIM_SP_fixed_32",
    "cuDPEFIM_SP_fixed_64",
    "cuDPEFIM_SP_fixed_128",
    "cuDPEFIM_SP_fixed_256",
    "cuDPEFIM_SP_fixed_512",
    "cuDPEFIM_SP_fixed_1024",
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

###############################################################################
# Plotting                                                                   #
###############################################################################
import os
import subprocess
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

###############################################################################
# Persistent cache (CSV) utilities                                            #
###############################################################################

CSV_CACHE = Path("iteration_times_cache_v2.csv")

CACHE_COLUMNS = [
    "dataset",  # absolute path of the dataset file
    "minutil",  # minimum utility threshold
    "method",  # cuDPEFIM_SP_fixed_32 | cuDPEFIM_SP_adaptive | ...
    "adaptive",  # bool
    "block_size",  # int or None
    "iteration",  # iteration index (int)
    "time_ms",  # execution time in ms (float)
    "gpu_mem_mb",  # GPU memory used in MB (int)
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
) -> Dict[int, Tuple[float, int]]:
    """Run cuDPEFIM once and return iteration → (time_ms, gpu_mem_mb) mapping."""

    cmd = [
        "/export/home1/ltarun/cuDPEFIM/build/cuDPEFIM",
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
            # Reconstruct dict and return
            return dict(
                zip(
                    subset["iteration"].astype(int),
                    zip(
                        subset["time_ms"].astype(float),
                        subset["gpu_mem_mb"].astype(int),
                    ),
                )
            )

    # Otherwise execute and store
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
    "Parse": "#2ca02c",
    "TWU": "#1f77b4",
    "Build": "#ffdd57",
    "Search": "#ff7f0e",
}

method_colors = {
    "DPHIM_DPHIM_dphim": "#1f77b4",  # Blue
    "cuDPEFIM_SP_adaptive": "#d62728",  # Red
}
method_markers = {
    "DPHIM_DPHIM_dphim": "^",
    "cuDPEFIM_SP_adaptive": "o",
}

static_methods = [
    "cuDPEFIM_SP_fixed_32",
    "cuDPEFIM_SP_fixed_64",
    "cuDPEFIM_SP_fixed_128",
    "cuDPEFIM_SP_fixed_256",
    "cuDPEFIM_SP_fixed_512",
    "cuDPEFIM_SP_fixed_1024",
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
    """
    Plot iteration-wise times and memory, saving the legend to a separate file.
    """
    configs = [("adaptive", None)] + [(f"static_{bs}", bs) for bs in block_sizes]

    all_iterations: set[int] = set()
    results: Dict[str, Dict[int, Tuple[float, int]]] = {}

    # Collect / compute data (cache aware)
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

    # ──────────────────────────────────────────────────────────── plotting ────
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # --- Plot Execution Times on ax1 ---
    for label, bs in configs:
        adaptive_flag = label == "adaptive"
        method_key = (
            "cuDPEFIM_SP_adaptive" if adaptive_flag else f"cuDPEFIM_SP_fixed_{bs}"
        )
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

    # ──────────────────────────────────────────────────────────────────────────
    # --- START OF CHANGE: Plot Memory as (Previous Iter + Current Iter) ---
    # ──────────────────────────────────────────────────────────────────────────

    # 1. Extract the raw memory usage for each iteration
    raw_adaptive_mems = [results["adaptive"].get(i, (0, 0))[1] for i in iterations]

    # 2. Calculate the memory to be plotted according to the "prev + current" rule
    plotted_mems = []
    if raw_adaptive_mems:
        # The first iteration has no "previous", so it's just its own memory
        plotted_mems.append(raw_adaptive_mems[0])
        # For all subsequent iterations, sum the current and previous memory values
        for i in range(1, len(raw_adaptive_mems)):
            plotted_mems.append(raw_adaptive_mems[i - 1] + raw_adaptive_mems[i])

    # 3. Plot the calculated memory footprint
    if plotted_mems:
        ax2.fill_between(
            iterations,
            plotted_mems,
            alpha=0.3,
            color="gray",
            label="Active GPU Memory (GA-HUIM)",  # More accurate label
        )

    # ──────────────────────────────────────────────────────────────────────────
    # --- END OF CHANGE ---
    # ──────────────────────────────────────────────────────────────────────────

    # --- Collect Legend Handles ---
    handles1, labels1 = ax1.get_legend_handles_labels()
    # Update label to reflect the new memory calculation
    mem_patch = Patch(color="gray", alpha=0.3, label="Active GPU Memory (GA-HUIM)")
    all_handles = handles1 + [mem_patch]
    all_labels = labels1 + ["Active GPU Memory"]

    # --- Formatting ---
    adaptive_times = [results["adaptive"].get(i, (0, 0))[0] for i in iterations]
    max_adaptive_time = max(adaptive_times) if adaptive_times else 1
    height_cutoff = 1.5 * max_adaptive_time
    ax1.set_ylim(0, height_cutoff)
    ax2.set_ylim(bottom=0)

    ax1.set_xlabel("Iteration", fontsize=14)
    ax1.set_ylabel("Execution Time (ms)", fontsize=14)
    ax2.set_ylabel("Active GPU Memory (MB)", fontsize=14, rotation=270, labelpad=20)

    ax1.tick_params(axis="y", labelsize=14)
    ax2.tick_params(axis="y", labelsize=14)
    ax1.tick_params(axis="x", labelsize=14)

    if iterations:
        ax1.set_xlim(min(iterations) - 0.2, max(iterations) + 0.2)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # --- Save Main Plot ---
    file_base = Path(input_file).stem
    file_name = f"{file_base}_minutil_{minutil}_iteration.png"

    # title
    title = file_base.split("_")[0]
    plt.title(f"Iteration-wise Execution Time and Memory on {title}", fontsize=16)

    # plt.tight_layout()
    plt.savefig(file_name, dpi=500)
    plt.close(fig)

    # --- Generate and Save Separate Legend Image ---
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
        plt.savefig(legend_filename, dpi=500, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig_legend)


if __name__ == "__main__":
    block_sizes = [32, 64, 128, 256, 512, 1024]

    process_data = [
        [
            "/export/home1/ltarun/cuDPEFIM/datasets/accidents_utility_spmf.txt",
            [3330000, 3340000, 3350000, 3360000, 3370000, 3380000, 3390000],
        ],
        [
            "/export/home1/ltarun/cuDPEFIM/datasets/connect_utility_spmf.txt",
            [7800000, 7900000, 8000000, 8100000, 8200000, 8300000, 8400000],
        ],
        [
            "/export/home1/ltarun/cuDPEFIM/datasets/pumsb_utility_spmf.txt",
            # 9150000, 9250000, 9350000,
            [9450000],
        ],
        [
            "/export/home1/ltarun/cuDPEFIM/datasets/BMS_utility_spmf.txt",
            # , 2002000,2004000,2006000
            [2000000],
        ],
        [
            "/export/home1/ltarun/cuDPEFIM/datasets/chess_utility_spmf.txt",
            [190000, 200000, 210000, 220000, 230000, 240000, 250000],
        ],
        [
            "/export/home1/ltarun/cuDPEFIM/datasets/kosarak_utility_spmf.txt",
            [770000, 780000, 790000, 800000, 810000, 820000, 830000],
        ],
        [
            "/export/home1/ltarun/cuDPEFIM/datasets/retail_utility_spmf.txt",
            [1430, 1435, 1440, 1445, 1450, 1455, 1460],
        ],
        [
            "/export/home1/ltarun/cuDPEFIM/datasets/mushroom_utility_spmf.txt",
            [9000, 10000, 11000, 12000, 13000, 14000, 15000],
        ],
        [
            "/export/home1/ltarun/cuDPEFIM/experiments/synthetic_hu/huT10M_ni100_avg5.txt",
            [5, 10, 15, 20, 25, 30, 35],
        ],
        [
            "/export/home1/ltarun/cuDPEFIM/experiments/synthetic_hu/huT10M_ni100_avg10.txt",
            [90000, 100000, 110000, 120000, 130000, 140000, 150000],
        ],
        [
            "/export/home1/ltarun/cuDPEFIM/experiments/synthetic_hu/huT10M_ni50_avg10.txt",
            [80000, 90000, 100000, 110000, 120000, 130000, 140000],
        ],
        [
            "/export/home1/ltarun/cuDPEFIM/experiments/synthetic_hu/huT10M_ni25_avg10.txt",
            [50000, 60000, 70000, 80000, 90000, 100000, 110000],
        ],
    ]

    for input_path, minutils in process_data:
        minutil = minutils[0]
        print(f"Processing {input_path} with minutil {minutil}")
        # if input_path.endswith("accidents_utility_spmf.txt"):
        plot_for_configs(input_path, minutil, block_sizes, include_legend=True)
        # else:
        # plot_for_configs(input_path, minutil, block_sizes)
