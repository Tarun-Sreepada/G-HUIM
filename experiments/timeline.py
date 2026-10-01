import os
import subprocess
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Union
import io

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

RES_DIR = Path("results")
RES_DIR.mkdir(exist_ok=True)


CSV_CACHE_DETAILED = RES_DIR / "iterations_detailed.csv"
CACHE_COLUMNS_DETAILED = [
    "dataset",
    "minutil",
    "iteration",
    "time_ms",
    "mem_before_mb",
    "mem_after_mb",
    "distribution",
]


def _load_detailed_cache() -> pd.DataFrame:
    """Load existing detailed cache or create an empty DataFrame."""
    if CSV_CACHE_DETAILED.exists():
        try:
            return pd.read_csv(CSV_CACHE_DETAILED)
        except Exception as e:
            print(
                f"Warning: Could not read cache file. It might be outdated. Error: {e}"
            )
            return pd.DataFrame(columns=CACHE_COLUMNS_DETAILED)
    return pd.DataFrame(columns=CACHE_COLUMNS_DETAILED)


def _append_to_detailed_cache(rows: list) -> None:
    """Append new rows to the detailed CSV cache."""
    df_new = pd.DataFrame(rows, columns=CACHE_COLUMNS_DETAILED)

    header = not (
        CSV_CACHE_DETAILED.exists() and os.path.getsize(CSV_CACHE_DETAILED) > 0
    )
    df_new.to_csv(CSV_CACHE_DETAILED, mode="a", header=header, index=False)


def run_and_parse_detailed(
    input_file: str,
    minutil: int,
    total_memory: str = "60G",
    method: str = "adaptive",
    adaptive: bool = True,
    block_size: int | None = None,
) -> pd.DataFrame:
    """
    Run G-HUIM once, parse the detailed output, and return a DataFrame.
    """
    cmd = [
        "../build/G-HUIM",
        "--input",
        input_file,
        "--minutil",
        str(minutil),
        "--total_memory",
        total_memory,
        "--output",
        "/dev/null",
    ]

    print(f"Executing: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        stdout = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command for {input_file} with minutil {minutil}")
        print(e.stdout)
        return pd.DataFrame(columns=CACHE_COLUMNS_DETAILED)

    parsed_rows = []
    for line in stdout.splitlines():
        if line.strip().startswith("Iteration"):
            line_stripped = line.strip()
            list_start_index = line_stripped.find("[")

            if list_start_index != -1:
                before_list_part = line_stripped[:list_start_index]
                parts = before_list_part.split()

                list_part_str = line_stripped[list_start_index:]
                dist_str_cleaned = list_part_str.strip("[]")
                distribution = []
                if dist_str_cleaned:
                    distribution = [int(x) for x in dist_str_cleaned.split(",")]

                it = int(parts[1])
                t_ms = float(parts[2])
                mem_before_mb = int(parts[3])
                mem_after_mb = int(parts[4])

                parsed_rows.append(
                    [
                        input_file,
                        minutil,
                        it,
                        t_ms,
                        mem_before_mb,
                        mem_after_mb,
                        str(distribution),
                    ]
                )

    return pd.DataFrame(parsed_rows, columns=CACHE_COLUMNS_DETAILED)


def get_or_run_detailed_data(input_file: str, minutil: int) -> pd.DataFrame:
    """Return detailed iteration data, reusing cached results when available."""
    cache = _load_detailed_cache()

    if not cache.empty:
        mask = (cache["dataset"] == input_file) & (cache["minutil"] == minutil)
        subset = cache[mask]
        if not subset.empty:
            print(f"--- Loading from cache for {Path(input_file).name} ---")
            subset = subset.copy()
            subset["distribution"] = subset["distribution"].apply(eval)
            return subset

    df_new = run_and_parse_detailed(input_file, minutil)
    if not df_new.empty:
        _append_to_detailed_cache(df_new.values.tolist())

    if not df_new.empty:
        df_new["distribution"] = df_new["distribution"].apply(eval)
    return df_new


def plot_log_timeline(df: pd.DataFrame, output_filename: str) -> None:
    """
    Plots the timeline using only the measured data points from the program.
    """
    if df.empty:
        print(f"No data to plot for {output_filename}.")
        return

    df = df.copy().sort_values(by="iteration").reset_index(drop=True)
    df["cumulative_time_ms"] = df["time_ms"].cumsum()
    df["start_time_ms"] = df["cumulative_time_ms"] - df["time_ms"]

    df["dataset"] = df["dataset"].apply(lambda x: Path(x).name)

    fig, ax1 = plt.subplots(figsize=(18, 9))
    ax2 = ax1.twinx()

    memory = []
    for _, row in df.iterrows():
        memory.append(
            (row["mem_after_mb"] - row["mem_before_mb"], row["cumulative_time_ms"])
        )

    usage_by_iteration = []
    for i, (mem, time) in enumerate(memory):
        if i == 0:
            usage_by_iteration.append((mem, time))
        else:
            usage_by_iteration.append((mem + memory[i - 1][0], time))

    ax1.plot(
        df["cumulative_time_ms"],
        [mem for mem, _ in usage_by_iteration],
        color="black",
        lw=2.5,
        marker="o",
        markersize=4,
        label="GPU Memory",
    )
    ax1.fill_between(
        df["cumulative_time_ms"],
        [mem for mem, _ in usage_by_iteration],
        color="gray",
        alpha=0.4,
    )

    num_dist = (
        len(df["distribution"].iloc[0])
        if not df.empty and df["distribution"].iloc[0]
        else 0
    )
    colors = plt.cm.YlOrRd(np.linspace(0.3, 1, num_dist))

    for _, row in df.iterrows():
        bottom = 0
        for i, task_count in enumerate(row["distribution"]):
            if task_count > 0:
                ax2.bar(
                    x=row["start_time_ms"],
                    height=task_count,
                    width=row["time_ms"] if row["time_ms"] > 0 else 0.1,
                    bottom=bottom if bottom > 0 else 1,
                    color=colors[i],
                    align="edge",
                    linewidth=0,
                    alpha=0.5,
                )
            bottom += task_count

    ax2.set_yscale("log")

    ax1.set_xlabel("Cumulative Time (ms)", fontsize=16)
    ax1.set_ylabel("GPU Memory (MB)", fontsize=16)
    ax2.set_ylabel("Task Distribution Count (Log Scale)", fontsize=16)

    ax1.tick_params(axis="y", labelsize=14)
    ax2.tick_params(axis="y", labelsize=14)
    ax1.tick_params(axis="x", labelsize=14, rotation=45)

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=1)
    ax1.set_xlim(left=0, right=df["cumulative_time_ms"].max() * 1.02)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    title = (
        Path(output_filename).stem.replace("utility_spmf", "").replace("_", " ").title()
    )

    plt.title(f"Iteration Timeline: {title}", fontsize=20, pad=20)

    thread_labels = ["1", "32", "64", "128", "256", "512", "1024", "1024+"]
    legend_elements = [Patch(facecolor="gray", alpha=0.4, label="GPU Memory")]
    if num_dist <= len(thread_labels):
        legend_elements += [
            Patch(facecolor=colors[i], label=f"{thread_labels[i]} threads")
            for i in range(num_dist)
        ]

    fig.legend(
        handles=legend_elements,
        title="Threads per Task (Yellow=Few)",
        loc="lower center",
        ncol=5,
        fontsize=12,
        title_fontsize=14,
    )

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(output_filename, dpi=300)
    plt.close(fig)
    print(f"Plot saved to {output_filename}")


if __name__ == "__main__":
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
        print(f"\nProcessing {Path(input_path).name} with minutil {minutil}")

        iteration_df = get_or_run_detailed_data(input_path, minutil)

        if not iteration_df.empty:
            file_base = Path(input_path).stem
            output_file = OUT_DIR / f"{file_base}_minutil_{minutil}_log_timeline.png"
            plot_log_timeline(iteration_df, output_file)
        else:
            print(
                f"Skipping plot for {input_path} as no data was generated or cache was invalid."
            )
