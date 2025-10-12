# plot_results.py

import argparse
import logging
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# --- Configuration ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Define a clear and consistent style scheme for plots
STYLE_CONFIG = {
    "method_labels": {
        "DPHIM_DPHIM_dphim": "DPHIM",
    "G-HUIM_SP_adaptive": "GA-HUIM",
        "PHA_PHA": "PHA",
        # Add fixed block size labels if you run those experiments
    "G-HUIM_SP_fixed_32": "GS-HUIM(32)", "G-HUIM_SP_fixed_64": "GS-HUIM(64)",
    "G-HUIM_SP_fixed_128": "GS-HUIM(128)", "G-HUIM_SP_fixed_256": "GS-HUIM(256)",
    "G-HUIM_SP_fixed_512": "GS-HUIM(512)", "G-HUIM_SP_fixed_1024": "GS-HUIM(1024)",
    },
    "phase_colors": {
        "Parse": "#2ca0c2", "TWU": "#1f77b4", "Build": "#ffdd57", "Search": "#ff7f0e"
    },
    "method_colors": {
        "DPHIM_DPHIM_dphim": "#1f77b4",  # Blue
    "G-HUIM_SP_adaptive": "#d62728",  # Red
        "PHA_PHA": "#2ca02c",  # Green
    },
    "method_markers": {
        "DPHIM_DPHIM_dphim": "*",
    "G-HUIM_SP_adaptive": "o",
        "PHA_PHA": "s",
    },
    "method_linestyles": {
        "DPHIM_DPHIM_dphim": "-",
    "G-HUIM_SP_adaptive": "--",
        "PHA_PHA": ":",
    }
}

# --- Plotting Functions ---

def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the DataFrame for plotting by creating combined columns and calculating totals."""
    df_processed = df.rename(columns={
        "Parse_Time": "Parse", "TWU_Time": "TWU",
        "Build_Time": "Build", "Search_Time": "Search",
    })
    df_processed["Minimum_Utility"] = df_processed["Minimum_Utility"].astype(int)
    df_processed["AlgoMeth"] = df_processed["Algorithm"] + "_" + df_processed["Method"]
    
    phases = ["Parse", "TWU", "Build", "Search"]
    df_processed["Total_Time"] = df_processed[phases].sum(axis=1)
    return df_processed

def _generate_legends(output_dir: str, all_methods: List[str]):
    """Generate and save separate legend images for phases and methods."""
    # Phase Legend
    fig_phase, ax_phase = plt.subplots(figsize=(4, 0.5))
    phases = STYLE_CONFIG["phase_colors"].keys()
    handles = [Patch(facecolor=STYLE_CONFIG["phase_colors"][p], label=p, edgecolor='black') for p in phases]
    ax_phase.legend(handles=handles, loc="center", ncol=len(phases), frameon=False, title="Execution Phases")
    ax_phase.axis("off")
    fig_phase.savefig(os.path.join(output_dir, "legend_phases.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_phase)

    # Method Legend for Line Plots
    fig_line, ax_line = plt.subplots(figsize=(5, 1.5))
    handles = []
    for m in all_methods:
        if m in STYLE_CONFIG["method_labels"]:
            handles.append(plt.Line2D(
                [0], [0],
                color=STYLE_CONFIG["method_colors"].get(m, "gray"),
                linestyle=STYLE_CONFIG["method_linestyles"].get(m, "-"),
                marker=STYLE_CONFIG["method_markers"].get(m, ""),
                label=STYLE_CONFIG["method_labels"][m]
            ))
    ax_line.legend(handles=handles, loc="center", ncol=2, frameon=False, title="Algorithms")
    ax_line.axis("off")
    fig_line.savefig(os.path.join(output_dir, "legend_lines.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_line)
    
    logging.info(f"Legends saved to {output_dir}")

def _plot_breakdown_chart(df_file: pd.DataFrame, all_methods: List[str], output_path: str):
    """Generate and save a stacked bar chart of execution time breakdown."""
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=300)
    
    min_util = df_file["Minimum_Utility"].min()
    df_subset = df_file[df_file["Minimum_Utility"] == min_util]
    
    positions = np.arange(len(all_methods))
    bar_width = 0.6
    
    for i, method in enumerate(all_methods):
        row = df_subset[df_subset["AlgoMeth"] == method]
        if row.empty: continue
        
        bottom = 0.0
        for phase, color in STYLE_CONFIG["phase_colors"].items():
            height = row[phase].iloc[0] / 1000.0
            ax.bar(positions[i], height, bar_width, bottom=bottom, color=color, edgecolor="black", linewidth=0.7)
            bottom += height
            
    ax.set_ylabel("Execution Time (s)", fontsize=12)
    ax.set_xticks(positions)
    ax.set_xticklabels([STYLE_CONFIG['method_labels'].get(m, m) for m in all_methods], rotation=45, ha="right")
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    file_base = df_file["File"].iloc[0]
    plt.title(f"Execution Time Breakdown on {file_base.capitalize()}")
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logging.info(f"Saved breakdown plot to {output_path}")

def _plot_total_time_chart(df_file: pd.DataFrame, all_methods: List[str], output_path: str):
    """Generate and save a line chart of total time vs. minimum utility."""
    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=300)
    
    utils = sorted(df_file["Minimum_Utility"].unique())
    
    for method in all_methods:
        df_m = df_file[df_file["AlgoMeth"] == method]
        if df_m.empty: continue
        
        times = df_m.set_index("Minimum_Utility")["Total_Time"].reindex(utils) / 1000.0
        ax.plot(
            utils, times,
            label=STYLE_CONFIG["method_labels"].get(method, method),
            linestyle=STYLE_CONFIG["method_linestyles"].get(method, "-"),
            marker=STYLE_CONFIG["method_markers"].get(method, ""),
            color=STYLE_CONFIG["method_colors"].get(method, "gray")
        )
        
    ax.set_xlabel("Minimum Utility", fontsize=12)
    ax.set_ylabel("Execution Time (s)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(bottom=0)
    ax.set_xticks(utils)
    ax.tick_params(axis='x', rotation=45)
    
    file_base = df_file["File"].iloc[0]
    plt.title(f"Performance on {file_base.capitalize()}")
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logging.info(f"Saved total time plot to {output_path}")

def _plot_patterns_chart(df_file: pd.DataFrame, all_methods: List[str], output_path: str):
    """Generate and save a line chart of frequent patterns vs. minimum utility."""
    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=300)

    utils = sorted(df_file["Minimum_Utility"].unique())

    for method in all_methods:
        df_m = df_file[df_file["AlgoMeth"] == method]
        if df_m.empty: continue

        patterns = df_m.set_index("Minimum_Utility")["Frequent_Patterns"].reindex(utils)
        ax.plot(
            utils, patterns,
            label=STYLE_CONFIG["method_labels"].get(method, method),
            linestyle=STYLE_CONFIG["method_linestyles"].get(method, "-"),
            marker=STYLE_CONFIG["method_markers"].get(method, ""),
            color=STYLE_CONFIG["method_colors"].get(method, "gray")
        )

    ax.set_xlabel("Minimum Utility", fontsize=12)
    ax.set_ylabel("Number of Patterns Found (Log Scale)", fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle='--', alpha=0.7)
    ax.set_xticks(utils)
    ax.tick_params(axis='x', rotation=45)

    file_base = df_file["File"].iloc[0]
    plt.title(f"Patterns Found on {file_base.capitalize()}")
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logging.info(f"Saved patterns plot to {output_path}")

def plot_all_results(csv_path: str, output_dir: str):
    """Main function to generate all plots from the results CSV."""
    if not os.path.exists(csv_path):
        logging.error(f"Results file not found: {csv_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    df = _preprocess_data(pd.read_csv(csv_path))
    
    all_methods = sorted([m for m in df["AlgoMeth"].unique() if m in STYLE_CONFIG["method_labels"]])
    
    for file_name in sorted(df["File"].unique()):
        df_file = df[df["File"] == file_name].copy()
        
        _plot_breakdown_chart(
            df_file, all_methods, 
            os.path.join(output_dir, f"{file_name}_breakdown.png")
        )
        
        _plot_total_time_chart(
            df_file, all_methods, 
            os.path.join(output_dir, f"{file_name}_total_time.png")
        )

        _plot_patterns_chart(
            df_file, all_methods,
            os.path.join(output_dir, f"{file_name}_patterns.png")
        )

    _generate_legends(output_dir, all_methods)

def main():
    parser = argparse.ArgumentParser(description="Plot High-Utility Itemset Mining benchmark results.")
    parser.add_argument(
        "input_csv", type=str, help="Path to the input results CSV file."
    )
    parser.add_argument(
        "--output-dir", type=str, default="plots", help="Directory to save the generated plots."
    )
    args = parser.parse_args()
    
    plot_all_results(args.input_csv, args.output_dir)
    logging.info("Plotting complete.")

if __name__ == '__main__':
    main()