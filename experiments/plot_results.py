# plot_results.py

import argparse
import logging
import os
from typing import List, Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --- Configuration ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# New method names and styles provided by the user
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

# Define styles
phases = ["Parse", "TWU", "Build", "Search"]
PHASE_COLORS = {
    "Parse": "#2ca02c",  # Green
    "TWU": "#1f77b4",    # Blue
    "Build": "#ffdd57",  # Yellow
    "Search": "#ff7f0e", # Orange
}

# Hatch patterns to apply to stacked bars (repeats if more phases than hatches)
HATCHES = ["/", "\\", "x", "o", ".", "*", "+", "-"]

METHOD_COLORS = {
    "DPHIM_DPHIM_dphim": "#1f77b4",  # Blue
    "G-HUIM_SP_adaptive": "#d62728",  # Red
}
METHOD_MARKERS = {
    "DPHIM_DPHIM_dphim": "^",
    "G-HUIM_SP_adaptive": "o",
}

STATIC_METHODS = [
    "G-HUIM_SP_fixed_32",
    "G-HUIM_SP_fixed_64",
    "G-HUIM_SP_fixed_128",
    "G-HUIM_SP_fixed_256",
    "G-HUIM_SP_fixed_512",
    "G-HUIM_SP_fixed_1024",
]

# Calculate colors for GS-HUIM methods (shades of green)
GS_HUIM_COLOR = "#008000" # A distinct dark green for all fixed methods

for method in STATIC_METHODS:
    METHOD_COLORS[method] = GS_HUIM_COLOR
    METHOD_MARKERS[method] = "" # Using 's' (square) for all fixed methods

LINE_STYLES = ["-", "--", ":", "-.", (0, (3, 5, 1, 5)), (0, (1, 1)), (0, (5, 10))]
ALL_METHOD_KEYS = list(method_nice_names.keys())
METHOD_LINESTYLES = {
    m: LINE_STYLES[i % len(LINE_STYLES)] for i, m in enumerate(ALL_METHOD_KEYS)
}

# --- Plotting Functions ---

def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the DataFrame for plotting by creating combined columns and calculating totals."""
    # Filter out PHA as requested
    df = df[df["Algorithm"] != "PHA"].copy()
    
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
    fig_phase, ax_phase = plt.subplots(figsize=(4.5, 0.5))
    phases = PHASE_COLORS.keys()
    handles = []
    for i, p in enumerate(phases):
        hatch = HATCHES[i % len(HATCHES)]
        handles.append(Patch(facecolor=PHASE_COLORS[p], label=p, edgecolor='black', hatch=hatch))
    ax_phase.legend(handles=handles, loc="center", ncol=len(phases), frameon=False, title="Execution Phases")
    ax_phase.axis("off")
    fig_phase.savefig(os.path.join(output_dir, "legend_phases.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_phase)

    # Method Legend for Line Plots
    fig_line, ax_line = plt.subplots(figsize=(7, 1.5))
    handles = []
    
    # Sort methods by name for a clear legend
    methods_for_legend = sorted([m for m in all_methods if m in method_nice_names], 
                                key=lambda x: method_nice_names[x])
    
    for m in methods_for_legend:
        handles.append(Line2D(
            [0], [0],
            color=METHOD_COLORS.get(m, "gray"),
            linestyle=METHOD_LINESTYLES.get(m, "-"),
            marker=METHOD_MARKERS.get(m, ""),
            label=method_nice_names[m]
        ))
    
    ax_line.legend(handles=handles, loc="center", ncol=3, frameon=False, title="Algorithms")
    ax_line.axis("off")
    fig_line.savefig(os.path.join(output_dir, "legend_lines.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_line)
    
    logging.info(f"Legends saved to {output_dir}")

def _plot_breakdown_chart(df_file: pd.DataFrame, output_path: str):
    """Generate and save a stacked bar chart of execution time breakdown with custom order/labels."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300) # Made more square
    
    min_util = df_file["Minimum_Utility"].min()
    df_subset = df_file[df_file["Minimum_Utility"] == min_util]
    
    # 1. Define custom plotting order and groups
    ga_huim = "G-HUIM_SP_adaptive"
    dphim = "DPHIM_DPHIM_dphim"
    gs_huim_methods = STATIC_METHODS # Already defined
    
    plot_methods = [ga_huim] + gs_huim_methods + [dphim]
    
    # 2. Calculate positions with improved spacing and centering
    bar_width = 0.8

    # Start positions (we'll center all bars by adding half bar_width)
    ga_pos = 0.0

    # Leave a slightly larger gap after GA-HUIM before the GS-HUIM group
    gs_pos_start = ga_pos + 1.8 * bar_width
    # gs_positions = np.arange(gs_pos_start, gs_pos_start + len(gs_huim_methods) * bar_width, bar_width)

    # --- AFTER ---
    num_gs_methods = len(gs_huim_methods)
    # Calculate the center of the *last* bar in the GS-HUIM group
    gs_pos_end = gs_pos_start + (num_gs_methods - 1) * bar_width
    # Generate exactly num_gs_methods points from start to end
    gs_positions = np.linspace(gs_pos_start, gs_pos_end, num=num_gs_methods)

    # Increase gap between the last GS-HUIM (1024) and DPHIM so they're not too close
    if gs_positions.size > 0:
        dphim_pos = gs_positions[-1] + 2.0 * bar_width
    else:
        dphim_pos = gs_pos_start + 2.0 * bar_width

    # Center the bar positions by adding half the bar width
    ga_center = ga_pos + (bar_width / 2.0)
    gs_positions_centered = gs_positions + (bar_width / 2.0)
    dphim_center = dphim_pos + (bar_width / 2.0)

    positions = np.array([ga_center] + list(gs_positions_centered) + [dphim_center])
    
    # 3. Plot the bars
    for method, pos in zip(plot_methods, positions):
        row = df_subset[df_subset["AlgoMeth"] == method]
        if row.empty: continue
        
        bottom = 0.0
        for i, (phase, color) in enumerate(PHASE_COLORS.items()):
            height = row[phase].iloc[0] / 1000.0
            hatch = HATCHES[i % len(HATCHES)]
            ax.bar(pos, height, bar_width, bottom=bottom, color=color, edgecolor="black", linewidth=0.7, hatch=hatch)
            bottom += height
            
    # 4. Set labels and ticks
    ax.set_ylabel("Execution Time (s)", fontsize=12)
    ax.set_xticks(positions)
    
    # Create the custom tick labels directly from plot_methods so the count matches positions
    xtick_labels = []
    for method in plot_methods:
        if method == ga_huim:
            xtick_labels.append(method_nice_names.get(method, method).split('(')[0])
        elif method in gs_huim_methods:
            label = method_nice_names.get(method, method)
            xtick_labels.append(label.split('(')[-1].replace(')', ''))
            # xtick_labels.append("")
        else:
            # DPHIM or any other method: use nice name if available
            xtick_labels.append(method_nice_names.get(method, method))

    ax.set_xticklabels(xtick_labels, rotation=0)
    
    # 5. Add "GS-HUIM" bar on top
    # Calculate the span for the GS-HUIM group
    # positions layout: [GA, GS_1, GS_2, ..., GS_n, DPHIM]
    # first GS index is 1, last GS index is 1 + (n-1) = n
    first_gs_idx = 1
    last_gs_idx = len(gs_huim_methods)
    gs_xmin_for_line = positions[first_gs_idx] - bar_width/2
    gs_xmax_for_line = positions[last_gs_idx] + bar_width/2
    gs_center = (gs_xmin_for_line + gs_xmax_for_line) / 2
    
    y_max = ax.get_ylim()[1]
    
    ax.text(gs_center, y_max * 1.05, "GS-HUIM", ha='center', va='bottom', fontsize=12, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle="square,pad=0.3"))

    ax.hlines(y=y_max*1.03, xmin=gs_xmin_for_line, xmax=gs_xmax_for_line, 
              color='black', linewidth=1)
            
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    file_base = df_file["File"].iloc[0]
    # plt.title(f"Execution Time Breakdown on {file_base.capitalize()}")
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logging.info(f"Saved breakdown plot to {output_path}")

def _plot_line_chart(df_file: pd.DataFrame, all_methods: List[str], output_path: str, y_col: str, y_label: str, title_suffix: str, log_scale: bool = False):
    """Generic function to generate line charts (total time, patterns found)."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300) # Made more square
    
    utils = sorted(df_file["Minimum_Utility"].unique())
    
    # Filter methods to only include those in the current result set and config
    plottable_methods = [m for m in all_methods if m in method_nice_names]

    # Plot the lines, ensuring all methods that *should* be plotted are, and using the correct styles
    for method in plottable_methods:
        df_m = df_file[df_file["AlgoMeth"] == method]
        if df_m.empty: continue
        
        y_data = df_m.set_index("Minimum_Utility")[y_col].reindex(utils)
        
        # Convert time to seconds for time plots
        if y_col == "Total_Time":
            y_data = y_data / 1000.0
            
        ax.plot(
            utils, y_data,
            label=method_nice_names.get(method, method),
            linestyle=METHOD_LINESTYLES.get(method, "-"),
            marker=METHOD_MARKERS.get(method, ""),
            color=METHOD_COLORS.get(method, "gray")
        )
        
    ax.set_xlabel("Minimum Utility", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    
    if log_scale:
        ax.set_yscale('log')
        ax.grid(True, which="both", linestyle='--', alpha=0.7)
    else:
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=0)
        
    ax.set_xticks(utils)
    ax.tick_params(axis='x', rotation=45)
    
    file_base = df_file["File"].iloc[0]
    plt.title(f"{title_suffix} on {file_base.capitalize()}")
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logging.info(f"Saved {y_col} plot to {output_path}")

# Wrapper functions for the generic line chart plotter
def _plot_total_time_chart(df_file: pd.DataFrame, all_methods: List[str], output_path: str):
    """Generate and save a line chart of total time vs. minimum utility."""
    _plot_line_chart(
        df_file, all_methods, output_path,
        y_col="Total_Time", y_label="Execution Time (s)", 
        title_suffix="Performance", log_scale=False
    )

def _plot_patterns_chart(df_file: pd.DataFrame, all_methods: List[str], output_path: str):
    """Generate and save a line chart of frequent patterns vs. minimum utility."""
    _plot_line_chart(
        df_file, all_methods, output_path,
        y_col="Frequent_Patterns", y_label="Number of Patterns Found (Log Scale)", 
        title_suffix="Patterns Found", log_scale=True
    )

def plot_all_results(csv_path: str, output_dir: str):
    """Main function to generate all plots from the results CSV."""
    if not os.path.exists(csv_path):
        logging.error(f"Results file not found: {csv_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    df = _preprocess_data(pd.read_csv(csv_path))
    
    # Only include methods in the new config and in the data
    all_methods = sorted([m for m in df["AlgoMeth"].unique() if m in method_nice_names])
    
    if not all_methods:
        logging.warning("No plottable methods found after filtering.")
        return

    for file_name in sorted(df["File"].unique()):
        df_file = df[df["File"] == file_name].copy()
        
        _plot_breakdown_chart(
            df_file,
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