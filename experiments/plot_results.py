import argparse
import logging
import os
from typing import List, Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

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
PHASE_COLORS = {
    "Parse": "#2ca02c",
    "TWU": "#1f77b4",
    "Build": "#ffdd57",
    "Search": "#ff7f0e",
}

HATCHES = ["/", "\\", "x", "o", ".", "*", "+", "-"]

METHOD_COLORS = {
    "DPHIM_DPHIM_dphim": "#1f77b4",
    "G-HUIM_SP_adaptive": "#d62728",
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

GS_HUIM_COLOR = "#008000"

for method in STATIC_METHODS:
    METHOD_COLORS[method] = GS_HUIM_COLOR
    METHOD_MARKERS[method] = ""

LINE_STYLES = ["-", "--", ":", "-.", (0, (3, 5, 1, 5)), (0, (1, 1)), (0, (5, 10))]
ALL_METHOD_KEYS = list(method_nice_names.keys())
METHOD_LINESTYLES = {
    m: LINE_STYLES[i % len(LINE_STYLES)] for i, m in enumerate(ALL_METHOD_KEYS)
}

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
        logging.warning("LaTeX test run failed: %s; disabling matplotlib usetex.", e)
        return False


# If LaTeX isn't fully available at runtime, disable text.usetex so matplotlib
# won't call out to latex for text measurement (which causes runtime failures
# such as missing type1ec.sty). We still keep saving PGF/PDF files, but matplotlib
# will use its internal font metrics rather than compiling TeX for labels.
if matplotlib.rcParams.get("text.usetex", False):
    if not _check_latex_working():
        matplotlib.rcParams["text.usetex"] = False


def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the DataFrame for plotting by creating combined columns and calculating totals."""
    df = df[df["Algorithm"] != "PHA"].copy()

    df_processed = df.rename(
        columns={
            "Parse_Time": "Parse",
            "TWU_Time": "TWU",
            "Build_Time": "Build",
            "Search_Time": "Search",
        }
    )
    df_processed["Minimum_Utility"] = df_processed["Minimum_Utility"].astype(int)
    df_processed["AlgoMeth"] = df_processed["Algorithm"] + "_" + df_processed["Method"]

    phases = ["Parse", "TWU", "Build", "Search"]
    df_processed["Total_Time"] = df_processed[phases].sum(axis=1)
    return df_processed


def _generate_legends(output_dir: str, all_methods: List[str]): 
    """Generate and save separate legend images for phases and methods.""" 
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir) 

    phases = list(PHASE_COLORS.keys()) 

    # --- Phase legend --- 
    phase_fig_width = max(12, len(phases) * 1.5) 
    fig_phase, ax_phase = plt.subplots(figsize=(phase_fig_width, 3)) 
    phase_handles = [ 
        Patch( 
            facecolor=PHASE_COLORS[p], 
            label=p, 
            edgecolor="black", 
            linewidth=1.5, 
        ) 
        for p in phases 
    ] 
    ax_phase.legend( 
        handles=phase_handles, 
        loc="center", 
        ncol=4, 
        frameon=False, 
        fontsize=20, 
        title_fontsize=22, 
        handlelength=4, 
        handleheight=2.5, 
        columnspacing=2.0, 
        borderpad=1.2, 
        labelspacing=1.5, 
    ) 
    ax_phase.axis("off") 
    fig_phase.savefig( 
        os.path.join(output_dir, "legend_phases.png"), dpi=500, bbox_inches="tight" 
    ) 
    plt.close(fig_phase) 

    # --- Algorithm line legend --- 
    available_methods = [m for m in all_methods if m in method_nice_names] 
    ordered_methods: List[str] = [] 

    def _append_if_present(method_name: str) -> None: 
        if method_name in available_methods and method_name not in ordered_methods: 
            ordered_methods.append(method_name) 

    _append_if_present("G-HUIM_SP_adaptive") 
    for static_method in STATIC_METHODS: 
        _append_if_present(static_method) 
    _append_if_present("DPHIM_DPHIM_dphim") 

    for method in available_methods: 
        _append_if_present(method) 

    methods_for_legend = ordered_methods if ordered_methods else available_methods 
    line_handles = [ 
        Line2D( 
            [0], 
            [0], 
            color=METHOD_COLORS.get(m, "gray"), 
            linestyle=METHOD_LINESTYLES.get(m, "-"), 
            marker=METHOD_MARKERS.get(m, ""), 
            label=method_nice_names[m], 
            linewidth=6, 
            markersize=16, 
        ) 
        for m in methods_for_legend 
    ] 

    line_fig_width = max(12, len(line_handles) * 2.5) 
    fig_line, ax_line = plt.subplots(figsize=(line_fig_width, 3)) 
    ax_line.legend( 
        handles=line_handles, 
        loc="center", 
        ncol=4, 
        frameon=False, 
        fontsize=20, 
        title_fontsize=22, 
        handlelength=2.5, 
        handleheight=1.8, 
        columnspacing=1, 
        # borderpad=1.4, 
        # labelspacing=1.8, 
    ) 
    ax_line.axis("off") 
    fig_line.savefig( 
        os.path.join(output_dir, "legend_lines.png"), dpi=500, bbox_inches="tight" 
    ) 
    plt.close(fig_line) 

    # --- Hatch legend --- 
    hatch_handles = [ 
        Patch( 
            facecolor="white", 
            edgecolor="black", 
            hatch=HATCHES[i % len(HATCHES)], 
            label=method_nice_names[m], 
            linewidth=1.5, 
        ) 
        for i, m in enumerate(methods_for_legend) 
    ] 

    hatch_fig_width = max(12, len(hatch_handles) * 2.5) 
    fig_hatch, ax_hatch = plt.subplots(figsize=(hatch_fig_width, 3)) 
    ax_hatch.legend( 
        handles=hatch_handles, 
        loc="center", 
        ncol=4, 
        frameon=False, 
        fontsize=20, 
        title_fontsize=22, 
        handlelength=1.5, 
        handleheight=1.5, 
        columnspacing=1, 
        # borderpad=1.4, 
        # labelspacing=1.8, 
    ) 
    ax_hatch.axis("off") 
    

    fig_hatch.savefig( 
        os.path.join(output_dir, "legend_hatches.png"), dpi=500, bbox_inches="tight" 
    ) 
    plt.close(fig_hatch) 

    logging.info(f"Legends saved to {output_dir}") 



def _plot_breakdown_chart(df_file: pd.DataFrame, output_path: str, show_labels: bool = True):
    """Generate and save a stacked bar chart of execution time breakdown with custom order/labels.

    If show_labels is False the chart will omit axis labels and the GS-HUIM annotation (used for subsequent
    datasets to reduce visual clutter)."""
    fig, ax = plt.subplots(figsize=(2, 2), dpi=500)

    min_util = df_file["Minimum_Utility"].min()
    df_subset = df_file[df_file["Minimum_Utility"] == min_util]

    ga_huim = "G-HUIM_SP_adaptive"
    dphim = "DPHIM_DPHIM_dphim"
    gs_huim_methods = STATIC_METHODS

    plot_methods = [ga_huim] + gs_huim_methods + [dphim]

    # slightly wider bars for readability while keeping a square figure
    bar_width = 0.95

    ga_pos = 0.0

    gs_pos_start = ga_pos + 1.8 * bar_width
    num_gs_methods = len(gs_huim_methods)
    gs_pos_end = gs_pos_start + (num_gs_methods - 1) * bar_width
    gs_positions = np.linspace(gs_pos_start, gs_pos_end, num=num_gs_methods)

    if gs_positions.size > 0:
        dphim_pos = gs_positions[-1] + 2.0 * bar_width
    else:
        dphim_pos = gs_pos_start + 2.0 * bar_width

    ga_center = ga_pos + (bar_width / 2.0)
    gs_positions_centered = gs_positions + (bar_width / 2.0)
    dphim_center = dphim_pos + (bar_width / 2.0)

    positions = np.array([ga_center] + list(gs_positions_centered) + [dphim_center])

    for method_idx, (method, pos) in enumerate(zip(plot_methods, positions)):
        row = df_subset[df_subset["AlgoMeth"] == method]
        if row.empty:
            continue

        bottom = 0.0
        hatch = HATCHES[method_idx % len(HATCHES)]
        for phase, color in PHASE_COLORS.items():
            height = row[phase].iloc[0] / 1000.0
            ax.bar(
                pos,
                height,
                bar_width,
                bottom=bottom,
                color=color,
                edgecolor="black",
                linewidth=0.7,
                hatch=hatch,
            )
            bottom += height

    # Determine GA-HUIM total (used to set a consistent y-axis cap)
    ga_row = df_subset[df_subset["AlgoMeth"] == "G-HUIM_SP_adaptive"]
    # compute totals for all present methods (in seconds)

    # if file name contains 'accidents' on y axis label put time
    if 'accidents' in df_file['File'].iloc[0].lower():
        ax.set_ylabel("Time (s)", fontsize=12)

    totals = []
    for method in plot_methods:
        row = df_subset[df_subset["AlgoMeth"] == method]
        if row.empty:
            continue
        total_time = (
            row["Parse"].iloc[0] + row["TWU"].iloc[0] + row["Build"].iloc[0] + row["Search"].iloc[0]
        ) / 1000.0
        totals.append(total_time)

    # If GA exists and totals have variance, set y cap to 2x GA; otherwise keep autoscaling
    apply_cap = False
    ga_total = None
    if not ga_row.empty and len(totals) > 0:
        ga_total = ga_row["Parse"].iloc[0] + ga_row["TWU"].iloc[0] + ga_row["Build"].iloc[0] + ga_row["Search"].iloc[0]
        ga_total = ga_total / 1000.0
        if max(totals) - min(totals) > 0:
            apply_cap = True

    # longest
    longest = max(totals) if len(totals) > 0 else None

    y_cap = None
    if apply_cap and ga_total is not None:
        # Use 4x GA-HUIM as the y-axis cap
        # y_cap = ga_total * 8.0
        y_cap = min(ga_total * 16.0, longest * 1.1 if longest is not None else 0)

    # annotate any bars that exceed the cap with a text label indicating the real value
    for method_idx, (method, pos) in enumerate(zip(plot_methods, positions)):
        row = df_subset[df_subset["AlgoMeth"] == method]
        if row.empty:
            continue
        total_time = (
            row["Parse"].iloc[0] + row["TWU"].iloc[0] + row["Build"].iloc[0] + row["Search"].iloc[0]
        ) / 1000.0
        if y_cap is not None and total_time > y_cap:
            ax.text(
                pos,
                y_cap * 0.98,
                f"{total_time:.0f}",
                ha="center",
                va="top",
                color="black",
                fontsize=9,
                rotation=0,
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
            )

    if y_cap is not None:
        ax.set_ylim(top=y_cap)

    if show_labels:
        ax.set_ylabel("Execution Time (s)", fontsize=12)
    ax.set_xticks(positions)

    xtick_labels = []
    for method in plot_methods:
        if method == ga_huim:
            xtick_labels.append(method_nice_names.get(method, method).split("(")[0])
        elif method in gs_huim_methods:
            label = method_nice_names.get(method, method)
            xtick_labels.append(label.split("(")[-1].replace(")", ""))
        else:
            xtick_labels.append(method_nice_names.get(method, method))

    ax.set_xticklabels(xtick_labels, rotation=0)

    first_gs_idx = 1
    last_gs_idx = len(gs_huim_methods)
    gs_xmin_for_line = positions[first_gs_idx] - bar_width / 2
    gs_xmax_for_line = positions[last_gs_idx] + bar_width / 2
    gs_center = (gs_xmin_for_line + gs_xmax_for_line) / 2

    y_max = ax.get_ylim()[1]


    if show_labels:
        ax.text(
            gs_center,
            y_max * 0.99,
            "GS-HUIM",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(
                facecolor="white", alpha=0.9, edgecolor="none", boxstyle="square,pad=0.3"
            ),
        )

        ax.hlines(
            y=y_max * 0.97,
            xmin=gs_xmin_for_line,
            xmax=gs_xmax_for_line,
            color="black",
            linewidth=1,
        )

        # set y_max a bit higher to accommodate the text
        ax.set_ylim(top=y_max * 1.06)

    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    if not show_labels:
        ax.set_xticklabels([])


    file_base = df_file["File"].iloc[0]
    # plt.title(f"Execution Time Breakdown on {file_base.capitalize()}")
    fig.tight_layout()
    # Save PNG for compatibility, and also save PGF/PDF for LaTeX-friendly inclusion.
    plt.savefig(output_path, dpi=300)
    base, ext = os.path.splitext(output_path)
    plt.close(fig)
    logging.info(f"Saved breakdown plot to {output_path}")


def _plot_line_chart(
    df_file: pd.DataFrame,
    all_methods: List[str],
    output_path: str,
    y_col: str,
    y_label: str,
    title_suffix: str,
    log_scale: bool = False,
    show_labels: bool = True,
):
    """Generic function to generate line charts (total time, patterns found).

    When show_labels is False the axis labels and x tick labels are suppressed for compact multi-panel
    output (used for datasets beyond the first one).
    """

    fig, ax = plt.subplots(figsize=(2, 2), dpi=500)
    if y_col == "Total_Time":
        fig, ax = plt.subplots(figsize=(3, 3), dpi=500)


    # ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=1000))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=10))
    ax.yaxis.set_minor_locator(plt.AutoLocator())


    utils = sorted(df_file["Minimum_Utility"].unique())

    plottable_methods = [m for m in all_methods if m in method_nice_names]

    for method in plottable_methods:
        df_m = df_file[df_file["AlgoMeth"] == method]
        if df_m.empty:
            continue

        y_data = df_m.set_index("Minimum_Utility")[y_col].reindex(utils)

        if y_col == "Total_Time":
            y_data = y_data / 1000.0

        ax.plot(
            utils,
            y_data,
            label=method_nice_names.get(method, method),
            linestyle=METHOD_LINESTYLES.get(method, "-"),
            marker=METHOD_MARKERS.get(method, ""),
            color=METHOD_COLORS.get(method, "gray"),
        )

    if show_labels:
        ax.set_xlabel("Minimum Utility", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)

        # if file name contains 'accidents' on y axis label put time
    if 'accidents' in df_file['File'].iloc[0].lower():
        ax.set_ylabel("Time (s)", fontsize=12)
        ax.set_xlabel("Minimum Utility", fontsize=12)



    if log_scale:
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.7)
    else:
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_ylim(bottom=0)

    ax.set_xticks(utils)
    # if show_labels:
    #     ax.tick_params(axis="x", rotation=45)
    # else:
    #     # hide x tick labels for subsequent datasets
    #     ax.set_xticklabels(["" for _ in utils])

    # For line charts, set y cap relative to GA-HUIM (if available) and annotate overflows
    if "G-HUIM_SP_adaptive" in plottable_methods:
        df_ga = df_file[df_file["AlgoMeth"] == "G-HUIM_SP_adaptive"]
        if not df_ga.empty:
            ga_vals = df_ga.set_index("Minimum_Utility")[y_col].reindex(utils)
            # convert to seconds if total time
            if y_col == "Total_Time":
                ga_vals = ga_vals / 1000.0
            # check variance: if all values are identical (or only one), skip applying cap
            ga_vals_clean = ga_vals.dropna()
            if len(ga_vals_clean) == 0 or (ga_vals_clean.max() - ga_vals_clean.min() == 0):
                ga_base = None
            else:
                # use the first non-null GA value as baseline
                ga_base = float(ga_vals_clean.iloc[0])
        else:
            ga_base = None
    else:
        ga_base = None

    # Do not apply GA-based y-axis capping for Frequent_Patterns charts
    if y_col == "Frequent_Patterns":
        ga_base = None

    # If GA exists and totals have variance, set y cap to 2x GA; otherwise keep autoscaling
    apply_cap = False
    for method in plottable_methods:
        df_m = df_file[df_file["AlgoMeth"] == method]
        if df_m.empty:
            continue
        y_data = df_m.set_index("Minimum_Utility")[y_col].reindex(utils)
        if y_col == "Total_Time":
            y_data = y_data / 1000.0
        if not y_data.isnull().all() and (y_data.max() - y_data.min() > 0):
            apply_cap = True
            break
        if y_col == "Frequent_Patterns":
            # For Frequent Patterns, we only apply cap if GA is available
            apply_cap = ga_base is not None

    # Only apply a GA-based cap when GA exists, the plotted values vary, and
    # log-scale is not requested. For Total_Time use 4x GA; otherwise keep autoscaling.
    if not log_scale and ga_base is not None and apply_cap and y_col == "Total_Time":
        logging.info(f"Applying GA-based y-cap for {output_path}")
        try:
            y_cap_line = float(ga_base) * 8.0
            # enforce the cap (set top to 4x GA)
            ax.set_ylim(top=y_cap_line)

            # Annotate DPHIM points that exceed the cap with their time
            dphim_key = "DPHIM_DPHIM_dphim"
            if dphim_key in plottable_methods:
                df_dphim = df_file[df_file["AlgoMeth"] == dphim_key]
                if not df_dphim.empty:
                    y_dphim = df_dphim.set_index("Minimum_Utility")[y_col].reindex(utils)
                    y_dphim = y_dphim / 1000.0
                    for ux, val in zip(utils, y_dphim.values):
                        if val is None or (isinstance(val, float) and np.isnan(val)):
                            continue
                        try:
                            v = float(val)
                        except Exception:
                            continue
                        if v > y_cap_line:
                            ax.text(
                                ux,
                                y_cap_line * 0.98,
                                f"{v:.2f}",
                                ha="center",
                                va="top",
                                fontsize=8,
                                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                            )
        except Exception:
            # if conversion fails, skip applying the cap
            logging.warning("Could not apply GA-based y-cap for %s", output_path)


    # disable the x-axis labels
    # ax.set_xticklabels([])

    # set_ylim
    ax.set_ylim(top=10**2)

    file_base = df_file["File"].iloc[0]
    # plt.title(f"{title_suffix} on {file_base.capitalize()}")
    fig.tight_layout()
    # Save PNG and LaTeX-friendly formats.
    plt.savefig(output_path, dpi=300)
    base, ext = os.path.splitext(output_path)
    plt.close(fig)
    logging.info(f"Saved {y_col} plot to {output_path}")


def _plot_total_time_chart(
    df_file: pd.DataFrame,
    all_methods: List[str],
    output_path: str,
    show_labels: bool = True,
    log_scale: bool = False,
):
    """Generate and save a line chart of total time vs. minimum utility."""
    _plot_line_chart(
        df_file,
        all_methods,
        output_path,
        y_col="Total_Time",
        y_label="Execution Time (s)",
        title_suffix="Performance",
        log_scale=log_scale,
        show_labels=show_labels,
    )


def _plot_patterns_chart(
    df_file: pd.DataFrame,
    all_methods: List[str],
    output_path: str,
    show_labels: bool = True,
    log_scale: bool = True,
):
    """Generate and save a line chart of frequent patterns vs. minimum utility."""
    _plot_line_chart(
        df_file,
        all_methods,
        output_path,
        y_col="Frequent_Patterns",
        # y_label="Number of Patterns Found (Log Scale)",
        y_label="",
        title_suffix="Patterns Found",
        log_scale=log_scale,
        show_labels=show_labels,
    )


def plot_all_results(csv_path: str, output_dir: str, use_line_log: bool = False):
    """Main function to generate all plots from the results CSV."""
    if not os.path.exists(csv_path):
        logging.error(f"Results file not found: {csv_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    df = _preprocess_data(pd.read_csv(csv_path))

    all_methods = sorted([m for m in df["AlgoMeth"].unique() if m in method_nice_names])

    if not all_methods:
        logging.warning("No plottable methods found after filtering.")
        return

    csv_files = sorted(df["File"].unique())
    for idx, file_name in enumerate(csv_files):
        df_file = df[df["File"] == file_name].copy()

        show_labels = idx == 0

        _plot_breakdown_chart(
            df_file,
            os.path.join(output_dir, f"{file_name}_breakdown.png"),
            show_labels=show_labels,
        )

        # total-time uses log scale when requested via CLI; patterns keep their default
        _plot_total_time_chart(
            df_file,
            all_methods,
            os.path.join(output_dir, f"{file_name}_total_time.png"),
            show_labels=show_labels,
            log_scale=True,
        )

        _plot_patterns_chart(
            df_file,
            all_methods,
            os.path.join(output_dir, f"{file_name}_patterns.png"),
            show_labels=show_labels,
            log_scale=True,
        )

    _generate_legends(output_dir, all_methods)


def main():
    parser = argparse.ArgumentParser(
        description="Plot High-Utility Itemset Mining benchmark results."
    )
    parser.add_argument(
        "input_csv", type=str, help="Path to the input results CSV file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save the generated plots.",
    )
    args = parser.parse_args()

    plot_all_results(args.input_csv, args.output_dir)
    logging.info("Plotting complete.")


if __name__ == "__main__":
    main()
