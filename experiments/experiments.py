import subprocess
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator

# --- Data Loading and Saving ---

def load_results(csv_path="combined_results.csv"):
    """Load results from CSV or create a new DataFrame if it doesn't exist."""
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    # IMPROVEMENT: Add Total_Time to the DataFrame to simplify comparisons.
    cols = [
        "Algorithm", "Method", "File", "Minimum_Utility",
        "Parse_Time", "TWU_Time", "Build_Time", "Search_Time",
        "Total_Time", "Frequent_Patterns"
    ]
    return pd.DataFrame(columns=cols)

def save_results(df, csv_path="combined_results.csv"):
    """Save results DataFrame to CSV."""
    # Ensure columns are in a consistent order before saving
    ordered_cols = [
        "Algorithm", "Method", "File", "Minimum_Utility",
        "Parse_Time", "TWU_Time", "Build_Time", "Search_Time",
        "Total_Time", "Frequent_Patterns"
    ]
    df[ordered_cols].to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

def append_if_best(df, entry, csv_path):
    """
    FIX: More robustly insert or replace the row that gives the lowest Total_Time
    for a given (Algorithm, Method, File, Minimum_Utility) key.
    """
    key_cols = ["Algorithm", "Method", "File", "Minimum_Utility"]
    
    # IMPROVEMENT: Calculate total time once and add it to the entry dictionary.
    entry["Total_Time"] = (
        entry["Parse_Time"] + entry["Build_Time"] + 
        entry["Search_Time"] + entry["TWU_Time"]
    )

    # Find if a row with the same key already exists.
    mask = (df[key_cols] == pd.Series(entry)[key_cols]).all(axis=1)

    if not mask.any():
        # First result for this key, append it.
        # Use pd.concat for a more modern and safer append.
        new_df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        # A result exists, check if the new one is better.
        cur_idx = df.index[mask].tolist()[0]
        if entry["Total_Time"] < df.loc[cur_idx, "Total_Time"]:
            # Replace the existing row with the better one.
            df.loc[cur_idx] = pd.Series(entry)
        new_df = df # Return the modified or original dataframe
        
    save_results(new_df, csv_path)
    return new_df # Return the updated dataframe for the next iteration


def has_run_before(df, algorithm, method, file_name, min_util):
    """Check if this combination has already been processed."""
    base = os.path.splitext(os.path.basename(file_name))[0]
    mask = (
        (df["Algorithm"] == algorithm)
        & (df["Method"] == method)
        & (df["File"] == base)
        & (df["Minimum_Utility"] == min_util)
    )
    return not df[mask].empty

# --- Parsers ---

def parse_metrics_cu(output):
    """Extract metrics from cuDPEFIM stdout."""
    metrics = {"read": 0, "twu": 0, "parse_temp": 0, "build": 0, "mining": 0, "patterns_count": 0}
    iteration_total = 0

    for line in output.splitlines():
        print([x for x in line.split() if x != ""])  # Debug: Print each line split into parts
        print("Iteration total so far:", iteration_total)  # Debug: Print current iteration total

        # IMPROVEMENT: More robust parsing that isn't strictly dependent on split index.
        if "Read" in line:
            metrics["read"] = int(line.split()[2])
        elif "CalcTWU" in line:
            metrics["twu"] = int(line.split()[1])
        elif "Parse" in line:
            metrics["parse_temp"] = int(line.split()[1])
        elif "Build" in line:
            metrics["build"] = int(line.split()[1])
        elif "Mining Completed" in line:
            metrics["mining"] = int(line.split()[2])
        elif line.startswith("Iteration"):
            try:
                print("Parsing iteration line has time:", line.split()[2], iteration_total)  # Debug
                iteration_total += int(line.split()[2])
            except (IndexError, ValueError):
                pass
        elif "Number of patterns:" in line:
            metrics["patterns_count"] = int(line.split(":")[-1].strip())

    parse_time = metrics["read"] + metrics["parse_temp"]
    build_time = metrics["build"]
    print("Iteration total time:", iteration_total, "Mining time:", metrics["mining"])
    search_time = iteration_total + metrics["mining"]
    twu_time = metrics["twu"]

    print(metrics)

    return parse_time, twu_time, build_time, search_time, metrics["patterns_count"]


def parse_metrics_dphim(output):
    """Extract metrics from DPHIM stdout."""
    metrics = {"parse": 0, "calc": 0, "build": 0, "search": 0, "patterns": 0}
    for line in output.splitlines():
        parts = [p.strip() for p in line.split()]
        if not parts: continue
        if "parse:" in parts:
            metrics["parse"] = int(parts[1])
        elif "calcTWU:" in parts:
            metrics["calc"] = int(parts[1])
        elif "Build:" in parts:
            metrics["build"] = int(parts[1])
        elif "Search:" in parts:
            metrics["search"] = int(parts[1])
        elif "count:" in parts and "itemsets" in parts:
            metrics["patterns"] = int(parts[-1])
    return metrics["parse"], metrics["calc"], metrics["build"], metrics["search"], metrics["patterns"]


# --- Runners ---

def run_cuDPEFIM(
    process_data, method, managed=False, adaptive=False,
    total_memory="78G", block_size=1024, repeats=10, csv_path="combined_results.csv"
):
    if method not in ("DP", "SP"):
        raise ValueError("Invalid method: choose 'DP' or 'SP'")
    df = load_results(csv_path)

    for file_path, min_utils in process_data:
        base = os.path.splitext(os.path.basename(file_path))[0]
        for min_util in min_utils:
            method_label = f"{method}_adaptive" if adaptive else f"{method}_fixed_{block_size}"
            
            if has_run_before(df, "cuDPEFIM", method_label, base, min_util):
                print(f"Skipping cuDPEFIM ({method_label}) {base} min_util={min_util}")
                continue

            best_run = None
            for i in range(repeats):
                cmd = [
                    "/export/home1/ltarun/cuDPEFIM/build/cuDPEFIM",
                    "--input", file_path, "--minutil", str(min_util),
                    "--threads", "24", "--total_memory", total_memory,
                    "--managed", str(managed).lower(), "--method", method,
                    "--adaptive", str(adaptive).lower(),
                ]
                if not adaptive:
                    cmd += ["--block_size", str(block_size)]

                print(f"Run {i+1}/{repeats}: {' '.join(cmd)}")
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    p, t, b, s, patterns = parse_metrics_cu(proc.stdout)
                    print(f"Parsed times - P: {p}, T: {t}, B: {b}, S: {s}, Patterns: {patterns}")
                    total = p + t + b + s
                    if best_run is None or total < best_run["total"]:
                        best_run = {"total": total, "p": p, "t": t, "b": b, "s": s, "patterns": patterns}
                except subprocess.CalledProcessError as e:
                    print(f"Error running command: {e.stderr}")
                    break # Stop repeating if there's a persistent error
            
            if best_run:
                entry = {
                    "Algorithm": "cuDPEFIM", "Method": method_label, "File": base,
                    "Minimum_Utility": min_util, "Parse_Time": best_run["p"],
                    "TWU_Time": best_run["t"], "Build_Time": best_run["b"],
                    "Search_Time": best_run["s"], "Frequent_Patterns": best_run["patterns"]
                }
                df = append_if_best(df, entry, csv_path)

def run_dphim(
    process_data, methods, cores, repeats=10, csv_path="combined_results.csv"
):
    df = load_results(csv_path)

    for file_path, min_utils in process_data:
        base = os.path.splitext(os.path.basename(file_path))[0]
        for min_util in min_utils:
            for meth in methods:
                label = f"DPHIM_{meth}"
                if has_run_before(df, "DPHIM", label, base, min_util):
                    print(f"Skipping DPHIM ({label}) {base} min_util={min_util}")
                    continue

                best_run = None
                for i in range(repeats):
                    cmd = [
                        "/export/home1/ltarun/dphim/build/run",
                        "-i", file_path, "-m", str(min_util),
                        "-s", meth, "-t", str(cores), "-o", "/dev/null"
                    ]
                    print(f"Run {i+1}/{repeats}: {' '.join(cmd)}")
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    p, t, b, s, pat = parse_metrics_dphim(proc.stdout)
                    total = p + t + b + s
                    if best_run is None or total < best_run["total"]:
                        best_run = {"total": total, "p": p, "t": t, "b": b, "s": s, "pat": pat}

                if best_run:
                    entry = {
                        "Algorithm": "DPHIM", "Method": label, "File": base,
                        "Minimum_Utility": min_util, "Parse_Time": best_run["p"],
                        "TWU_Time": best_run["t"], "Build_Time": best_run["b"],
                        "Search_Time": best_run["s"], "Frequent_Patterns": best_run["pat"]
                    }
                    df = append_if_best(df, entry, csv_path)

# --- Plotting ---
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_total(csv_path="combined_results.csv"):
    """
    Generates bar and line plots to visualize algorithm performance results from a CSV file.
    
    This function creates two types of plots for each dataset ('File') in the CSV:
    1. A stacked bar chart showing the execution time breakdown. Key algorithms (DPHIM, GA-HUIM)
       have a consistent hatch pattern matching their line plot markers.
    2. A line chart comparing total execution time. The x-axis formatting is unified:
       scientific notation is only used if all utility values have trailing zeros.
    
    Legends are generated as separate image files to declutter the main plots.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found.")
        return

    # 1. Load and preprocess the data
    df = pd.read_csv(csv_path).rename(columns={
        "Parse_Time": "Parse", "TWU_Time": "TWU",
        "Build_Time": "Build", "Search_Time": "Search",
    })
    df["Minimum_Utility"] = df["Minimum_Utility"].astype(int)
    df["AlgoMeth"] = df["Algorithm"] + "_" + df["Method"]
    
    phases = ["Parse", "TWU", "Build", "Search"]
    df["Total_Time"] = df[phases].sum(axis=1)

    def sort_key(method_name):
        parts = method_name.split("_")
        algo, suffix = parts[0], parts[-1].lower()
        group_order = {"DPHIM": 0, "cuDPEFIM": 1}.get(algo, 2)
        num_order = float("inf") if suffix == "adaptive" else int(suffix) if suffix.isdigit() else 0
        return (group_order, num_order)

    all_methods = sorted(df["AlgoMeth"].unique(), key=sort_key)
    n_methods = len(all_methods)

    # 2. Define plotting styles with a unified scheme
    method_nice_names = {
        "DPHIM_DPHIM_dphim": "DPHIM",
        "cuDPEFIM_SP_fixed_32": "GS-HUIM(32)", "cuDPEFIM_SP_fixed_64": "GS-HUIM(64)",
        "cuDPEFIM_SP_fixed_128": "GS-HUIM(128)", "cuDPEFIM_SP_fixed_256": "GS-HUIM(256)",
        "cuDPEFIM_SP_fixed_512": "GS-HUIM(512)", "cuDPEFIM_SP_fixed_1024": "GS-HUIM(1024)",
        "cuDPEFIM_SP_adaptive": "GA-HUIM",
    }
    
    phase_colors = {"Parse": "#2ca0c2", "TWU": "#1f77b4", "Build": "#ffdd57", "Search": "#ff7f0e"}
    
    dphim_key = "DPHIM_DPHIM_dphim"
    adaptive_key = "cuDPEFIM_SP_adaptive"
    static_method_keys = [m for m in all_methods if m not in [dphim_key, adaptive_key]]

    # --- Unified Style Definitions for DPHIM, GA-HUIM, and GS-HUIM ---
    # Colors
    method_colors = {"DPHIM_DPHIM_dphim": "#1f77b4", "cuDPEFIM_SP_adaptive": "#d62728"}
    green_shades = plt.cm.Greens(np.linspace(0.4, 0.9, len(static_method_keys)))
    for method, color in zip(static_method_keys, green_shades):
        method_colors[method] = color

    # Markers
    method_markers = {"DPHIM_DPHIM_dphim": "*", "cuDPEFIM_SP_adaptive": "o"}
    for method in static_method_keys:
        method_markers[method] = ""

    # Hatches
    method_hatches = {"DPHIM_DPHIM_dphim": "*", "cuDPEFIM_SP_adaptive": "o"}
    static_hatches = ['/', '\\', '|', '-', 'x', '.']
    for i, method in enumerate(static_method_keys):
        method_hatches[method] = static_hatches[i % len(static_hatches)]

    # Linestyles
    method_linestyles = {"DPHIM_DPHIM_dphim": "-", "cuDPEFIM_SP_adaptive": "--"}
    static_linestyles = [':', '-.', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (1, 1)), (0, (3, 5, 1, 5))]
    for i, method in enumerate(static_method_keys):
        method_linestyles[method] = static_linestyles[i % len(static_linestyles)]

    legends_generated = False

    # 3. Generate a plot for each file in the dataset
    for file_name in sorted(df["File"].unique()):
        df_file = df[df["File"] == file_name].copy()
        utils = sorted(df_file["Minimum_Utility"].unique())
        if not utils: continue
        
        # --- Plot 1: Bar Chart for Execution Time Breakdown ---
        fig, ax = plt.subplots(figsize=(5, 4.5), dpi=300)
        bar_width, gap_size, current_pos = 0.8, 0.5, 0.0
        positions = []
        for i in range(n_methods):
            if i > 0:
                m_prev, m_curr = all_methods[i-1], all_methods[i]
                if ("DPHIM" in m_prev and "fixed" in m_curr) or \
                   ("fixed" in m_prev and "adaptive" in m_curr):
                    current_pos += gap_size
            positions.append(current_pos)
            current_pos += 1.0
        
        static_bar_heights = {}
        for i, method in enumerate(all_methods):
            row = df_file[(df_file["Minimum_Utility"] == utils[0]) & (df_file["AlgoMeth"] == method)]
            if row.empty: continue
            bottom = 0.0
            for phase in phases:
                height = row[phase].iloc[0] / 1000.0
                ax.bar(positions[i], height, bar_width, bottom=bottom,
                       color=phase_colors[phase], edgecolor="black", 
                       linewidth=0.7, hatch=method_hatches[method])
                bottom += height
            if method in static_method_keys:
                static_bar_heights[i] = bottom
        
        static_indices = [i for i, m in enumerate(all_methods) if m in static_method_keys]
        if static_indices:
            left_pos = positions[static_indices[0]] - bar_width / 2
            right_pos = positions[static_indices[-1]] + bar_width / 2
            ax.text((left_pos + right_pos) / 2, max(static_bar_heights.values() or [0]) * 1.05,
                    "|<----------- GS-HUIM (static) ----------->|", ha="center", va="bottom", fontsize=9)
        
        ax.set_ylabel("Execution Time (s)", fontsize=12)
        ax.set_xticks(positions)
        ax.set_xticklabels([])


        title = file_name.split("_")[0].capitalize()
        plt.title(f"{title}")
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
        fig.tight_layout()



        plt.savefig(f"{file_name}_breakdown.png", dpi=300)
        plt.close(fig)

        # --- Plot 2: Line Chart for Total Time vs. Utility ---
        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)


        for method in all_methods:
            df_m = df_file[df_file["AlgoMeth"] == method]
            if df_m.empty: continue
            times = df_m.set_index("Minimum_Utility")["Total_Time"].reindex(utils) / 1000.0
            ax.plot(utils, times, label=method_nice_names.get(method, method),
                    linestyle=method_linestyles[method], marker=method_markers[method],
                    color=method_colors[method])
        
        ax.set_xlabel("Minimum Utility", fontsize=12)
        ax.set_ylabel("Execution Time (s)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        # set the y-axis to 0 start and the top as 10% more than max
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.1)

        # --- UNIFIED X-AXIS FORMATTING ---
        # Use scientific notation ONLY if ALL utility values end in '0'.
        # If any value does not (e.g., 1435), use plain integer format for all labels.
        use_plain_format = any(u % 10 != 0 for u in utils if u != 0)
        
        if use_plain_format:
            xtick_labels = [str(u) for u in utils]
        else:
            def format_sigfig_label(u):
                if u == 0: return "0"
                s_u = str(u)
                s_stripped = s_u.rstrip('0')
                if not s_stripped: return "0"
                sig_figs = len(s_stripped)
                exponent = len(s_u) - 1
                mantissa = u / (10**exponent)
                mantissa_str = f"{mantissa:.{sig_figs - 1}f}".rstrip('0').rstrip('.')
                return f"${mantissa_str} \\times 10^{{{exponent}}}$"
            xtick_labels = [format_sigfig_label(u) for u in utils]

        ax.set_xticks(utils)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=9)
        plt.title(f"Total Execution Time vs. Minimum Utility on {title}")

        
        fig.tight_layout()
        # title = file_base.split("_")[0]
        title = file_name.split("_")[0]

        plt.savefig(f"{file_name}_total_time_vs_util.png", dpi=300)
        plt.close(fig)

        # Generate separate legends only once
        if not legends_generated:
            # Phase Legend
            fig_phase_leg, ax_phase_leg = plt.subplots(figsize=(3, 1.5))
            phase_handles = [Patch(facecolor=phase_colors[p], label=p) for p in phases]
            ax_phase_leg.axis("off")
            ax_phase_leg.legend(handles=phase_handles, title="Phases", loc="center", ncol=len(phases), frameon=False)
            plt.tight_layout()
            plt.savefig("legend_phases.png", bbox_inches="tight", pad_inches=0.1, dpi=300)
            plt.close(fig_phase_leg)

            # Method Legend
            fig_method_leg, ax_method_leg = plt.subplots(figsize=(len(all_methods) * 1.5 / 2, 2.5))
            method_handles = []
            for m in all_methods:
                nice_name = method_nice_names.get(m, m)
                if m in [dphim_key, adaptive_key]:
                    handle = plt.Line2D([0], [0], color=method_colors[m], 
                                        linestyle=method_linestyles[m], marker=method_markers[m], 
                                        markersize=8, label=nice_name)
                else:
                    handle = Patch(facecolor=method_colors[m], edgecolor="black",
                                   hatch=method_hatches[m], label=nice_name)
                method_handles.append(handle)

            ax_method_leg.axis("off")
            ax_method_leg.legend(handles=method_handles, title="Method", loc="center", ncol=2, frameon=False)
            plt.tight_layout()
            plt.savefig("legend_methods.png", bbox_inches="tight", pad_inches=0.1, dpi=300)
            plt.close(fig_method_leg)



            # Line Legend
            fig_line_leg, ax_line_leg = plt.subplots(figsize=(len(all_methods) * 1.5 / 2, 2.5))
            line_handles = []
            for m in all_methods:
                nice_name = method_nice_names.get(m, m)
                handle = plt.Line2D([0], [0], color=method_colors[m], 
                                    linestyle=method_linestyles[m], marker=method_markers[m], 
                                    markersize=8, label=nice_name)
                line_handles.append(handle)
            ax_line_leg.axis("off")
            ax_line_leg.legend(handles=line_handles, title="Method", loc="center", ncol=2, frameon=False)
            plt.tight_layout()
            plt.savefig("legend_lines.png", bbox_inches="tight", pad_inches=0.1, dpi=300)
            plt.close(fig_line_leg)

            
            legends_generated = True

    print("Finished generating all plots and legends.")

if __name__ == '__main__':
        
    process_data = [
            # [
            #     "/export/home1/ltarun/cuDPEFIM/datasets/accidents_utility_spmf.txt",
            #     [3330000, 3340000, 3350000, 3360000, 3370000, 3380000, 3390000]
            # ],
            [
                "/export/home1/ltarun/cuDPEFIM/datasets/connect_utility_spmf.txt",
                [7800000, 7900000, 8000000, 8100000, 8200000, 8300000, 8400000]
            ],
            # [
            #     "/export/home1/ltarun/cuDPEFIM/datasets/pumsb_utility_spmf.txt",
            #     # 9150000, 9250000, 9350000, 
            #     [9450000],
            # ],
            # [
            #     "/export/home1/ltarun/cuDPEFIM/datasets/BMS_utility_spmf.txt",
            #     # , 2002000,2004000,2006000
            #     [2000000],
            # ],
            [
                "/export/home1/ltarun/cuDPEFIM/datasets/chess_utility_spmf.txt",
                [190000, 200000, 210000, 220000, 230000, 240000, 250000]
            ],
            # [
            #     "/export/home1/ltarun/cuDPEFIM/datasets/kosarak_utility_spmf.txt",
            #     [770000, 780000, 790000, 800000, 810000, 820000, 830000]
            # ],
            [
                "/export/home1/ltarun/cuDPEFIM/datasets/retail_utility_spmf.txt",
                [1430, 1435, 1440, 1445, 1450, 1455, 1460]
            ],
            [
                "/export/home1/ltarun/cuDPEFIM/datasets/mushroom_utility_spmf.txt",
                [9000, 10000, 11000, 12000, 13000, 14000, 15000]
            ]
        ]

    results_df = load_results()
    # block_sizes = [32, 64, 128, 256, 512, 1024]
    # block_sizes = [32, 64, 128, 256, 512, 1024]
    # for bs in block_sizes:
    #     run_cuDPEFIM(process_data, "SP", adaptive=False, block_size=bs, repeats=1)
    run_cuDPEFIM(process_data, "SP", adaptive=True, repeats=1)
    run_dphim(process_data, methods=["dphim"], cores=48, repeats=1)
    plot_total()
# plot_cuDPEFIM_speedup_by_file()

