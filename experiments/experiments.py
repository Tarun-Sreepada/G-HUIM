import subprocess
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator


# Initialize results DataFrame with key metric columns
def load_results(csv_path="combined_results.csv"):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    cols = [
        "Algorithm",
        "Method",
        "File",
        "Minimum_Utility",
        "Parse_Time",
        "TWU_Time",
        "Build_Time",
        "Search_Time",
        "Frequent_Patterns",
    ]
    return pd.DataFrame(columns=cols)


def append_if_best(df, entry, csv_path):
    """Insert or replace the row that gives the lowest total_time
    for (Algorithm, Method, File, Minimum_Utility)."""
    key_cols = ["Algorithm", "Method", "File", "Minimum_Utility"]
    total_time = (
        entry["Parse_Time"]
        + entry["Build_Time"]
        + entry["Search_Time"]
        + entry["TWU_Time"]
    )

    mask = (df[key_cols] == pd.Series(entry)[key_cols]).all(axis=1)

    if not mask.any():
        # first result for this key → simply append
        df.loc[len(df)] = entry
    else:
        cur_idx = df.index[mask][0]
        cur_total = df.loc[
            cur_idx, ["Parse_Time", "TWU_Time", "Build_Time", "Search_Time"]
        ].sum()
        if total_time < cur_total:
            df.loc[cur_idx] = entry  # replace with faster one
    save_results(df, csv_path)


def save_results(df, csv_path="combined_results.csv"):
    """Save results DataFrame to CSV."""
    print(df)
    df.to_csv(csv_path, index=False)


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

def parse_metrics_cu(output):
    """Extract parse, build, and search times + pattern count from stdout."""
    metrics = {
        k: 0
        for k in [
            "read",
            "parse_temp",
            "twu",
            "build",
            "alloc",
            "patterns",
            "mining",
            "save",
            "patterns_count",
        ]
    }

    iteration_total = 0

    for line in output.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        key = parts[0]

        if key == "Read":
            metrics["read"] = int(parts[1])
        elif key == "CalcTWU":
            metrics["twu"] = int(parts[1])
        elif key == "Parse":
            metrics["parse_temp"] = int(parts[1])
        elif key == "Build":
            metrics["build"] = int(parts[1])
        elif key == "Allocator" and "Initialized" in line:
            metrics["alloc"] = int(parts[-3])
        elif key == "Patterns" and "Allocated" in line:
            metrics["patterns"] = int(parts[-3])
        elif key == "Mining" and "Completed" in line:
            metrics["mining"] = int(parts[2])
        elif key == "Iteration":
            try:
                print(parts)
                iteration_total += int(parts[2])
            except (IndexError, ValueError):
                pass
        elif key == "Save":
            metrics["save"] = int(parts[1])
        elif "Number of patterns:" in line:
            metrics["patterns_count"] = int(
                line.split("Number of patterns:")[-1].strip()
            )
        elif "Allocated" in line:
            print("Allocated memory:", line.split("Allocated")[-1].strip())

    parse_time = metrics["read"] + metrics["parse_temp"]
    build_time = metrics["build"]
    # Include sum of iteration times in search time
    search_time = iteration_total + metrics["mining"]
    twu_time = metrics["twu"]

    return parse_time, twu_time, build_time, search_time, metrics["patterns_count"]

def run_cuDPEFIM(
    process_data,
    method,
    managed=False,
    adaptive=False,
    total_memory="78G",
    block_size=1024,
    repeats=10,
    csv_path="combined_results.csv",
):
    if method not in ("DP", "SP"):
        raise ValueError("Invalid method: choose 'DP' or 'SP'")
    df = load_results(csv_path)

    for file_path, min_utils in process_data:
        base = os.path.splitext(os.path.basename(file_path))[0]
        for min_util in min_utils:
            # Build method_label
            if method == "SP":
                adapt_label = "adaptive" if adaptive else f"fixed_{block_size}"
                method_label = f"{method}_{adapt_label}"
            else:
                # for DP we don’t care about adaptive/block_size
                method_label = f"{method}"

            # skip if we already have a record
            if has_run_before(df, "cuDPEFIM", method_label, base, min_util):
                print(
                    f"Skipping cuDPEFIM ({method_label})  {base}  min_util={min_util}"
                )
                continue

            best = None
            for _ in range(repeats):
                cmd = [
                    "/export/home1/ltarun/cuDPEFIM/build/cuDPEFIM",
                    "--input",
                    file_path,
                    "--minutil",
                    str(min_util),
                    "--threads",
                    "24",
                    "--total_memory",
                    total_memory,
                    "--managed",
                    str(managed).lower(),
                    "--method",
                    method,
                    "--adaptive",
                    str(adaptive).lower(),
                ]
                # only specify block_size when not using adaptive blocks
                if not adaptive:
                    cmd += ["--block_size", str(block_size)]

                print("Running:", " ".join(cmd))
                try:
                    proc = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True,
                    )
                    p, t, b, s, patterns = parse_metrics_cu(proc.stdout)
                    total = p + b + s + t
                    if best is None or total < best["total"]:
                        best = dict(total=total, p=p, b=b, s=s, patterns=patterns, t=t)
                except subprocess.CalledProcessError as e:
                    print(f"Error: {e}; continuing with other runs")

            if best:
                entry = {
                    "Algorithm": "cuDPEFIM",
                    "Method": method_label,
                    "File": base,
                    "Minimum_Utility": min_util,
                    "Parse_Time": best["p"],
                    "TWU_Time": best["t"],
                    "Build_Time": best["b"],
                    "Search_Time": best["s"],
                    "Frequent_Patterns": best["patterns"],
                }
                append_if_best(df, entry, csv_path)


def parse_metrics_dphim(output):
    # High-level counts
    parse_ms = 0
    calc_ms = 0
    build_ms = 0
    search_ms = 0
    patterns = 0
    # read lines
    for line in output.splitlines():
        line = line.strip().split()
        line = [x for x in line if x]
        if "parse:" in line:
            parse_ms = int(line[1])
        elif "calcTWU:" in line:
            calc_ms = int(line[1])
        elif "Build:" in line:
            build_ms = int(line[1])
        elif "Search:" in line:
            search_ms = int(line[1])
        elif "count:" in line and "itemsets" in line:
            patterns = int(line[-1])

    parse_time = parse_ms
    calc_time = calc_ms
    build_time = build_ms
    search_time = search_ms

    return parse_time, calc_time, build_time, search_time, patterns


def run_dphim(
    process_data,
    methods,
    cores,
    repeats=10,
    csv_path="combined_results.csv",
):
    df = load_results(csv_path)

    for file_path, min_utils in process_data:
        base = os.path.splitext(os.path.basename(file_path))[0]
        for min_util in min_utils:
            for meth in methods:
                label = f"DPHIM_{meth}"

                # ──► skip if we already have a record for this key
                if has_run_before(df, "DPHIM", label, base, min_util):
                    print(f"Skipping DPHIM ({label})  {base}  min_util={min_util}")
                    continue

                best = None
                for _ in range(repeats):
                    cmd = [
                        "/export/home1/ltarun/dphim/build/run",
                        "-i",
                        file_path,
                        "-m",
                        str(min_util),
                        "-s",
                        meth,
                        "-t",
                        str(cores),
                        "-o",
                        "/dev/null",
                    ]
                    print("Running:", " ".join(cmd))
                    proc = subprocess.run(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )
                    p, t, b, s, pat = parse_metrics_dphim(proc.stdout)
                    total = p + b + s + t
                    if best is None or total < best["total"]:
                        best = dict(total=total, p=p, b=b, s=s, pat=pat, t=t)

                if best:  # save the fastest run
                    entry = {
                        "Algorithm": "DPHIM",
                        "Method": label,
                        "File": base,
                        "Minimum_Utility": min_util,
                        "Parse_Time": best["p"],
                        "TWU_Time": best["t"],
                        "Build_Time": best["b"],
                        "Search_Time": best["s"],
                        "Frequent_Patterns": best["pat"],
                    }
                    append_if_best(df, entry, csv_path)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

def plot_total(csv_path="combined_results.csv"):
    if not os.path.exists(csv_path):
        print(f"CSV file '{csv_path}' not found.")
        return

    # Load and preprocess data
    df = pd.read_csv(csv_path).rename(columns={
        "Parse_Time": "Parse",
        "TWU_Time": "TWU",
        "Build_Time": "Build",
        "Search_Time": "Search",
    })
    df["Minimum_Utility"] = df["Minimum_Utility"].astype(int)
    df["AlgoMeth"] = df["Algorithm"] + "_" + df["Method"]

    def sort_key(x):
        parts = x.split("_")
        algo, suffix = parts[0], parts[-1].lower()
        num = float("inf") if suffix == "adaptive" else int(suffix) if suffix.isdigit() else 0
        grp = {"DPHIM": 0, "cuDPEFIM": 1}.get(algo, 2)
        return (grp, num)

    all_methods = sorted(df["AlgoMeth"].unique(), key=sort_key)
    n_methods = len(all_methods)

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
    green_steps = ["5F", "5F", "5F", "5F", "5F", "5F"]

    for method, green_hex in zip(static_methods, green_steps):
        hex_color = f"#00{green_hex}00"
        method_colors[method] = hex_color
        method_markers[method] = ""

    line_styles = ["-", "--", ":", "-.", (0, (3, 5, 1, 5)), (0, (1, 1)), (0, (5, 10))]
    method_linestyles = {m: line_styles[i % len(line_styles)] for i, m in enumerate(all_methods)}

    linestyles = ["-", "--", ":", "-."]
    hatches = ["/", "\\", "x", "o", ".", "*", "+", "-"]
    method_style = {
        m: {"linestyle": linestyles[i % len(linestyles)], "hatch": hatches[i % len(hatches)]}
        for i, m in enumerate(all_methods)
    }

    FIG_W_IN, FIG_H_IN = 4, 4
    DPI = 500
    BAR_FACTOR = 0.5
    GAP_MULT = 1.1

    bar_width = BAR_FACTOR / n_methods

    transition_gap_map = {}

    for i in range(n_methods - 1):
        m1, m2 = all_methods[i].lower(), all_methods[i + 1].lower()

        if "dphim" in m1 and any(m2.endswith(str(d)) for d in range(10)):
            transition_gap_map[i] = bar_width * 0.6  # normal DPHIM → static gap
        elif any(m1.endswith(str(d)) for d in range(10)) and "adaptive" in m2:
            transition_gap_map[i] = bar_width * 0.6  # smaller static → GA-HUIM gap

    extra_spacing = [0.0] * n_methods
    acc = 0.0
    for i in range(1, n_methods):
        acc += transition_gap_map.get(i - 1, 0.0)
        extra_spacing[i] = acc

    first_file = True
    for file_name in sorted(df["File"].unique()):
        df_file = df[df["File"] == file_name]
        utils = sorted(df_file["Minimum_Utility"].unique())
        if not utils:
            continue
        min_util = utils[0]

        fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
        fig.subplots_adjust(right=0.95)

        x0 = np.array([0.0])
        for i, method in enumerate(all_methods):
            xpos = x0[0] + (i - (n_methods - 1)/2) * bar_width + extra_spacing[i]
            row = df_file[(df_file["Minimum_Utility"] == min_util) &
                          (df_file["AlgoMeth"] == method)]
            if row.empty:
                continue
            bottom = 0.0
            for phase in phases:
                height = row[phase].iloc[0] / 1000.0
                ax.bar(xpos, height, bar_width, bottom=bottom,
                       color=phase_colors[phase], edgecolor="black", linewidth=1,
                       linestyle=method_style[method]["linestyle"],
                       hatch=method_style[method]["hatch"])
                bottom += height
            
            friendly_name = method_nice_names.get(method, method)
            if method.startswith("cuDPEFIM_SP_fixed_"):
                label = f"({friendly_name.split('(')[-1]}"  # Only the number in parentheses
                # remove brackets from label
                label = label.replace("(", "").replace(")", "")
            else:
                label = friendly_name

            if first_file:
                ax.text(xpos, bottom + 0.002, label, ha='center', va='bottom', fontsize=10, rotation=0)

        static_indices = [
            i for i, m in enumerate(all_methods)
            if m.startswith("cuDPEFIM_SP_fixed_") and m.endswith(tuple(str(p) for p in [32, 64, 128, 256, 512, 1024]))
        ]

        if static_indices and first_file:
            # Compute center x position of the static bars
            left_x = x0[0] + (static_indices[0] - (n_methods - 1)/2) * bar_width + extra_spacing[static_indices[0]]
            right_x = x0[0] + (static_indices[-1] - (n_methods - 1)/2) * bar_width + extra_spacing[static_indices[-1]]
            center_x = (left_x + right_x) / 2
            y_pos = 2.475  # adjust this if it's too close or too far from the bars

            ax.text(
                center_x, y_pos, "|<--- GS-HUIM (static) --->|",
                ha="center", va="top", fontsize=10, transform=ax.transData
            )

        # get curent y-axis limits
        y_min, y_max = ax.get_ylim()
        # set y-axis limits to be 10% larger than the max value
        y_max = y_max * 1.05
        ax.set_ylim(y_min, y_max)


        ax.set_xticks([])
        ax.grid(True)

        if first_file:
            ax.legend(handles=[Patch(facecolor=phase_colors[p], label=p) for p in phases], fontsize=10, loc="upper right")
            # bring it a little bit down
            ax.legend_.set_bbox_to_anchor((1.0, 0.95))
            ax.set_ylabel("Execution Time (s)", fontsize=14)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.savefig(f"{file_name}_min_util.png", bbox_inches="tight", dpi=DPI)
        plt.close(fig)

        df_total = df_file.copy()
        df_total["TotalTime"] = df_total[phases].sum(axis=1) / 1000.0
        methods_in_file = sorted(df_total["AlgoMeth"].unique(), key=sort_key)

        fig, ax = plt.subplots(figsize=(4, 4))
        for method in methods_in_file:
            df_m = df_total[df_total["AlgoMeth"] == method]
            if df_m.empty:
                continue
            times = [df_m[df_m["Minimum_Utility"] == u]["TotalTime"].iloc[0]
                     if not df_m[df_m["Minimum_Utility"] == u].empty else np.nan
                     for u in utils]
            color = method_colors.get(method, "black")
            marker = method_markers.get(method, "")
            linestyle = method_linestyles.get(method, "-")
            label = method_nice_names.get(method, method)
            # remove brackets from label

            ax.plot(utils, times, label=label,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=2,
                    color=color,
                    markersize=8)

        log_utils = [np.log10(u) for u in utils]
        rounded_exponents = [int(np.floor(x)) for x in log_utils]

        if max(rounded_exponents) - min(rounded_exponents) <= 1 and max(utils) > 10000:
            common_exp = int(np.median(rounded_exponents)) - 2
            scale_factor = 10 ** common_exp
            xtick_labels = [f"{u / scale_factor:.1f}".rstrip("0").rstrip(".") for u in utils]
            ax.set_xlabel(f"Minimum Utility (×$10^{common_exp}$)", fontsize=14)
        else:
            xtick_labels = [str(u) for u in utils]
            ax.set_xlabel("Minimum Utility", fontsize=14)

        ax.set_xticks(utils)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right")

        ax.grid(True)
        if first_file:
            ax.legend(fontsize=10)
            ax.set_ylabel("Execution Time (s)", fontsize=14)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.savefig(f"{file_name}_total_time_vs_util.png", bbox_inches="tight", dpi=500)
        plt.close(fig)

        if first_file:
            fig_method_leg, ax_method_leg = plt.subplots(figsize=(len(all_methods) * 1.5, 2))
            method_handles = [
                Patch(facecolor=method_colors.get(m, "white"), edgecolor="black",
                      hatch=method_style[m]["hatch"],
                      label=method_nice_names.get(m, m))
                for m in all_methods
            ]
            ax_method_leg.axis("off")
            ax_method_leg.legend(
                handles=method_handles, title="Method", loc="center", ncol=len(method_handles),
                frameon=False, handlelength=3, handleheight=2, fontsize="small"
            )
            plt.tight_layout()
            plt.savefig(f"{file_name}_method_legend.png", bbox_inches="tight", pad_inches=0.1, dpi=500)
            plt.close(fig_method_leg)

        first_file = False

        
process_data = [
    # # [
    # #     "/export/home1/ltarun/cuEFIM/datasets/chainstore.txt",
    # #     [2000000, 2050000, 2100000, 2150000, 2200000, 2500000, 3000000],
    # # ],
    # [
    #     "/export/home1/ltarun/cuEFIM/datasets/accidents_utility_spmf.txt",
    #     [8000000, 9000000, 10000000, 11000000, 12000000, 15000000, 20000000],
    # ],
    # # [
    # #     "/export/home1/ltarun/cuEFIM/datasets/kosarak_utility_spmf.txt",
    # #     [760000, 770000, 780000, 790000, 800000, 900000, 1000000],
    # # ],
    # # [
    # #     "/export/home1/ltarun/cuEFIM/datasets/BMS_utility_spmf.txt",
    # #     [2025000, 2030000, 2035000, 2040000, 2045000],
    # # ],
    # [
    #     "/export/home1/ltarun/cuEFIM/datasets/connect_utility_spmf.txt",
    #     [10000000, 10500000, 11000000, 11500000, 12000000, 15000000, 16000000],
    # ],
    # # [
    # #     "/export/home1/ltarun/cuEFIM/datasets/chess_utility.txt",
    # #     [300000, 310000, 320000, 330000, 340000, 350000],
    # # ],
    # # [
    # #     "/export/home1/ltarun/cuEFIM/datasets/mushroom.txt",
    # #     [35000, 40000, 45000, 50000, 55000],
    # # ],
    # [
    #     "/export/home1/ltarun/cuEFIM/datasets/pumsb_utility.txt",
    #     [11000000, 11100000, 11200000, 11300000, 11400000],
    # ],
    # [
    #     "/export/home1/ltarun/cuEFIM/datasets/retail.txt",
    #     [1345, 1346, 1347, 1348, 1349],
    # ],
]


results_df = load_results()
block_sizes = [32, 64, 128, 256, 512, 1024]
for bs in block_sizes:
    run_cuDPEFIM(process_data, "SP", adaptive=False, block_size=bs, repeats=1)
run_cuDPEFIM(process_data, "SP", adaptive=True, repeats=1)
run_dphim(process_data, methods=["dphim"], cores=48, repeats=1)
plot_total()
# plot_cuDPEFIM_speedup_by_file()
