import subprocess
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


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
        "Build_Time",
        "Search_Time",
        "Frequent_Patterns",
    ]
    return pd.DataFrame(columns=cols)


def append_if_best(df, entry, csv_path):
    """Insert or replace the row that gives the lowest total_time
    for (Algorithm, Method, File, Minimum_Utility)."""
    key_cols = ["Algorithm", "Method", "File", "Minimum_Utility"]
    total_time = entry["Parse_Time"] + entry["Build_Time"] + entry["Search_Time"]

    mask = (df[key_cols] == pd.Series(entry)[key_cols]).all(axis=1)

    if not mask.any():
        # first result for this key → simply append
        df.loc[len(df)] = entry
    else:
        cur_idx = df.index[mask][0]
        cur_total = df.loc[cur_idx, ["Parse_Time", "Build_Time", "Search_Time"]].sum()
        if total_time < cur_total:
            df.loc[cur_idx] = entry  # replace with faster one
    save_results(df, csv_path)


def save_results(df, csv_path="combined_results.csv"):
    """Save results DataFrame to CSV."""
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
            "build",
            "alloc",
            "patterns",
            "mining",
            "save",
            "patterns_count",
        ]
    }

    for line in output.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        key = parts[0]
        if key == "Read":
            metrics["read"] = int(parts[1])
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
    search_time = metrics["alloc"] + metrics["patterns"] + metrics["mining"]
    return parse_time, build_time, search_time, metrics["patterns_count"]

def run_cuDPEFIM(
    process_data,
    method,
    managed=False,
    total_memory="78G",
    repeats=10,
    csv_path="combined_results.csv",
):
    if method not in ("DP", "SP"):
        raise ValueError("Invalid method: choose 'DP' or 'SP'")
    df = load_results(csv_path)

    for file_path, min_utils in process_data:
        base = os.path.splitext(os.path.basename(file_path))[0]
        for min_util in min_utils:
            method_label = f"{method}_{'managed' if managed else 'device'}"

            # ──► skip if we already have a record for this key
            if has_run_before(df, "cuDPEFIM", method_label, base, min_util):
                print(f"Skipping cuDPEFIM ({method_label})  {base}  min_util={min_util}")
                continue

            best = None
            for _ in range(repeats):
                cmd = [
                    "/export/home1/ltarun/cuDPEFIM/build/cuDPEFIM",
                    "--input", file_path,
                    "--minutil", str(min_util),
                    "--threads", "24",
                    "--block_size", "1024",
                    "--total_memory", total_memory,
                    "--managed", str(managed).lower(),
                    "--method", method,
                ]
                print("Running:", " ".join(cmd))
                try:
                    proc = subprocess.run(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, check=True
                    )
                    p, b, s, patterns = parse_metrics_cu(proc.stdout)
                    total = p + b + s
                    if best is None or total < best["total"]:
                        best = dict(total=total, p=p, b=b, s=s, patterns=patterns)
                except subprocess.CalledProcessError as e:
                    print(f"Error: {e}; continuing with other runs")

            if best:  # save the fastest run
                entry = {
                    "Algorithm": "cuDPEFIM",
                    "Method": method_label,
                    "File": base,
                    "Minimum_Utility": min_util,
                    "Parse_Time": best["p"],
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

    parse_time = parse_ms + calc_ms
    build_time = build_ms
    search_time = search_ms
    return parse_time, build_time, search_time, patterns

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
                        "-i", file_path,
                        "-m", str(min_util),
                        "-s", meth,
                        "-t", str(cores),
                        "-o", "/dev/null",
                    ]
                    print("Running:", " ".join(cmd))
                    proc = subprocess.run(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )
                    p, b, s, pat = parse_metrics_dphim(proc.stdout)
                    total = p + b + s
                    if best is None or total < best["total"]:
                        best = dict(total=total, p=p, b=b, s=s, pat=pat)

                if best:  # save the fastest run
                    entry = {
                        "Algorithm": "DPHIM",
                        "Method": label,
                        "File": base,
                        "Minimum_Utility": min_util,
                        "Parse_Time": best["p"],
                        "Build_Time": best["b"],
                        "Search_Time": best["s"],
                        "Frequent_Patterns": best["pat"],
                    }
                    append_if_best(df, entry, csv_path)



def plot_clustered_by_util(csv_path="combined_results.csv"):
    if not os.path.exists(csv_path):
        print(f"CSV file '{csv_path}' not found.")
        return

    # rename column Parse_Time to Parse+TWU

    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Parse_Time": "Parse+TWU"})
    df["Minimum_Utility"] = df["Minimum_Utility"].astype(int)
    df["AlgoMeth"] = df["Algorithm"] + "_" + df["Method"]

    # methods are still global
    all_methods = sorted(df["AlgoMeth"].unique())
    n_methods = len(all_methods)

    # styling maps (same as before)
    phases = ["Parse+TWU", "Build_Time", "Search_Time"]
    phase_colors = {
        "Parse+TWU": "#2ca02c",
        "Build_Time": "#ffdd57",
        "Search_Time": "#ff7f0e",
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

    for file_name in sorted(df["File"].unique()):
        df_file = df[df["File"] == file_name]

        # recompute utils *per file*:
        utils = sorted(df_file["Minimum_Utility"].unique())
        n_utils = len(utils)
        x = np.arange(n_utils)
        total_width = 0.8
        bar_width = total_width / n_methods

        fig, ax = plt.subplots(figsize=(1.2 * n_utils, 6))

        # draw bars
        for j, util in enumerate(utils):
            for i, method in enumerate(all_methods):
                xpos = x[j] + (i - (n_methods - 1) / 2) * bar_width
                bottom = 0
                style = method_style[method]
                for phase in phases:
                    row = df_file[
                        (df_file["Minimum_Utility"] == util)
                        & (df_file["AlgoMeth"] == method)
                    ]
                    height = row[phase].iloc[0] if not row.empty else 0
                    ax.bar(
                        xpos,
                        height,
                        bar_width,
                        bottom=bottom,
                        color=phase_colors[phase],
                        edgecolor="black",
                        linewidth=1,
                        linestyle=style["linestyle"],
                        hatch=style["hatch"],
                        label=phase if (j == 0 and i == 0) else None,
                    )
                    bottom += height

        # only the utils that actually exist in this file
        ax.set_xticks(x)
        ax.set_xticklabels([str(u) for u in utils], rotation=45, ha="right")

        ax.set_xlabel("Minimum Utility (int)")
        ax.set_ylabel("Execution Time (ms)")
        ax.set_title(f"Parse/Build/Search by Utility for '{file_name}'")

        # Phase legend
        phase_handles = [
            Patch(facecolor=phase_colors[p], label=p.replace("_", " ")) for p in phases
        ]
        leg1 = ax.legend(
            handles=phase_handles,
            title="Phase",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

        # Method legend
        method_handles = [
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch=method_style[m]["hatch"],
                linestyle=method_style[m]["linestyle"],
                label=m,
            )
            for m in all_methods
        ]
        leg2 = ax.legend(
            handles=method_handles,
            title="Method",
            bbox_to_anchor=(1.02, 0.5),
            loc="upper left",
        )
        ax.add_artist(leg1)

        plt.tight_layout()
        plt.savefig(f"{file_name}_clustered_by_util.png")
        plt.close(fig)


def plot_search_only(csv_path="combined_results.csv"):
    if not os.path.exists(csv_path):
        print(f"CSV file '{csv_path}' not found.")
        return

    df = pd.read_csv(csv_path)
    df["Minimum_Utility"] = df["Minimum_Utility"].astype(int)
    df["AlgoMeth"] = df["Algorithm"] + "_" + df["Method"]

    all_methods = sorted(df["AlgoMeth"].unique())
    n_methods = len(all_methods)
    total_width = 0.8
    bar_width = total_width / n_methods

    # Assign each method a distinct color from the tab10 palette
    palette = plt.get_cmap("tab10").colors
    method_color = {m: palette[i % len(palette)] for i, m in enumerate(all_methods)}

    # And a hatch pattern per method
    hatches = ["/", "\\", "x", "o", ".", "*", "+", "-"]
    method_hatch = {m: hatches[i % len(hatches)] for i, m in enumerate(all_methods)}

    for file_name in sorted(df["File"].unique()):
        df_file = df[df["File"] == file_name]
        utils = sorted(df_file["Minimum_Utility"].unique())
        x = np.arange(len(utils))

        fig, ax = plt.subplots(figsize=(1.2 * len(utils), 6))

        # Draw search‐only bars, colored and striped per method
        for j, util in enumerate(utils):
            for i, method in enumerate(all_methods):
                xpos = x[j] + (i - (n_methods - 1) / 2) * bar_width
                row = df_file[
                    (df_file["Minimum_Utility"] == util)
                    & (df_file["AlgoMeth"] == method)
                ]
                height = row["Search_Time"].iloc[0] if not row.empty else 0
                ax.bar(
                    xpos,
                    height,
                    bar_width,
                    color=method_color[method],
                    edgecolor="black",
                    hatch=method_hatch[method],
                    label=method if (j == 0) else None,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([str(u) for u in utils], rotation=45, ha="right")
        ax.set_xlabel("Minimum Utility")
        ax.set_ylabel("Search Time (ms)")
        ax.set_title(f"Search‐Only Times by Utility for '{file_name}'")

        # Legend: one patch per method showing its color & hatch
        handles = [
            Patch(
                facecolor=method_color[m],
                edgecolor="black",
                hatch=method_hatch[m],
                label=m,
            )
            for m in all_methods
        ]
        ax.legend(
            handles=handles, title="Method", bbox_to_anchor=(1.02, 1), loc="upper left"
        )

        plt.tight_layout()
        plt.savefig(f"{file_name}_search_only.png")
        plt.close(fig)


process_data = [
    [
        "/export/home1/ltarun/cuEFIM/datasets/chainstore.txt",
        [2100000, 2200000, 2300000, 2400000, 2500000],
    ],
    [
        "/export/home1/ltarun/cuEFIM/datasets/ecommerce_retail.txt",
        [750000, 760000, 770000, 780000, 790000],
    ],
    [
        "/export/home1/ltarun/cuEFIM/datasets/accidents_utility_spmf.txt",
        [8000000, 9000000, 10000000, 11000000, 12000000, 15000000, 20000000],
    ],
    [
        "/export/home1/ltarun/cuEFIM/datasets/kosarak_utility_spmf.txt",
        [720000, 730000, 740000, 750000, 760000],
    ],
    [
        "/export/home1/ltarun/cuEFIM/datasets/BMS_utility_spmf.txt",
        [2025000, 2030000, 2035000, 2040000, 2045000, 2050000],
    ],
    [
        "/export/home1/ltarun/cuEFIM/datasets/connect_utility_spmf.txt",
        [12000000, 12500000, 13000000, 13500000, 14000000, 14500000],
    ],
]


results_df = load_results()
run_cuDPEFIM(process_data, "DP", repeats=3)
run_cuDPEFIM(process_data, "SP", repeats=3)
run_dphim(process_data, methods=["dphim"], cores=48, repeats=3)
plot_clustered_by_util()
plot_search_only()
