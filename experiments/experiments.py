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

def plot_total(csv_path="combined_results.csv"):
    if not os.path.exists(csv_path):
        print(f"CSV file '{csv_path}' not found.")
        return

    # ─── load & rename ──────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "Parse_Time":  "Parse",
        "TWU_Time":    "TWU",
        "Build_Time":  "Build",
        "Search_Time": "Search",
    })
    df["Minimum_Utility"] = df["Minimum_Utility"].astype(int)
    df["AlgoMeth"] = df["Algorithm"] + "_" + df["Method"]

    # ─── sort methods so that fixed_{num} come before adaptive ─────────────────
    def sort_key(x):
        parts = x.split("_")
        algo, suffix = parts[0], parts[-1].lower()
        # group all methods with numeric suffix as "fixed", adaptive last
        if suffix.isdigit():
            num = int(suffix)
        elif suffix == "adaptive":
            num = float("inf")
        else:
            num = 0
        # primary sort by algorithm if you still want DPHIM first, etc.
        # remove or tweak this if you only care about the fixed/adaptive ordering:
        if algo == "DPHIM":
            grp = 0
        elif algo == "cuDPEFIM":
            grp = 1
        else:
            grp = 2
        return (grp, num)

    all_methods = sorted(df["AlgoMeth"].unique(), key=sort_key)
    n_methods   = len(all_methods)

    # ─── styling maps ──────────────────────────────────────────────────────────
    phases = ["Parse", "TWU", "Build", "Search"]
    phase_colors = {
        "Parse":  "#2ca02c",
        "TWU":    "#1f77b4",
        "Build":  "#ffdd57",
        "Search": "#ff7f0e",
    }
    linestyles = ["-", "--", ":", "-."]
    hatches     = ["/", "\\", "x", "o", ".", "*", "+", "-"]
    method_style = {
        m: {
            "linestyle": linestyles[i % len(linestyles)],
            "hatch":     hatches[i % len(hatches)],
        }
        for i, m in enumerate(all_methods)
    }

    for file_name in sorted(df["File"].unique()):
        df_file = df[df["File"] == file_name]
        utils   = sorted(df_file["Minimum_Utility"].unique())
        n_utils = len(utils)
        x       = np.arange(n_utils)

        total_width = 0.8
        bar_width   = total_width / n_methods

        # ─── compute “gap only at the two transitions” ──────────────────────────
        gap_indices = set()
        for i in range(n_methods - 1):
            curr = all_methods[i].lower()
            nxt  = all_methods[i+1].lower()
            # after any “dphim_*” → first fixed_{num}
            if "dphim"   in curr and nxt.endswith(tuple(str(d) for d in range(10))):
                gap_indices.add(i)
            # after last fixed_{num} → adaptive
            if curr.endswith(tuple(str(d) for d in range(10))) and "adaptive" in nxt:
                gap_indices.add(i)

        gap_width = bar_width * 0.3
        extra     = [0.0] * n_methods
        acc       = 0.0
        for i in range(1, n_methods):
            if (i - 1) in gap_indices:
                acc += gap_width
            extra[i] = acc

        # ─── draw ──────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(6 * n_utils, 6))
        # leave 25% margin on right for method legend
        fig.subplots_adjust(right=0.75)

        for j, util in enumerate(utils):
            for i, method in enumerate(all_methods):
                xpos   = x[j] + (i - (n_methods-1)/2) * bar_width + extra[i]
                bottom = 0
                style  = method_style[method]
                for phase in phases:
                    row    = df_file[
                        (df_file["Minimum_Utility"] == util) &
                        (df_file["AlgoMeth"]         == method)
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

        # ─── axes labels & title ───────────────────────────────────────────────
        ax.set_xticks(x)
        ax.set_xticklabels([str(u) for u in utils], rotation=45, ha="right")
        ax.set_xlabel("Minimum Utility (int)")
        ax.set_ylabel("Execution Time (ms)")
        ax.set_title(f"Parse/Build/Search by Utility for '{file_name}'")

        # ─── phase legend (colored blocks) ────────────────────────────────────
        phase_handles = [Patch(facecolor=phase_colors[p], label=p) for p in phases]
        leg1 = ax.legend(
            handles=phase_handles,
            title="Phase",
            loc="upper left",
        )
        ax.add_artist(leg1)

        # ─── method legend (hatches & lines), bigger boxes ───────────────────
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
        ax.legend(
            handles=method_handles,
            title="Method",
            loc="upper right",
            handlelength=3,    # wider
            handleheight=3,    # taller
            fontsize="small",
            frameon=False,
        )

        plt.savefig(f"{file_name}_total.png", bbox_inches="tight")
        plt.close(fig)


def plot_phase(csv_path="combined_results.csv"):
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

    # phase definitions & colors
    phase_cols = ["TWU_Time", "Build_Time", "Search_Time"]
    phase_labels = {"TWU_Time": "TWU", "Build_Time": "Build", "Search_Time": "Search"}
    phase_colors = {
        "TWU_Time": "#1f77b4",
        "Build_Time": "#ffdd57",
        "Search_Time": "#ff7f0e",
    }

    # === your styles ===
    linestyles = ["-", "--", ":", "-."]
    hatches = ["/", "\\", "x", "o", ".", "*", "+", "-"]
    method_style = {
        m: {
            "linestyle": linestyles[i % len(linestyles)],
            "hatch": hatches[i % len(hatches)],
        }
        for i, m in enumerate(all_methods)
    }
    # ====================

    # say you want ~1 inch per (util × methods) grouping:

    for file_name in sorted(df["File"].unique()):
        df_file = df[df["File"] == file_name]
        utils = sorted(df_file["Minimum_Utility"].unique())
        x = np.arange(len(utils))

        # --- 1) Stacked bar chart of phases ---
        n_utils = len(utils)
        n_methods = len(all_methods)
        width = n_utils * n_methods * 1.0
        height = 6
        fig, ax = plt.subplots(figsize=(width, height))
        for j, util in enumerate(utils):
            for i, method in enumerate(all_methods):
                xpos = x[j] + (i - (n_methods - 1) / 2) * bar_width
                row = df_file[
                    (df_file["Minimum_Utility"] == util)
                    & (df_file["AlgoMeth"] == method)
                ]
                if row.empty:
                    continue

                bottom = 0
                for ph in phase_cols:
                    h = row[ph].iloc[0]
                    ax.bar(
                        xpos,
                        h,
                        bar_width,
                        bottom=bottom,
                        color=phase_colors[ph],
                        edgecolor="black",
                        hatch=method_style[method]["hatch"],
                        label=phase_labels[ph] if (j == 0 and i == 0) else None,
                    )
                    bottom += h

        ax.set_xticks(x)
        ax.set_xticklabels([str(u) for u in utils], rotation=45, ha="right")
        ax.set_xlabel("Minimum Utility")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"Phase Breakdown for '{file_name}'")
        handles = [
            Patch(facecolor=phase_colors[ph], edgecolor="black", label=phase_labels[ph])
            for ph in phase_cols
        ]

        ax.legend(
            handles=handles, title="Phase", bbox_to_anchor=(1.02, 1), loc="upper left"
        )
        # plt.tight_layout()

        phase_handles = [
            Patch(facecolor=phase_colors[ph], edgecolor="black", label=phase_labels[ph])
            for ph in phase_cols
        ]
        phase_legend = ax.legend(
            handles=phase_handles,
            title="Phase",
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
        )

        # now add a second legend for methods (using hatches)
        method_handles = [
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch=method_style[m]["hatch"],
                label=m,
            )
            for m in all_methods
        ]

        # add the legend for methods with bigger text + bigger markers
        ax.legend(
            handles=method_handles,
            title="Method",
            title_fontsize=14,  # make the legend title bigger
            fontsize=12,  # make the text labels bigger
            markerscale=2,  # scale up marker (dot/line) sizes
            handlelength=2.5,  # length of the line or box
            handletextpad=1.0,  # padding between marker and text
            bbox_to_anchor=(1.02, 0.5),
            loc="upper left",
        )
        ax.add_artist(phase_legend)  # make sure the phase legend stays
        ax.legend(
            handles=method_handles,
            title="Method",
            bbox_to_anchor=(1.02, 0.5),
            loc="upper left",
        )

        plt.savefig(f"{file_name}_phases.png")
        plt.close(fig)

def plot_cuDPEFIM_speedup_by_file(csv_path="combined_results.csv"):
    if not os.path.exists(csv_path):
        print(f"CSV file '{csv_path}' not found.")
        return

    # 1) load + compute total time
    df = pd.read_csv(csv_path)
    df["Minimum_Utility"] = df["Minimum_Utility"].astype(int)
    df["Total_Time"] = (
        df["Parse_Time"] + df["TWU_Time"] + df["Build_Time"] + df["Search_Time"]
    )

    # 2) keep only cuDPEFIM rows
    df_cu = df[df["Algorithm"] == "cuDPEFIM"]

    # 3) pick baseline once (fixed_32 or SP_fixed_32)
    baselines = ["fixed_32", "SP_fixed_32"]
    chosen_baseline = next((b for b in baselines if b in df_cu["Method"].unique()), None)
    if chosen_baseline is None:
        print("Baseline 'fixed_32' or 'SP_fixed_32' not found.")
        return

    # 4) for each file, build pivot, compute speedup, plot
    for file_name in sorted(df_cu["File"].unique()):
        df_file = df_cu[df_cu["File"] == file_name]

        # pivot with mean to collapse duplicates
        pivot = df_file.pivot_table(
            index="Minimum_Utility",
            columns="Method",
            values="Total_Time",
            aggfunc="mean"
        )

        # keep only fixed_* and adaptive
        methods = [m for m in pivot.columns
                   if m.startswith("SP_fixed_") or m.startswith("SP_adaptive")]
        if chosen_baseline not in methods:
            methods.insert(0, chosen_baseline)
        pivot = pivot[methods]
        print(f"Filtered pivot for '{file_name}':")
        print(pivot)

        # speedup ratio
        # speedup = pivot.div(pivot[chosen_baseline], axis=0) this is like 80% we want 1.25x
        speedup = pivot[methods].copy()
        for method in methods:
            speedup[method] = pivot[chosen_baseline] / pivot[method]
        speedup = speedup.rename(columns={chosen_baseline: "Baseline"})
        print(f"Speedup for '{file_name}':")
        print(speedup)
        print()

        # plot
        plt.figure(figsize=(8, 5))
        cmap = plt.get_cmap("tab10")
        lines = [m for m in methods if m != chosen_baseline]
        print(f"Methods for '{file_name}': {lines}")
        colors = cmap(np.linspace(0, 1, len(lines)))

        for color, method in zip(colors, lines):
            plt.plot(
                speedup.index,
                speedup[method],
                marker="o",
                label=method,
                color=color
            )

        # baseline reference
        plt.axhline(1.0, linestyle="--", color="gray", linewidth=1)

        plt.xlabel("Minimum Utility (int)")
        plt.ylabel(f"Speedup over {chosen_baseline}")
        plt.title(f"cuDPEFIM Speedup for '{file_name}' (baseline={chosen_baseline})")
        plt.legend(title="Method", loc="best", fontsize="small")
        plt.grid(True)
        plt.tight_layout()

        out_name = f"Speedup_{file_name}_{chosen_baseline}.png"
        plt.savefig(out_name)
        plt.close()

        print(f"→ Saved speedup plot for '{file_name}' as: {out_name}")

process_data = [
    [
        "/export/home1/ltarun/cuEFIM/datasets/chainstore.txt",
        [2000000, 2050000, 2100000, 2150000, 2200000, 2500000, 3000000],
    ],
    [
        "/export/home1/ltarun/cuEFIM/datasets/accidents_utility_spmf.txt",
        [8000000, 9000000, 10000000, 11000000, 12000000, 15000000, 20000000],
    ],
    [
        "/export/home1/ltarun/cuEFIM/datasets/kosarak_utility_spmf.txt",
        [760000, 770000, 780000, 790000, 800000, 900000, 1000000],
    ],
    # [
    #     "/export/home1/ltarun/cuEFIM/datasets/BMS_utility_spmf.txt",
    #     [2025000, 2030000, 2035000, 2040000, 2045000],
    # ],
    [
        "/export/home1/ltarun/cuEFIM/datasets/connect_utility_spmf.txt",
        [10000000, 10500000, 11000000, 11500000, 12000000, 15000000, 16000000],
    ],
    [
        "/export/home1/ltarun/cuEFIM/datasets/chess_utility.txt",
        [300000, 310000, 320000, 330000, 340000, 350000],
    ],
    [
        "/export/home1/ltarun/cuEFIM/datasets/mushroom.txt",
        [35000, 40000, 45000, 50000, 55000],
    ],
    [
        "/export/home1/ltarun/cuEFIM/datasets/pumsb_utility.txt",
        [11000000, 11100000, 11200000, 11300000, 11400000],
    ],
    [
        "/export/home1/ltarun/cuEFIM/datasets/retail.txt",
        [1345, 1346, 1347, 1348, 1349],
    ]
]


results_df = load_results()
block_sizes = [32, 64, 128, 256, 512, 1024]
for bs in block_sizes:
    run_cuDPEFIM(process_data, "SP", adaptive=False, block_size=bs, repeats=1)
run_cuDPEFIM(process_data, "SP", adaptive=True, repeats=1)
run_dphim(process_data, methods=["dphim"], cores=48, repeats=1)
plot_total()
plot_cuDPEFIM_speedup_by_file()
# plot_phase()
