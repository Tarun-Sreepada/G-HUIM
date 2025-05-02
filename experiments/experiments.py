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
    df = df.rename(
        columns={
            "Parse_Time": "Parse",
            "TWU_Time": "TWU",
            "Build_Time": "Build",
            "Search_Time": "Search",
        }
    )
    df["Minimum_Utility"] = df["Minimum_Utility"].astype(int)
    df["AlgoMeth"] = df["Algorithm"] + "_" + df["Method"]

    # ─── sort methods ────────────────────────────────────────────────────────────
    def sort_key(x):
        parts = x.split("_")
        algo, suffix = parts[0], parts[-1].lower()
        if suffix.isdigit():
            num = int(suffix)
        elif suffix == "adaptive":
            num = float("inf")
        else:
            num = 0
        if algo == "DPHIM":
            grp = 0
        elif algo == "cuDPEFIM":
            grp = 1
        else:
            grp = 2
        return (grp, num)

    all_methods = sorted(df["AlgoMeth"].unique(), key=sort_key)
    print("Methods:", all_methods)
    n_methods = len(all_methods)

    method_nice_names = {
        "DPHIM_DPHIM_dphim": "DPHIM",
        "cuDPEFIM_SP_fixed_32": "cuDPEFIM Block size 32",
        "cuDPEFIM_SP_fixed_64": "cuDPEFIM Block size 64",
        "cuDPEFIM_SP_fixed_128": "cuDPEFIM Block size 128",
        "cuDPEFIM_SP_fixed_256": "cuDPEFIM Block size 256",
        "cuDPEFIM_SP_fixed_512": "cuDPEFIM Block size 512",
        "cuDPEFIM_SP_fixed_1024": "cuDPEFIM Block size 1024",
        "cuDPEFIM_SP_adaptive": "cuDPEFIM Adaptive",
    }

    # ─── styles ─────────────────────────────────────────────────────────────────
    phases = ["Parse", "TWU", "Build", "Search"]
    phase_colors = {
        "Parse": "#2ca02c",
        "TWU": "#1f77b4",
        "Build": "#ffdd57",
        "Search": "#ff7f0e",
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

    # ─── per file ───────────────────────────────────────────────────────────────
    for file_name in sorted(df["File"].unique()):
        df_file = df[df["File"] == file_name]
        utils = sorted(df_file["Minimum_Utility"].unique())
        if not utils:
            continue

        min_util = utils[0]  # <- minimum utility

        ## ─────────────────────────────
        ## 1. FIRST IMAGE: min util stacked bars for all methods
        ## ─────────────────────────────
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.subplots_adjust(right=0.95)

        total_width = 0.8
        bar_width = total_width / n_methods

        gap_indices = set()
        for i in range(n_methods - 1):
            curr = all_methods[i].lower()
            nxt = all_methods[i + 1].lower()
            if "dphim" in curr and nxt.endswith(tuple(str(d) for d in range(10))):
                gap_indices.add(i)
            if curr.endswith(tuple(str(d) for d in range(10))) and "adaptive" in nxt:
                gap_indices.add(i)

        gap_width = bar_width * 0.3
        extra = [0.0] * n_methods
        acc = 0.0
        for i in range(1, n_methods):
            if (i - 1) in gap_indices:
                acc += gap_width
            extra[i] = acc

        x = np.array([0])

        for i, method in enumerate(all_methods):
            xpos = x[0] + (i - (n_methods - 1) / 2) * bar_width + extra[i]
            bottom = 0
            style = method_style[method]
            row = df_file[
                (df_file["Minimum_Utility"] == min_util)
                & (df_file["AlgoMeth"] == method)
            ]
            if row.empty:
                continue
            for phase in phases:
                height = row[phase].iloc[0]
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
                )
                bottom += height

        ax.set_xticks([])
        # ax.set_xlabel(f"Minimum Utility = {min_util}", fontsize=14)
        ax.set_ylabel("Execution Time (ms)", fontsize=14)
        # ax.set_title(f"All Methods at Minimum Utility for '{file_name}'")
        ax.grid(True)

                # make all font larger
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # set label font size
        for label in ax.get_xticklabels():
            label.set_fontsize(14)
        for label in ax.get_yticklabels():
            label.set_fontsize(14)
            
 

        plt.savefig(f"{file_name}_min_util.png", bbox_inches="tight", dpi=500)
        plt.close(fig)

        ## ─────────────────────────────
        ## 2. SECOND IMAGE: adaptive methods over min_util
        ## ─────────────────────────────

        # df_adaptive = df_file[df_file["AlgoMeth"].str.lower().str.contains("adaptive")]
        # if df_adaptive.empty:
        #     continue
        # adaptive_methods = sorted(df_adaptive["AlgoMeth"].unique(), key=sort_key)

        # n_adaptive = len(adaptive_methods)
        # total_width = 0.8
        # bar_width = total_width / n_adaptive

        # utils = sorted(df_adaptive["Minimum_Utility"].unique())
        # x_base = np.arange(len(utils))

        # fig, ax = plt.subplots(figsize=(4, 4))

        # util_to_idx = {u: idx for idx, u in enumerate(utils)}

        # for i, method in enumerate(adaptive_methods):
        #     style = method_style.get(method, {"linestyle": "-", "hatch": "/"})
        #     df_meth = df_adaptive[df_adaptive["AlgoMeth"] == method]

        #     xpos = [
        #         util_to_idx[u] + (i - (n_adaptive - 1) / 2) * bar_width
        #         for u in df_meth["Minimum_Utility"].values
        #     ]

        #     bottom = np.zeros_like(xpos, dtype=float)

        #     for phase in phases:
        #         heights = df_meth[phase].values
        #         ax.bar(
        #             xpos,
        #             heights,
        #             bar_width,
        #             bottom=bottom,
        #             color=phase_colors[phase],
        #             edgecolor="black",
        #             linewidth=1,
        #             hatch=style["hatch"],
        #         )
        #         bottom += heights

        # ax.set_xticks(x_base)
        # ax.set_xticklabels([str(u) for u in utils], rotation=45, ha="right")
        # ax.set_xlabel("Minimum Utility", fontsize=14)
        # ax.set_ylabel("Execution Time (ms)", fontsize=14)
        # # ax.set_title(f"Adaptive Methods over Utility for '{file_name}'")
        # ax.grid(True)

        #         # make all font larger
        # plt.xticks(fontsize=14)
        # plt.yticks(fontsize=14)
        # # set label font size
        # for label in ax.get_xticklabels():
        #     label.set_fontsize(14)
        # for label in ax.get_yticklabels():
        #     label.set_fontsize(14)
            


        # plt.savefig(f"{file_name}_adaptive_vs_util.png", bbox_inches="tight", dpi=500)
        # plt.close(fig)
        df_adaptive = df_file[df_file["AlgoMeth"].str.lower().str.contains("adaptive")]
        if df_adaptive.empty:
            continue
        adaptive_methods = sorted(df_adaptive["AlgoMeth"].unique(), key=sort_key)

        n_adaptive = len(adaptive_methods)
        total_width = 0.8
        bar_width = total_width / n_adaptive

        utils = sorted(df_adaptive["Minimum_Utility"].unique())
        x_base = np.arange(len(utils))

        fig, ax = plt.subplots(figsize=(4, 4))

        util_to_idx = {u: idx for idx, u in enumerate(utils)}

        for i, method in enumerate(adaptive_methods):
            style = method_style.get(method, {"linestyle": "-", "hatch": "/"})
            df_meth = df_adaptive[df_adaptive["AlgoMeth"] == method]

            xpos = [
                util_to_idx[u] + (i - (n_adaptive - 1) / 2) * bar_width
                for u in df_meth["Minimum_Utility"].values
            ]

            bottom = np.zeros_like(xpos, dtype=float)

            for phase in phases:
                heights = df_meth[phase].values
                ax.bar(
                    xpos,
                    heights,
                    bar_width,
                    bottom=bottom,
                    color=phase_colors[phase],
                    edgecolor="black",
                    linewidth=1,
                    hatch=style["hatch"],
                )
                bottom += heights

        # Determine scaling factor
        log_utils = [np.log10(u) for u in utils]
        rounded_exponents = [int(np.floor(x)) for x in log_utils]

        # Check if all utils have the same or similar exponent (±1 allowed)
        if max(rounded_exponents) - min(rounded_exponents) <= 1 and max(utils) > 10000:
            common_exp = int(np.median(rounded_exponents))
            scale_factor = 10 ** common_exp
            xtick_labels = [f"{u / scale_factor:.1f}".rstrip("0").rstrip(".") for u in utils]
            ax.set_xlabel(f"Minimum Utility (×$10^{common_exp}$)", fontsize=14)
        else:
            xtick_labels = [str(u) for u in utils]
            ax.set_xlabel("Minimum Utility", fontsize=14)

        ax.set_xticks(x_base)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
        ax.set_ylabel("Execution Time (ms)", fontsize=14)
        ax.grid(True)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        for label in ax.get_xticklabels():
            label.set_fontsize(14)
        for label in ax.get_yticklabels():
            label.set_fontsize(14)

        plt.savefig(f"{file_name}_adaptive_vs_util.png", bbox_inches="tight", dpi=500)
        plt.close(fig)


        ## ─────────────────────────────
        ## 3. SEPARATE LEGEND IMAGES
        ## ─────────────────────────────
        # Phase Legend (vertical)
        fig_phase_leg, ax_phase_leg = plt.subplots(figsize=(2, 3))
        phase_handles = [Patch(facecolor=phase_colors[p], label=p) for p in phases]
        ax_phase_leg.axis("off")
        leg1 = ax_phase_leg.legend(
            handles=phase_handles,
            title="Phase",
            loc="center left",
            frameon=False,
        )
        plt.tight_layout()
        plt.savefig(f"{file_name}_phase_legend.png", bbox_inches="tight", pad_inches=0.1, dpi=500)
        plt.close(fig_phase_leg)

        # Method Legend (horizontal)
        fig_method_leg, ax_method_leg = plt.subplots(figsize=(len(all_methods) * 1.5, 2))  # auto-wide
        method_handles = [
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch=method_style[m]["hatch"],
                label=method_nice_names.get(m, m),
            )
            for m in all_methods
        ]
        ax_method_leg.axis("off")
        leg2 = ax_method_leg.legend(
            handles=method_handles,
            title="Method",
            loc="center",
            ncol=len(method_handles),  # all in 1 row horizontally
            frameon=False,
            handlelength=3,
            handleheight=2,
            fontsize="small",
        )
        plt.tight_layout()
        plt.savefig(f"{file_name}_method_legend.png", bbox_inches="tight", pad_inches=0.1, dpi=500)
        plt.close(fig_method_leg)

process_data = [
    # [
    #     "/export/home1/ltarun/cuEFIM/datasets/chainstore.txt",
    #     [2000000, 2050000, 2100000, 2150000, 2200000, 2500000, 3000000],
    # ],
    [
        "/export/home1/ltarun/cuEFIM/datasets/accidents_utility_spmf.txt",
        [8000000, 9000000, 10000000, 11000000, 12000000, 15000000, 20000000],
    ],
    # [
    #     "/export/home1/ltarun/cuEFIM/datasets/kosarak_utility_spmf.txt",
    #     [760000, 770000, 780000, 790000, 800000, 900000, 1000000],
    # ],
    # [
    #     "/export/home1/ltarun/cuEFIM/datasets/BMS_utility_spmf.txt",
    #     [2025000, 2030000, 2035000, 2040000, 2045000],
    # ],
    [
        "/export/home1/ltarun/cuEFIM/datasets/connect_utility_spmf.txt",
        [10000000, 10500000, 11000000, 11500000, 12000000, 15000000, 16000000],
    ],
    # [
    #     "/export/home1/ltarun/cuEFIM/datasets/chess_utility.txt",
    #     [300000, 310000, 320000, 330000, 340000, 350000],
    # ],
    # [
    #     "/export/home1/ltarun/cuEFIM/datasets/mushroom.txt",
    #     [35000, 40000, 45000, 50000, 55000],
    # ],
    [
        "/export/home1/ltarun/cuEFIM/datasets/pumsb_utility.txt",
        [11000000, 11100000, 11200000, 11300000, 11400000],
    ],
    [
        "/export/home1/ltarun/cuEFIM/datasets/retail.txt",
        [1345, 1346, 1347, 1348, 1349],
    ],
]


results_df = load_results()
block_sizes = [32, 64, 128, 256, 512, 1024]
for bs in block_sizes:
    run_cuDPEFIM(process_data, "SP", adaptive=False, block_size=bs, repeats=1)
run_cuDPEFIM(process_data, "SP", adaptive=True, repeats=1)
run_dphim(process_data, methods=["dphim"], cores=48, repeats=1)
plot_total()
# plot_cuDPEFIM_speedup_by_file()
