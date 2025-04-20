import os
import subprocess
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import plotly.graph_objects as go

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





EXECUTABLE = "/export/home1/ltarun/cuDPEFIM/build/cuDPEFIM"

TOTAL_MEMORY = "78G"
MANAGED = False
METHOD = "DP"


def run_cuDPEFIM_once(input_file, min_util, block_size, num_block):
    cmd = [
        EXECUTABLE,
        "--input", input_file,
        "--minutil", str(min_util),
        "--threads", "24",
        "--block_size", str(block_size),
        "--num_blocks", str(num_block),
        "--total_memory", TOTAL_MEMORY,
        "--managed", str(MANAGED).lower(),
        "--method", METHOD,
    ]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        p, b, s, patterns = parse_metrics_cu(proc.stdout) 
        return s  # search time only
    except subprocess.CalledProcessError as e:
        print(f"Error running {cmd}: {e}")
        return None
    
def run_dphim_once(input_file, min_util, method, cores):
    cmd = [
        "/export/home1/ltarun/dphim/build/run",
        "-i", input_file,
        "-m", str(min_util),
        "-s", method,
        "-t", str(cores),
        "-o", "/dev/null",
    ]

    print("Running command:", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        p, b, s, patterns = parse_metrics_dphim(proc.stdout)
        return s  # search time only
    except subprocess.CalledProcessError as e:
        print(f"Error running {cmd}: {e}")
        return None


def plot_3d(file_path: str):
    # --- 1) Read metadata + data ---
    with open(file_path, "r") as f:
        lines = f.readlines()
    input_file = lines[0].split(": ", 1)[1].strip()
    # min utility (not used in plotting)
    min_util = int(lines[1].split(": ", 1)[1].strip())
    # DPHIM baseline time to compute speedup
    dphim_time = float(lines[3].split(": ", 1)[1].strip())

    # data starts at line 7 (0‑indexed row 6)
    df = pd.read_csv(file_path, skiprows=6)
    df["block_size"] = df["block_size"].astype(int)
    df["num_blocks"] = df["num_blocks"].astype(int)
    df["search_time"] = df["search_time"].astype(float)

    # --- 2) Compute speedup relative to smallest block size and smallest num blocks ---
    min_block_size = df["block_size"].min()
    min_num_blocks = df["num_blocks"].min()

    df["speedup"] = (
        df["search_time"]
        / df.loc[
            (df["block_size"] == min_block_size)
            & (df["num_blocks"] == min_num_blocks),
            "search_time",
        ].values[0]
    )


    # --- 3) Helper to normalize and map to "rgb(r,g,b)" strings ---
    def make_rgb(xs, ys, zs):
        xn = (xs - xs.min()) / (xs.max() - xs.min()) if xs.max() != xs.min() else 0
        yn = (ys - ys.min()) / (ys.max() - ys.min()) if ys.max() != ys.min() else 0
        zn = (zs - zs.min()) / (zs.max() - zs.min()) if zs.max() != zs.min() else 0
        return [
            f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
            for r, g, b in zip(xn, zn, yn)
        ]

    colors_time     = make_rgb(df["block_size"], df["num_blocks"], df["search_time"])
    colors_speedup = make_rgb(df["block_size"], df["num_blocks"], df["speedup"])

    # --- 4) First 3D scatter: search_time ---
    fig1 = go.Figure(
        go.Scatter3d(
            x=df["block_size"],
            y=df["num_blocks"],
            z=df["search_time"],
            mode="markers",
            marker=dict(size=6, color=colors_time)
        )
    )
    fig1.update_layout(
        scene=dict(
            xaxis=dict(title="Block Size", color="red"),
            yaxis=dict(title="Num Blocks", color="blue"),
            zaxis=dict(title="Search Time", color="green")
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    out_prefix = input_file.split("/")[-1].rsplit(".", 1)[0]
    fig1.write_html(f"{out_prefix}_search_time_plot.html")

    # --- 5) Second 3D scatter: speedup ---
    fig2 = go.Figure(
        go.Scatter3d(
            x=df["block_size"],
            y=df["num_blocks"],
            z=df["speedup"],
            mode="markers",
            marker=dict(size=6, color=colors_speedup)
        )
    )
    fig2.update_layout(
        scene=dict(
            xaxis=dict(title="Block Size", color="red"),
            yaxis=dict(title="Num Blocks", color="blue"),
            zaxis=dict(title="Speedup", color="green")
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    fig2.write_html(f"{out_prefix}_speedup_plot.html")

def sweep(input_file, min_util, block_sizes, num_blocks):

    save_file = input_file.split("/")[-1].split(".")[0] + "_sweep.csv"
    print("Saving to", save_file)

    if os.path.exists(save_file):
        # print("File already exists, deleting...")
        # os.remove(save_file)
        print("File already exists, skipping...")

        plot_3d(save_file)

        return

    
    with open(save_file, "w") as f:
        f.write("File: " + input_file.split("/")[-1] + "\n")
        f.write("Minimum Utility: " + str(min_util) + "\n")

        f.write("DPHIM Baseline\n")
        s = run_dphim_once(input_file, min_util, "dphim", 48)
        f.write(f"search_time: {s}\n\n")

        f.flush()
        os.fsync(f.fileno())

        f.write("cuDPEFIM Sweep\n")
        f.write("block_size,num_blocks,search_time\n")
        for block_size in block_sizes:
            for num_block in num_blocks:
                search_time = run_cuDPEFIM_once(input_file, min_util, block_size, num_block)
                if search_time is not None:
                    f.write(f"{block_size},{num_block},{search_time}\n")
                    print(f"block_size: {block_size}, num_blocks: {num_block}, search_time: {search_time}")
                else:
                    print(f"Failed to run with block_size: {block_size}, num_blocks: {num_block}")
                f.flush()
                os.fsync(f.fileno())

    plot_3d(save_file)



# Sweep settings
block_sizes = [2**i for i in range(5, 11)]  # 1,2,4,...,1024
num_blocks = [1, 2, 4, 8, 16, 32, 64, 114]

process_data = [
    [
        "/export/home1/ltarun/cuEFIM/datasets/chainstore.txt",
        4000000,
    ],
    [
        "/export/home1/ltarun/cuEFIM/datasets/ecommerce_retail.txt",
        700000,
    ],
    [
        "/export/home1/ltarun/cuEFIM/datasets/accidents_utility_spmf.txt",
        10000000,
    ],
    [
        "/export/home1/ltarun/cuEFIM/datasets/kosarak_utility_spmf.txt",
        740000,
    ],
    [
        "/export/home1/ltarun/cuEFIM/datasets/BMS_utility_spmf.txt",
        2020000,
    ],
    [
        "/export/home1/ltarun/cuEFIM/datasets/connect_utility_spmf.txt",
        12000000,
    ],
]


for input_file, min_util in process_data:
    print("Processing", input_file)
    sweep(input_file, min_util, block_sizes, num_blocks)
    print("Finished processing", input_file)