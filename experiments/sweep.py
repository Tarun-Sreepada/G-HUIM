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
    # Read metadata and data
    lines = open(file_path, "r").readlines()
    input_file = lines[0].split(": ")[1].strip()
    min_util = int(lines[1].split(": ")[1].strip())
    dphim_time = float(lines[3].split(": ")[1].strip())
    df = pd.read_csv(file_path, skiprows=6)
    
    # Cast types
    df["block_size"] = df["block_size"].astype(int)
    df["num_blocks"] = df["num_blocks"].astype(int)
    df["search_time"] = df["search_time"].astype(float)
    
    # Normalize coordinates for coloring
    x = df["block_size"]
    y = df["num_blocks"]
    z = df["search_time"]
    x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x * 0
    y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() != y.min() else y * 0
    z_norm = (z - z.min()) / (z.max() - z.min()) if z.max() != z.min() else z * 0
    
    # Map normalized coords to RGB: X->R, Z->G, Y->B
    colors = [
        f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
        for r, g, b in zip(x_norm, z_norm, y_norm)
    ]
    
    # Create 3D scatter plot
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(size=6, color=colors)
            )
        ]
    )
    
    # Color the axes
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Block Size", color="red"),
            yaxis=dict(title="Num Blocks", color="blue"),
            zaxis=dict(title="Search Time (ms)", color="green"),
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    
    # Save interactive HTML
    fig.write_html(f"{file_path.split('.')[0]}_3d_plot.html")
    print("Saved interactive 3D plot to ", f"{file_path.split('.')[0]}_3d_plot.html")



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