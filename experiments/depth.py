import subprocess
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

def run_iteration_times(input_file, minutil, adaptive=False, block_size=None, total_memory="78G", method="SP"):
    cmd = [
        "/export/home1/ltarun/cuDPEFIM/build/cuDPEFIM",
        "--input", input_file,
        "--minutil", str(minutil),
        "--method", method,
        "--total_memory", total_memory,
        "--output", "/dev/null"
    ]
    if adaptive:
        cmd += ["--adaptive", "true"]
    else:
        cmd += ["--adaptive", "false", "--block_size", str(block_size)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    times = {}
    for line in result.stdout.splitlines():
        # print(line)
        if line.startswith("Iteration"):
            parts = line.split()
            it = int(parts[1])
            t = float(parts[2])
            times[it] = t
    return times



import matplotlib.pyplot as plt
def plot_for_configs(input_file, minutil, block_sizes, total_memory="78G", method="SP"):
    configs = [("adaptive", None)] + [(f"static_{bs}", bs) for bs in block_sizes]
    all_iterations = set()
    results = {}

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

    # Gather iteration times for each config
    for label, bs in configs:
        times = run_iteration_times(
            input_file, minutil,
            adaptive=(label == "adaptive"),
            block_size=bs,
            total_memory=total_memory,
            method=method
        )
        results[label] = times
        all_iterations.update(times.keys())

    iterations = sorted(all_iterations)
    plt.figure(figsize=(4, 4))

    linestyles = ['-', '--', ':', '-.']
    legend_handles = []
    adaptive_times = [results["adaptive"].get(i, 0) for i in iterations]
    max_adaptive_time = max(adaptive_times) if adaptive_times else 0
    height_cutoff = 1.5 * max_adaptive_time

    
    for idx, (label, bs) in enumerate(configs):
        y = [results[label].get(i, 0) for i in iterations]
        style = linestyles[idx % len(linestyles)]

        method_key = f"cuDPEFIM_SP_adaptive" if label == "adaptive" else f"cuDPEFIM_SP_fixed_{bs}"
        display_name = method_nice_names.get(method_key, label)

        line, = plt.plot(iterations, y,
                        label=display_name,
                        linestyle=style,
                        marker='o' if label == "adaptive" else None)

        # Use actual line color from plot
        legend_handles.append(
            Line2D([0], [0],
                linestyle=style,
                marker='o' if label == "adaptive" else None,
                color=line.get_color(),  # fix: apply plot color
                label=display_name)
        )

    plt.ylim(0, height_cutoff)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Time (ms)", fontsize=14)
    plt.tight_layout()

    # add caption
    # plt.figtext(0.5, 0.01, f"Minimum Utility: {minutil}", ha="center", fontsize=14)

    # make all font larger
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontsize(14)

    # Save main plot without legend
    file_base = os.path.splitext(os.path.basename(input_file))[0]
    file_name = f"iteration_times_{file_base}_minutil_{minutil}.png"
    plt.savefig(file_name, dpi=500)
    plt.close()

    # === Create a separate legend figure ===
    fig_leg, ax_leg = plt.subplots(figsize=(max(6, len(legend_handles) * 1.5), 1.5))
    ax_leg.axis("off")
    # leg = ax_leg.legend(handles=legend_handles, loc="center", frameon=False, ncol=len(legend_handles))
    leg = ax_leg.legend(handles=legend_handles, loc="center", frameon=False, ncol=1)
    legend_file = f"legend_{file_base}_minutil_{minutil}.png"
    plt.savefig(legend_file, bbox_inches="tight", pad_inches=0.1, dpi=500)
    plt.close(fig_leg)


# Example usage
if __name__ == "__main__":
    block_sizes = [32, 64, 128, 256, 512, 1024]
    
        
        
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
        #     [740000, 770000, 780000, 790000, 800000, 900000, 1000000],
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
            [9500000, 11300000, 11400000],
        ],
        [
            "/export/home1/ltarun/cuEFIM/datasets/retail.txt",
            [1345, 1346, 1347, 1348, 1349],
        ]
    ]

        
    
    # plot_for_configs(input_path, minutil, block_sizes)
    for input_path, minutils in process_data:
        minutil = minutils[0]
        print(f"Processing {input_path} with minutil {minutil}")
        plot_for_configs(input_path, minutil, block_sizes)
