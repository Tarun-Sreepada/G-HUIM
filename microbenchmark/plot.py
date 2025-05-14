import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import io

EXE_PATH = "./thread_usage"

# Target work values
work_vals = [8192, 32 * 1024, 128 * 1024, 512 * 1024]
sync_times = 128  # fixed

def get_df(work_done: int, sync_times: int) -> pd.DataFrame:
    cmd = [EXE_PATH, f"{work_done}", f"{sync_times}"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return pd.read_csv(io.StringIO(result.stdout))


first = True
for work_done in work_vals:
    df = get_df(work_done, sync_times)

    # ---- Linear plot ----
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot(df.block_size, df.mixed_ms, marker="o", linestyle="-")
    ax.set_xlabel("Threads per Block")

    if first:
        ax.set_ylabel("Execution Time (ms)")

    ax.set_xlim(0, df.block_size.max() * 1.1)
    ax.set_ylim(0, df.mixed_ms.max() * 1.1)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    # ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f"mixed_w{work_done}_s{sync_times}_linear.jpg", dpi=500, bbox_inches="tight")
    plt.close(fig)

    # ---- Log plot ----
    fig, ax = plt.subplots(figsize=(2, 3))
    ax.plot(df.block_size, df.mixed_ms, marker="o", linestyle="-")
    ax.set_xscale("log", base=2)
    # ax.set_xlabel("Threads per Block")
    # ax.set_ylabel("Execution Time (ms)")
    # add the first tick label
    ax.xaxis.set_major_locator(plt.LogLocator(base=2, numticks=10))

    if first:
        # ax.set_ylabel("Execution Time (ms)")
        first = False
    ax.set_xlim(df.block_size.min() * 0.9, df.block_size.max() * 1.1)
    ax.set_ylim(0, df.mixed_ms.max() * 1.1)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    # ax.legend(loc="upper left")
    fig.tight_layout()

    # remove all white space
    

    fig.savefig(f"mixed_w{work_done}_s{sync_times}_log.jpg", dpi=500, bbox_inches="tight")
    plt.close(fig)
