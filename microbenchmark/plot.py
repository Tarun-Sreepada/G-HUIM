import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import io

# path to your executable
EXE_PATH = "./thread_usage"

# ranges for work_done and sync_times
work_vals = [2**i for i in range(10, 25)]   # 4, 8, …, 1<<20
sync_vals = [2**i for i in range(0, 15)]   

def get_df(work_done: int, sync_times: int) -> pd.DataFrame:
    """
    Runs the executable with the given parameters, captures its stdout,
    and returns a DataFrame.
    Assumes the exe prints CSV to stdout with a header row.
    """
    # build your command-line args; adjust flags as needed
    cmd = [EXE_PATH,
           f"{work_done}",
           f"{sync_times}"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    csv_out = result.stdout
    # if your exe outputs space- or tab-delimited, specify sep='\s+' or '\t'
    return pd.read_csv(io.StringIO(csv_out))


first = True

for work_done in work_vals:
    for sync_times in sync_vals:
        
        if work_done < sync_times:
            # skip if work done is less than sync times
            continue
        
        df = get_df(work_done, sync_times)

        # ---- LINEAR PLOT ----
        fig, ax1 = plt.subplots(figsize=(5, 5))
        ax1.plot(df.block_size, df.compute_ms, marker="o", linestyle="-", label="Compute-only")
        ax1.plot(df.block_size, df.sync_ms,    marker="s", linestyle="--", label=f"Sync-only ({sync_times})")
        ax1.plot(df.block_size, df.mixed_ms,   marker="^", linestyle=":", label="Mixed")
        ax1.set_xlabel("Threads per Block")
        ax1.set_ylabel("Execution Time (ms)")
        ax1.set_xlim(0, df.block_size.max() * 1.1)
        
        ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
        

            

        ax2 = ax1.twinx()
        ax2.plot(df.block_size, [work_done]*len(df), color="gray",
                 linestyle="-.", marker="x", label="Total Work")
        ax2.set_ylabel("Total Work (Iterations)")
        
        if first:
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")
            
            

        fig.tight_layout()
        fig.savefig(f"w_{work_done}_{sync_times}_linear.jpg", dpi=500)
        plt.close(fig)

        # ---- LOG PLOT ----
        fig, ax1 = plt.subplots(figsize=(5, 5))
        ax1.plot(df.block_size, df.compute_ms, marker="o", linestyle="-", label="Compute-only")
        ax1.plot(df.block_size, df.sync_ms,    marker="s", linestyle="--", label=f"Sync-only ({sync_times})")
        ax1.plot(df.block_size, df.mixed_ms,   marker="^", linestyle=":", label="Mixed")
        ax1.set_xscale("log", base=2)
        ax1.set_xlabel("Threads per Block (log₂)")
        ax1.set_ylabel("Execution Time (ms)")
        ax1.set_xlim(df.block_size.min() * 0.9, df.block_size.max() * 1.1)
        ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
        # ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(df.block_size, [work_done]*len(df), color="gray",
                 linestyle="-.", marker="x", label="Total Work")
        ax2.set_ylabel("Total Work (Iterations)")
        # ax2.legend(loc="upper right")
        
        if first:
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")
            first = False

        fig.tight_layout()
        fig.savefig(f"w_{work_done}_{sync_times}_log.jpg", dpi=500)
        plt.close(fig)
