import pandas as pd
import matplotlib.pyplot as plt

work_done = 1 << 20
sync_times = 1 << 13

df = pd.read_csv("results.csv")

# ---------- linear plot ----------------------------------------------------
fig, ax1 = plt.subplots()

# set figure size
fig.set_size_inches(5, 5)

ax1.plot(df.block_size, df.compute_ms, marker="o", label="compute-only ↓")
ax1.plot(df.block_size, df.sync_ms, marker="o", label=f"sync-only ({sync_times}) ↑")
ax1.plot(
    df.block_size,
    df.mixed_ms,
    marker="o",
    label="mixed (sync interveaved with compute)",
)

ax1.set_xlabel("threads per block")
ax1.set_ylabel("time (ms)")
ax1.set_xlim(0, df.block_size.max() * 1.1)
ax1.set_ylim(0, df.mixed_ms.max() * 1.1)
ax1.grid(True, which="both")
ax1.legend(loc="upper left")

# ---- secondary y-axis (right) --------------------------------------------
ax2 = ax1.twinx()
# plot a ingle line at work_done
ax2.plot(
    df.block_size,
    [work_done] * len(df.block_size),
    color="gray",
    linestyle="--",
    marker="x",
    label="work per block",
)

ax2.set_ylabel("work done (arithmetic)")
ax2.legend(loc="upper right")

fig.tight_layout()
fig.savefig("plot.png", dpi=500)

# ---------- logarithmic plot ----------------------------------------------
fig, ax1 = plt.subplots()

# set figure size
fig.set_size_inches(5, 5)

ax1.plot(df.block_size, df.compute_ms, marker="o", label="compute-only ↓")
ax1.plot(df.block_size, df.sync_ms, marker="o", label=f"sync-only ({sync_times}) ↑")
ax1.plot(
    df.block_size,
    df.mixed_ms,
    marker="o",
    label="mixed (sync interveaved with compute)",
)

ax1.set_xscale("log", base=2)
ax1.set_xlabel("threads per block (log₂)")
ax1.set_ylabel("time (ms)")
xmin = df.block_size.min() * 0.9
xmax = df.block_size.max() * 1.1
ax1.set_xlim(xmin, xmax)
ax1.grid(True, which="both")
ax1.legend(loc="upper left")

# reuse the same work-curve on a twin axis
ax2 = ax1.twinx()
ax2.plot(
    df.block_size,
    [work_done] * len(df.block_size),
    color="gray",
    linestyle="--",
    marker="x",
    label="work per block",
)
ax2.set_ylabel("work done (iterations)")
ax2.legend(loc="upper right")

fig.tight_layout()
fig.savefig("plot_log.png", dpi=500)
