import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import io

EXE_PATH = "./thread_usage"

import matplotlib

# Configure Matplotlib for LaTeX-friendly output (PGF/PDF).
# Note: rendering PGF with LaTeX requires a LaTeX installation when compiling
# the generated .pgf files into a document. Saving .pdf provides a ready-to-use
# vector figure as well.
matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "pgf.rcfonts": False,
    }
)


def _check_latex_working() -> bool:
    """Return True if a minimal LaTeX run succeeds (used by matplotlib tex rendering).

    Matplotlib with text.usetex=True will attempt to run 'latex' to measure text.
    Some TeX Live installations may miss small packages (e.g., type1ec) and fail.
    We try a minimal run and fall back to disabling usetex on failure.
    """
    import subprocess

    tex = r"""\documentclass{article}
\begin{document}
lp
\end{document}
"""
    try:
        p = subprocess.run(
            ["latex", "-interaction=nonstopmode", "--halt-on-error", "file.tex"],
            input=tex.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except Exception as e:
        return False


# If LaTeX isn't fully available at runtime, disable text.usetex so matplotlib
# won't call out to latex for text measurement (which causes runtime failures
# such as missing type1ec.sty). We still keep saving PGF/PDF files, but matplotlib
# will use its internal font metrics rather than compiling TeX for labels.
if matplotlib.rcParams.get("text.usetex", False):
    if not _check_latex_working():
        matplotlib.rcParams["text.usetex"] = False

                    
work_vals = [8192, 32 * 1024, 128 * 1024, 512 * 1024]
sync_times = 128         

def get_df(work_done: int, sync_times: int) -> pd.DataFrame:
    cmd = [EXE_PATH, f"{work_done}", f"{sync_times}"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return pd.read_csv(io.StringIO(result.stdout))


first = True
for work_done in work_vals:
    df = get_df(work_done, sync_times)

                           
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot(df.block_size, df.mixed_ms, marker="o", linestyle="-")
    ax.set_xlabel("Threads per Block")

    if first:
        ax.set_ylabel("Execution Time (ms)")

    ax.set_xlim(0, df.block_size.max() * 1.1)
    ax.set_ylim(0, df.mixed_ms.max() * 1.1)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
                                 
    fig.tight_layout()
    fig.savefig(f"mixed_w{work_done}_s{sync_times}_linear.jpg", dpi=500, bbox_inches="tight")
    plt.close(fig)

                        
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot(df.block_size, df.mixed_ms, marker="o", linestyle="-")
    ax.set_xscale("log", base=2)
                                        
                                          
                              
    ax.xaxis.set_major_locator(plt.LogLocator(base=2, numticks=10))

    if first:
                                              
        first = False
    ax.set_xlim(df.block_size.min() * 0.9, df.block_size.max() * 1.1)
    ax.set_ylim(0, df.mixed_ms.max() * 1.1)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
                                 
    fig.tight_layout()

                            
    

    fig.savefig(f"mixed_w{work_done}_s{sync_times}_log.jpg", dpi=500, bbox_inches="tight")
    plt.close(fig)
