import argparse
import logging
import os
import re
import subprocess
from typing import Dict, List, Tuple

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


GHUIM_BIN = "../build/G-HUIM"
DPHIM_BIN = "../../dphim/build/run"
PHA_BIN = "../datasets/pha"


ORDERED_COLS = [
    "Algorithm",
    "Method",
    "File",
    "Minimum_Utility",
    "Parse_Time",
    "TWU_Time",
    "Build_Time",
    "Search_Time",
    "Total_Time",
    "Frequent_Patterns",
]


def load_results(csv_path: str) -> pd.DataFrame:
    """Load results from CSV or create a new DataFrame if it doesn't exist."""
    if os.path.exists(csv_path):
        logging.info(f"Loading existing results from {csv_path}")
        return pd.read_csv(csv_path)
    logging.info(f"No results file found. Creating a new DataFrame.")
    return pd.DataFrame(columns=ORDERED_COLS)


def save_results(df: pd.DataFrame, csv_path: str):
    """Save results DataFrame to CSV, ensuring a consistent column order."""

    output_dir = os.path.dirname(csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df[ORDERED_COLS].to_csv(csv_path, index=False)
    logging.info(f"Results saved to {csv_path}")


def update_results(df: pd.DataFrame, entry: Dict, csv_path: str) -> pd.DataFrame:
    """
    Insert or update a result entry, keeping only the one with the best (lowest)
    Total_Time for a given configuration.
    """
    key_cols = ["Algorithm", "Method", "File", "Minimum_Utility"]

    entry["Total_Time"] = sum(
        entry.get(k, 0) for k in ["Parse_Time", "Build_Time", "Search_Time", "TWU_Time"]
    )

    new_row = pd.DataFrame([entry])
    combined_df = pd.concat([df, new_row], ignore_index=True)

    combined_df.sort_values(by=key_cols + ["Total_Time"], ascending=True, inplace=True)

    updated_df = combined_df.drop_duplicates(subset=key_cols, keep="first")

    save_results(updated_df, csv_path)
    return updated_df


def has_run_before(
    df: pd.DataFrame, algorithm: str, method: str, file_name: str, min_util: int
) -> bool:
    """Check if a result for this specific configuration already exists."""
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    mask = (
        (df["Algorithm"] == algorithm)
        & (df["Method"] == method)
        & (df["File"] == base_name)
        & (df["Minimum_Utility"] == min_util)
    )
    return not df[mask].empty


def parse_metrics_cu(output: str) -> Tuple[float, float, float, float, int]:
    """Extract metrics from G-HUIM stdout."""
    metrics = {
        "read": 0,
        "twu": 0,
        "parse_temp": 0,
        "build": 0,
        "mining": 0,
        "patterns": 0,
    }
    iteration_total_ms = 0

    for line in output.splitlines():
        line = re.sub(r"^\[[^]]+\]\s*", "", line)
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split()]
        try:
            if "Read" in line:
                metrics["read"] = float(parts[2])
            elif "CalcTWU" in line:
                metrics["twu"] = float(parts[1])
            elif "Parse" in line:
                metrics["parse_temp"] = float(parts[1])
            elif "Build" in line:
                metrics["build"] = float(parts[1])
            elif "Mining Completed" in line:
                metrics["mining"] = float(parts[2])
            elif "Iteration" in line:
                iteration_total_ms += float(parts[2])
            elif "Number of patterns:" in line:
                metrics["patterns"] = int(parts[-1])
        except (IndexError, ValueError) as e:
            logging.warning(f"Could not parse line '{line}': {e}")

    parse_time = metrics["read"] + metrics["parse_temp"]
    search_time = iteration_total_ms + metrics["mining"]
    return (
        parse_time,
        metrics["twu"],
        metrics["build"],
        search_time,
        metrics["patterns"],
    )


def parse_metrics_dphim(output: str) -> Tuple[float, float, float, float, int]:
    """Extract metrics from DPHIM stdout."""
    metrics = {"parse": 0, "twu": 0, "build": 0, "search": 0, "patterns": 0}
    for line in output.splitlines():
        parts = [p.strip() for p in line.split()]
        if not parts:
            continue
        try:
            if "parse:" in parts:
                metrics["parse"] = int(parts[1])
            elif "calcTWU:" in parts:
                metrics["twu"] = int(parts[1])
            elif "Build:" in parts:
                metrics["build"] = int(parts[1])
            elif "Search:" in parts:
                metrics["search"] = int(parts[1])
            elif "count:" in parts and "itemsets" in parts:
                metrics["patterns"] = int(parts[-1])
        except (IndexError, ValueError) as e:
            logging.warning(f"Could not parse line '{line}': {e}")
    return (
        metrics["parse"],
        metrics["twu"],
        metrics["build"],
        metrics["search"],
        metrics["patterns"],
    )


def parse_metrics_pha(output: str) -> Tuple[float, float, float, float, int]:
    """Extract metrics from PHA stdout."""
    gpu_ms, cpu_ms, patterns = 0.0, 0.0, 0
    for line in output.splitlines():
        try:
            if "Total High-Utility Itemsets Found" in line:
                patterns = int(line.split(":")[-1].strip())
            elif "Total Time on GPU" in line:
                gpu_ms = float(re.search(r"([\d\.]+)", line).group(1))
            elif "Total Time on CPU" in line:
                cpu_ms = float(re.search(r"([\d\.]+)", line).group(1))
        except (AttributeError, IndexError, ValueError) as e:
            logging.warning(f"Could not parse PHA line '{line}': {e}")

    return 0.0, 0.0, cpu_ms, gpu_ms, patterns


def run_experiment(cmd: List[str], parser_func, repeats: int) -> Dict:
    """A generic function to run a command multiple times and return the best result."""
    if parser_func is parse_metrics_cu and "--log-level" not in cmd:
        cmd = [*cmd, "--log-level", "debug"]
    best_run = None
    for i in range(repeats):
        logging.info(f"Run {i + 1}/{repeats}: {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            p, t, b, s, patterns = parser_func(proc.stdout + "\n" + proc.stderr)
            total_time = p + t + b + s

            if best_run is None or total_time < best_run["Total_Time"]:
                best_run = {
                    "Parse_Time": p,
                    "TWU_Time": t,
                    "Build_Time": b,
                    "Search_Time": s,
                    "Frequent_Patterns": patterns,
                    "Total_Time": total_time,
                }
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed with error:\n{e.stderr}")
            return None
    return best_run


def run_all_experiments(
    process_data, repeats: int, csv_path: str, gpu_memory: str = "64G"
):
    """Main function to orchestrate all algorithm benchmarks."""
    df = load_results(csv_path)

    block_sizes = [32, 64, 128, 256, 512, 1024]

    for file_path, min_utils in process_data:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        for min_util in min_utils:

            method_label = "SP_adaptive"
            if not has_run_before(df, "G-HUIM", method_label, base_name, min_util):
                cmd = [
                    GHUIM_BIN,
                    "--input",
                    file_path,
                    "--minutil",
                    str(min_util),
                    "--method",
                    "SP",
                    "--adaptive",
                    "true",
                    "--total_memory",
                    gpu_memory,
                    "--output",
                    "/dev/null",
                ]
                best_run = run_experiment(cmd, parse_metrics_cu, repeats)
                if best_run:
                    entry = {
                        "Algorithm": "G-HUIM",
                        "Method": method_label,
                        "File": base_name,
                        "Minimum_Utility": min_util,
                        **best_run,
                    }
                    df = update_results(df, entry, csv_path)

            for block_size in block_sizes:
                method_label = f"SP_fixed_{block_size}"
                if not has_run_before(df, "G-HUIM", method_label, base_name, min_util):
                    cmd = [
                        GHUIM_BIN,
                        "--input",
                        file_path,
                        "--minutil",
                        str(min_util),
                        "--method",
                        "SP",
                        "--adaptive",
                        "false",
                        "--pattern_memory",
                        "2G",
                        "--block_size",
                        str(block_size),
                        "--total_memory",
                        gpu_memory,
                        "--output",
                        "/dev/null",
                    ]
                    best_run = run_experiment(cmd, parse_metrics_cu, repeats)
                    if best_run:
                        entry = {
                            "Algorithm": "G-HUIM",
                            "Method": method_label,
                            "File": base_name,
                            "Minimum_Utility": min_util,
                            **best_run,
                        }
                        df = update_results(df, entry, csv_path)

            method_label = "DPHIM_dphim"
            if not has_run_before(df, "DPHIM", method_label, base_name, min_util):
                cmd = [
                    DPHIM_BIN,
                    "-i",
                    file_path,
                    "-m",
                    str(min_util),
                    "-s",
                    "dphim",
                    "-t",
                    "48",
                    "-o",
                    "/dev/null",
                ]
                best_run = run_experiment(cmd, parse_metrics_dphim, repeats)
                if best_run:
                    entry = {
                        "Algorithm": "DPHIM",
                        "Method": method_label,
                        "File": base_name,
                        "Minimum_Utility": min_util,
                        **best_run,
                    }
                    df = update_results(df, entry, csv_path)

            # method_label = "PHA"
            # if not has_run_before(df, "PHA", method_label, base_name, min_util):
            #     cmd = [PHA_BIN, file_path, str(min_util)]
            #     best_run = run_experiment(cmd, parse_metrics_pha, repeats)
            #     if best_run:
            #         entry = {"Algorithm": "PHA", "Method": method_label, "File": base_name, "Minimum_Utility": min_util, **best_run}
            #         df = update_results(df, entry, csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run High-Utility Itemset Mining benchmarks."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to repeat each experiment.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/benchmarks.csv",
        help="Path to save the results CSV file.",
    )
    args = parser.parse_args()

    datasets_to_process = [
        [
            "../datasets/pumsb_utility_spmf.txt",
            [9680000, 9690000, 9700000, 9710000, 9720000, 9730000],
        ],
        [
            "../datasets/BMS_utility_spmf.txt",
            [1999000, 2000000, 2001000, 2003000, 2004000, 2005000],
        ],
        [
            "../datasets/kosarak_utility_spmf.txt",
            [720000, 730000, 740000, 750000, 760000, 770000],
        ],
        [
            "../datasets/accidents_utility_spmf.txt",
            [3900000, 4000000, 4100000, 4200000, 4300000, 4400000],
        ],
        [
            "../datasets/connect_utility_spmf.txt",
            [7400000, 7500000, 7600000, 7700000, 7800000, 7900000],
        ],
        [
            "../datasets/chess_utility_spmf.txt",
            [170000, 175000, 180000, 185000, 190000, 195000],
        ],
        [
            "../datasets/retail_utility_spmf.txt",
            [1440, 1445, 1450, 1455, 1460, 1465],
        ],
        [
            "../datasets/mushroom_utility_spmf.txt",
            [5500, 6000, 6500, 7000, 7500, 8000],
        ],
    ]

    run_all_experiments(
        datasets_to_process, args.repeats, args.output_csv, gpu_memory="60G"
    )
    logging.info("All experiments completed.")


if __name__ == "__main__":
    main()
