#!/usr/bin/env python3
"""
BTRA Overhead Analysis - generates overhead table and plot from perf data.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplot2tikz
import polars as pl
from tabulate import tabulate

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

from r2c_common import (
    BENCH_FULL_TO_SHORT,
    load_json_data,
    extract_benchmarks_from_json,
    extract_configs_from_json,
    natural_sort_key,
)


def load_data_as_polars(json_path: str, pass_num: int = 1) -> pl.DataFrame:
    """Load JSON data and convert to a polars DataFrame."""
    data = load_json_data(json_path)

    pass_data = data.get(str(pass_num), {})
    benchmarks = extract_benchmarks_from_json(data, str(pass_num))
    configs = extract_configs_from_json(data, str(pass_num))
    configs = sorted(configs, key=natural_sort_key)

    rows = []
    for bench in benchmarks:
        if bench not in pass_data:
            continue
        for config in configs:
            if config not in pass_data[bench]:
                continue
            row = {
                "benchmark": bench,
                "bench_short": BENCH_FULL_TO_SHORT.get(bench, bench),
                "config": config,
            }
            for event, event_data in pass_data[bench][config].items():
                if isinstance(event_data, dict) and "mean" in event_data:
                    row[event] = event_data["mean"]
            rows.append(row)

    return pl.DataFrame(rows)


def compute_overhead_table(df: pl.DataFrame) -> pl.DataFrame:
    """Compute cycles/instructions overhead and IPC for all configs."""
    baseline = df.filter(pl.col("config") == "baseline").select([
        "benchmark",
        pl.col("cycles").alias("baseline_cycles"),
        pl.col("instructions").alias("baseline_instructions"),
    ])

    result = (
        df.join(baseline, on="benchmark")
        .with_columns([
            ((pl.col("cycles") / pl.col("baseline_cycles") - 1) * 100).alias("cycles_overhead"),
            ((pl.col("instructions") / pl.col("baseline_instructions") - 1) * 100).alias("instr_overhead"),
            (pl.col("instructions") / pl.col("cycles")).alias("ipc"),
        ])
    )

    return result


def build_summary_table(df: pl.DataFrame) -> pl.DataFrame:
    """
    Build a summary table with benchmarks as rows and metrics per config as columns.
    """
    result = compute_overhead_table(df)

    benchmarks = df.select("benchmark").unique().to_series().to_list()
    configs = sorted(df.select("config").unique().to_series().to_list(), key=natural_sort_key)
    btra_configs = [c for c in configs if c != "baseline"]

    rows = []
    for bench in benchmarks:
        bench_short = BENCH_FULL_TO_SHORT.get(bench, bench)
        row = {"Benchmark": bench_short}

        bench_data = result.filter(pl.col("benchmark") == bench)

        for config in btra_configs:
            step = config.replace("r2c-btra", "")
            config_data = bench_data.filter(pl.col("config") == config)

            if config_data.height == 0:
                row[f"Instr {step}"] = None
                row[f"Cycles {step}"] = None
                row[f"IPC {step}"] = None
            else:
                data = config_data.row(0, named=True)
                row[f"Instr {step}"] = data.get("instr_overhead")
                row[f"Cycles {step}"] = data.get("cycles_overhead")
                row[f"IPC {step}"] = data.get("ipc")

        rows.append(row)

    return pl.DataFrame(rows)


def write_latex_rows(summary: pl.DataFrame, output_path: Path) -> None:
    """Write LaTeX table rows (benchmarks as rows, Cycles and IPC per BTRA config)."""
    lines = []
    for row in summary.iter_rows(named=True):
        cells = [row["Benchmark"]]
        for col in summary.columns:
            if col == "Benchmark" or col.startswith("Instr"):
                continue
            val = row[col]
            if val is None:
                cells.append("--")
            elif col.startswith("IPC"):
                cells.append(f"{val:.3f}")
            else:
                cells.append(f"{val:.1f}")
        lines.append(" & ".join(cells) + r" \\")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")


def plot_instr_vs_cycles(df: pl.DataFrame, output_path: Path) -> None:
    """Scatter plot of instructions overhead vs cycles overhead."""
    result = compute_overhead_table(df)

    # Filter to only BTRA configs
    btra_data = result.filter(pl.col("config") != "baseline")

    fig, ax = plt.subplots(figsize=(5, 4))

    benchmarks = sorted(btra_data.select("bench_short").unique().to_series().to_list())

    for bench in benchmarks:
        bench_df = btra_data.filter(pl.col("bench_short") == bench)
        x = bench_df["instr_overhead"].to_list()
        y = bench_df["cycles_overhead"].to_list()
        ax.plot(x, y, "o-", label=bench, linewidth=1.5, markersize=5)

    # Add diagonal line (y = x)
    max_val = max(btra_data["instr_overhead"].max(), btra_data["cycles_overhead"].max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5, label='y = x')

    ax.set_xlabel("Instructions Overhead (%)")
    ax.set_ylabel("Cycles Overhead (%)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    matplot2tikz.save(output_path)
    plt.close()


def plot_overhead_vs_ipc(df: pl.DataFrame, output_path: Path) -> None:
    """Create scatter plot of cycles overhead vs IPC change, with lines per benchmark."""
    result = compute_overhead_table(df)

    # Filter to only BTRA configs (not baseline)
    btra_data = result.filter(pl.col("config") != "baseline")

    # Extract step number for ordering
    btra_data = btra_data.with_columns(
        pl.col("config").str.replace("r2c-btra", "").cast(pl.Int32).alias("steps")
    ).sort(["benchmark", "steps"])

    fig, ax = plt.subplots(figsize=(5, 3.5))

    benchmarks = sorted(btra_data.select("bench_short").unique().to_series().to_list())

    for bench in benchmarks:
        bench_df = btra_data.filter(pl.col("bench_short") == bench)
        x = bench_df["cycles_overhead"].to_list()
        ipc_values = bench_df["ipc"].to_list()

        # Calculate IPC change relative to first point (2 BTRAs)
        ipc_baseline = ipc_values[0]
        y = [(ipc / ipc_baseline - 1) * 100 for ipc in ipc_values]

        ax.plot(x, y, "o-", label=bench, linewidth=1.5, markersize=5)

    ax.set_xlabel("Cycles Overhead (%)")
    ax.set_ylabel("IPC Change (%)")
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    matplot2tikz.save(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="BTRA Overhead Analysis")
    parser.add_argument("json_path", type=str, help="Path to perf data JSON file")
    parser.add_argument(
        "--latex-output", type=str, default=None, help="LaTeX output directory."
    )
    parser.add_argument(
        "--plot-output", type=str, default=None, help="Output path for IPC change plot (PDF/PNG)."
    )
    parser.add_argument(
        "--plot-instr-cycles", type=str, default=None, help="Output path for instr vs cycles plot (PDF/PNG)."
    )
    args = parser.parse_args()

    df = load_data_as_polars(args.json_path, pass_num=1)
    summary = build_summary_table(df)

    # Print markdown to stdout
    headers = summary.columns
    rows = summary.rows()
    print(tabulate(rows, headers=headers, tablefmt="github", floatfmt=".2f"))

    # Write LaTeX to file if requested
    if args.latex_output:
        output_path = Path(args.latex_output) / "btra-step-cycles.tex"
        write_latex_rows(summary, output_path)

    # Generate plots if requested
    if args.plot_output:
        plot_overhead_vs_ipc(df, Path(args.plot_output))

    if args.plot_instr_cycles:
        plot_instr_vs_cycles(df, Path(args.plot_instr_cycles))


if __name__ == "__main__":
    main()
