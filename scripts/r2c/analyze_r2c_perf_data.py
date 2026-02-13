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

from common import (
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
    """Compute cycles/instructions overhead and IPC change for all configs."""
    baseline = df.filter(pl.col("config") == "baseline").select([
        "benchmark",
        pl.col("cycles").alias("baseline_cycles"),
        pl.col("instructions").alias("baseline_instructions"),
        (pl.col("instructions") / pl.col("cycles")).alias("baseline_ipc"),
    ])

    result = (
        df.join(baseline, on="benchmark")
        .with_columns([
            ((pl.col("cycles") / pl.col("baseline_cycles") - 1) * 100).alias("cycles_overhead"),
            ((pl.col("instructions") / pl.col("baseline_instructions") - 1) * 100).alias("instr_overhead"),
            (pl.col("instructions") / pl.col("cycles")).alias("ipc"),
        ])
        .with_columns([
            ((pl.col("ipc") / pl.col("baseline_ipc") - 1) * 100).alias("ipc_change"),
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
                row[f"IPC {step}"] = data.get("ipc_change")

        rows.append(row)

    return pl.DataFrame(rows)


def write_latex_rows(summary: pl.DataFrame, output_path: Path) -> None:
    """Write LaTeX table rows (benchmarks as rows, Cycles and IPC change per BTRA config)."""
    lines = []
    for row in summary.iter_rows(named=True):
        cells = [row["Benchmark"]]
        for col in summary.columns:
            if col == "Benchmark" or col.startswith("Instr"):
                continue
            val = row[col]
            if val is None:
                cells.append("--")
            else:
                # Both cycles overhead and IPC change are percentages
                cells.append(f"{val:.1f}")
        lines.append(" & ".join(cells) + r" \\")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")


def write_ipc_progression_rows(df: pl.DataFrame, output_path: Path) -> None:
    """Write LaTeX table rows with absolute IPC values and relative change in parens."""
    result = compute_overhead_table(df)

    # Get sorted configs
    configs = sorted(
        df.select("config").unique().to_series().to_list(),
        key=natural_sort_key
    )

    # Get benchmarks sorted by baseline IPC (descending, so CPU-bound first)
    baseline_df = result.filter(pl.col("config") == "baseline").select([
        "bench_short", "ipc"
    ]).sort("ipc", descending=True)
    benchmarks = baseline_df["bench_short"].to_list()

    lines = []
    for bench in benchmarks:
        bench_data = result.filter(pl.col("bench_short") == bench)
        cells = [bench]

        baseline_ipc = None

        for config in configs:
            config_row = bench_data.filter(pl.col("config") == config)
            if config_row.height == 0:
                cells.append("--")
            else:
                ipc = config_row["ipc"][0]
                if config == "baseline":
                    baseline_ipc = ipc
                    cells.append(f"{ipc:.2f}")
                else:
                    # Show absolute IPC with relative change in parens
                    if baseline_ipc:
                        delta = ((ipc / baseline_ipc) - 1) * 100
                        sign = "+" if delta >= 0 else ""
                        cells.append(f"{ipc:.2f} ({sign}{delta:.0f})")
                    else:
                        cells.append(f"{ipc:.2f}")

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


def load_tma_l1_data(tma_json_path: str) -> dict:
    """Load TMA Level 1 data from authoritative JSON."""
    data = load_json_data(tma_json_path)
    return data.get("benchmarks", {})


def plot_tma_l1_summary(tma_json_path: str, output_path: Path, mode: str) -> None:
    """Plot TMA L1 summary (baseline vs btra10) in stacked or delta mode."""
    tma = load_tma_l1_data(tma_json_path)
    benches = sorted(
        [b for b in tma.keys() if b in BENCH_FULL_TO_SHORT],
        key=lambda b: BENCH_FULL_TO_SHORT.get(b, b),
    )

    categories = ["retiring", "bad_spec", "frontend", "backend"]
    category_labels = {
        "retiring": "Retiring",
        "bad_spec": "Bad Spec",
        "frontend": "Frontend",
        "backend": "Backend",
    }
    # Match colorblind palette from plot_r2c_benchmark_data.py
    colors = {
        "retiring": "#0173b2",   # blue
        "bad_spec": "#56b4e9",   # light blue
        "frontend": "#949494",   # gray
        "backend": "#de8f05",    # orange
    }

    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = list(range(len(benches)))

    if mode == "stacked":
        width = 0.32
        gap = 0.06
        legend_handles = []
        for i, bench in enumerate(benches):
            base = tma[bench]["baseline"]
            btra = tma[bench]["r2c-btra10"]

            base_bottom = 0.0
            btra_bottom = 0.0
            for cat in categories:
                base_val = base[cat]
                btra_val = btra[cat]
                handle = ax.bar(
                    i - (width / 2 + gap / 2),
                    base_val,
                    width,
                    bottom=base_bottom,
                    color=colors[cat],
                )
                ax.bar(
                    i + (width / 2 + gap / 2),
                    btra_val,
                    width,
                    bottom=btra_bottom,
                    color=colors[cat],
                )
                if i == 0:
                    legend_handles.append(handle[0])
                base_bottom += base_val
                btra_bottom += btra_val

        ax.set_ylabel("TMA L1 Share (%)")
        ax.set_xticks(x)
        ax.set_xticklabels([BENCH_FULL_TO_SHORT[b] for b in benches], rotation=45, ha="right")
        ax.legend(
            handles=legend_handles,
            labels=[category_labels[c] for c in categories],
            ncol=4,
            fontsize=8,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.40),
            borderaxespad=0.0,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=1.0,
            handlelength=1.0,
            handleheight=1.0,
            handletextpad=0.4,
        )
        ax.set_xlim(-0.5, len(benches) - 0.5)
        ax.set_ylim(0, 100)
    else:
        width = 0.18
        offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
        for idx, cat in enumerate(categories):
            vals = []
            for bench in benches:
                base = tma[bench]["baseline"][cat]
                btra = tma[bench]["r2c-btra10"][cat]
                vals.append(btra - base)
            ax.bar(
                [i + offsets[idx] for i in x],
                vals,
                width,
                label=category_labels[cat],
                color=colors[cat],
            )

        ax.set_ylabel("TMA L1 Delta (pp, BTRA10 - baseline)")
        ax.set_xticks(x)
        ax.set_xticklabels([BENCH_FULL_TO_SHORT[b] for b in benches], rotation=30, ha="right")
        ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.12))
        ax.axhline(0, color="gray", linewidth=0.8, alpha=0.6)

    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    matplot2tikz.save(output_path)
    plt.close()
    if mode == "stacked":
        inject_tikz_legend(output_path, category_labels, colors)


def compute_cache_absorption(all_passes_json: str, tma_json_path: str) -> list[dict]:
    """Compute cache absorption metrics for L1i/L2 and frontend deltas."""
    data = load_json_data(all_passes_json)
    pass1 = data.get("1", {})
    pass2 = data.get("2", {})
    tma = load_tma_l1_data(tma_json_path)

    rows = []
    for bench, bench_data in pass1.items():
        if bench not in BENCH_FULL_TO_SHORT:
            continue
        if bench not in pass2 or bench not in tma:
            continue
        if "baseline" not in pass1[bench] or "r2c-btra10" not in pass1[bench]:
            continue
        if "baseline" not in pass2[bench] or "r2c-btra10" not in pass2[bench]:
            continue

        base_l1i = pass1[bench]["baseline"]["L1-icache-load-misses"]["mean"]
        btra_l1i = pass1[bench]["r2c-btra10"]["L1-icache-load-misses"]["mean"]
        l1i_delta = (btra_l1i / base_l1i - 1) * 100 if base_l1i else None

        base_hit = pass2[bench]["baseline"]["l2_rqsts.code_rd_hit"]["mean"]
        base_miss = pass2[bench]["baseline"]["l2_rqsts.code_rd_miss"]["mean"]
        btra_hit = pass2[bench]["r2c-btra10"]["l2_rqsts.code_rd_hit"]["mean"]
        btra_miss = pass2[bench]["r2c-btra10"]["l2_rqsts.code_rd_miss"]["mean"]

        base_rate = base_hit / (base_hit + base_miss) if (base_hit + base_miss) else None
        btra_rate = btra_hit / (btra_hit + btra_miss) if (btra_hit + btra_miss) else None
        l2_rate_delta = (btra_rate - base_rate) * 100 if base_rate is not None and btra_rate is not None else None

        fe_delta = tma[bench]["r2c-btra10"]["frontend"] - tma[bench]["baseline"]["frontend"]

        rows.append({
            "benchmark": bench,
            "bench_short": BENCH_FULL_TO_SHORT.get(bench, bench),
            "l1i_delta": l1i_delta,
            "l2_hit_delta": l2_rate_delta,
            "fe_delta": fe_delta,
        })

    return rows


def inject_tikz_legend(output_path: Path, category_labels: dict, colors: dict) -> None:
    """Inject explicit legend entries into matplot2tikz output."""
    lines = output_path.read_text().splitlines()
    insert_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "]":
            insert_idx = i + 1
            break
    if insert_idx is None:
        return

    entries = []
    for key in ["retiring", "bad_spec", "frontend", "backend"]:
        hex_color = colors[key].lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        entries.append(
            f"\\addlegendimage{{area legend, fill={{rgb,255:red,{r};green,{g};blue,{b}}}}}"
        )
        entries.append(f"\\addlegendentry{{{category_labels[key]}}}")

    lines[insert_idx:insert_idx] = entries
    output_path.write_text("\n".join(lines) + "\n")


def plot_cache_absorption(all_passes_json: str, tma_json_path: str, output_path: Path, mode: str) -> None:
    """Plot cache absorption metrics."""
    rows = compute_cache_absorption(all_passes_json, tma_json_path)
    rows = sorted(rows, key=lambda r: r["bench_short"])

    fig, ax = plt.subplots(figsize=(7, 3.8))

    if mode == "scatter":
        x = [r["l1i_delta"] for r in rows]
        y = [r["l2_hit_delta"] for r in rows]
        c = [r["fe_delta"] for r in rows]
        sc = ax.scatter(x, y, c=c, cmap="coolwarm", s=60, edgecolors="k", linewidths=0.3)
        for r in rows:
            ax.annotate(r["bench_short"], (r["l1i_delta"], r["l2_hit_delta"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Frontend Bound Δ (pp)")
        ax.set_xlabel("L1i Misses Δ (%)")
        ax.set_ylabel("L2 Code Hit Rate Δ (pp)")
        ax.axhline(0, color="gray", linewidth=0.8, alpha=0.6)
        ax.axvline(0, color="gray", linewidth=0.8, alpha=0.6)
    else:
        x = list(range(len(rows)))
        width = 0.35
        l1i = [r["l1i_delta"] for r in rows]
        l2 = [r["l2_hit_delta"] for r in rows]
        fe = [r["fe_delta"] for r in rows]

        ax.bar([i - width / 2 for i in x], l1i, width, label="L1i Misses Δ (%)", color="#4C78A8")
        ax.bar([i + width / 2 for i in x], l2, width, label="L2 Hit Rate Δ (pp)", color="#F58518")
        ax.set_ylabel("Δ (%) / Δ (pp)")
        ax.set_xticks(x)
        ax.set_xticklabels([r["bench_short"] for r in rows], rotation=30, ha="right")
        ax.axhline(0, color="gray", linewidth=0.8, alpha=0.6)

        ax2 = ax.twinx()
        ax2.plot(x, fe, color="#E45756", marker="o", linewidth=1.3, label="Frontend Bound Δ (pp)")
        ax2.set_ylabel("Frontend Bound Δ (pp)")

        # Combined legend
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=3)

    ax.grid(True, axis="y", alpha=0.3)
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
        "--tma-json", type=str, default=None, help="Path to authoritative TMA JSON (for TMA L1 plot)."
    )
    parser.add_argument(
        "--passes-json", type=str, default=None, help="Path to all_passes_data.json (for cache absorption plot)."
    )
    parser.add_argument(
        "--plot-mode",
        type=str,
        default=None,
        choices=[
            "overhead-vs-ipc",
            "instr-vs-cycles",
            "tma-l1-stacked",
            "tma-l1-delta",
            "cache-absorption-bars",
            "cache-absorption-scatter",
        ],
        help="Generate a single plot (mutually exclusive modes).",
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

        ipc_output_path = Path(args.latex_output) / "ipc-progression.tex"
        write_ipc_progression_rows(df, ipc_output_path)

    # Generate a single plot if requested
    if args.plot_mode:
        if not args.latex_output:
            raise SystemExit("--plot-mode requires --latex-output to set the output directory")
        plot_path = Path(args.latex_output) / f"{args.plot_mode}.tex"

        if args.plot_mode == "overhead-vs-ipc":
            plot_overhead_vs_ipc(df, plot_path)
        elif args.plot_mode == "instr-vs-cycles":
            plot_instr_vs_cycles(df, plot_path)
        elif args.plot_mode in {"tma-l1-stacked", "tma-l1-delta"}:
            if not args.tma_json:
                raise SystemExit("--plot-mode tma-l1-* requires --tma-json")
            mode = "stacked" if args.plot_mode.endswith("stacked") else "delta"
            plot_tma_l1_summary(args.tma_json, plot_path, mode)
        elif args.plot_mode in {"cache-absorption-bars", "cache-absorption-scatter"}:
            if not args.passes_json or not args.tma_json:
                raise SystemExit("--plot-mode cache-absorption-* requires --passes-json and --tma-json")
            mode = "bars" if args.plot_mode.endswith("bars") else "scatter"
            plot_cache_absorption(args.passes_json, args.tma_json, plot_path, mode)


if __name__ == "__main__":
    main()
