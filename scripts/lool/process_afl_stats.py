import argparse
import polars as pl
from pathlib import Path


pl.Config.set_fmt_str_lengths(1200)
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(1200)
pl.Config.set_fmt_float("full")


def aggregate_fuzzer_stats_with_polars(stats_dir: str):
    """
    Finds all fuzzer_stats files, extracts time-based metrics,
    and aggregates them by experiment using Polars.
    """
    stats_path = Path(stats_dir)
    if not stats_path.is_dir():
        print(f"Error: Directory '{stats_path}' not found.")
        return

    all_stats_data = []
    metrics = [
        "cmplog_time",
        "fuzz_time",
        "calibration_time",
        "start_time",
        "run_time",
        "sync_time",
        "trim_time",
        "stability",
        "var_byte_count",
        "execs_per_sec",
        "execs_done",
        "corpus_found",
        "edges_found",
        "bitmap_cvg",
    ]
    for stats_file in stats_path.rglob("**/fuzzing/*/output/default/fuzzer_stats"):
        try:
            # Extract experiment and campaign names from the path
            campaign_dir = stats_file.parent.parent.parent
            experiment_dir = campaign_dir.parent.parent

            experiment_name = experiment_dir.name
            campaign_name = campaign_dir.name

            with open(stats_file, "r") as f:
                for line in f:
                    try:
                        # Split line into key and value
                        key_raw, value_str = line.split(":", 1)
                        key = key_raw.strip()

                        # Only process metrics that end with _time
                        if key in metrics:
                            all_stats_data.append(
                                {
                                    "experiment": experiment_name,
                                    "campaign": campaign_name,
                                    "metric": key,
                                    "value": float(value_str.replace("%", "").strip()),
                                }
                            )
                    except (ValueError, IndexError):
                        # Ignore lines that are malformed or not in key:value format
                        pass
        except IndexError:
            # Path structure might not be as expected, skip this file
            print(f"Warning: Could not parse path structure for {stats_file}")
            continue

    if not all_stats_data:
        print("No fuzzer statistics ending in '_time' were found.")
        return

    # Create a Polars DataFrame from the collected data
    df = pl.DataFrame(all_stats_data)

    # Group by experiment and metric, then sum the values
    aggregated_df = df.group_by(["experiment", "metric"]).agg(
        pl.median("value").alias("total_time_s")
    )

    # Pivot the DataFrame to make each metric a column
    pivoted_df = aggregated_df.pivot(
        index="experiment", columns="metric", values="total_time_s"
    ).sort("experiment")

    # Enforce a predictable column order for readability/stability
    column_order = ["experiment"] + [m for m in metrics if m in pivoted_df.columns]
    pivoted_df = pivoted_df.select(column_order)

    # For each column ending with "_time", append the percentage of run_time, e.g., "2 (20)"
    if "run_time" in pivoted_df.columns:
        time_cols = [c for c in pivoted_df.columns if c.endswith("_time")]
        exprs = []
        for c in time_cols:
            exprs.append(
                pl.when(pl.col("run_time") > 0)
                .then(
                    pl.format(
                        "{} ({})",
                        pl.col(c).round(0).cast(pl.Int64),
                        ((pl.col(c) / pl.col("run_time")) * 100)
                        .round(0)
                        .cast(pl.Int64),
                    )
                )
                .otherwise(
                    pl.format(
                        "{} (0)",
                        pl.col(c).round(0).cast(pl.Int64),
                    )
                )
                .alias(c)
            )
        if exprs:
            pivoted_df = pivoted_df.with_columns(exprs)

    print("Aggregated Fuzzer Statistics (seconds with % of run_time in parentheses):")
    print(pivoted_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""This script aggregates AFL++ fuzzer statistics from experiment directories.
It finds all 'fuzzer_stats' files within a given directory, extracts
time-based metrics and other relevant statistics, and aggregates them
by experiment using Polars. The final output is a table showing the
median values for each metric per experiment, with time-based metrics
also expressed as a percentage of the total run time.
""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--stats-dir",
        type=str,
        default="data/statistics",
        help="The path to the directory containing experiment statistics.",
    )
    args = parser.parse_args()

    # Ensure you have polars installed: pip install polars
    aggregate_fuzzer_stats_with_polars(args.stats_dir)
