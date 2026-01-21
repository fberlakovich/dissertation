import argparse
import json
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import polars
import polars as pl

from utils import define_latex_table, load_name_map


# --- Custom Logger Setup ---
class InfoFilter(logging.Filter):
    def filter(self, rec):
        return rec.levelno in (logging.DEBUG, logging.INFO)


# --- Natural Sort Key ---
def natural_sort_key_for_series(s: pl.Series) -> pl.Series:
    """Applies a natural sort key for a Polars Series, assuming numeric names."""
    return s.cast(pl.Int64)


# --- Core Statistics Calculation Functions ---


def apply_name_map(df: pl.DataFrame, name_map: dict | None) -> pl.DataFrame:
    """If provided, apply a mapping to the 'Experiment' column.
    Unknown names remain unchanged.
    """
    if df.is_empty() or name_map is None or "Experiment" not in df.columns:
        return df
    return df.with_columns(pl.col("Experiment").replace_strict(name_map))


def calculate_outcome_stats(df: pl.DataFrame) -> pl.DataFrame:
    """Calculates raw test outcome statistics per campaign."""
    if "Outcome" not in df.columns:
        return pl.DataFrame({"Campaign": df["Campaign"].unique()})
    total_tests = df.group_by("Campaign").agg(pl.len().alias("total"))
    outcome_counts = df.pivot(
        index="Campaign", on="Outcome", values="Test", aggregate_function="len"
    ).fill_null(0)
    return outcome_counts.join(total_tests, on="Campaign")


def calculate_increase_stats(df: pl.DataFrame) -> pl.DataFrame:
    """Calculates raw coverage increase statistics per campaign."""
    cov_cols = [
        "CumulativeCoverage",
        "CumulativeGlobalCounterCoverage",
        "CumulativeMethodCounterCoverage",
    ]
    exprs = [
        (pl.col(c).diff() > 0).sum().alias(f'Increases_{c.replace("Cumulative", "")}')
        for c in cov_cols
        if c in df.columns
    ]
    if not exprs:
        return pl.DataFrame({"Campaign": df["Campaign"].unique()})

    total_tests = df.group_by("Campaign").agg(pl.len().alias("total"))
    return df.group_by("Campaign").agg(exprs).join(total_tests, on="Campaign")


def calculate_source_stats(df: pl.DataFrame) -> pl.DataFrame:
    """Calculates raw source and parameter file statistics per campaign."""
    if df.is_empty():
        return pl.DataFrame()

    total_tests = df.group_by("Campaign").agg(pl.len().alias("total"))
    all_stats_df = total_tests.clone()

    # Source Stats
    if "LevenshteinDistance" in df.columns:
        zero_lev = df.group_by("Campaign").agg(
            (pl.col("LevenshteinDistance") == 0).sum().alias("Identical")
        )
        all_stats_df = all_stats_df.join(zero_lev, on="Campaign", how="left")

    # Parameter Stats
    if "NumParameterFiles" in df.columns:
        param_df = df.filter(pl.col("NumParameterFiles") > 0)
        total_with_params = param_df.group_by("Campaign").agg(
            pl.len().alias("total_with_params")
        )

        param_counts = param_df.pivot(
            index="Campaign",
            on="NumParameterFiles",
            values="Test",
            aggregate_function="len",
        ).fill_null(0)

        ORDINAL_ADVERBS = {
            1: "Once",
            2: "Twice",
            3: "Thrice",
            4: "Four Times",
            5: "Five Times",
            6: "Six Times",
        }

        rename_dict = {
            str(i): f"{ORDINAL_ADVERBS[i-1]}" if i - 1 > 0 else "First"
            for i in range(1, len(ORDINAL_ADVERBS) + 1)
        }
        actual_rename_dict = {
            k: v for k, v in rename_dict.items() if k in param_counts.columns
        }
        param_counts = param_counts.rename(actual_rename_dict)

        all_stats_df = all_stats_df.join(param_counts, on="Campaign", how="left")
        all_stats_df = all_stats_df.join(total_with_params, on="Campaign", how="left")

    return all_stats_df.fill_null(0)


def calculate_runtime_stats(
    df: pl.DataFrame, group_by_col: str | None = None
) -> pl.DataFrame:
    """Calculates test runtime and LoC statistics (median, mean, min)."""
    agg_exprs = []

    # Runtime stats
    if "Elapsed Time (s)" not in df.columns:
        if "StartTime" in df.columns and "EndTime" in df.columns:
            df = df.with_columns(
                ((pl.col("EndTime") - pl.col("StartTime")) / 1000.0).alias(
                    "Elapsed Time (s)"
                )
            )

    if "Elapsed Time (s)" in df.columns:
        agg_exprs.extend(
            [
                pl.col("Elapsed Time (s)").median().alias("Median Runtime (s)"),
                pl.col("Elapsed Time (s)").mean().alias("Mean Runtime (s)"),
                pl.col("Elapsed Time (s)").min().alias("Min Runtime (s)"),
                pl.col("Elapsed Time (s)").max().alias("Max Runtime (s)"),
            ]
        )

    # LoC stats
    if "LoC" in df.columns:
        agg_exprs.extend(
            [
                pl.col("LoC").median().alias("Median LoC"),
                pl.col("LoC").mean().alias("Mean LoC"),
                pl.col("LoC").max().alias("Max LoC"),
            ]
        )

    if not agg_exprs:
        if group_by_col and group_by_col in df.columns:
            return pl.DataFrame({group_by_col: df[group_by_col].unique()})
        return pl.DataFrame()

    if group_by_col:
        return df.group_by(group_by_col).agg(agg_exprs)
    else:
        return df.select(agg_exprs)


# --- Formatting and Presentation Functions ---


def format_and_print_stats(
    stats_df: pl.DataFrame, title: str, relative_to: dict = None
):
    """Formats a raw statistics DataFrame into percentages and prints it."""
    if stats_df.is_empty() or "Campaign" not in stats_df.columns:
        return

    logging.info(title)
    if relative_to is None:
        relative_to = {}

    stat_cols = [
        c
        for c in stats_df.columns
        if c not in ["Campaign", "total", "total_with_params"]
    ]

    formatted_cols = []
    for col_name in sorted(stat_cols):
        total_col = relative_to.get(col_name, "total")
        if total_col not in stats_df.columns:
            continue

        perc = (
            pl.when(pl.col(total_col) > 0)
            .then((pl.col(col_name) / pl.col(total_col) * 100).round(1))
            .otherwise(0.0)
        )
        formatted_cols.append(
            pl.format(f"{{}} ({{}}%)", pl.col(col_name), perc).alias(col_name)
        )

    if not formatted_cols:
        print("No statistics to display for this section.")
        return

    formatted_df = stats_df.select(["Campaign"] + formatted_cols).sort(
        by=pl.col("Campaign").pipe(natural_sort_key_for_series)
    )

    with_median = add_median_row(formatted_df)
    print(with_median)


def format_and_print_simple_stats(df: pl.DataFrame, title: str):
    """Formats a numeric stats DataFrame by rounding and then prints it."""
    if df.is_empty():
        return

    logging.info(title)

    numeric_cols = [c for c in df.columns if df[c].dtype in pl.NUMERIC_DTYPES]

    if not numeric_cols:
        print(df)
        return

    formatted_df = df.with_columns([pl.col(c).round(3) for c in numeric_cols])

    # Try to sort naturally if a 'Campaign' or 'Experiment' column exists
    if "Campaign" in formatted_df.columns:
        formatted_df = formatted_df.sort(
            by=pl.col("Campaign").pipe(natural_sort_key_for_series)
        )
    elif "Experiment" in formatted_df.columns:
        formatted_df = formatted_df.sort("Experiment")

    print(formatted_df)





def add_median_row(df: pl.DataFrame) -> pl.DataFrame:
    """Calculates and appends a median summary row to a formatted statistics DataFrame."""
    if df.is_empty() or df.height < 2:
        return df

    median_stats = {}
    for col_name in df.columns:
        if col_name in ["Campaign", "sort_key"]:
            continue

        counts = df.select(
            pl.col(col_name).str.extract(r"(\d+) \(").cast(pl.Int64)
        ).to_series()
        percs = df.select(
            pl.col(col_name).str.extract(r"\((\d+\.\d+)%\)").cast(pl.Float64)
        ).to_series()

        median_count = counts.median()
        median_perc = percs.median()

        if median_count is not None and median_perc is not None:
            median_stats[col_name] = f"{median_count:.0f} ({median_perc:.1f}%)"

    if not median_stats:
        return df

    median_row = pl.DataFrame([median_stats])
    median_row = median_row.with_columns(pl.lit("Median").alias("Campaign"))

    median_row = median_row.select(df.columns)

    return pl.concat([df, median_row])


# --- Main Execution Logic ---


def run_single_analysis(args):
    """Handles the original single-file analysis workflow."""
    parquet_path = Path(args.parquet_file)
    if not parquet_path.is_file():
        logging.critical(f"Parquet file not found at: {parquet_path}")
        sys.exit(1)

    try:
        full_df = pl.read_parquet(parquet_path)
        logging.info(f"Successfully loaded data from {parquet_path} using Polars")
    except Exception as e:
        logging.critical(f"Could not read Parquet file: {e}")
        sys.exit(1)

    if args.campaign:
        full_df = full_df.filter(pl.col("Campaign") == args.campaign)

        if full_df.is_empty():
            logging.critical(
                f"Specified campaign '{args.campaign}' not found in the data."
            )
            sys.exit(1)

    # --- Post-Processing and Metric Calculation ---
    cols_to_cast = [
        "CumulativeCoverage",
        "CumulativeGlobalCounterCoverage",
        "CumulativeMethodCounterCoverage",
        "LoC",
        "LevenshteinDistance",
        "FileSize",
        "SeedFileSize",
        "SeedFileFirstDiff",
        "SeedFileDiffCount",
        "NumParameterFiles",
    ]
    existing_cols_to_cast = [c for c in cols_to_cast if c in full_df.columns]
    full_df = full_df.with_columns(
        [pl.col(c).cast(pl.Int64, strict=False) for c in existing_cols_to_cast]
    )

    if "StartTime" in full_df.columns and "EndTime" in full_df.columns:
        full_df = full_df.with_columns(
            ((pl.col("EndTime") - pl.col("StartTime")) / 1000.0).alias(
                "Elapsed Time (s)"
            )
        ).with_columns(
            pl.col("Elapsed Time (s)")
            .cum_sum()
            .over("Campaign")
            .alias("Cumulative Time (s)")
        )

    if "CumulativeCoverage" in full_df.columns:
        prev_cumulative = pl.col("CumulativeCoverage").shift(1).over("Campaign")
        increase = pl.col("CumulativeCoverage") - prev_cumulative
        full_df = full_df.with_columns(
            ((increase / prev_cumulative) * 100).alias("Coverage Increase (%)")
        )

    # --- Prepare DataFrame for Console Output ---
    df_to_print = full_df
    if args.show_changes_only and "Coverage Increase (%)" in df_to_print.columns:
        logging.info("\n--- Filtering for New Coverage Discovery ---")
        df_to_print = df_to_print.filter(
            (pl.col("Coverage Increase (%)") > 0)
            | (pl.col("Coverage Increase (%)").is_null())
        )

    # --- Display and Save Results ---
    logging.info("\n--- Coverage Analysis Complete ---")
    if not df_to_print.is_empty():
        display_cols = [
            "Campaign",
            "Test",
            "CumulativeCoverage",
            "CumulativeGlobalCounterCoverage",
            "CumulativeMethodCounterCoverage",
            "Coverage Increase (%)",
            "LoC",
            "LevenshteinDistance",
            "SeedFileSize",
            "SeedFileFirstDiff",
            "SeedFileDiffCount",
            "NumParameterFiles",
            "Elapsed Time (s)",
            "Cumulative Time (s)",
            "CoverageSource",
        ]
        existing_display_cols = [c for c in display_cols if c in df_to_print.columns]
        print(df_to_print.select(existing_display_cols))

    # --- Display Aggregate Statistics using refactored functions ---
    if not full_df.is_empty():
        outcome_stats = calculate_outcome_stats(full_df)
        format_and_print_stats(
            outcome_stats, "\n--- Per-Campaign Test Outcome Statistics ---"
        )

        increase_stats = calculate_increase_stats(full_df)
        format_and_print_stats(
            increase_stats, "\n--- Per-Campaign Coverage Increase Statistics ---"
        )

        source_stats = calculate_source_stats(full_df)
        param_cols = [
            c
            for c in source_stats.columns
            if c not in ["Campaign", "total", "Identical"]
        ]
        relative_to_map = {col: "total_with_params" for col in param_cols}
        format_and_print_stats(
            source_stats,
            "\n--- Per-Campaign Source & Parameter File Statistics ---",
            relative_to=relative_to_map,
        )

        runtime_stats = calculate_runtime_stats(full_df, group_by_col="Campaign")
        format_and_print_simple_stats(
            runtime_stats, "\n--- Per-Campaign Test Runtime & LoC Statistics ---"
        )

    else:
        logging.info("No data available to analyze.")


def process_single_experiment(
    parquet_path_str: str, agg_methods: list[str]
) -> dict | None:
    """
    Worker function to process a single parquet file.
    Calculates statistics and aggregates them using multiple specified methods.
    """
    parquet_path = Path(parquet_path_str)
    if not parquet_path.is_file():
        logging.warning(f"File not found, skipping: {parquet_path}")
        return None

    experiment_name = str(parquet_path.parent.name)

    try:
        df = pl.read_parquet(parquet_path)
    except Exception as e:
        logging.error(f"Could not read {parquet_path}: {e}")
        return None

    outcome_stats = calculate_outcome_stats(df)
    increase_stats = calculate_increase_stats(df)
    source_stats = calculate_source_stats(df)
    runtime_stats = calculate_runtime_stats(df)

    agg_map = {
        "median": pl.Expr.median,
        "mean": pl.Expr.mean,
        "sum": pl.Expr.sum,
        "min": pl.Expr.min,
        "max": pl.Expr.max,
    }

    def aggregate_stats_df(stats_df: pl.DataFrame) -> pl.DataFrame:
        if stats_df.is_empty():
            return pl.DataFrame()

        agg_exprs = []
        for col_name in stats_df.columns:
            if col_name == "Campaign":
                continue

            # Create a list of individual aggregation expressions for the current column
            single_col_aggs = [
                agg_map[method](pl.col(col_name)).round(1).cast(pl.String)
                for method in agg_methods
                if method in agg_map
            ]

            # Combine them into a single string column separated by " / "
            agg_exprs.append(
                pl.concat_str(single_col_aggs, separator=" / ").alias(col_name)
            )

        if not agg_exprs:
            return pl.DataFrame()

        return stats_df.select(agg_exprs)

    results = {"name": experiment_name}
    results["outcome"] = aggregate_stats_df(outcome_stats).with_columns(
        pl.lit(experiment_name).alias("Experiment")
    )
    results["increase"] = aggregate_stats_df(increase_stats).with_columns(
        pl.lit(experiment_name).alias("Experiment")
    )
    results["source"] = aggregate_stats_df(source_stats).with_columns(
        pl.lit(experiment_name).alias("Experiment")
    )
    results["runtime"] = runtime_stats.with_columns(
        pl.lit(experiment_name).alias("Experiment")
    )

    return results


def run_multi_experiment_summary(args):
    """Handles the new multi-file summary workflow in parallel."""
    all_stats = {
        "outcome": [],
        "increase": [],
        "source": [],
        "runtime": [],
    }

    logging.info(
        f"Starting parallel processing for {len(args.summarize_experiments)} experiments..."
    )

    worker_func = partial(process_single_experiment, agg_methods=args.agg_method)

    with ProcessPoolExecutor() as executor:
        results = executor.map(worker_func, args.summarize_experiments)

        for result in results:
            if result:
                logging.info(f"Completed processing for experiment: {result['name']}")
                if "outcome" in result and not result["outcome"].is_empty():
                    all_stats["outcome"].append(result["outcome"])
                if "increase" in result and not result["increase"].is_empty():
                    all_stats["increase"].append(result["increase"])
                if "source" in result and not result["source"].is_empty():
                    all_stats["source"].append(result["source"])
                if "runtime" in result and not result["runtime"].is_empty():
                    all_stats["runtime"].append(result["runtime"])

    aggregation_order = "/".join(m.upper() for m in args.agg_method)
    logging.info(f"\n--- Experiment Summary (Aggregations: {aggregation_order}) ---")

    # --- Merge and Display Tables ---
    name_map = getattr(args, "name_map_dict", None)
    final_outcome_df = (
        pl.concat(all_stats["outcome"], how="diagonal")
        if all_stats["outcome"]
        else pl.DataFrame()
    )
    final_increase_df = (
        pl.concat(all_stats["increase"], how="diagonal")
        if all_stats["increase"]
        else pl.DataFrame()
    )
    final_source_df = (
        pl.concat(all_stats["source"], how="diagonal")
        if all_stats["source"]
        else pl.DataFrame()
    )
    final_runtime_df = (
        pl.concat(all_stats["runtime"]) if all_stats["runtime"] else pl.DataFrame()
    )

    # Apply name mapping for display if provided
    final_outcome_df = apply_name_map(final_outcome_df, name_map)
    final_increase_df = apply_name_map(final_increase_df, name_map)
    final_source_df = apply_name_map(final_source_df, name_map)
    final_runtime_df = apply_name_map(final_runtime_df, name_map)


    def to_pascal_case(name: str) -> str:
        # Split on underscores, spaces, or hyphens, capitalize each, and join
        parts = re.split(r"[_\s\-]+", name)
        return "".join(word.capitalize() for word in parts if word)

    # Merge outcome and source stats
    if not final_outcome_df.is_empty() and not final_source_df.is_empty():
        merged_df = final_outcome_df.join(final_source_df, on="Experiment", how="left")
        columns_to_remove = ["total_with_params", "total_right"]

        # Use the .drop() method to remove the columns
        merged_df = merged_df.drop(columns_to_remove)

        overview_cols = [
            "Experiment",
            "total",
            "passed",
            "timeout",
            "skipped",
            "failed",
        ]

        # Split into two tables as requested
        afl_cols = [
            "Experiment",
            "total",
            "skipped",
            "Identical",
            "First",
            "Once",
            "Twice",
            "Thrice",
            "Four Times",
            "Five Times",
        ]

        # Table A: Remaining columns (everything except the special subset)
        remaining_table = merged_df.select(overview_cols)
        logging.info(
            "\n--- Merged Outcome, Source & Parameter Statistics (sans Zero-Edit/Original/Reduced variants) ---"
        )
        test_overview_table = remaining_table.rename(
            {col: to_pascal_case(col) for col in remaining_table.columns}
        )

        unique_bugs = {}
        if args.unique_bugs:
            with open(args.unique_bugs, "r") as f:
                bugs = json.load(f)
                for old_name, bugs in bugs.items():
                    unique_bugs[name_map[old_name]] = str(bugs)

        test_overview_table = (
            test_overview_table.with_columns(
                pl.when(pl.col("Experiment").is_in(unique_bugs.keys()))
                .then(
                    pl.col("Failed").cast(polars.datatypes.String)
                    + " ("
                    + pl.col("Experiment").replace_strict(
                        unique_bugs, default="Unmapped"
                    )
                    + ")"
                )
                .otherwise(pl.col("Failed").cast(polars.datatypes.String))
                .alias("Failed (Unique)")
            )
            .drop("Failed")
            .fill_null("0")
        ).join(
            final_runtime_df.drop(
                [
                    c
                    for c in final_runtime_df.columns
                    if "Min" in c or "Max" in c or "Mean" in c
                ]
            ),
            on="Experiment",
            how="left",
        )
        print(test_overview_table)
        define_latex_table(
            args,
            "test-overview",
            test_overview_table,
        )

        # Table B: Only the special subset, filtered to rows where at least one is non-zero/non-null
        if afl_cols:
            subset_table = merged_df.select(afl_cols)
            # Extract all numeric values from aggregated strings and test if any > 0 per column
            any_positive_per_col = [
                pl.col(c)
                .str.extract_all(r"-?\d+(?:\.\d+)?")
                .list.eval(pl.element().cast(pl.Float64) > 0)
                .list.any()
                .fill_null(False)
                for c in afl_cols
                if c not in ["total", "skipped", "First"]
            ]
            non_zero_mask = pl.any_horizontal(any_positive_per_col)
            filtered_subset = subset_table.filter(non_zero_mask)
            if not filtered_subset.is_empty():
                logging.info(
                    "\n--- Zero-Edit/Original/Reduced Statistics (only rows with any non-zero) ---"
                )

                subset_table = filtered_subset.select(
                    "Experiment", pl.all().exclude("Experiment")
                ).fill_null("0")
                print(subset_table)
                define_latex_table(
                    args,
                    "afl-test-overview",
                    subset_table,
                )
            else:
                logging.info(
                    "\n--- Zero-Edit/Original/Reduced Statistics: no rows with non-zero values ---"
                )
        else:
            logging.info(
                "\n--- Zero-Edit/Original/Reduced Statistics: columns not present ---"
            )
    elif not final_outcome_df.is_empty():
        logging.info("\n--- Test Outcome Statistics ---")
        print(final_outcome_df)
    elif not final_source_df.is_empty():
        logging.info("\n--- Source & Parameter File Statistics ---")
        print(final_source_df)

    # Display increase stats separately
    if not final_increase_df.is_empty():
        logging.info("\n--- Coverage Increase Statistics ---")
        print(final_increase_df.select("Experiment", pl.all().exclude("Experiment")))

    # Display runtime stats separately
    if not final_runtime_df.is_empty():
        format_and_print_simple_stats(
            final_runtime_df.select("Experiment", pl.all().exclude("Experiment")),
            "\n--- Experiment Test Runtime & LoC Statistics ---",
        )




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Process coverage and runtime statistics per test and summarize across experiments.

This script supports two modes:
- Single analysis mode (default): Given one preprocessed Parquet file for a fuzzing
  campaign, it prints detailed per-test information (coverage, runtime,
  parameters/source info) and several per-campaign summaries. It can optionally
  export a CSV summary and LaTeX tables.
- Multi-experiment summary mode (--summarize-experiments): Given multiple Parquet
  files (each treated as one experiment), it aggregates selected statistics across
  campaigns using one or more methods (median/mean/sum/min/max), applies an optional
  experiment name mapping, and prints compact overview tables (with optional LaTeX).

Inputs (depending on mode):
- preprocessed_summary.parquet (required)
- name map JSON and unique-bugs JSON (optional, for display)

Outputs:
- Console tables with formatted statistics
- LaTeX tables when --latex-output is provided
""",
        formatter_class=argparse.RawTextHelpFormatter
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "parquet_file",
        nargs="?",
        default=None,
        help="The path to a single 'preprocessed_summary.parquet' file for detailed analysis.",
    )
    group.add_argument(
        "--summarize-experiments",
        nargs="+",
        help="A list of parquet files to summarize. Each file is treated as one experiment.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG level) logging.",
    )
    parser.add_argument(
        "--show-changes-only",
        action="store_true",
        help="Only display tests where the cumulative coverage has increased (single file mode).",
    )

    parser.add_argument(
        "--campaign",
        help="Only process a single campaign with this name (single file mode).",
    )
    parser.add_argument(
        "--latex-output", type=str, default=None, help="LaTeX output filename."
    )
    parser.add_argument(
        "--name-map",
        type=str,
        default=None,
        help="Path to a JSON file mapping experiment folder names to display names.",
    )

    parser.add_argument(
        "--unique-bugs",
        type=str,
        default=None,
        help="Path to a JSON file containing the number of unique bugs for each experiment.",
    )

    parser.add_argument(
        "--agg-method",
        nargs="+",
        choices=["median", "mean", "sum", "min", "max"],
        default=["median"],
        help="One or more aggregation methods to use across campaigns for each experiment summary.",
    )
    args = parser.parse_args()

    # Load optional experiment name map
    name_map = load_name_map(args.name_map)
    args.name_map_dict = name_map

    # --- Configure Logger ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.addFilter(InfoFilter())
    stdout_handler.setFormatter(logging.Formatter("[%(processName)s] %(message)s"))
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(
        logging.Formatter("[%(levelname)s] [%(processName)s] %(message)s")
    )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

    # --- Configure Polars Display ---
    pl.Config.set_fmt_str_lengths(1200)
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(1200)
    pl.Config.set_fmt_table_cell_list_len(3)
    pl.Config.set_fmt_float("full")
    pl.Config.set_thousands_separator(",")

    # --- Route to appropriate function based on mode ---
    if args.summarize_experiments:
        run_multi_experiment_summary(args)
    elif args.parquet_file:
        run_single_analysis(args)
    else:
        parser.print_help()
        sys.exit(1)
