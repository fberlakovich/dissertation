import argparse
import gc
import json
import logging
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import polars as pl
from tqdm import tqdm

pl.Config.set_fmt_str_lengths(1200)
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(1200)
pl.Config.set_fmt_table_cell_list_len(3)
pl.Config.set_fmt_float("full")
pl.Config.set_float_precision(10)
pl.Config.set_thousands_separator(",")


def apply_name_map(df: pl.DataFrame, name_map: dict[str, str] | None) -> pl.DataFrame:
    """Applies a name map to the 'Experiment' column of a DataFrame."""
    if name_map and "Experiment" in df.columns:
        return df.with_columns(
            pl.col("Experiment").replace(name_map, default=pl.col("Experiment"))
        )
    return df


def process_file_for_opts(
    file_path_str: str,
    vocab_filename: str,
    id_col: str,
    count_col: str,
    delete_cols: list[str],
) -> tuple[str, pl.DataFrame] | None:
    """
    Minimal version of process_file, only for extracting opt counts for rarity calculation.
    """
    try:
        input_path = Path(file_path_str)
        vocab_path = input_path.parent / vocab_filename
        with open(vocab_path, "r") as f:
            mapping = json.load(f)
            id_to_name_map = {
                int(k): str(tuple(v)) if isinstance(v, list) else v
                for k, v in mapping.items()
            }
        experiment_name = input_path.parent.name

        input_df = (
            pl.read_parquet(input_path)
            .with_columns(
                pl.duration(
                    milliseconds=(
                        pl.col("StartTime") - pl.col("StartTime").min().over("Campaign")
                    )
                ).alias("Elapsed"),
            )
            .drop(
                [
                    "LoC",
                    "FileSize",
                    "LevenshteinDistance",
                    "SeedFileSize",
                    "SeedFileFirstDiff",
                    "SeedFileDiffCount",
                    "NumParameterFiles",
                    "CoverageSource",
                    "NewCoverageFileIDs",
                    "Time",
                    "StartTime",
                    "EndTime",
                ]
            )
        )

        df = (
            input_df.drop(delete_cols)
            .explode([id_col, count_col])
            .with_columns(
                pl.col(id_col).cast(pl.Int64), pl.col(count_col).cast(pl.Int64)
            )
            .drop_nulls(subset=[id_col, count_col, "Campaign"])
            .filter(pl.col(count_col) > 0)
            .with_columns(
                pl.col(id_col)
                .replace_strict(id_to_name_map, default="Unmapped")
                .alias("Contributor")
            )
            .sort(["Campaign", "Elapsed"])
        )

        df_long_cum = df.with_columns(
            pl.col(count_col)
            .cum_sum()
            .over(["Campaign", "Contributor"])
            .alias("CumulativeCount")
        )

        state_per_campaign_at_events = df_long_cum.pivot(
            index=["Elapsed", "Campaign"],
            columns="Contributor",
            values="CumulativeCount",
        )
        del df_long_cum
        gc.collect()

        dense_state_per_campaign = state_per_campaign_at_events.with_columns(
            pl.all()
            .exclude(["Campaign", "Elapsed"])
            .sort_by(by="Elapsed", descending=False)
            .fill_null(strategy="forward")
            .over("Campaign")
        ).fill_null(0)

        return (experiment_name, dense_state_per_campaign)

    except Exception as e:
        logging.error(f"Failed to process file {file_path_str}: {e}", exc_info=True)
        return None


def combine_experiment_data_for_opts(
    input_files: list[str],
    vocab_filename: str,
    id_col: str,
    count_col: str,
    delete_cols=None,
    name_map: dict[str, str] = None,
) -> dict[str, pl.DataFrame]:
    """Minimal version of combine_experiment_data for rarity calculation."""
    worker_func = partial(
        process_file_for_opts,
        vocab_filename=vocab_filename,
        id_col=id_col,
        count_col=count_col,
        delete_cols=delete_cols,
    )

    logging.info(
        f"Loading and processing data from {len(input_files)} experiments in parallel..."
    )

    todo = []
    cumulative_dfs = {}
    processed_names = {}
    for file in input_files:
        processed_file = Path(Path(file).parent, f"wide_timeline_{id_col}.parquet")
        experiment_name = Path(file).parent.name
        if name_map is not None:
            experiment_name = name_map.get(experiment_name, experiment_name)
        processed_names[experiment_name] = processed_file
        if processed_file.is_file():
            logging.info(f"Using cached file: {processed_file}")
            cumulative_dfs[experiment_name] = pl.read_parquet(processed_file)
        else:
            logging.info(f"Adding {file} to processing queue")
            todo.append(file)

    if todo:
        with Pool(processes=3) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(worker_func, todo),
                    total=len(todo),
                    desc="Processing experiments",
                )
            )

        valid_results = [r for r in results if r is not None]
        for name, df in valid_results:
            if name_map is not None:
                name = name_map.get(name, name)

            # Don't write parquet cache if name mapping is used, to avoid conflicts
            if name_map is None:
                df.write_parquet(processed_names[name])
            cumulative_dfs[name] = df

    return cumulative_dfs


def extract_final_campaign_counts(
    experiment_dfs: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """Minimal version of extract_final_campaign_counts."""
    logging.info(
        f"Extracting final counts for each campaign from {len(experiment_dfs)} experiments..."
    )
    final_counts_list = []
    for experiment, df in experiment_dfs.items():
        if df.is_empty():
            continue
        per_campaign_finals = (
            df.sort("Elapsed")
            .group_by("Campaign", maintain_order=True)
            .last()
            .drop("Elapsed")
        )

        final_counts_list.append(
            per_campaign_finals.with_columns(pl.lit(experiment).alias("Experiment"))
        )

    if not final_counts_list:
        return pl.DataFrame()

    result = pl.concat(final_counts_list, how="diagonal_relaxed").select(
        "Experiment", "Campaign", pl.all().exclude("Experiment", "Campaign")
    )
    return result.fill_null(0)


def load_test_data_for_correlation(
    file_path_str: str, rare_opts_list: list[str]
) -> pl.DataFrame:
    """
    Loads raw test data for correlation analysis:
    - NewCoverage (bool)
    - FoundRareOpt (bool)
    """
    try:
        input_path = Path(file_path_str)
        experiment_name = input_path.parent.name

        try:
            vocab_path = input_path.parent / "opt_vocabulary.json"
            with open(vocab_path, "r") as f:
                mapping = json.load(f)
                opt_vocab_map = {
                    int(k): str(tuple(v)) if isinstance(v, list) else v
                    for k, v in mapping.items()
                }
        except Exception as e:
            logging.error(
                f"Failed to load opt_vocabulary.json for {experiment_name}: {e}"
            )
            return pl.DataFrame()

        df = pl.read_parquet(
            file_path_str,
            # Read the necessary source columns
            columns=[
                "Campaign",
                "StartTime",
                "Test",
                "Outcome",
                "Time",
                "OptIDs",
                "NewCoverageFileIDs",
            ],
        )

        if df.is_empty():
            return pl.DataFrame()

        # 1. Calculate 'Elapsed' column
        df = df.with_columns(
            pl.col("Time").cast(pl.Datetime),
            pl.duration(
                milliseconds=(
                    pl.col("StartTime") - pl.col("StartTime").min().over("Campaign")
                )
            ).alias("Elapsed"),
        )

        # 2. Robustly compute IsNewCoverage from NewCoverageFileIDs (which may be str or list)
        df = (
            df.filter(~pl.col("Outcome").is_in(['timeout', 'skipped']))
            .with_columns(
                NewCovIDs_casted=pl.col("NewCoverageFileIDs").cast(
                    pl.List(pl.Int64), strict=False
                ),
            )
            .with_columns(
                NewCovIDs_filled=pl.coalesce(
                    [pl.col("NewCovIDs_casted"), pl.lit([], dtype=pl.List(pl.Int64))]
                )
            )
            .with_columns(IsNewCoverage=pl.col("NewCovIDs_filled").list.len() > 0)
        )

        # 3. Handle rare opts
        # Build a literal list expression to avoid dtype issues in list operations
        rare_opts_lit = pl.lit(rare_opts_list, dtype=pl.List(pl.Utf8))

        per_test_df = (
            df.with_columns(
                # 1. Cast OptIDs to List[i64]. This turns Strings and other types to null.
                OptIDs_casted=pl.col("OptIDs").cast(pl.List(pl.Int64), strict=False),
            )
            .with_columns(
                # 2. Fill all nulls (original or from cast) with an empty list
                OptIDs_filled=pl.coalesce(
                    [pl.col("OptIDs_casted"), pl.lit([], dtype=pl.List(pl.Int64))]
                )
            )
            .with_columns(
                # 3. Map IDs to names; drop unknown/None
                OptNames=pl.col("OptIDs_filled")
                .list.eval(pl.element().replace_strict(opt_vocab_map, default=None))
                .list.drop_nulls()
            )
            .with_columns(
                # Check if any rare opts are in the list
                FoundRareOpt=pl.col("OptNames")
                .list.set_intersection(rare_opts_lit)
                .list.len()
                > 0
            )
            .select(
                pl.lit(experiment_name).alias("Experiment"),
                "Campaign",
                "Test",
                "Elapsed",
                "IsNewCoverage",
                "FoundRareOpt",
            )
        )
        # --- END FIX ---

        return per_test_df
    except Exception as e:
        logging.error(
            f"Failed to process {file_path_str} for correlation: {e}", exc_info=True
        )
        return pl.DataFrame()


def handle_correlation(args: argparse.Namespace) -> None:
    """
    Handler for the 'correlate' command.
    Calculates the correlation between finding new coverage and triggering rare optimizations.
    """
    logging.info("Starting Coverage-Optimization Correlation Analysis...")

    # --- 1. Get Rarity List (MODIFIED) ---
    logging.info("Calculating optimization rarity...")
    experiment_dfs_opts = combine_experiment_data_for_opts(
        args.input_files,
        vocab_filename="opt_vocabulary.json",
        id_col="OptIDs",
        count_col="OptCounts",
        delete_cols=["PairIDs", "PairCounts"],
        name_map=args.name_map_dict,
    )
    final_opts_counts = extract_final_campaign_counts(experiment_dfs_opts)
    if final_opts_counts.is_empty():
        logging.error("No optimization data found. Cannot calculate rarity.")
        return

    # Unpivot to get total counts per optimization
    long_counts = final_opts_counts.unpivot(
        index=["Experiment", "Campaign"],
        variable_name="Optimization",
        value_name="Count",
    )

    total_per_item = (
        long_counts.group_by("Optimization")
        .agg(pl.sum("Count").alias("TotalCount"))
        .filter(pl.col("TotalCount") > 0)
    )  # Filter out optimizations that never ran

    if total_per_item.is_empty():
        logging.warning("No optimization triggers found across all experiments.")
        return

    # Calculate the percentile threshold
    percentile = args.correlation_percentile / 100.0

    rarity_threshold_value = total_per_item["TotalCount"].quantile(percentile, "linear")

    if rarity_threshold_value is None:
        logging.warning("Could not determine rarity threshold. Using minimum count.")
        rarity_threshold_value = total_per_item["TotalCount"].min()
        if rarity_threshold_value is None:
            logging.error("No optimization counts found.")
            return

    rare_opts_list = total_per_item.filter(
        pl.col("TotalCount") <= rarity_threshold_value
    )["Optimization"].to_list()
    print(rare_opts_list)

    if not rare_opts_list:
        logging.warning(
            f"No rare optimizations found in the bottom {args.correlation_percentile}%% percentile (count <= {rarity_threshold_value})."
        )
        return

    logging.info(
        f"Found {len(rare_opts_list)} rare optimizations in the bottom {args.correlation_percentile}%% percentile (total trigger count <= {rarity_threshold_value})."
    )
    # --- End of Rarity Logic ---

    # --- 2. Load Optimization Vocabulary ---
    # We just need one vocab file
    try:
        vocab_path = Path(args.input_files[0]).parent / "opt_vocabulary.json"
        with open(vocab_path, "r") as f:
            mapping = json.load(f)
            opt_vocab_map = {
                int(k): str(tuple(v)) if isinstance(v, list) else v
                for k, v in mapping.items()
            }
    except Exception as e:
        logging.error(f"Failed to load opt_vocabulary.json: {e}")
        return

    # --- 3. Load test data in parallel ---
    logging.info("Loading per-test correlation data in parallel...")
    all_test_data = []

    worker_func = partial(
        load_test_data_for_correlation,
        rare_opts_list=rare_opts_list,
    )

    with Pool() as pool:
        results = list(
            tqdm(
                pool.imap_unordered(worker_func, args.input_files),
                total=len(args.input_files),
                desc="Processing test data",
            )
        )
        all_test_data = [r for r in results if not r.is_empty()]

    if not all_test_data:
        logging.error("No correlation data could be loaded.")
        return

    full_correlation_df = pl.concat(all_test_data)

    # --- 4. Apply name map ---
    full_correlation_df = apply_name_map(full_correlation_df, args.name_map_dict)

    # --- 5. Build Contingency Table ---
    logging.info("Building contingency table...")
    contingency_table = (
        full_correlation_df.group_by("Experiment")
        .agg(
            pl.when(pl.col("IsNewCoverage") & pl.col("FoundRareOpt"))
            .then(1)
            .otherwise(0)
            .sum()
            .alias("NewCov_RareOpt"),
            pl.when(pl.col("IsNewCoverage") & ~pl.col("FoundRareOpt"))
            .then(1)
            .otherwise(0)
            .sum()
            .alias("NewCov_NoRareOpt"),
            pl.when(~pl.col("IsNewCoverage") & pl.col("FoundRareOpt"))
            .then(1)
            .otherwise(0)
            .sum()
            .alias("NoNewCov_RareOpt"),
            pl.when(~pl.col("IsNewCoverage") & ~pl.col("FoundRareOpt"))
            .then(1)
            .otherwise(0)
            .sum()
            .alias("NoNewCov_NoRareOpt"),
        )
        .with_columns(
            (
                pl.col("NewCov_RareOpt")
                + pl.col("NewCov_NoRareOpt")
                + pl.col("NoNewCov_RareOpt")
                + pl.col("NoNewCov_NoRareOpt")
            ).alias("TotalTests")
        )
        .sort("Experiment")
    )

    logging.info(
        f"\n--- Coverage-Optimization Correlation (Rarest {args.correlation_percentile}%% Percentile) ---"
    )
    print(contingency_table)

    # --- 6. Calculate and print percentages/ratios (WITH NEW COLUMNS) ---
    logging.info("\n--- Correlation Ratios ---")
    ratios_table = (
        contingency_table.with_columns(
            # P(RareOpt | NewCov)
            (
                pl.col("NewCov_RareOpt")
                / (pl.col("NewCov_RareOpt") + pl.col("NewCov_NoRareOpt"))
            ).alias("P(RareOpt|NewCov)"),
            # P(NewCov | RareOpt)
            (
                pl.col("NewCov_RareOpt")
                / (pl.col("NewCov_RareOpt") + pl.col("NoNewCov_RareOpt"))
            ).alias("P(NewCov|RareOpt)"),
            # P(Opt_Rare)
            (
                (pl.col("NewCov_RareOpt") + pl.col("NoNewCov_RareOpt"))
                / pl.col("TotalTests")
            ).alias("P(Opt_Rare)"),
            # P(newCov)
            (
                (pl.col("NewCov_RareOpt") + pl.col("NewCov_NoRareOpt"))
                / pl.col("TotalTests")
            ).alias("P(newCov)"),
        )
        .select(
            "Experiment",
            "P(RareOpt|NewCov)",
            "P(NewCov|RareOpt)",
            "P(Opt_Rare)",
            "P(newCov)",
            "TotalTests",
        )
        .sort("P(Opt_Rare)", descending=True)
    )

    print(ratios_table)


def main(argv: list[str] | None = None) -> None:
    """Main function to parse arguments and dispatch commands."""
    parser = argparse.ArgumentParser(
        description="Correlation analysis for fuzzer experiments.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="One or more paths to 'preprocessed_summary.parquet' files.",
    )
    parser.add_argument(
        "--name-map",
        type=str,
        default=None,
        help="Path to a JSON file mapping experiment folder names to display names.",
    )
    parser.add_argument(
        "--correlation-percentile",
        type=float,
        default=10.0,
        help="The percentile of rarity to use as a threshold (e.g., 10 means the 10%% rarest optimizations) (default: 10.0).",
    )

    # Add a dummy func attribute to args
    parser.set_defaults(func=handle_correlation)

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s - %(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    name_map = None
    if args.name_map:
        try:
            name_map_path = Path(args.name_map)
            if not name_map_path.is_file():
                raise FileNotFoundError
            with open(name_map_path, "r") as f:
                name_map = json.load(f)
            logging.info(f"Loaded experiment name map from {args.name_map}")
        except FileNotFoundError:
            logging.error(f"Name map file not found: {args.name_map}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logging.error(
                f"Error decoding JSON from name map file {args.name_map}: {e}"
            )
            sys.exit(1)
    args.name_map_dict = name_map

    args.func(args)


if __name__ == "__main__":
    main()
