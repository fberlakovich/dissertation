import argparse
import datetime
import gc
import json
import logging
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

import utils


import polars.selectors


import colorcet as cc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import polars as pl
import seaborn as sns
from tqdm import tqdm
import matplot2tikz


def get_hours_minutes_formatter():
    """Return a formatter that renders float hours as H or H:MM (e.g., 3.5 -> 3:30)."""

    def _fmt(x, pos):
        if x == int(x):
            return f"{int(x)}"
        hours = int(x)
        minutes = int(round((x - hours) * 60))
        return f"{hours}:{minutes:02d}"

    return ticker.FuncFormatter(_fmt)


def save_plot_as_tikz(figure, output: str | None) -> None:
    """Save the figure via matplot2tikz when an output path is provided."""
    if not output:
        return
    matplot2tikz.save(output, axis_width="\\linewidth")
    logging.info(f"Plot saved to {output}")


# --- Polars Configuration ---
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


def run_cross_experiment_analysis(
    final_campaign_counts: pl.DataFrame,
    entity_name: str,
):
    """
    Performs cross-experiment analysis. Uniqueness is calculated as the median of per-campaign scores.
    """
    # --- Part 1: Cross-experiment stats based on median experiment state ---
    logging.info(f"Aggregating total {entity_name} counts across all experiments...")

    final_experiment_counts = final_campaign_counts.group_by("Experiment").agg(
        pl.all().exclude(["Experiment", "Campaign"]).median()
    )

    all_zero = [
        c
        for c in final_experiment_counts.columns
        if c != "Experiment" and (final_experiment_counts[c] == 0).all()
    ]
    final_experiment_counts = final_experiment_counts.drop(all_zero)

    long_df = final_experiment_counts.unpivot(
        index="Experiment", variable_name=entity_name, value_name="Count"
    )
    num_experiments = len(final_experiment_counts["Experiment"].unique())

    agg_df = long_df.group_by(entity_name).agg(
        pl.sum("Count").alias("TotalCount"),
        pl.min("Count").alias("Min_per_exp"),
        pl.max("Count").alias("Max_per_exp"),
        pl.median("Count").alias("Median_per_exp"),
        pl.std("Count").alias("StdDev_per_exp"),
    )

    agg_df = agg_df.with_columns(
        (pl.col("StdDev_per_exp") / (pl.col("TotalCount") / num_experiments)).alias(
            "RelStdDev_%_(CV)"
        )
    )
    agg_df = agg_df.with_columns((pl.col("RelStdDev_%_(CV)") * 100).fill_nan(0))

    results_df = agg_df.sort("TotalCount", descending=True).select(
        pl.col(entity_name),
        pl.col("TotalCount"),
        pl.col("Min_per_exp"),
        pl.col("Max_per_exp"),
        pl.col("Median_per_exp").cast(pl.Int64),
        pl.col("StdDev_per_exp").round(2),
        pl.col("RelStdDev_%_(CV)").round(2),
    )

    logging.info(
        f"\n--- Total Cross-Experiment {entity_name} Counts (from Median Campaign State) ---"
    )
    print(results_df)


def calculate_per_campaign_uniqueness(
    final_counts: pl.DataFrame, entity_name: str
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Calculates the per-campaign uniqueness score, rarity weights,
    and the full weighted contributions.
    This function correctly filters all-zero entities before calculating weights.
    """
    logging.info(f"Calculating per-campaign uniqueness for {entity_name}s...")

    # 1. Filter all-zero entities
    all_zero = [
        c
        for c in final_counts.columns
        if c not in ["Experiment", "Campaign"] and (final_counts[c] == 0).all()
    ]
    filtered_final_counts = final_counts.drop(all_zero)

    # 2. Unpivot
    long_campaign_df = filtered_final_counts.unpivot(
        index=["Experiment", "Campaign"],
        variable_name=entity_name,
        value_name="Count",
    )

    # 3. Calculate weights
    rarity_weights = calculate_rarity_weights(long_campaign_df, entity_name)
    logging.info(f"\nRarity weights (top 10):")
    print(rarity_weights.sort(by="Weight", descending=True).head(10))

    # 4. Calculate weighted contributions
    weighted_contributions = long_campaign_df.join(
        rarity_weights, on=entity_name, how="left"
    ).with_columns(
        (pl.col("Count").add(1).log10() * pl.col("Weight")).alias(
            "WeightedContribution"
        )
    )

    # 5. Calculate per-campaign uniqueness
    per_campaign_uniqueness = weighted_contributions.group_by(
        ["Experiment", "Campaign"]
    ).agg(pl.sum("WeightedContribution").alias("Uniqueness"))

    return per_campaign_uniqueness, weighted_contributions, rarity_weights


def uniqueness_analysis(
    entity_name: str, final_campaign_counts: pl.DataFrame, show_contributors: int | None
) -> tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame, pl.DataFrame]:
    """
    Performs campaign-level uniqueness analysis.
    Returns experiment ratings, contributors, weights, and raw per-campaign scores.
    """

    per_campaign_uniqueness, weighted_contributions, rarity_weights = (
        calculate_per_campaign_uniqueness(final_campaign_counts, entity_name)
    )

    experiment_ratings = (
        per_campaign_uniqueness.group_by("Experiment")
        .agg(
            pl.median("Uniqueness").alias("Median_Uniqueness"),
            pl.mean("Uniqueness").alias("Mean_Uniqueness"),
            pl.min("Uniqueness").alias("Min_Uniqueness"),
            pl.max("Uniqueness").alias("Max_Uniqueness"),
        )
        .sort("Median_Uniqueness", descending=True)
    )

    contributors = None
    if show_contributors and show_contributors > 0:
        median_campaigns = (
            per_campaign_uniqueness.join(experiment_ratings, on="Experiment")
            .with_columns(
                (pl.col("Uniqueness") - pl.col("Median_Uniqueness")).abs().alias("diff")
            )
            .sort("diff")
            .group_by("Experiment", maintain_order=True)
            .first()
            .select(["Experiment", "Campaign", "Median_Uniqueness"])
        )

        median_campaign_contributions = weighted_contributions.join(
            median_campaigns, on=["Experiment", "Campaign"], how="inner"
        )

        contributors = (
            median_campaign_contributions.sort("WeightedContribution", descending=True)
            .filter(
                pl.col("WeightedContribution").cum_sum().over("Experiment")
                <= ((float(show_contributors) / 100) * pl.col("Median_Uniqueness"))
            )
            .select(
                [
                    "Experiment",
                    entity_name,
                    "WeightedContribution",
                    "Weight",
                    "Count",
                    "Median_Uniqueness",
                ]
            )
            .sort(
                by=["Median_Uniqueness", "Experiment", "WeightedContribution"],
                descending=True,
            )
            .drop("Median_Uniqueness")
        )

    return experiment_ratings, contributors, rarity_weights, per_campaign_uniqueness


def process_file(
    file_path_str: str,
    vocab_filename: str,
    id_col: str,
    count_col: str,
    delete_cols: list[str],
) -> tuple[str, pl.DataFrame] | None:
    """
    Processes a single parquet file, returning its name and a wide pivoted DataFrame
    representing the state of each contributor in each campaign over time.
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
                pl.col("Time").cast(pl.Datetime),
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
                    "Time",
                    "StartTime",
                    "EndTime",
                ]
            )
        )

        logging.info(f"[{experiment_name}] Loaded dataframe")

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

        unmapped = df.filter(pl.col("Contributor") == "Unmapped")
        if len(unmapped) > 0:
            logging.warning(
                f"[{experiment_name}] Found {len(unmapped)} unmapped contributors."
            )

        logging.info(f"[{experiment_name}] Translated id columns")

        df_long_cum = df.with_columns(
            pl.col(count_col)
            .cum_sum()
            .over(["Campaign", "Contributor"])
            .alias("CumulativeCount")
        )
        print(
            f"[{experiment_name}] Long DataFrame memory: {df_long_cum.estimated_size('mb'):.2f} MB"
        )

        state_per_campaign_at_events = df_long_cum.pivot(
            index=["Elapsed", "Campaign"],
            columns="Contributor",
            values="CumulativeCount",
        )
        del df_long_cum
        gc.collect()
        print(
            f"[{experiment_name}] Pivoted DF memory: {state_per_campaign_at_events.estimated_size('mb'):.2f} MB"
        )

        dense_state_per_campaign = state_per_campaign_at_events.with_columns(
            pl.all()
            .exclude(["Campaign", "Elapsed"])
            .sort_by(by="Elapsed", descending=False)
            .fill_null(strategy="forward")
            .over("Campaign")
        ).fill_null(0)
        print(
            f"[{experiment_name}] Filled DF memory: {dense_state_per_campaign.estimated_size('mb'):.2f} MB"
        )

        return (experiment_name, dense_state_per_campaign)

    except Exception as e:
        logging.error(f"Failed to process file {file_path_str}: {e}", exc_info=True)
        return None


def combine_experiment_data(
    input_files: list[str],
    vocab_filename: str,
    id_col: str,
    count_col: str,
    delete_cols=None,
    name_map: dict[str, str] = None,
) -> dict[str, pl.DataFrame]:
    """Aggregate final counters and collect full cumulative DataFrames in parallel."""
    worker_func = partial(
        process_file,
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
            experiment_name = name_map[experiment_name]
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
                name = name_map[name]

            df.write_parquet(processed_names[name])

    return cumulative_dfs


def plot_uniqueness_boxplot(
    df: pl.DataFrame,
    entity_name: str,
    output: str | None,
) -> None:
    """
    Generates and saves a boxplot of final uniqueness scores.
    'df' is expected to be the per_campaign_uniqueness dataframe.
    """
    figure, ax = plt.subplots(figsize=(14, 8))
    logging.info(f"Generating uniqueness boxplot...")

    try:
        # df is the per_campaign_uniqueness dataframe
        df_to_plot = df
        order = (
            df_to_plot.group_by("Experiment")
            .agg(pl.median("Uniqueness").alias("median_uniqueness"))
            .sort("median_uniqueness", descending=True)["Experiment"]
            .to_list()
        )
        final_time_data_pd = df_to_plot.to_pandas()
        title = f"{entity_name} Uniqueness Distribution (Final State)"
        # Swap axes: plot Uniqueness on x-axis and Experiments on y-axis
        ax = sns.boxplot(
            color=".8",
            linecolor="#137",
            ax=ax,
            data=final_time_data_pd,
            x="Uniqueness",
            y="Experiment",
            order=order,
        )
        ax.set_title(title, fontsize=20)
        ax.set_xlabel("Uniqueness Score", fontsize=20)
        ax.set_ylabel(None)
        ax.tick_params(axis="both", labelsize=12)

        # No rotation needed since labels are now on the y-axis
        figure.tight_layout()

        save_plot_as_tikz(figure, output)

        figure.show()

    except Exception as e:
        logging.error(f"Failed to generate boxplot: {e}", exc_info=True)


def plot_timeline(
    df: pl.DataFrame,
    entity_name: str,
    show_range: bool,
    plot_mode: str,
    output: str | None,
) -> None:
    """
    Generates and saves a plot of uniqueness over time (overlay or grid).
    'df' is expected to be the campaign_uniqueness_timeline dataframe.
    """
    figure, ax = plt.subplots(figsize=(14, 8))
    logging.info(f"Generating uniqueness timeline plot using '{plot_mode}' mode...")
    try:
        # Input 'df' is the campaign_uniqueness_timeline dataframe
        # It has "Time", "Experiment", "Campaign", "Uniqueness"
        df_to_plot = df.with_columns(
            (pl.col("Time").dt.total_hours()).alias("Time (hours)")
        )

        title = f"{entity_name} Uniqueness Over Time"
        palette = sns.color_palette(
            cc.glasbey, n_colors=len(df_to_plot["Experiment"].unique())
        )

        hm_formatter = get_hours_minutes_formatter()

        if plot_mode == "grid":
            df_pd = df_to_plot.to_pandas()
            g = sns.relplot(
                ax=ax,
                data=df_pd,
                x="Time (hours)",
                y="Uniqueness",
                col="Experiment",
                hue="Experiment",
                col_wrap=4,
                kind="line",
                estimator="median",
                errorbar=("pi", 50) if show_range else None,
                legend=False,
                height=3,
                aspect=1.5,
            )

            x_min, x_max = df_pd["Time (hours)"].min(), df_pd["Time (hours)"].max()
            if x_max > x_min:
                tick_locations = np.linspace(x_min, x_max, 24)
                for ax in g.axes.flat:
                    ax.set_xticks(tick_locations)
                    ax.xaxis.set_major_formatter(hm_formatter)
                    ax.tick_params(axis="x")

            if show_range:
                title += " (Median with Min/Max Campaign Range)"
            g.fig.suptitle(title, y=1.02)
            g.set_titles(col_template="{col_name}")
            g.set_axis_labels("Time (hours)", "Uniqueness Score")
            g.tight_layout(w_pad=1)

        elif plot_mode == "overlay":  # overlay mode
            final_time = df_to_plot["Time (hours)"].max()
            if final_time is not None:
                legend_order = (
                    df_to_plot.filter(pl.col("Time (hours)") == final_time)
                    .group_by("Experiment")
                    .agg(pl.median("Uniqueness").alias("FinalUniqueness"))
                    .sort("FinalUniqueness", descending=True)["Experiment"]
                    .to_list()
                )
            else:
                legend_order = df_to_plot["Experiment"].unique().to_list()

            df_pd = df_to_plot.to_pandas()
            ax = sns.lineplot(
                ax=ax,
                data=df_pd,
                x="Time (hours)",
                y="Uniqueness",
                hue="Experiment",
                hue_order=legend_order,
                estimator="median",
                errorbar=("pi", 50) if show_range else None,
                legend="full",
                linewidth=2.5,
                palette=palette,
            )

            x_min, x_max = df_pd["Time (hours)"].min(), df_pd["Time (hours)"].max()
            if x_max > x_min:
                ax.set_xticks(np.linspace(x_min, x_max, 24))
                ax.xaxis.set_major_formatter(hm_formatter)

            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Time (hours)", fontsize=12)
            ax.set_ylabel("Uniqueness Score", fontsize=12)
            ax.legend(title="Experiment", bbox_to_anchor=(1.05, 1), loc="upper left")
            figure.tight_layout()

        save_plot_as_tikz(figure, output)

        figure.show()

    except Exception as e:
        logging.error(f"Failed to generate plot: {e}", exc_info=True)


def handle_consistency(args: argparse.Namespace) -> None:
    """
    Calculates the internal consistency of each experiment's runs
    using only rare, weighted optimization vectors.
    """
    if args.entity == "opts":
        vocab_filename = "opt_vocabulary.json"
        id_col, count_col, entity_name = "OptIDs", "OptCounts", "Optimization"
        delete_cols = ["PairIDs", "PairCounts"]
    else:  # pairs
        vocab_filename = "pair_vocabulary.json"
        id_col, count_col, entity_name = "PairIDs", "PairCounts", "OptimizationPair"
        delete_cols = ["OptIDs", "OptCounts"]

    experiment_dfs = combine_experiment_data(
        args.input_files,
        vocab_filename,
        id_col,
        count_col,
        delete_cols,
        args.name_map_dict,
    )
    final_counts = extract_final_campaign_counts(experiment_dfs, args.group_opts)

    if final_counts.height < 2:
        logging.warning(
            "Not enough data to calculate consistency. Need at least 2 runs."
        )
        return

    # 1. Get rarity weights (calculated from all experiments)
    _, _, rarity_weights = calculate_per_campaign_uniqueness(final_counts, entity_name)

    # Filter for rare optimizations to be used in consistency calculation.
    weight_threshold = args.consistency_weight_threshold
    logging.info(
        f"--- Calculating consistency using optimizations with Weight > {weight_threshold} ---"
    )
    rare_optimizations = rarity_weights.filter(pl.col("Weight") > weight_threshold)[
        entity_name
    ].to_list()

    if not rare_optimizations:
        logging.warning(
            f"No optimizations found with Weight > {weight_threshold}. Cannot calculate consistency."
        )
        return
    else:
        logging.info("Rare optimizations:")
        print(rarity_weights.filter(pl.col("Weight") > weight_threshold))

    # 2. Create the weighted contribution vectors
    long_counts = final_counts.unpivot(
        index=["Experiment", "Campaign"], variable_name=entity_name, value_name="Count"
    )

    weighted_vectors_long = (
        long_counts.join(rarity_weights, on=entity_name, how="left")
        .with_columns(pl.col("Weight").fill_null(0.0))
        .with_columns(
            (pl.col("Count").add(1).log10() * pl.col("Weight")).alias(
                "WeightedContribution"
            )
        )
    )

    # Pivot back to wide format
    vector_data_all = weighted_vectors_long.pivot(
        index=["Experiment", "Campaign"],
        columns=entity_name,
        values="WeightedContribution",
    ).fill_null(0.0)

    # Select only the columns corresponding to rare optimizations for the analysis.
    rare_cols_present = [
        col for col in rare_optimizations if col in vector_data_all.columns
    ]
    if not rare_cols_present:
        logging.warning(
            f"Optimizations with Weight > {weight_threshold} were found, but none exist in the run data. Cannot calculate consistency."
        )
        return

    vector_data_rare = vector_data_all.select(
        ["Experiment", "Campaign"] + rare_cols_present
    )

    # 3. Calculate similarities based on these new "rare-only" weighted vectors
    results = []
    for exp_name in vector_data_rare["Experiment"].unique().to_list():
        exp_vectors = vector_data_rare.filter(pl.col("Experiment") == exp_name).drop(
            "Experiment", "Campaign"
        )

        if exp_vectors.height < 2:
            results.append(
                {"Experiment": exp_name, "AvgCentroidSim": None, "AvgPairwiseSim": None}
            )
            continue

        exp_data_np = exp_vectors.to_numpy()

        # Method 1: Average Centroid Similarity
        centroid = np.mean(exp_data_np, axis=0).reshape(1, -1)
        centroid_sims = cosine_similarity(exp_data_np, centroid)
        avg_centroid_sim = np.mean(centroid_sims)

        # Method 2: Average Pairwise Similarity
        pair_matrix = cosine_similarity(exp_data_np)
        upper_triangle_indices = np.triu_indices(n=pair_matrix.shape[0], k=1)
        pairwise_scores = pair_matrix[upper_triangle_indices]
        avg_pairwise_sim = (
            np.mean(pairwise_scores) if len(pairwise_scores) > 0 else None
        )

        results.append(
            {
                "Experiment": exp_name,
                "AvgCentroidSim": avg_centroid_sim,
                "AvgPairwiseSim": avg_pairwise_sim,
            }
        )

    results_df = pl.from_dicts(results).sort("AvgPairwiseSim", descending=True)
    logging.info(f"--- {entity_name} Run Consistency (Weighted, Rare-Only) ---")
    print(results_df)


def handle_uniqueness_timeline(args: argparse.Namespace) -> None:
    """Calculates and optionally plots uniqueness scores over time."""
    if args.entity == "opts":
        vocab_filename = "opt_vocabulary.json"
        id_col, count_col, entity_name = "OptIDs", "OptCounts", "Optimization"
        delete_cols = ["PairIDs", "PairCounts"]
    else:  # pairs
        vocab_filename = "pair_vocabulary.json"
        id_col, count_col, entity_name = "PairIDs", "PairCounts", "OptimizationPair"
        delete_cols = ["OptIDs", "OptCounts"]

    experiment_dfs = combine_experiment_data(
        args.input_files,
        vocab_filename,
        id_col,
        count_col,
        delete_cols,
        args.name_map_dict,
    )
    if not experiment_dfs or len(experiment_dfs) < 2:
        logging.warning("Uniqueness analysis requires at least 2 experiments. Exiting.")
        return

    final_counts = extract_final_campaign_counts(experiment_dfs)

    # We need the weights from the final counts to apply to the timeline
    _, _, rarity_weights = calculate_per_campaign_uniqueness(final_counts, entity_name)

    all_timeline_dfs = [
        df.with_columns(pl.lit(name).alias("Experiment"))
        for name, df in experiment_dfs.items()
        if not df.is_empty()
    ]
    if not all_timeline_dfs:
        logging.warning("No valid timeline data found. Exiting.")
        return

    combined_timeline = pl.concat(all_timeline_dfs, how="diagonal").sort("Elapsed")
    max_time = combined_timeline["Elapsed"].max()
    if not max_time:
        logging.warning("No time data available in experiments. Exiting.")
        return

    time_step = datetime.timedelta(minutes=args.time_step)
    time_steps_list = (
        pl.datetime_range(
            start=datetime.datetime.min,
            end=datetime.datetime.min + max_time,
            interval=time_step,
            eager=True,
        )
        - datetime.datetime.min
    )
    time_grid = pl.DataFrame(pl.Series("Time", time_steps_list, dtype=pl.Duration))

    unique_groups = combined_timeline.select(["Experiment", "Campaign"]).unique()
    sampling_grid = time_grid.join(unique_groups, how="cross")

    sampled_states = sampling_grid.join_asof(
        combined_timeline,
        left_on="Time",
        right_on="Elapsed",
        by=["Experiment", "Campaign"],
    ).fill_null(0)

    long_states_over_time = sampled_states.drop("Elapsed").unpivot(
        index=["Time", "Experiment", "Campaign"],
        variable_name=entity_name,
        value_name="Count",
    )

    # This join now uses the correctly calculated rarity_weights
    campaign_uniqueness_timeline = (
        long_states_over_time.join(rarity_weights, on=entity_name, how="left")
        .with_columns(
            (pl.col("Count").add(1).log10() * pl.col("Weight")).alias(
                "WeightedContribution"
            )
        )
        .group_by(["Time", "Experiment", "Campaign"])
        .agg(pl.sum("WeightedContribution").alias("Uniqueness"))
        .sort(["Time", "Experiment"])
    )

    aggregated_timeline = (
        campaign_uniqueness_timeline.group_by(["Time", "Experiment"])
        .agg(
            pl.median("Uniqueness").alias("Median_Uniqueness"),
            pl.min("Uniqueness").alias("Min_Uniqueness"),
            pl.max("Uniqueness").alias("Max_Uniqueness"),
            pl.std("Uniqueness").alias("StdDev_Uniqueness"),
        )
        .sort(["Time", "Experiment"])
    )

    logging.info("\n--- Uniqueness Timeline (Median and Range) ---")
    print(aggregated_timeline)

    if args.plot:
        plot_timeline(
            campaign_uniqueness_timeline,
            entity_name,
            show_range=args.show_campaign_range,
            plot_mode=args.plot_mode,
            output=args.output,
        )


def calculate_rarity_weights(
    long_counts: pl.DataFrame, entity_name: str
) -> pl.DataFrame:
    """Calculates static rarity weights based on final total counts of each entity."""
    logging.info("Calculating static rarity weights based on final counts...")

    # NOTE: This function assumes `long_counts` has already been filtered
    # to remove all-zero entities.

    total_per_item = long_counts.group_by(entity_name).agg(
        pl.sum("Count").alias("Total")
    )
    grand_total = total_per_item["Total"].sum()

    if grand_total == 0:
        logging.warning(
            "Grand total of all counts is zero. Rarity weights will be zero."
        )
        return total_per_item.with_columns(pl.lit(0.0).alias("Weight")).select(
            entity_name, "Weight"
        )

    rarity_weights = (
        total_per_item.filter(pl.col("Total") > 0)
        .with_columns((pl.lit(grand_total) / pl.col("Total")).log10().alias("Weight"))
        .select(entity_name, "Weight")
    )
    return rarity_weights


def add_shared_args(p: argparse.ArgumentParser) -> None:
    """Adds arguments shared across 'opts' and 'pairs' subparsers."""
    p.add_argument(
        "input_files",
        nargs="+",
        help="One or more paths to 'preprocessed_summary.parquet' files.",
    )
    p.add_argument(
        "--show-contributors",
        type=int,
        metavar="N",
        help="Show the top N contributors to the uniqueness score (for --uniqueness-method=score).",
    )

    p.add_argument(
        "--plot-boxplot",
        action="store_true",
        help="Generate and show a boxplot of the final uniqueness score distribution.",
    )


def extract_final_campaign_counts(
    experiment_dfs: dict[str, pl.DataFrame], group=False
) -> pl.DataFrame:
    """Extracts the final state for each campaign in each experiment."""
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
        if group:
            prefixes = {
                c.split("_")[0] for c in per_campaign_finals.columns if "_" in c
            }
            per_campaign_finals = per_campaign_finals.select(
                [
                    pl.sum_horizontal([polars.selectors.starts_with(f"{p}")]).alias(p)
                    for p in prefixes
                ]
                + ["Campaign"]
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


def print_exclusive_optimizations_table(
    entity_name: str,
    final_counts: pl.DataFrame,
    rarity_weights: pl.DataFrame,
    experiment_ratings: pl.DataFrame,
    name_map_dict: dict[str, str] | None,
    min_own_avg_log_count: float,
    max_other_avg_log_count: float,
):
    """
    Prints a table of optimizations that are (almost) exclusive to each experiment,
    based on average log10(Count).
    """
    logging.info(
        f"\n--- Exclusive {entity_name} Analysis (Own AvgLogCount >= {min_own_avg_log_count}, Other AvgLogCount <= {max_other_avg_log_count}) ---"
    )
    try:
        # 1. Unpivot to long format
        # final_counts is already dense (filled with 0s)
        long_counts = final_counts.unpivot(
            index=["Experiment", "Campaign"],
            variable_name=entity_name,
            value_name="Count",
        )

        # 2. Calculate average log10(Count) for all (Experiment, Entity) pairs
        avg_log_count_all = (
            long_counts.with_columns(pl.col("Count").add(1).log10().alias("LogCount"))
            .group_by(["Experiment", entity_name])
            .agg(pl.mean("LogCount").alias("AvgLogCount"))
        )

        # 3. For each (Experiment, Entity), find the max avg log(count) in *other* experiments
        max_other_avg_log_count_df = (
            avg_log_count_all.join(avg_log_count_all, on=entity_name, suffix="_other")
            .filter(pl.col("Experiment") != pl.col("Experiment_other"))
            .group_by(["Experiment", entity_name])
            .agg(
                pl.col("AvgLogCount_other")
                .max()
                .fill_null(0.0)
                .alias("MaxAvgLogCountInOthers")
            )
        )

        # 4. Join back and filter for "exclusives"
        # Use a left join in case an entity *only* exists in one experiment
        comparison_df = avg_log_count_all.join(
            max_other_avg_log_count_df,
            on=["Experiment", entity_name],
            how="left",
        ).with_columns(
            pl.col("MaxAvgLogCountInOthers").fill_null(
                0.0
            )  # Fill nulls for entities exclusive to one experiment
        )

        exclusive_df = comparison_df.filter(
            (pl.col("AvgLogCount") >= min_own_avg_log_count)
            & (pl.col("MaxAvgLogCountInOthers") <= max_other_avg_log_count)
        )

        if exclusive_df.is_empty():
            logging.info(
                "No (almost) exclusive entities found with current thresholds."
            )
            return

        # 5. Format for printing
        exp_order = experiment_ratings.sort("Median_Uniqueness", descending=True)[
            "Experiment"
        ].to_list()

        exclusive_df_formatted = (
            exclusive_df.join(rarity_weights, on=entity_name, how="left")
            .with_columns(pl.col("Weight").fill_null(0.0))
            .select(
                "Experiment",
                entity_name,
                pl.col("AvgLogCount").round(4).alias("OwnAvgLogCount"),
                pl.col("MaxAvgLogCountInOthers").round(4).alias("MaxOtherAvgLogCount"),
                pl.col("Weight").round(4),
            )
        )

        # Get mapped names order
        mapped_order = apply_name_map(
            pl.DataFrame({"Experiment": exp_order}), name_map_dict
        )["Experiment"].to_list()

        final_table = exclusive_df_formatted.with_columns(
            pl.col("Experiment").cast(pl.Enum(categories=mapped_order))
        ).sort(by=["Experiment", "Weight"], descending=[False, True])

        print(final_table)

    except Exception as e:
        logging.error(
            f"Failed to generate exclusive optimizations table: {e}", exc_info=True
        )


def print_uniqueness(
    entity_name: str,
    final_counts: pl.DataFrame,
    show_contributors: int,
    name_map_dict: dict[str, str] | None,
    plot_boxplot: bool,
    output_file: str | None,
    show_exclusive_opts: bool,
    exclusive_min_own_avg_log_count: float,
    exclusive_max_other_avg_log_count: float,
):
    """
    Calculates, prints, and optionally plots uniqueness analysis results.
    """
    # final_counts is unmapped
    experiment_ratings, contributors, rarity_weights, per_campaign_uniqueness = (
        uniqueness_analysis(entity_name, final_counts, show_contributors)
    )

    # Call the new function here, after ratings and weights are calculated
    if show_exclusive_opts:
        print_exclusive_optimizations_table(
            entity_name,
            final_counts,  # Unmapped counts
            rarity_weights,
            experiment_ratings,  # Unmapped ratings (for sorting)
            name_map_dict,
            exclusive_min_own_avg_log_count,
            exclusive_max_other_avg_log_count,
        )

    logging.info(f"\n--- {entity_name} Uniqueness (Median of Campaign Scores) ---")
    print(experiment_ratings)

    if contributors is not None:
        logging.info(
            f"--- Top {show_contributors} contributors to median campaign uniqueness ---"
        )
        print(contributors)

    if plot_boxplot:
        logging.info("Generating uniqueness boxplot...")
        # Call the new dedicated boxplot function
        plot_uniqueness_boxplot(per_campaign_uniqueness, entity_name, output_file)

    return (
        experiment_ratings,
        contributors,
        rarity_weights,
    )


def handle_compare(args: argparse.Namespace) -> None:
    """
    Performs a differential analysis between two specific experiments,
    including run-level details for top impacting optimizations.
    """
    if len(args.experiments) != 2:
        logging.error("The 'compare' command requires exactly two experiment names.")
        sys.exit(1)

    exp_a_name, exp_b_name = args.experiments[0], args.experiments[1]
    logging.info(f"Performing differential analysis: '{exp_a_name}' vs. '{exp_b_name}'")

    if args.entity == "opts":
        vocab_filename = "opt_vocabulary.json"
        id_col, count_col, entity_name = "OptIDs", "OptCounts", "Optimization"
        delete_cols = ["PairIDs", "PairCounts"]
    else:  # pairs
        vocab_filename = "pair_vocabulary.json"
        id_col, count_col, entity_name = "PairIDs", "PairCounts", "OptimizationPair"
        delete_cols = ["OptIDs", "OptCounts"]

    experiment_dfs = combine_experiment_data(
        args.input_files,
        vocab_filename,
        id_col,
        count_col,
        delete_cols,
        args.name_map_dict,
    )
    final_counts = extract_final_campaign_counts(experiment_dfs, args.group_opts)

    if final_counts.is_empty():
        logging.warning("No data found. Exiting comparison.")
        return

    # 1. Get rarity weights (calculated from all experiments)
    _, _, rarity_weights = calculate_per_campaign_uniqueness(final_counts, entity_name)

    # 2. Calculate average log10(Count) for all
    long_counts = final_counts.unpivot(
        index=["Experiment", "Campaign"], variable_name=entity_name, value_name="Count"
    ).with_columns(
        # Calculate LogCount here to use later for run-level details
        pl.col("Count")
        .add(1)
        .log10()
        .alias("LogCount")
    )

    avg_log_count_all = long_counts.group_by(["Experiment", entity_name]).agg(
        pl.mean("LogCount").alias("AvgLogCount")
    )

    # 3. Filter average data for just the two experiments
    exp_a_avg_data = avg_log_count_all.filter(pl.col("Experiment") == exp_a_name)
    exp_b_avg_data = avg_log_count_all.filter(pl.col("Experiment") == exp_b_name)

    # 4. Join average data together
    comparison_df = exp_a_avg_data.join(
        exp_b_avg_data, on=entity_name, how="full", suffix="_B"
    ).with_columns(
        pl.col("AvgLogCount").fill_null(0.0),
        pl.col("AvgLogCount_B").fill_null(0.0),
    )

    # 5. Calculate average impact
    impact_df = (
        comparison_df.join(rarity_weights, on=entity_name, how="left")
        .with_columns(pl.col("Weight").fill_null(0.0))
        .with_columns(
            (pl.col("AvgLogCount") - pl.col("AvgLogCount_B")).alias("LogCountDiff"),
        )
        .with_columns(
            (pl.col("LogCountDiff") * pl.col("Weight")).alias("ScoreImpact"),
            (pl.lit(10.0).pow(pl.col("LogCountDiff").abs())).alias("TriggerRatio"),
        )
        .filter(pl.col("ScoreImpact").abs() > 0.01)  # Filter out noise
        .select(
            entity_name,
            pl.col("ScoreImpact").round(4),
            pl.col("Weight").round(4),
            pl.col("LogCountDiff").round(4),
            pl.col("TriggerRatio").round(2).alias("TriggerRatio(x)"),
            pl.col("AvgLogCount").round(4).alias(f"AvgLog_{exp_a_name[:10]}"),
            pl.col("AvgLogCount_B").round(4).alias(f"AvgLog_{exp_b_name[:10]}"),
        )
        .sort("ScoreImpact", descending=True)
    )

    logging.info(f"--- Top 25 advantages for '{exp_a_name}' over '{exp_b_name}' ---")
    print(impact_df.head(25))

    logging.info(f"--- Top 25 advantages for '{exp_b_name}' over '{exp_a_name}' ---")
    print(impact_df.tail(25).sort("ScoreImpact", descending=False))

    # Optionally, show run-level details for the most impactful optimizations.
    n_details = args.compare_run_details
    if n_details > 0:
        top_impact_opts = (
            impact_df.with_columns(pl.col("ScoreImpact").abs().alias("AbsImpact"))
            .sort("AbsImpact", descending=True)
            .head(n_details)[entity_name]
            .to_list()
        )

        logging.info(
            f"\n--- Run-level log10(Count+1) for Top {n_details} Impacting Optimizations ---"
        )

        run_level_data = (
            long_counts.filter(
                pl.col(entity_name).is_in(top_impact_opts)
                & pl.col("Experiment").is_in([exp_a_name, exp_b_name])
            )
            .select("Experiment", "Campaign", entity_name, "LogCount")
            # Pivot to show runs side-by-side
            .pivot(
                index=["Campaign", entity_name], columns="Experiment", values="LogCount"
            )
            # Add weight back for context
            .join(
                rarity_weights.select(entity_name, "Weight"), on=entity_name, how="left"
            )
            # Sort by Opt name, then Campaign for consistent view
            .sort("Weight", "Campaign", descending=True)
            .select(  # Reorder columns for clarity
                entity_name,
                pl.col("Weight").round(2),
                "Campaign",
                exp_a_name,
                exp_b_name,
            )
        )
        print(run_level_data)


def handle_opts(args: argparse.Namespace) -> None:
    """Handler for the 'opts' command."""
    experiment_dfs = combine_experiment_data(
        args.input_files,
        vocab_filename="opt_vocabulary.json",
        id_col="OptIDs",
        count_col="OptCounts",
        delete_cols=["PairIDs", "PairCounts"],
        name_map=args.name_map_dict,
    )
    final_counts = extract_final_campaign_counts(
        experiment_dfs, group=args.group_opts
    )  # Unmapped

    entity_name = "Optimization"
    run_cross_experiment_analysis(
        final_counts,  # Pass unmapped data
        entity_name=entity_name,
    )
    # Note: run_cross_experiment_analysis would also need mapping before print
    # if name_map_dict is used, but this is separate from uniqueness logic.

    print_uniqueness(
        entity_name,
        final_counts,  # Pass unmapped data
        args.show_contributors,
        args.name_map_dict,  # Pass the map
        args.plot_boxplot,  # Pass boxplot flag
        args.output,  # Pass output file
        args.show_exclusive_opts,
        args.exclusive_min_own_avg_log_count,
        args.exclusive_max_other_avg_log_count,
    )


def handle_pairs(args: argparse.Namespace) -> None:
    """Handler for the 'pairs' command."""
    experiment_dfs = combine_experiment_data(
        args.input_files,
        vocab_filename="pair_vocabulary.json",
        id_col="PairIDs",
        count_col="PairCounts",
        delete_cols=["OptIDs", "OptCounts"],
        name_map=args.name_map_dict,
    )
    final_counts = extract_final_campaign_counts(experiment_dfs)  # Unmapped

    entity_name = "OptimizationPair"
    run_cross_experiment_analysis(
        final_counts, entity_name=entity_name
    )  # Pass unmapped data

    # Call the unified print/plot function
    experiment_ratings, contributors, rarity_weights = print_uniqueness(
        entity_name,
        final_counts,  # Pass unmapped data
        args.show_contributors,
        args.name_map_dict,  # Pass the map
        args.plot_boxplot,
        args.output,
        args.show_exclusive_opts,
        args.exclusive_min_own_avg_log_count,
        args.exclusive_max_other_avg_log_count,
    )

    # --- The rest of this logic uses the unmapped 'contributors' and 'final_counts' ---
    # This is correct, as PMI/SDI should be calculated on raw data
    if contributors is None:  # print_uniqueness might not return it
        _, contributors, _, _ = uniqueness_analysis(
            entity_name, final_counts, args.show_contributors
        )
        if contributors is None:
            logging.warning("Cannot perform PMI analysis: no contributors found.")
            return

    opt_dfs = combine_experiment_data(
        args.input_files,
        vocab_filename="opt_vocabulary.json",
        id_col="OptIDs",
        count_col="OptCounts",
        delete_cols=["PairIDs", "PairCounts"],
    )
    opt_counts = extract_final_campaign_counts(opt_dfs)  # Unmapped

    split = final_counts.unpivot(
        index=["Experiment", "Campaign"],
        variable_name="Optimization",
        value_name="Count",
    ).with_columns(
        [
            pl.col("Optimization")
            .str.strip_chars("()")
            .str.split(", ")
            .list.get(0)
            .str.replace_all("'", "")
            .alias("first"),
            pl.col("Optimization")
            .str.strip_chars("()")
            .str.split(", ")
            .list.get(1)
            .str.replace_all("'", "")
            .alias("second"),
        ]
    )

    single_probs = (
        opt_counts.unpivot(
            index=["Experiment", "Campaign"],
            variable_name="Optimization",
            value_name="Count",
        )
        .group_by("Optimization")
        .agg(pl.sum("Count").alias("Count"))
        .filter(pl.col("Count") > 0)
        .with_columns(SingleProb=pl.col("Count") / pl.col("Count").sum())
    )
    pair_probs = (
        split.group_by("first", "second", "Optimization")
        .agg(pl.sum("Count").alias("Count"))
        .filter(pl.col("Count") > 0)
        .with_columns(PairProb=pl.col("Count") / pl.col("Count").sum())
    )
    result = (
        (
            pair_probs.join(
                single_probs,
                how="inner",
                left_on="first",
                right_on="Optimization",
                suffix="_left",
            )
            .join(
                single_probs,
                how="inner",
                left_on="second",
                right_on="Optimization",
                suffix="_right",
            )
            .with_columns(
                PMI=(
                    pl.col("PairProb")
                    / (pl.col("SingleProb") * pl.col("SingleProb_right")).log10()
                ),
                R_Pair=-pl.col("PairProb").log10(),
                R_Left=-pl.col("SingleProb").log10(),
                R_Right=-pl.col("SingleProb_right").log10(),
            )
            .with_columns(
                R_Max=pl.max_horizontal(["R_Left", "R_Right"]),
            )
        )
        .with_columns(
            Even=1
            - (
                (
                    (pl.col("R_Pair") - pl.col("R_Left")).abs()
                    - (pl.col("R_Pair") - pl.col("R_Right")).abs()
                ).abs()
                / (
                    (pl.col("R_Pair") - pl.col("R_Left")).abs()
                    + (pl.col("R_Pair") - pl.col("R_Right")).abs()
                )
            )
        )
        .with_columns(
            SDI=(
                (pl.col("R_Pair") - pl.max_horizontal("R_Left", "R_Right"))
                / (
                    pl.col("R_Left")
                    + pl.col("R_Right")
                    - pl.max_horizontal("R_Left", "R_Right")
                )
            )
        )
    )

    _, _, opt_weights, _ = uniqueness_analysis(
        "Optimization", opt_counts, args.show_contributors
    )

    opt_weights = dict(zip(opt_weights["Optimization"], opt_weights["Weight"]))

    def map_to_weights(pair):
        try:
            tup = eval(pair)
            return str(
                (
                    opt_weights.get(tup[0], 0.0),
                    opt_weights.get(tup[1], 0.0),
                )
            )
        except Exception:
            return str((0.0, 0.0))

    contributors_with_pmi = contributors.with_columns(
        pl.col("OptimizationPair")
        .map_elements(map_to_weights, return_dtype=pl.String)
        .alias("PairElemScores")
    ).join(
        result, left_on="OptimizationPair", right_on="Optimization", suffix="_other"
    )

    logging.info("--- Top Contributors with PMI/SDI data ---")
    print(contributors_with_pmi)


def handle_crossover(args: argparse.Namespace) -> None:
    """
    For each experiment, finds the crossover between its top N optimization
    contributors and the optimizations present in its top N pair contributors.
    """
    logging.info(f"Starting crossover analysis for Top {args.top_n} contributors...")

    logging.info("Loading and processing optimization data...")
    opts_dfs = combine_experiment_data(
        args.input_files,
        vocab_filename="opt_vocabulary.json",
        id_col="OptIDs",
        count_col="OptCounts",
        delete_cols=["PairIDs", "PairCounts"],
        name_map=args.name_map_dict,
    )
    final_opts_counts = extract_final_campaign_counts(opts_dfs)

    logging.info("Loading and processing optimization pair data...")
    pairs_dfs = combine_experiment_data(
        args.input_files,
        vocab_filename="pair_vocabulary.json",
        id_col="PairIDs",
        count_col="PairCounts",
        delete_cols=["OptIDs", "OptCounts"],
        name_map=args.name_map_dict,
    )
    final_pairs_counts = extract_final_campaign_counts(pairs_dfs)

    split = final_pairs_counts.unpivot(
        index=["Experiment", "Campaign"],
        variable_name="Optimization",
        value_name="Count",
    ).with_columns(
        [
            pl.col("Optimization")
            .str.strip_chars("()")
            .str.split(", ")
            .list.get(0)
            .str.replace_all("'", "")
            .alias("first"),
            pl.col("Optimization")
            .str.strip_chars("()")
            .str.split(", ")
            .list.get(1)
            .str.replace_all("'", "")
            .alias("second"),
        ]
    )

    single_probs = (
        final_opts_counts.unpivot(
            index=["Experiment", "Campaign"],
            variable_name="Optimization",
            value_name="Count",
        )
        .group_by("Optimization")
        .agg(pl.sum("Count"))
        .with_columns(SingleProb=pl.col("Count") / pl.col("Count").sum())
    )
    pair_probs = (
        split.group_by("first", "second")
        .agg(pl.sum("Count"))
        .with_columns(PairProb=pl.col("Count") / pl.col("Count").sum())
    )
    result = (
        pair_probs.join(
            single_probs,
            how="inner",
            left_on="first",
            right_on="Optimization",
            suffix="_left",
        )
        .join(
            single_probs,
            how="inner",
            left_on="second",
            right_on="Optimization",
            suffix="_right",
        )
        .with_columns(
            PMI=(
                pl.col("PairProb")
                / (pl.col("SingleProb") * pl.col("SingleProb_right")).log10()
            ),
            R_Pair=-pl.col("PairProb").log10(),
            R_Left=-pl.col("SingleProb").log10(),
            R_Right=-pl.col("SingleProb_right").log10(),
        )
        .with_columns(
            R_Max=pl.max_horizontal(["R_Left", "R_Right"]),
        )
    )

    print("=== Pearson Correlation Coefficient R_Pair vs R_Max ===")
    print(np.corrcoef(result["R_Pair"].to_numpy(), result["R_Max"].to_numpy()))

    print("=== Spearman Correlation Coefficient R_Pair vs R_Max ===")
    from scipy.stats import spearmanr

    rpairmaxcorrelation = spearmanr(
        result["R_Pair"].to_numpy(), result["R_Max"].to_numpy()
    )
    print(rpairmaxcorrelation)
    utils.define_latex_var(args, "rpairvsmaxcorrelation", rpairmaxcorrelation[0])

    pair_elem_predictors = np.column_stack(
        [
            np.ones(result.height),
            result["R_Left"].to_numpy(),
            result["R_Right"].to_numpy(),
        ]
    )
    pair_prob = result["R_Pair"].to_numpy()
    beta, residuals, rank, s = np.linalg.lstsq(
        pair_elem_predictors, pair_prob, rcond=None
    )  # beta = [alpha, beta1, beta2]
    yhat = pair_elem_predictors @ beta
    residuals2 = pair_prob - yhat
    ss_res = np.sum(residuals2**2)
    ss_tot = np.sum((pair_prob - pair_prob.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    regression_summary = {
        "alpha": float(beta[0]),
        "beta_R_A": float(beta[1]),
        "beta_R_X": float(beta[2]),
        "R2": float(r2),
        "n_pairs": int(result.height),
    }

    r_pair = result["R_Pair"].to_numpy()
    r_max = result["R_Max"].to_numpy()

    corr_Rpair_Rmax = float(np.corrcoef(r_pair, r_max)[0, 1])

    print("=== Regression R_pair ~ R_A + R_X ===")
    for k, v in regression_summary.items():
        print(f"{k}: {v}")

    utils.define_latex_var(args, "regressionr", r2)

    print("\n=== Correlation tests ===")
    print(f"corr(R_pair, R_max): {corr_Rpair_Rmax:.4f}")

    # -----------------------------
    # 6) OPTIONAL: PMI distribution quick stats
    # -----------------------------
    pmi_stats = result.select(
        [
            pl.col("PMI").mean().alias("PMI_mean"),
            pl.col("PMI").median().alias("PMI_median"),
            pl.col("PMI").std().alias("PMI_std"),
            pl.quantile("PMI", 0.1).alias("PMI_p10"),
            pl.quantile("PMI", 0.9).alias("PMI_p90"),
        ]
    )
    print("\n=== PMI stats ===")
    print(pmi_stats)

    p_star_left = pair_probs.group_by("first").agg(
        pl.col("PairProb").sum().alias("P_Left*")
    )
    p_star_right = pair_probs.group_by("second").agg(
        pl.col("PairProb").sum().alias("P_Right*")
    )

    pairs_star = (
        pair_probs.join(p_star_left, on="first", how="left")
        .join(p_star_right, on="second", how="left")
        .with_columns(
            (pl.col("PairProb") / (pl.col("P_Left*") * pl.col("P_Right*")))
            .log10()
            .alias("PMI_star"),
        )
    )

    pmi_stats = pairs_star.select(
        [
            pl.col("PMI_star").mean().alias("PMI_mean"),
            pl.col("PMI_star").median().alias("PMI_median"),
            pl.col("PMI_star").std().alias("PMI_std"),
            pl.quantile("PMI_star", 0.1).alias("PMI_p10"),
            pl.quantile("PMI_star", 0.9).alias("PMI_p90"),
        ]
    )
    print("\n=== PMI* stats ===")
    print(pmi_stats)
    utils.define_latex_var(args, "pmimean", pmi_stats["PMI_mean"][0])

    if final_opts_counts.is_empty() or final_pairs_counts.is_empty():
        logging.warning("No data found for opts or pairs. Exiting crossover analysis.")
        return

    experiment_names = final_opts_counts["Experiment"].unique().to_list()
    results = []

    opts_uniqueness, opts_contributors, _, _ = uniqueness_analysis(
        "Optimization", final_opts_counts, args.top_n
    )
    pairs_uniqueness, pairs_contributors, _, _ = uniqueness_analysis(
        "OptimizationPair", final_pairs_counts, args.top_n
    )

    opts_order = opts_uniqueness.select(
        "Experiment", Order=pl.int_range(1, pl.len() + 1)
    )
    pairs_order = pairs_uniqueness.select(
        "Experiment", Order=pl.int_range(1, pl.len() + 1)
    )

    merged = opts_uniqueness.join(pairs_uniqueness, on="Experiment", how="inner")
    experiment_rank_correlation = spearmanr(
        merged["Median_Uniqueness"].to_numpy(),
        merged["Median_Uniqueness_right"].to_numpy(),
    )

    print("=== Experiment Rank Correlation ===")
    print(experiment_rank_correlation)
    utils.define_latex_var(
        args, "experimentrankcorrelation", experiment_rank_correlation[0]
    )

    # Ensure contributors were found before proceeding
    if opts_contributors is None or pairs_contributors is None:
        logging.warning(
            "Could not calculate contributors, skipping crossover analysis."
        )
        return

    logging.info("Calculating crossover for each experiment...")
    for exp_name in tqdm(experiment_names, desc="Analyzing Experiments"):
        top_pairs = (
            pairs_contributors.filter(pl.col("Experiment") == exp_name)
            .drop("Experiment")["OptimizationPair"]
            .to_list()
        )
        top_opts = (
            opts_contributors.filter(pl.col("Experiment") == exp_name)
            .drop("Experiment")["Optimization"]
            .to_list()
        )

        crossovers = list(
            [pair for pair in top_pairs if any([opt in pair for opt in top_opts])]
        )

        crossover_ratio = (
            (len(crossovers) / len(top_pairs)) * 100 if len(top_pairs) > 0 else 0
        )

        results.append(
            {
                "Experiment": exp_name,
                "OptUniqueness": opts_uniqueness.filter(
                    pl.col("Experiment") == exp_name
                )["Median_Uniqueness"][0],
                "OptUniquenessOrder": opts_order.filter(
                    pl.col("Experiment") == exp_name
                )["Order"][0],
                "PairUniqueness": pairs_uniqueness.filter(
                    pl.col("Experiment") == exp_name
                )["Median_Uniqueness"][0],
                "PairsUniquenessOrder": pairs_order.filter(
                    pl.col("Experiment") == exp_name
                )["Order"][0],
                "CrossoverCount": len(crossovers),
                "CrossoverRatio": f"{crossover_ratio:.0f}",
                "CrossoverOptimizations": sorted(crossovers),
            }
        )

    if not results:
        logging.warning("No results to display.")
        return

    results_df = pl.from_dicts(results).sort("OptUniquenessOrder", descending=False)

    logging.info(f"\n--- Crossover Analysis Results (Top {args.top_n}) ---")
    print(results_df)

    latex_table = results_df.select(
        pl.col("Experiment"),
        pl.col("OptUniqueness").alias("Optimization Score"),
        pl.col("PairUniqueness").alias("Opt. Pair Score"),
        pl.col("PairsUniquenessOrder").alias("Opt. Pair Rank"),
    )

    utils.define_latex_table(args, "optimization-scores", latex_table)


def main(argv: list[str] | None = None) -> None:
    """Main function to parse arguments and dispatch commands."""
    parser = argparse.ArgumentParser(
        description="""This script provides a suite of tools for analyzing optimization and optimization pair
data from preprocessed fuzzing campaign summaries. It can perform several types of
analysis, including:
- Cross-experiment counts: Aggregating total counts of optimizations/pairs.
- Uniqueness analysis: Calculating a uniqueness score for each experiment based on
  the rarity of its optimizations/pairs. This can be visualized as a boxplot or
  a timeline.
- Differential analysis: Comparing two experiments to find optimizations that
  contribute most to the difference in their uniqueness scores.
- Crossover analysis: Examining the intersection between top contributing
  optimizations and top contributing optimization pairs.
""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--name-map",
        type=str,
        default=None,
        help="Path to a JSON file mapping experiment folder names to display names.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename for plots (e.g., 'my_plot.png' or 'my_plot.pgf').",
    )

    parser.add_argument(
        "--group-opts",
        default=None,
        action="store_true",
        help="Group optimizations belonging to the same phase",
    )

    parser.add_argument(
        "--show-exclusive-opts",
        action="store_true",
        help="Show a table of optimizations that are (almost) exclusive to each experiment.",
    )
    parser.add_argument(
        "--exclusive-min-own-avg-log-count",
        type=float,
        default=0.2,
        help="Minimum average log10(count) in an experiment's own runs to be considered 'exclusive' (default: 0.2).",
    )
    parser.add_argument(
        "--exclusive-max-other-avg-log-count",
        type=float,
        default=0.05,
        help="Maximum average log10(count) in *other* experiments to be considered 'exclusive' (default: 0.05).",
    )

    parser.add_argument(
        "--latex-output", type=str, default=None, help="LaTeX output filename."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- opts sub-command ---
    p_opts = subparsers.add_parser(
        "opts",
        help="Analyze optimization counts and final uniqueness.",
        description="""Analyzes individual optimizations. This includes calculating cross-experiment
statistics, and performing uniqueness analysis based on the rarity of optimizations.
It can display top contributors to uniqueness and plot the final uniqueness
distribution as a boxplot.""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_shared_args(p_opts)
    p_opts.set_defaults(func=handle_opts)

    # --- pairs sub-command ---
    p_pairs = subparsers.add_parser(
        "pairs",
        help="Analyze optimization pair counts and final uniqueness.",
        description="""Analyzes co-occurring pairs of optimizations. This includes calculating
cross-experiment statistics and performing uniqueness analysis based on the
rarity of pairs. It can also calculate and display PMI (Pointwise Mutual Information)
and SDI (Symmetric Dependence from Independence) for top contributing pairs.""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_shared_args(p_pairs)
    p_pairs.set_defaults(func=handle_pairs)

    p_consistency = subparsers.add_parser(
        "consistency",
        help="Calculate internal consistency of runs within each experiment.",
        description="""Calculates the internal consistency of runs within each experiment.
This is done by creating weighted contribution vectors for each run,
based on rare optimizations/pairs, and then calculating the average
cosine similarity between these vectors. It provides both average
centroid similarity and average pairwise similarity.""",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    add_shared_args(p_consistency)

    p_consistency.add_argument(
        "--entity",
        choices=["opts", "pairs"],
        required=True,
        help="Specify whether to analyze 'opts' or 'pairs'.",
    )

    p_consistency.add_argument(
        "--consistency-weight-threshold",
        type=float,
        default=1.0,
        help="Weight (rarity) threshold for including optimizations in consistency calculation (default: 1.0).",
    )

    p_consistency.set_defaults(func=handle_consistency)

    # Sub-parser for comparing two specific experiments.
    p_compare = subparsers.add_parser(
        "compare",
        help="Show a differential analysis between two specific experiments.",
        description="""Performs a differential analysis between two specific experiments.
It identifies the optimizations or pairs that contribute most to the
difference in uniqueness scores between the two experiments. The output
highlights the top advantages for each experiment and can show run-level
details for the most impactful entities.""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p_compare.add_argument(
        "experiments",
        nargs=2,
        metavar=("EXP_A", "EXP_B"),
        help="The two experiment names to compare.",
    )
    p_compare.add_argument(
        "--compare-run-details",
        type=int,
        default=5,
        metavar="N",
        help="Show run-level log10(Count+1) details for the top N impacting optimizations (default: 5).",
    )
    p_compare.add_argument(
        "--entity",
        choices=["opts", "pairs"],
        required=True,
        help="Specify whether to analyze 'opts' or 'pairs'.",
    )
    p_compare.add_argument("input_files", nargs="+", help="Paths to summary files.")
    p_compare.set_defaults(func=handle_compare)

    # --- crossover sub-command ---
    p_crossover = subparsers.add_parser(
        "crossover",
        help="Analyze the crossover between top optimizations and top optimization pairs.",
        description="""Analyzes the crossover between the top N contributing optimizations and the
optimizations present in the top N contributing optimization pairs for each
experiment. This helps understand the relationship between individual
optimization importance and their importance in pairs.""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p_crossover.add_argument(
        "input_files",
        nargs="+",
        help="One or more paths to 'preprocessed_summary.parquet' files.",
    )
    p_crossover.add_argument(
        "--top-n",
        type=int,
        default=20,
        metavar="N",
        help="The number of top contributors to consider for both opts and pairs (default: 20).",
    )
    p_crossover.set_defaults(func=handle_crossover)

    # --- uniqueness-timeline sub-command ---
    p_ut = subparsers.add_parser(
        "uniqueness-timeline",
        help="Analyze uniqueness scores over time.",
        description="""Analyzes and plots the evolution of uniqueness scores over the duration
of the fuzzing campaigns. It samples the state of each campaign at regular
time intervals, calculates the uniqueness at each step, and can plot the
resulting timeline as an overlay of all experiments or as a grid of
individual experiment plots.""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p_ut.add_argument(
        "--entity",
        choices=["opts", "pairs"],
        required=True,
        help="Specify whether to analyze 'opts' or 'pairs'.",
    )
    p_ut.add_argument(
        "input_files",
        nargs="+",
        help="One or more paths to 'preprocessed_summary.parquet' files.",
    )
    p_ut.add_argument(
        "--time-step",
        type=int,
        default=60,
        help="Time step in minutes for analysis (default: 60).",
    )
    p_ut.add_argument(
        "--plot",
        action="store_true",
        help="Generate and show a plot of the uniqueness timeline.",
    )
    p_ut.add_argument(
        "--show-campaign-range",
        action="store_true",
        help="Show shaded min/max uniqueness range of campaigns on the plot.",
    )
    p_ut.add_argument(
        "--plot-mode",
        choices=["overlay", "grid"],
        default="overlay",
        help="Plotting mode for the uniqueness timeline: 'overlay' or 'grid'.",
    )
    p_ut.set_defaults(func=handle_uniqueness_timeline)

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s - %(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    name_map = utils.load_name_map(args.name_map)
    args.name_map_dict = name_map

    args.func(args)


if __name__ == "__main__":
    main()
