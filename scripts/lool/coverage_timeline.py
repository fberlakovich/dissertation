import argparse
import json
import logging
import sys
from datetime import timedelta, datetime
from pathlib import Path

import colorcet as cc
import matplot2tikz
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import polars as pl
import seaborn as sns

from scripts.utils import define_latex_table

pl.Config.set_fmt_str_lengths(1200)
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(1200)


def plot_combined_coverage_timeline(
    ranking_info: list[tuple[str, pl.DataFrame, float]],
    check_cols: list[str],
    min_hours: float,
    max_hours: float,
    output: str | None = None,
    show_campaigns: bool = False,
    num_ticks: int = 24,
    tick_label_fontsize: int = 16,
    args: argparse.Namespace = None,
):
    """
    Generates and saves a plot comparing multiple coverage timelines.
    (This version uses a single pane with dynamic Y-limits)

    Args:
        plot_data_list: A list of tuples, where each tuple contains a
                        name for the experiment and its timeline stats DataFrame.
        min_hours: The minimum time in hours for the x-axis.
        max_hours: The maximum time in hours for the x-axis.
        output: Optional path to save the output file.
    """

    if show_campaigns:
        check_cols.extend(["Coverage_Q1", "Coverage_Q3"])

    palette = sns.color_palette(cc.glasbey, n_colors=len(ranking_info))

    bottom_min_calc = float("inf")
    bottom_max_calc = float("-inf")

    for _, df, _ in ranking_info:
        # Filter data to be within the plot's x-axis range
        df_in_range = df.filter(
            (pl.col("ElapsedHours") >= min_hours)
            & (pl.col("ElapsedHours") <= max_hours)
        )

        for col in check_cols:
            if col in df_in_range.columns:
                # Get stats for all data in range
                col_stats = df_in_range.select(
                    pl.col(col).min().alias("min"), pl.col(col).max().alias("max")
                )
                col_min = col_stats.item(0, "min")
                col_max = col_stats.item(0, "max")

                if col_min is not None:
                    bottom_min_calc = min(bottom_min_calc, col_min)
                if col_max is not None:
                    bottom_max_calc = max(bottom_max_calc, col_max)

    # Handle edge cases
    if bottom_min_calc == float("inf"):
        bottom_min_calc = 0.0
    if bottom_max_calc == float("-inf"):
        bottom_max_calc = bottom_min_calc + 1.0  # Default range if empty

    data_range = bottom_max_calc - bottom_min_calc
    # Add a 5% margin, but ensure a minimum margin if range is 0 or negative
    margin = max(data_range * 0.01, abs(bottom_min_calc * 0.01), 1.0)
    if data_range == 0:
        margin = abs(bottom_min_calc * 0.05) if bottom_min_calc != 0 else 1.0

    bottom_ylim = (bottom_min_calc - margin, bottom_max_calc + margin)

    fig, ax_bottom = plt.subplots(figsize=(12, 8))
    ax_bottom.set_ylim(bottom_ylim)

    for rank, (name, plot_df, _final_y) in enumerate(ranking_info, start=1):
        color = palette[rank - 1]
        base_label_name = name
        legend_label = f"{rank}. \\propername{{{base_label_name}}}"
        plot_df_pd = plot_df.to_pandas()

        # Always plot on the bottom axis
        sns.lineplot(
            x="ElapsedHours",
            y="MedianCoverage",
            data=plot_df_pd,
            ax=ax_bottom,
            label=legend_label,
            color=color,
            linewidth=2.5,
        )

        if show_campaigns:
            ax_bottom.fill_between(
                plot_df_pd["ElapsedHours"],
                plot_df_pd["Coverage_Q1"],
                plot_df_pd["Coverage_Q3"],
                alpha=0.15,
                color=color,
            )

    y_formatter = mticker.FuncFormatter(lambda v, _pos=None: rf"\num{{{int(v)}}}")

    if max_hours is not None and max_hours > min_hours:
        time_window_hours = float(max_hours - min_hours)
        tick_interval_hours = time_window_hours / num_ticks
        ax_bottom.set_xlim(float(min_hours), float(max_hours))
        ax_bottom.xaxis.set_major_locator(mticker.MultipleLocator(tick_interval_hours))

        def _format_time_from_hours(v: float, _pos=None) -> str:
            if v < 0:
                return ""
            total_seconds = int(round(v * 3600))
            h, rem = divmod(total_seconds, 3600)
            m, s = divmod(rem, 60)
            if tick_interval_hours >= 1.0:
                return f"{int(round(v))}"
            elif tick_interval_hours >= (1.0 / 60.0):
                return f"{h}:{m:02d}"
            else:
                return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"

        ax_bottom.xaxis.set_major_formatter(
            mticker.FuncFormatter(_format_time_from_hours)
        )
    else:
        ax_bottom.xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax_bottom.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _pos=None: f"{int(round(v))}")
        )

    # --- Formatting for SINGLE axis plot ---
    ax_bottom.set_title(
        "Cumulative Coverage Comparison Over Relative Time", fontsize=16
    )
    ax_bottom.set_xlabel("Time Elapsed (Hours)", fontsize=16)
    ax_bottom.set_ylabel("Cumulative Coverage", fontsize=16)

    handles, labels = ax_bottom.get_legend_handles_labels()
    from collections import OrderedDict

    legend_dict = OrderedDict(zip(labels, handles))
    ax_bottom.legend(legend_dict.values(), legend_dict.keys(), title="Experiment")

    ax_bottom.tick_params(axis="both", labelsize=tick_label_fontsize)
    ax_bottom.yaxis.set_major_formatter(y_formatter)
    ax_bottom.grid(
        True,
        which="major",
        axis="x",
        linestyle=":",
        linewidth=1,
        color="grey",
        alpha=0.6,
    )

    plt.tight_layout()

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        matplot2tikz.save(output_path, show_info=True)
        if args and args.save_as_image:
            image_output_path = output_path.with_suffix(".png")
            fig.savefig(image_output_path, dpi=300)
            logging.info(f"Plot also saved as {image_output_path}")

    plt.show()

    logging.info(f"Combined coverage timeline plot logic finished for '{output_path}'")


def plot_individual_campaigns_timeline(
    experiment_name: str,
    timeline_stats: pl.DataFrame,
    coverage_column_base: str,
    min_hours: float,
    max_hours: float,
    output: str | None = None,
    num_ticks: int = 24,
    tick_label_fontsize: int = 16,
):
    """
    Generates and saves a plot showing all individual campaign timelines for a
    single experiment.

    Args:
        experiment_name: The name of the experiment.
        timeline_stats: The timeline stats DataFrame, which must contain
                        pivoted columns for each campaign.
        coverage_column_base: The base name of the coverage column (e.g., "CumulativeCoverage").
        min_hours: The minimum time in hours for the x-axis.
        max_hours: The maximum time in hours for the x-axis.
        output: Optional path to save the output file.
        num_ticks: Number of ticks on the x-axis.
        tick_label_fontsize: Font size for tick labels.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    plot_df = timeline_stats.clone().with_columns(
        (pl.col("Elapsed").dt.total_seconds() / 3600).alias("ElapsedHours")
    )
    plot_df = plot_df.sort("ElapsedHours")

    campaign_cols = [
        c for c in plot_df.columns if c.startswith(f"{coverage_column_base}_Campaign_")
    ]

    if not campaign_cols:
        logging.error(
            "Could not find individual campaign columns to plot. "
            "Ensure 'calculate_timeline_stats' was run with show_campaigns=True."
        )
        return

    melted_df = plot_df.melt(
        id_vars=["ElapsedHours"],
        value_vars=campaign_cols,
        variable_name="Campaign",
        value_name="Coverage",
    )

    plot_df_pd = plot_df.to_pandas()
    melted_df_pd = melted_df.to_pandas()

    # Plot all individual campaign lines
    sns.lineplot(
        x="ElapsedHours",
        y="Coverage",
        hue="Campaign",
        data=melted_df_pd,
        ax=ax,
        legend=False,  # Hide legend for individual lines
        alpha=0.5,
        linewidth=1.0,
        palette=sns.color_palette(cc.glasbey, n_colors=len(campaign_cols)),
    )

    # Plot the median line on top
    sns.lineplot(
        x="ElapsedHours",
        y="MedianCoverage",
        data=plot_df_pd,
        ax=ax,
        label="Median",
        color="black",
        linewidth=1.0,
        linestyle="--",
    )

    title_name = experiment_name
    ax.set_title(f"Individual Campaign Coverage for '{title_name}'", fontsize=16)
    ax.set_xlabel("Time Elapsed (Hours)", fontsize=16)
    ax.set_ylabel("Cumulative Coverage", fontsize=16)
    ax.legend(title="Legend")

    # Increase tick label font sizes
    ax.tick_params(axis="both", labelsize=tick_label_fontsize)
    # Apply LaTeX formatting to y-axis ticks
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _pos=None: rf"\num{{{int(v)}}}")
    )

    # --- X-axis time formatting (copied from combined plot) ---
    if max_hours is not None and max_hours > min_hours:
        time_window_hours = float(max_hours - min_hours)
        tick_interval_hours = time_window_hours / num_ticks
        ax.set_xlim(float(min_hours), float(max_hours))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(tick_interval_hours))

        def _format_time_from_hours(v: float, _pos=None) -> str:
            if v < 0:
                return ""
            total_seconds = int(round(v * 3600))
            h, rem = divmod(total_seconds, 3600)
            m, s = divmod(rem, 60)

            if tick_interval_hours >= 1.0:
                return f"{int(round(v))}"
            elif tick_interval_hours >= (1.0 / 60.0):
                return f"{h}:{m:02d}"
            else:
                return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_time_from_hours))
    else:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _pos=None: f"{int(round(v))}")
        )

    ax.grid(
        True,
        which="major",
        axis="x",
        linestyle=":",
        linewidth=1,
        color="grey",
        alpha=0.6,
    )
    plt.tight_layout()

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        matplot2tikz.save(output_path)
        logging.info(f"Individual campaign plot saved to '{output_path}'")

    plt.show()


def plot_final_coverage_boxplot(
    boxplot_data_df: pl.DataFrame,
    ranking_info: list[tuple[str, pl.DataFrame, float]],
    coverage_type_name: str,
    max_hours: float,
    output: str | None = None,
    tick_label_fontsize: int = 16,
):
    """
    Generates and saves a boxplot of the final coverage distribution for
    each experiment, ordered by the median final coverage.

    Args:
        boxplot_data_df: DataFrame with columns "Campaign", "FinalCoverage", "Experiment".
        ranking_info: The sorted list of tuples (name, df, final_y) used for ordering.
        coverage_type_name: Readable name for the coverage (e.g., "Cumulative Code Coverage").
        max_hours: The max_hours cutoff used.
        output: Optional base path to save the output file.
        tick_label_fontsize: Font size for tick labels.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get the sorted list of experiment names from the ranking info
    sorted_names = [name for name, _, _ in ranking_info]
    palette = sns.color_palette(cc.glasbey, n_colors=len(sorted_names))

    plot_df_pd = boxplot_data_df.to_pandas()

    # Plot the boxplots, ordered by the ranking
    sns.boxplot(
        x="Experiment",
        y="FinalCoverage",
        data=plot_df_pd,
        ax=ax,
        order=sorted_names,  # Use PGF names for order
        palette=palette,
        showfliers=False,  # The stripplot will show outliers
    )

    # Overlay a stripplot to show individual campaign data points
    sns.stripplot(
        x="Experiment",
        y="FinalCoverage",
        data=plot_df_pd,
        ax=ax,
        order=sorted_names,  # Use PGF names for order
        color=".25",
        size=3,
        alpha=0.6,
    )

    title_name = coverage_type_name

    ax.set_title(f"Final {title_name} Distribution (at {max_hours:.1f}h)", fontsize=16)
    ax.set_xlabel("Experiment", fontsize=16)
    ax.set_ylabel(title_name, fontsize=16)

    # Rotate x-labels if they are long or numerous
    if len(sorted_names) > 5:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Formatting
    ax.tick_params(axis="both", labelsize=tick_label_fontsize)
    y_formatter = mticker.FuncFormatter(lambda v, _pos=None: rf"\num{{{int(v)}}}")
    ax.yaxis.set_major_formatter(y_formatter)
    ax.grid(
        True,
        which="major",
        axis="y",  # Horizontal gridlines
        linestyle=":",
        linewidth=1,
        color="grey",
        alpha=0.6,
    )

    plt.tight_layout()

    if output:
        output_path = Path(output)
        new_stem = output_path.stem + "_boxplot"
        output_path = output_path.with_stem(new_stem)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        matplot2tikz.save(output_path, show_info=True)
        logging.info(f"Final coverage boxplot saved to '{output_path}'")


def calculate_timeline_stats(
    df: pl.DataFrame,
    time_step_minutes: int,
    show_campaigns: bool,
    min_hours: float | None = None,
    max_hours: float | None = None,
    column="CumulativeCoverage",
) -> pl.DataFrame:
    """
    Calculates the timeline statistics for a single experiment DataFrame.
    """
    if df.is_empty():
        return pl.DataFrame()

    df = df.sort(["Campaign", "Time"]).with_columns(
        pl.col(column).cum_max().over("Campaign"),
        (pl.col("Time").cum_count().over("Campaign") + 1).alias("CumulativeTests"),
    )

    max_elapsed = df.select(pl.col("Elapsed").max()).item()
    if max_elapsed is not None:
        end_time = max_elapsed
        if max_hours is not None:
            end_time = min(end_time, timedelta(hours=float(max_hours)))
    else:
        end_time = (
            timedelta(hours=float(max_hours))
            if max_hours is not None
            else timedelta(hours=24)
        )
    start_time = (
        timedelta(hours=float(min_hours))
        if min_hours is not None and min_hours > 0
        else timedelta(seconds=0)
    )

    epoch = datetime(1970, 1, 1)
    time_bins_dt = pl.datetime_range(
        start=epoch + start_time,
        end=epoch + end_time,
        interval=f"{time_step_minutes}m",
        eager=True,
    )
    time_bins = time_bins_dt - epoch

    bins_df = pl.DataFrame({"Elapsed": time_bins})

    unique_campaigns = df.select(pl.col("Campaign").unique())
    grid_df = bins_df.join(unique_campaigns, how="cross")
    df_sorted = df.sort("Campaign", "Elapsed")

    merged_df = grid_df.join_asof(
        df_sorted.select(["Elapsed", "Campaign", column, "CumulativeTests"]),
        on="Elapsed",
        by="Campaign",
    )

    timeline_stats = merged_df.group_by("Elapsed").agg(
        pl.col(column).min().alias("MinCoverage"),
        pl.col(column).quantile(0.25).alias("Coverage_Q1"),
        pl.col(column).median().alias("MedianCoverage"),
        pl.col(column).quantile(0.75).alias("Coverage_Q3"),
        pl.col(column).max().alias("MaxCoverage"),
        pl.col("CumulativeTests").median().alias("MedianTests"),
    )
    timeline_stats = timeline_stats.with_columns(
        pl.col("MedianTests").fill_null(0).cast(pl.Int64)
    ).sort("Elapsed")

    if show_campaigns:
        # The 'columns' argument is deprecated and replaced with 'on'.
        campaign_pivot_df = merged_df.pivot(
            index="Elapsed", on="Campaign", values=[column, "CumulativeTests"]
        )
        renamed_cols = {}
        for col_name in campaign_pivot_df.columns:
            if col_name != "Elapsed":
                parts = col_name.split("_", 1)
                if len(parts) == 2:
                    value_name, campaign_id = parts
                    new_name = f"{value_name}_Campaign_{campaign_id}"
                    renamed_cols[col_name] = new_name
        campaign_pivot_df = campaign_pivot_df.rename(renamed_cols)
        timeline_stats = timeline_stats.join(campaign_pivot_df, on="Elapsed")

    return timeline_stats


def print_binned_summary_table(
    args, ranking_info: list[tuple[str, pl.DataFrame, float]], time_step_minutes: int
):
    """
    Prints a table of median coverage at each time bin, with one column
    per experiment, sorted by final rank.
    """
    if not ranking_info:
        return

    if not ranking_info[0][1].is_empty():
        # Select both, as Elapsed is needed for joining
        base_df = ranking_info[0][1].select("Elapsed", "ElapsedHours")
    else:
        logging.warning("First experiment data is empty, cannot build summary table.")
        return

    for name, timeline_df, _ in ranking_info:
        if "MedianCoverage" not in timeline_df.columns:
            continue

        # Prepare the experiment's data
        exp_df = timeline_df.select(
            "Elapsed",  # Use Elapsed for joining
            pl.col("MedianCoverage")
            .fill_null(0)
            .fill_nan(0)
            .cast(pl.Int64)
            .alias(name),
        )

        # Join onto the base DataFrame
        if name not in base_df.columns:
            base_df = base_df.join(exp_df, on="Elapsed", how="outer_coalesce")

    # Sort by ElapsedHours
    final_table = base_df.sort("ElapsedHours")

    tick_interval_hours = time_step_minutes / 60.0

    def _format_time_from_hours(v: float) -> str:
        if v < 0:
            return ""
        total_seconds = int(round(v * 3600))
        h, rem = divmod(total_seconds, 3600)
        m, s = divmod(rem, 60)

        # This logic matches the plot axis formatter
        if tick_interval_hours >= 1.0:
            return f"{int(round(v))}h"
        elif tick_interval_hours >= (1.0 / 60.0):
            return f"{h}:{m:02d}"
        else:
            return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"

    final_table = final_table.with_columns(
        pl.col("ElapsedHours")
        .map_elements(_format_time_from_hours, return_dtype=pl.String)
        .alias("Time")
    )

    all_columns = final_table.columns

    # Get ranked names from ranking_info
    sorted_names = [name for name, _, _ in ranking_info if name in all_columns]

    final_table = final_table.select(
        pl.col("Time"),  # Use new formatted time column
        *sorted_names,  # Select columns in ranked order
    )

    logging.info(f"\n--- Median Coverage by Time Bin (Ranked) ---")
    print(final_table)
    define_latex_table(args, "coverage-development", final_table)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze median cumulative code coverage over a relative timeline.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="One or more paths to 'preprocessed_summary.parquet' files.",
    )
    parser.add_argument(
        "--time-step",
        type=int,
        default=30,
        help="Time step in minutes for analysis (default: 30).",
    )
    parser.add_argument(
        "--show-campaigns",
        action="store_true",
        help="Include a column for each individual campaign in the output table.\n"
        "If combined with --plot and a *single* input file, this will plot all\n"
        "individual campaign timelines instead of the median comparison plot.",
    )
    parser.add_argument(
        "--type",
        choices=["code", "global", "method"],
        default="code",
        help="The type of coverage to plot",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a plot of the coverage timeline(s).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename for the saved plot.",
    )
    parser.add_argument(
        "--latex-output", type=str, default=None, help="LaTeX output filename."
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=24.0,
        help="Maximum time horizon in hours to analyze/plot (default: 24.0).",
    )
    parser.add_argument(
        "--min-hours",
        type=float,
        default=0.0,
        help="Minimum time horizon in hours to analyze/plot (default: 0.0).",
    )
    parser.add_argument(
        "--num-ticks",
        type=int,
        default=24,
        help="Number of ticks on the x axis (default: 24).",
    )
    parser.add_argument(
        "--tick-label-fontsize",
        type=int,
        default=12,
        help="Font size for x/y tick labels (default: 12).",
    )
    parser.add_argument(
        "--name-map",
        type=str,
        default=None,
        help="Path to a JSON file mapping folder names to readable experiment names.",
    )
    parser.add_argument(
        "--save-as-image",
        action="store_true",
        help="Save the plot as a PNG image in addition to the TikZ figure.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s - %(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    name_map = {}
    if args.name_map:
        name_map_path = Path(args.name_map)
        if name_map_path.exists():
            try:
                with open(name_map_path, "r") as f:
                    name_map = json.load(f)
                logging.info(f"Loaded name mappings from '{name_map_path}'")
            except json.JSONDecodeError:
                logging.error(f"Error: Could not decode JSON from '{name_map_path}'")
        else:
            logging.warning(f"Warning: Name map file not found at '{name_map_path}'")

    plot_data_for_combined_graph = []
    all_final_coverages = []
    single_file_campaign_plot = (
        args.plot and args.show_campaigns and len(args.input_files) == 1
    )

    for file_path_str in args.input_files:
        input_path = Path(file_path_str)
        if not input_path.exists():
            logging.error(f"Error: Input file not found at '{input_path}'")
            continue

        folder_name = input_path.parent.name
        experiment_name = name_map.get(folder_name, folder_name)
        logging.info(f"Processing '{input_path}' (Experiment: '{experiment_name}')...")

        df = (
            pl.read_parquet(input_path)
            .with_columns(pl.col("Time").cast(pl.Datetime))
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
                ]
            )
        )
        df = df.with_columns(
            pl.duration(
                milliseconds=(
                    pl.col("StartTime") - pl.col("StartTime").min().over("Campaign")
                )
            ).alias("Elapsed")
        )

        column = {
            "code": "CumulativeCoverage",
            "global": "CumulativeGlobalCounterCoverage",
            "method": "CumulativeMethodCounterCoverage",
        }.get(args.type, "CumulativeCoverage")

        if args.plot and not single_file_campaign_plot:
            df_with_elapsed_hours = df.with_columns(
                (pl.col("Elapsed").dt.total_seconds() / 3600).alias("ElapsedHours"),
                pl.col(column).cum_max().over("Campaign").alias(column),
            )
            df_filtered = df_with_elapsed_hours.filter(
                pl.col("ElapsedHours") <= args.max_hours
            )

            # Ensure we have data after filtering
            if df_filtered.is_empty() and not df_with_elapsed_hours.is_empty():
                # If max_hours filtered everything, take the latest available from *all* data
                logging.warning(
                    f"No data within max_hours for {experiment_name}. "
                    "Using latest available data for boxplot."
                )
                df_filtered = df_with_elapsed_hours
            elif df_filtered.is_empty():
                logging.warning(f"No data for {experiment_name} to add to boxplot.")
                continue

            # Get the max coverage *per campaign* within the time limit
            final_cov_df = (
                df_filtered.group_by("Campaign")
                .agg(pl.col(column).max().alias("FinalCoverage"))
                .with_columns(pl.lit(experiment_name).alias("Experiment"))
            )
            all_final_coverages.append(final_cov_df)

        timeline_stats = calculate_timeline_stats(
            df,
            args.time_step,
            args.show_campaigns,
            args.min_hours,
            args.max_hours,
            column,
        )

        logging.info(f"\n--- Cumulative Coverage For: {experiment_name} ---")

        total_seconds = pl.col("Elapsed").dt.total_seconds()
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        timeline_stats_printable = timeline_stats.clone().with_columns(
            (
                hours.cast(pl.String).str.zfill(2)
                + ":"
                + minutes.cast(pl.String).str.zfill(2)
                + ":"
                + seconds.cast(pl.String).str.zfill(2)
            ).alias("Elapsed")
        )

        print(timeline_stats_printable)

        if args.plot and single_file_campaign_plot:
            logging.info(
                f"Generating individual campaign plot for '{experiment_name}'..."
            )
            plot_individual_campaigns_timeline(
                experiment_name,
                timeline_stats,
                column,
                args.min_hours,
                args.max_hours,
                args.output,
                args.num_ticks,
                args.tick_label_fontsize,
            )
        elif args.plot:
            plot_data_for_combined_graph.append((experiment_name, timeline_stats))

    ranking_info: list[tuple[str, pl.DataFrame, float]] = []
    if args.plot and plot_data_for_combined_graph:
        check_cols = ["MedianCoverage"]
        if args.show_campaigns:
            check_cols.extend(["Coverage_Q1", "Coverage_Q3"])

        for name, timeline_df in plot_data_for_combined_graph:
            df = timeline_df.clone().with_columns(
                (pl.col("Elapsed").dt.total_seconds() / 3600).alias("ElapsedHours")
            )
            df = df.sort("ElapsedHours")

            within_max = df.filter(pl.col("ElapsedHours") <= args.max_hours)
            if not within_max.is_empty():
                final_y = within_max.select("MedianCoverage").tail(1).item()
            else:
                final_y = df.select("MedianCoverage").tail(1).item()

            # Handle case where final_y might be None
            if final_y is None:
                final_y = 0.0

            ranking_info.append((name, df, final_y))

        ranking_info.sort(key=lambda t: (-t[2], t[0]))

        print_binned_summary_table(args, ranking_info, args.time_step)

        if all_final_coverages:
            boxplot_data_df = pl.concat(all_final_coverages)

            # Create a readable name for the plot
            if args.type == "global":
                coverage_type_name = "Cumulative Global Counter Coverage"
            elif args.type == "method":
                coverage_type_name = "Cumulative Method Counter Coverage"
            else:
                coverage_type_name = "Cumulative Code Coverage"

            plot_final_coverage_boxplot(
                boxplot_data_df,
                ranking_info,
                coverage_type_name,
                args.max_hours,
                args.output,
                args.tick_label_fontsize,
            )

        plot_combined_coverage_timeline(
            ranking_info,
            check_cols,
            args.min_hours,
            args.max_hours,
            args.output,
            args.show_campaigns,
            args.num_ticks,
            args.tick_label_fontsize,
        )
    elif args.plot and single_file_campaign_plot:
        logging.info("Individual campaign plot generated. Skipping combined plot.")


if __name__ == "__main__":
    main()
