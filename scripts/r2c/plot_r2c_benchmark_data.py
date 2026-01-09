import argparse
import json
import os

import numpy as np
import pandas
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
from scipy.stats import gmean
import matplot2tikz

from r2c_common import (
    BENCHMARK_NAMES,
    BENCHMARK_NAMES_INT,
    GEOMEAN_SETS,
    natural_sort_key,
)

plt.style.use("petroff10")


def count_datapoints(data, field):
    for instance in data:
        instance_count = None
        instance_data = data[instance]
        for key in instance_data:
            count = len(instance_data[key][field])
            if instance_count is None:
                instance_count = count
                print(f"First benchmark ({key}) in {instance} has {count} data points")
            else:
                if instance_count != count:
                    print(
                        f"WARNING: Benchmark {key} in {instance} has {count} data points, but previous entries had {instance_count}"
                    )


def build_dataframe(data, field) -> pandas.DataFrame:
    df_data = dict()
    for instance in data:
        df_data.setdefault(instance, dict())
        instance_data = data[instance]
        for key in instance_data:
            df_data[instance][key] = instance_data[key][field]

    df = pd.DataFrame(df_data)
    return df


def build_actual_totals(actual_path, actual_machine, row_to_key, geomean_sets):
    """Load actual totals (runtime:median) and return percent overheads per row label, including geomeans."""
    if actual_path is None:
        return None
    try:
        with open(actual_path, "r") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"WARNING: could not read actual totals from {actual_path}: {exc}")
        return None

    machine_data = data.get(actual_machine)
    if machine_data is None:
        print(f"WARNING: machine '{actual_machine}' not found in {actual_path}")
        return None

    actual_ratios = dict()
    for row_label, key in row_to_key.items():
        entry = machine_data.get(key)
        if entry and "runtime:median" in entry:
            actual_ratios[row_label] = entry["runtime:median"]
        else:
            print(
                f"WARNING: missing runtime:median for {key} in machine {actual_machine}"
            )

    # Geomeans
    for name, subset in geomean_sets.items():
        vals = [actual_ratios[r] for r in subset if r in actual_ratios]
        if vals:
            actual_ratios[name] = gmean(vals)
        else:
            print(f"WARNING: missing actual values to compute geomean for {name}")

    # Convert to percent overhead
    return {label: (ratio - 1.0) * 100 for label, ratio in actual_ratios.items()}


def summarize_columns(df, name_mapping):
    """Return summary statistics per column (mapped names)."""
    rows = []
    for col in df.columns:
        vals = df[col].astype(float)
        rows.append(
            {
                "component": name_mapping.get(col, col),
                "count": len(vals),
                "mean": vals.mean(),
                "median": vals.median(),
                "std": vals.std(ddof=1),
                "cv": vals.std(ddof=1) / vals.mean() if vals.mean() != 0 else np.nan,
                "min": vals.min(),
                "max": vals.max(),
            }
        )
    return pd.DataFrame(rows)


def summarize_samples(data, name_mapping):
    """Return summary statistics per component/benchmark from runtime:all samples."""
    rows = []
    for comp, benches in data.items():
        comp_label = name_mapping.get(comp, comp)
        for bench, entry in benches.items():
            samples = entry.get("runtime:all")
            if not samples:
                continue
            arr = np.asarray(samples, dtype=float)
            if len(arr) < 1:
                continue
            rows.append(
                {
                    "component": comp_label,
                    "benchmark": bench,
                    "count": len(arr),
                    "mean": arr.mean(),
                    "median": np.median(arr),
                    "std": arr.std(ddof=1) if len(arr) > 1 else 0.0,
                    "cv": (arr.std(ddof=1) / arr.mean()) if arr.mean() != 0 and len(arr) > 1 else np.nan,
                    "min": arr.min(),
                    "max": arr.max(),
                }
            )
    return pd.DataFrame(rows)


def build_cv_map(data, row_to_key):
    """Return cv (%) per (benchmark, component-key) from runtime:all."""
    cv_map = {}
    for comp_key, benches in data.items():
        for row_label, bench_key in row_to_key.items():
            entry = benches.get(bench_key)
            if not entry or "runtime:all" not in entry:
                continue
            arr = np.asarray(entry["runtime:all"], dtype=float)
            if len(arr) < 2:
                continue
            mean_val = np.mean(arr)
            if mean_val == 0:
                continue
            cv_pct = (np.std(arr, ddof=1) / mean_val) * 100.0
            cv_map.setdefault(row_label, {})[comp_key] = cv_pct
    return cv_map


def build_actual_cv(actual_path, actual_machine, row_to_key):
    """Return cv (%) per benchmark for the actual machine using runtime:all."""
    if not actual_path:
        return {}
    try:
        with open(actual_path, "r") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"WARNING: could not read actual totals from {actual_path} for CV: {exc}")
        return {}
    machine_data = data.get(actual_machine)
    if machine_data is None:
        print(f"WARNING: machine '{actual_machine}' not found in {actual_path}")
        return {}
    cv_map = {}
    for row_label, bench_key in row_to_key.items():
        entry = machine_data.get(bench_key)
        if not entry or "runtime:all" not in entry:
            continue
        arr = np.asarray(entry["runtime:all"], dtype=float)
        if len(arr) < 2:
            continue
        mean_val = np.mean(arr)
        if mean_val == 0:
            continue
        cv_map[row_label] = (np.std(arr, ddof=1) / mean_val) * 100.0
    return cv_map


def draw_graphs(
    df,
    plot_name,
    row_labels,
    name_mapping,
    output,
    geomean_sets=None,
    stack_components=None,
    stack_label=None,
    bar_thickness_factor=1.0,
    bar_padding=0.0,
    actual_totals=None,
    actual_label="Actual total",
    error_map=None,
    error_color="black",
):
    plt.figure()

    sns.set_context(
        "paper",
        rc={
            "pgf.texsystem": "xelatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": True,
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.titlesize": 16,
            ## 'ytick.major.pad': 20,
            ## "figure.autolayout": True,
        },
    )

    no_index = df.reset_index()
    bmark_index = no_index.rename(columns={"index": "bmark"})
    bmark_index["bmark"] = row_labels

    if geomean_sets is None:
        geomean_sets = {"Geomean": row_labels}

    for name, subset in geomean_sets.items():
        geomean_row = []
        geomean_row.append(name)
        for column in bmark_index.columns.values[1:]:
            filtered_rows = bmark_index.loc[bmark_index["bmark"].isin(subset)]
            values = filtered_rows[column]
            geomean_row.append(gmean(values))

        bmark_index.loc[len(bmark_index)] = geomean_row
    geomean_labels = list(geomean_sets.keys())

    melted = pd.melt(bmark_index, id_vars=["bmark"], value_vars=name_mapping.keys())
    melted["value"] = melted["value"].transform(lambda x: (x - 1) * 100)
    melted["variable"] = melted["variable"].map(name_mapping)

    colorblind_palette = [
        "#0173b2",  # blue
        "#56b4e9",  # light blue
        "#949494",  # gray
        "#de8f05",  # orange
        "#cc79a7",  # reddish purple
    ]

    # Set as default color cycle
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colorblind_palette)

    base_labels = list(name_mapping.values())

    hue_order = None
    if stack_components and stack_label:
        # Get all mapped names
        all_labels = [name_mapping.get(k, k) for k in name_mapping.keys()]

        # Filter out the stack label to sort the rest
        # This ensures the original components keep their sorted order (and thus colors)
        # matching the default Seaborn behavior.
        standard_labels = [l for l in all_labels if l != stack_label]
        # standard_labels.sort()

        hue_order = standard_labels + [stack_label]

    # Force palette to cycle colorblind_palette for all modes (stacked or not)
    n_colors = len(base_labels)
    extended_palette = colorblind_palette * (
        int(n_colors / len(colorblind_palette)) + 1
    )
    extended_palette = extended_palette[:n_colors]

    # Manual grouped bars to control thickness and spacing
    axis = plt.gca()
    bmark_order = list(row_labels) + geomean_labels
    variable_order = hue_order if hue_order else base_labels

    # Map variables to colors
    color_lookup = {var: extended_palette[idx] for idx, var in enumerate(base_labels)}

    bar_width = bar_thickness_factor  # treat thickness as absolute height
    inner_offset = 0.9 * bar_width  # distance between bars inside one group
    group_height = (len(variable_order) - 1) * inner_offset + bar_width
    # bar_padding is an absolute gap between groups (in axis units)
    group_spacing = group_height + bar_padding

    # Keep figure height moderate so bars look thick
    fig = plt.gcf()
    w, _ = fig.get_size_inches()
    target_h = max(4.0, 2.0 + 0.55 * len(bmark_order))
    fig.set_size_inches(w, target_h)

    # Pivot for quick lookup of values per benchmark/variable
    pivot = melted.pivot(index="bmark", columns="variable", values="value")

    # Track positions for stacked bars
    stack_positions = dict()

    for b_idx, bmark in enumerate(bmark_order):
        base_y = b_idx * group_spacing
        for v_idx, var in enumerate(variable_order):
            y = base_y + (v_idx - (len(variable_order) - 1) / 2) * inner_offset
            val = pivot.loc[bmark, var]

            if stack_components and stack_label and var == stack_label:
                stack_positions[bmark] = (y, bar_width)
                continue

            err = None
            if error_map:
                err = error_map.get((bmark, var))
            axis.barh(
                y=y,
                width=val,
                height=bar_width,
                color=color_lookup[var],
                label=var,
                zorder=2,
                xerr=err,
                error_kw={
                    "ecolor": error_color,
                    "elinewidth": 1.6,
                    "capsize": 4.0,
                    "capthick": 1.6,
                },
            )

    # Put readable labels back onto the spaced positions
    y_ticks = [i * group_spacing for i in range(len(bmark_order))]
    axis.set_yticks(y_ticks)
    axis.set_yticklabels(bmark_order)
    if len(y_ticks) > 0:
        axis.set_ylim(-group_spacing / 2, y_ticks[-1] + group_spacing / 2)
    axis.invert_yaxis()  # keep the original top-to-bottom order

    if stack_components and stack_label:
        # Colors for stack components reuse the mapped labels
        component_colors = {}
        for comp_key in stack_components:
            mapped = name_mapping.get(comp_key, comp_key)
            component_colors[comp_key] = color_lookup.get(mapped, (0, 0, 0, 1))

        bmark_rows = bmark_index.set_index("bmark")

        for bmark, (y, height) in stack_positions.items():
            row_data = bmark_rows.loc[bmark]
            current_x = 0
            for comp_key in stack_components:
                val = (row_data[comp_key] - 1) * 100
                color = component_colors.get(comp_key, (0, 0, 0, 1))
                axis.barh(
                    y=y,
                    width=val,
                    height=height,
                    left=current_x,
                    color=color,
                    edgecolor=None,
                    zorder=3,
                    align="center",
                )
                current_x += val

        # Legend without the combined stack label
        handles = [
            plt.Line2D([0], [0], color=color_lookup[v], lw=6)
            for v in variable_order
            if v != stack_label
        ]
        labels = [v for v in variable_order if v != stack_label]
    else:
        # Legend for normal grouped bars
        handles = [
            plt.Line2D([0], [0], color=color_lookup[v], lw=6) for v in variable_order
        ]
        labels = list(variable_order)

    # Overlay actual totals as dashed outlines, if provided
    if actual_totals:
        outline_height = bar_width * 0.9
        center_map = dict(zip(bmark_order, y_ticks))

        # Prefer the stack bar y (so the outline sits exactly over the stacked total)
        def overlay_y(label):
            if stack_positions and label in stack_positions:
                return stack_positions[label][0]
            return center_map.get(label)

        for bmark in bmark_order:
            y = overlay_y(bmark)
            actual_val = actual_totals.get(bmark)
            if y is None or actual_val is None:
                continue
            axis.barh(
                y=y,
                width=actual_val,
                height=outline_height,
                left=0,
                facecolor="none",
                edgecolor="black",
                linestyle="--",
                linewidth=1.0,
                zorder=7,
                align="center",
            )
        from matplotlib.lines import Line2D

        handles.append(Line2D([0], [0], color="black", lw=1.0, linestyle="--"))
        labels.append(actual_label)

    axis.legend(handles, labels)

    axis.xaxis.grid(True, which="minor", linestyle="dotted", color="gainsboro")
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.xaxis.grid(True, which="major", linestyle="dashed", color="gainsboro")
    axis.xaxis.set_major_locator(MultipleLocator(5))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%d"))

    for item in axis.get_yticklabels():
        ## item.set_rotation(30)
        item.set_ha("right")
        item.set_verticalalignment("center")

    bottom, top = plt.xlim()
    plt.xlim(0, top)

    plt.xlabel("Performance Impact (%)")
    plt.ylabel(None)

    # ## plot_name = metric_name.replace("..", "-") + '-relative.' + format
    # h, l = axis.get_legend_handles_labels()
    #
    # axis.legend(h, l, title='Configuration', loc='lower right', prop={'size': 14},
    #             title_fontsize=14)
    plt.tight_layout()

    imagepath = os.path.join("pictures", plot_name)
    print("Saving image to " + imagepath)

    ## relative_plot.yaxis.grid(True, clip_on=False)
    sns.despine(left=True, bottom=True)

    matplot2tikz.save(output)
    plt.show()
    plt.close()
    return melted


def plot_r2c(
    df,
    name_mapping,
    output,
    stack_components=None,
    stack_label=None,
    bar_thickness_factor=1.0,
    bar_padding=0.0,
    actual_path=None,
    actual_machine="epyc",
    actual_label="Actual total",
    error_map=None,
    error_color="black",
    actual_cv=None,
    residual_bootstrap=False,
    bootstrap_iters=1000,
    data_raw=None,
    row_to_key=None,
    baseline_path=None,
    baseline_machine=None,
):
    benchmark_names = BENCHMARK_NAMES
    geomean_sets = GEOMEAN_SETS

    row_to_key = dict(zip(benchmark_names, df.index))
    actual_totals = (
        build_actual_totals(actual_path, actual_machine, row_to_key, geomean_sets)
        if actual_path
        else None
    )

    plot_result = draw_graphs(
        df,
        "comp-full.pgf",
        benchmark_names,
        name_mapping,
        output,
        geomean_sets=geomean_sets,
        stack_components=stack_components,
        stack_label=stack_label,
        bar_thickness_factor=bar_thickness_factor,
        bar_padding=bar_padding,
        actual_totals=actual_totals,
        actual_label=actual_label,
        error_map=error_map,
        error_color=error_color,
    )
    bootstrap_table = None
    if residual_bootstrap and actual_path and stack_components and row_to_key and data_raw is not None:
        # Build bootstrap CIs for residuals
        rows = []
        # Preload actual runtime arrays
        try:
            with open(actual_path, "r") as f:
                actual_data = json.load(f)
            actual_machine_data = actual_data.get(actual_machine, {})
        except Exception as exc:
            print(f"WARNING: bootstrap skipped; could not read actual data: {exc}")
            actual_machine_data = {}

        for row_label in benchmark_names:
            bench_key = row_to_key.get(row_label)
            if not bench_key:
                continue
            # Collect component arrays
            comp_arrays = []
            comp_baselines = []
            for comp in stack_components:
                entry = data_raw.get(comp, {}).get(bench_key)
                if not entry or "runtime:all" not in entry:
                    comp_arrays = []
                    break
                arr_abs = np.asarray(entry["runtime:all"], dtype=float)
                # Drop measurements faster than baseline estimate later
                if len(arr_abs) == 0:
                    comp_arrays = []
                    break
                # Derive baseline from ratio median in df
                if bench_key not in df.index or comp not in df.columns:
                    comp_arrays = []
                    break
                ratio_median = df.loc[bench_key, comp]
                baseline_median = np.median(arr_abs) / ratio_median if ratio_median != 0 else None
                if baseline_median is None or baseline_median == 0:
                    comp_arrays = []
                    break
                ratios = arr_abs / baseline_median
                ratios = ratios[ratios >= 1.0]  # ignore faster-than-baseline
                if len(ratios) == 0:
                    comp_arrays = []
                    break
                comp_arrays.append(ratios)  # ratios per sample
                comp_baselines.append(baseline_median)
            if not comp_arrays:
                continue
            # Actual array
            actual_entry = actual_machine_data.get(bench_key)
            if not actual_entry or "runtime:all" not in actual_entry:
                continue
            actual_arr_abs = np.asarray(actual_entry["runtime:all"], dtype=float)
            baseline_actual = None
            # derive from stored actual ratio median
            if actual_totals:
                actual_ratio_median = 1.0 + (actual_totals.get(row_label, None) / 100.0) if actual_totals else None
                if actual_ratio_median and actual_ratio_median != 0:
                    baseline_actual = np.median(actual_arr_abs) / actual_ratio_median
            if baseline_actual is None or baseline_actual == 0:
                continue
            actual_arr = actual_arr_abs / baseline_actual  # ratios per sample
            actual_arr = actual_arr[actual_arr >= 1.0]  # ignore faster-than-baseline
            if len(actual_arr) == 0:
                continue

            # Bootstrap
            preds = []
            for _ in range(bootstrap_iters):
                pred_ratio = 1.0
                valid = True
                for arr in comp_arrays:
                    if len(arr) == 0:
                        valid = False
                        break
                    sample_ratio = np.mean(np.random.choice(arr, size=len(arr), replace=True))
                    pred_ratio += (sample_ratio - 1.0)
                if not valid:
                    continue
                actual_sample_ratio = np.mean(np.random.choice(actual_arr, size=len(actual_arr), replace=True))
                pred_pct = (pred_ratio - 1.0) * 100.0
                actual_pct = (actual_sample_ratio - 1.0) * 100.0
                preds.append(pred_pct - actual_pct)
            if not preds:
                continue
            lower = np.percentile(preds, 2.5)
            upper = np.percentile(preds, 97.5)
            rows.append(
                {
                    "benchmark": row_label,
                    "residual_pct_lower": lower,
                    "residual_pct_upper": upper,
                    "ci_excludes_zero": lower > 0 or upper < 0,
                }
            )
        if rows:
            bootstrap_table = pd.DataFrame(rows)

    return plot_result, bootstrap_table


def min_max(df, name_mapping):
    result = pd.DataFrame(columns=["component", "max", "geomean"])
    for column in df.columns.values:
        values = df[column]
        geomean = np.round(gmean(values), 2)
        # min_value = np.round(min(values), 2)
        max_value = np.round(max(values), 2)
        result.loc[len(result)] = [name_mapping[column], max_value, geomean]

    # enclose names in \propername
    names = list(name_mapping.values())
    result = result.set_index("component")
    # sort rows the same way as the name mapping
    result = result.sort_index(key=lambda index: index.map(lambda i: names.index(i)))
    result.index = result.index.map(lambda name: f"\\propername{{{name}}}")
    return result


def build_table(df, name_mapping, sort_key=None, as_percentage=True):
    """Build a table with configurations as columns and benchmarks as rows.

    Args:
        df: DataFrame with benchmarks as index and configurations as columns
        name_mapping: dict mapping column keys to display names
        sort_key: optional function to sort configuration columns (applied to keys)
        as_percentage: if True, display as percentage overhead; otherwise as ratio

    Returns:
        DataFrame with benchmarks + geomean as rows, configurations as columns
    """
    # Get columns in order
    cols = list(df.columns)
    if sort_key:
        cols = sorted(cols, key=sort_key)

    # Build result dataframe
    result_data = {}
    for col in cols:
        col_name = name_mapping.get(col, col)
        if as_percentage:
            result_data[col_name] = (df[col] - 1.0) * 100
        else:
            result_data[col_name] = df[col]

    result = pd.DataFrame(result_data, index=df.index)

    # Add geomean row
    geomean_row = {}
    for col in cols:
        col_name = name_mapping.get(col, col)
        if as_percentage:
            geomean_row[col_name] = (gmean(df[col]) - 1.0) * 100
        else:
            geomean_row[col_name] = gmean(df[col])

    result.loc["geomean"] = geomean_row

    return result


class KeyValParser(argparse.Action):
    def __call__(self, parser, namespace, parts, option_string=None):
        setattr(namespace, self.dest, dict())
        for part in parts:
            key, value = part.split(":")
            getattr(namespace, self.dest)[key] = value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot benchmark data from instrumentation-infra."
    )
    parser.add_argument("data", metavar="DATA", type=argparse.FileType("r"))
    parser.add_argument("--names", nargs="*", action=KeyValParser)
    parser.add_argument(
        "--mode", choices=["plot", "minmax", "table"], action="store", default="plot"
    )
    parser.add_argument(
        "--sort-key",
        type=str,
        help="Regex pattern to extract sort key from column names (e.g., '(\\d+)$' to sort by trailing number)",
    )
    parser.add_argument(
        "--natural-sort",
        action="store_true",
        help="Use natural sorting for column names (e.g., btra2, btra10, btra20 instead of btra10, btra2, btra20)",
    )
    parser.add_argument(
        "--table-format",
        choices=["markdown", "latex"],
        default=None,
        help="Output format for table mode (default: markdown). Implies --mode table.",
    )
    parser.add_argument("--output", type=str, action="store", required=True)
    parser.add_argument("--stack", nargs="*", help="List of component keys to stack")
    parser.add_argument(
        "--stack-label", type=str, default="Combined", help="Label for the stacked bar"
    )
    parser.add_argument(
        "--bar-thickness-factor",
        type=float,
        default=1.0,
        help="Scales the bar thickness (default: 1.0)",
    )
    parser.add_argument(
        "--bar-padding",
        type=float,
        default=0.0,
        help="Extra spacing between benchmark groups (default: 0.0)",
    )
    parser.add_argument(
        "--actual-total",
        type=str,
        help="Path to dataset containing actual total overheads (runtime:median)",
    )
    parser.add_argument(
        "--actual-machine",
        type=str,
        default="epyc",
        help="Machine key to read from the actual-total dataset",
    )
    parser.add_argument(
        "--actual-label",
        type=str,
        default="Actual total",
        help="Legend label for actual totals overlay",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics (mean/median/std/min/max) per column",
    )
    parser.add_argument(
        "--samples-summary",
        action="store_true",
        help="Print summary statistics per component/benchmark using runtime:all samples",
    )
    parser.add_argument(
        "--error-bars",
        action="store_true",
        help="Add symmetric error bars using relative std from runtime:all samples.",
    )
    parser.add_argument(
        "--error-bar-color",
        type=str,
        default="black",
        help="Color for error bars (default: black)",
    )
    parser.add_argument(
        "--residual-report",
        action="store_true",
        help="Report predicted (stacked) vs actual overhead residuals and correlation with variability",
    )
    parser.add_argument(
        "--residual-bootstrap",
        action="store_true",
        help="Bootstrap residuals per benchmark (stacked vs actual) to get a CI; requires --stack and --actual-total",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=1000,
        help="Bootstrap iterations for residual bootstrap (default: 1000)",
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        help="Path to baseline JSON (full-r2c style) to derive per-run ratios for bootstrap",
    )
    parser.add_argument(
        "--baseline-machine",
        type=str,
        help="Machine key inside baseline JSON (e.g., epyc)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress info messages (useful for table mode)",
    )

    args = parser.parse_args()

    # If --table-format is specified but mode is still default, switch to table mode
    if args.table_format and args.mode == "plot":
        args.mode = "table"

    data = json.load(args.data)
    if not args.quiet:
        count_datapoints(data, "runtime:all")
    df = build_dataframe(data, "runtime:median")

    name_mapping = args.names

    if name_mapping is None:
        name_mapping = {name: name for name in df.columns.values}

    if args.stack:
        # Calculate stacked column
        cols_to_sum = args.stack
        # Verify columns exist
        missing = [c for c in cols_to_sum if c not in df.columns]
        if missing:
            print(f"WARNING: Stack components not found in data: {missing}")

        valid_cols = [c for c in cols_to_sum if c in df.columns]

        # (Sum of overheads) + 1.0
        # overhead = val - 1.0
        df["stacked_col"] = (df[valid_cols] - 1.0).sum(axis=1) + 1.0

        name_mapping["stacked_col"] = args.stack_label

    if args.error_bars and args.stack:
        print("NOTE: error bars are only drawn for unstacked bars; stacking is enabled, so error bars will be skipped.")

    if args.mode == "plot":
        error_map = None
        bootstrap_table = None
        if args.error_bars:
            benchmark_names = BENCHMARK_NAMES
            row_to_key = dict(zip(benchmark_names, df.index))
            error_map = {}
            for comp_key, comp_data in data.items():
                var_label = name_mapping.get(comp_key, comp_key)
                for row_label, bench_key in row_to_key.items():
                    entry = comp_data.get(bench_key)
                    if not entry or "runtime:all" not in entry:
                        continue
                    arr = np.asarray(entry["runtime:all"], dtype=float)
                    if len(arr) < 2:
                        continue
                    mean_val = np.mean(arr)
                    if mean_val != 0:
                        std_pct = (np.std(arr, ddof=1) / mean_val) * 100.0
                        error_map[(row_label, var_label)] = std_pct
        # Residual report: compare predicted stacked vs actual and correlate with variability
        if args.residual_report:
            if not args.stack:
                print("Residual report requires --stack to define which components to sum; skipping.")
            elif not args.actual_total:
                print("Residual report requires --actual-total; skipping.")
            else:
                benchmark_names = BENCHMARK_NAMES
                row_to_key = dict(zip(benchmark_names, df.index))
                geomean_sets = GEOMEAN_SETS
                actual_totals = build_actual_totals(args.actual_total, args.actual_machine, row_to_key, geomean_sets)
                actual_cv = build_actual_cv(args.actual_total, args.actual_machine, row_to_key)
                # Predicted from stacked col (already added)
                preds = (df["stacked_col"] - 1.0) * 100
                rows = []
                cv_map = build_cv_map(data, row_to_key)
                for idx, row_label in enumerate(benchmark_names):
                    pred = preds.iloc[idx] if idx < len(preds) else None
                    actual = actual_totals.get(row_label) if actual_totals else None
                    if pred is None or actual is None:
                        continue
                    residual = pred - actual
                    act_cv = actual_cv.get(row_label, np.nan)
                    comp_cvs = []
                    for comp_key in args.stack:
                        cv_entry = cv_map.get(row_label, {}).get(comp_key)
                        if cv_entry is not None:
                            comp_cvs.append(cv_entry)
                    cv_mean = np.mean(comp_cvs) if comp_cvs else np.nan
                    cv_max = np.max(comp_cvs) if comp_cvs else np.nan
                    combined_cv = np.sqrt(np.nanmean([cv_mean ** 2, act_cv ** 2])) if not np.isnan(cv_mean) or not np.isnan(act_cv) else np.nan
                    noise_band = 1.5 * np.nanmax([cv_mean, act_cv]) if not np.isnan(cv_mean) or not np.isnan(act_cv) else np.nan
                    within_noise = (
                        abs(residual) <= noise_band if not np.isnan(noise_band) else False
                    )
                    rows.append(
                        {
                            "benchmark": row_label,
                            "pred_pct": pred,
                            "actual_pct": actual,
                            "residual_pct": residual,
                            "abs_residual_pct": abs(residual),
                            "cv_mean_pct": cv_mean,
                            "cv_max_pct": cv_max,
                            "actual_cv_pct": act_cv,
                            "combined_cv_pct": combined_cv,
                            "noise_band_pct": noise_band,
                            "within_noise": within_noise,
                            "cv_count": len(comp_cvs),
                        }
                    )
                if rows:
                    residual_df = pd.DataFrame(rows)
                    print("\nResidual report (predicted stacked vs actual):")
                    print(residual_df.to_markdown(index=False, floatfmt=".4f"))
                    # Correlations
                    corr_rows = []
                    for col in ["cv_mean_pct", "cv_max_pct", "actual_cv_pct", "combined_cv_pct"]:
                        subset = residual_df[["abs_residual_pct", col]].dropna()
                        if len(subset) >= 2:
                            corr = subset.corr(method="pearson").iloc[0, 1]
                            corr_rows.append({"metric": f"|residual| vs {col}", "pearson": corr})
                    if corr_rows:
                        print("\nCorrelation (Pearson) between |residual| and CV metrics:")
                        print(pd.DataFrame(corr_rows).to_markdown(index=False, floatfmt=".4f"))

        result = plot_r2c(
            df,
            name_mapping,
            args.output,
            stack_components=args.stack if args.stack else None,
            stack_label=args.stack_label if args.stack else None,
            bar_thickness_factor=args.bar_thickness_factor,
            bar_padding=args.bar_padding,
            actual_path=args.actual_total,
            actual_machine=args.actual_machine,
            actual_label=args.actual_label,
            error_map=error_map,
            error_color=args.error_bar_color,
            residual_bootstrap=args.residual_bootstrap,
            bootstrap_iters=args.bootstrap_iters,
            data_raw=data,
            row_to_key=row_to_key if args.stack else None,
            baseline_path=args.baseline_path,
            baseline_machine=args.baseline_machine,
        )
        if isinstance(result, tuple):
            melted, bootstrap_table = result
        else:
            melted = result
        print(melted.to_markdown())

    if args.summary:
        comp_summary = summarize_columns(df, name_mapping)
        print("\nComponent summary (overhead ratios):")
        print(comp_summary.to_markdown(index=False, floatfmt=".4f"))

        if args.actual_total:
            benchmark_names = BENCHMARK_NAMES
            row_to_key = dict(zip(benchmark_names, df.index))
            geomean_sets = GEOMEAN_SETS
            actual_totals = build_actual_totals(args.actual_total, args.actual_machine, row_to_key, geomean_sets)
            if actual_totals:
                actual_df = pd.DataFrame({"actual_total_pct": actual_totals})
                stats = {
                    "count": actual_df.shape[0],
                    "mean": actual_df["actual_total_pct"].mean(),
                    "median": actual_df["actual_total_pct"].median(),
                    "std": actual_df["actual_total_pct"].std(ddof=1),
                    "min": actual_df["actual_total_pct"].min(),
                    "max": actual_df["actual_total_pct"].max(),
                }
                print(f"\nActual total summary (percent overhead, machine: {args.actual_machine}):")
                print(pd.DataFrame([stats]).to_markdown(index=False, floatfmt=".4f"))

    if args.samples_summary:
        samples_df = summarize_samples(data, name_mapping)
        if not samples_df.empty:
            print("\nPer-benchmark sample summary (runtime:all):")
            print(samples_df.to_markdown(index=False, floatfmt=".4f"))
        else:
            print("\nNo runtime:all samples found for samples summary.")

    if args.residual_bootstrap and bootstrap_table is not None:
        print("\nResidual bootstrap 95% CI (predicted stacked - actual):")
        print(bootstrap_table.to_markdown(index=False, floatfmt=".4f"))

    if args.mode == "minmax":
        result = min_max(df, name_mapping)
        styler = result.style.format(decimal=".", thousands=",", precision=2)
        print(styler.to_latex())

    if args.mode == "table":
        import re

        sort_key = None
        if args.natural_sort:
            sort_key = natural_sort_key
        elif args.sort_key:
            pattern = re.compile(args.sort_key)

            def sort_key(col):
                match = pattern.search(col)
                if match:
                    try:
                        return int(match.group())
                    except ValueError:
                        return match.group()
                return col

        result = build_table(df, name_mapping, sort_key=sort_key, as_percentage=True)
        table_format = args.table_format or "markdown"
        if table_format == "latex":
            # Generate LaTeX manually to avoid jinja2 dependency
            cols = result.columns.tolist()
            col_spec = "l" + "r" * len(cols)
            lines = [
                "\\begin{tabular}{" + col_spec + "}",
                "\\toprule",
                " & " + " & ".join(str(c) for c in cols) + " \\\\",
                "\\midrule",
            ]
            for idx, row in result.iterrows():
                vals = [f"{v:.1f}" if isinstance(v, float) else str(v) for v in row]
                lines.append(f"{idx} & " + " & ".join(vals) + " \\\\")
            lines.append("\\bottomrule")
            lines.append("\\end{tabular}")
            print("\n".join(lines))
        else:
            print(result.to_markdown(floatfmt=".1f"))
