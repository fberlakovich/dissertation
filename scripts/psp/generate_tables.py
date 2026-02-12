#!/usr/bin/env python3
"""Generate LaTeX table body files from PSP data packages.

Usage (subcommands):
    generate_tables.py edge-coverage  <experiment.tar.zst>  FUZZER:BASELINE [...]  [flags]
    generate_tables.py bug-data       <experiment.tar.zst>  FUZZER:BASELINE [...]  [flags]
    generate_tables.py entanglement   <analysis.tar.zstd>   [flags]
    generate_tables.py stepping-stones       <analysis.tar.zstd>   [flags]
    generate_tables.py partition-progression <analysis.tar.zstd>   [flags]

Outputs LaTeX table body files (siunitx-compatible, no header) to --latex-output directory.
"""

import argparse
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_fuzzers import (
    load_data_package,
    parse_comparisons,
    compare_coverage,
    vargha_delaney_a12,
    mann_whitney_test,
)
from utils import define_latex_table, define_latex_var, load_name_map


# ---------------------------------------------------------------------------
# Helpers (shared across subcommands)
# ---------------------------------------------------------------------------

def latex_escape(s):
    """Escape LaTeX special characters in a string."""
    return s.replace("_", r"\_")


def display_name(name, name_map):
    """Map a name to its display form: use name_map if available, else escape for LaTeX."""
    if name_map and name in name_map:
        return name_map[name]
    return latex_escape(name)


def table_name(kind, fuzzer, baseline, prefix=None):
    """Build a LaTeX-safe file name (no underscores) for a table."""
    parts = [kind]
    if prefix:
        parts.append(prefix)
    parts.extend([fuzzer, "vs", baseline])
    return "-".join(parts).replace("_", "-")


def sig_stars(p_value, significant):
    """Return significance stars for a p-value."""
    if not significant or p_value is None:
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    return "*"


def _coverage_row(benchmark, cov, name_map):
    """Build a single coverage comparison row dict."""
    fmed = cov["fuzzer_median"]
    bmed = cov["baseline_median"]
    diff = fmed - bmed
    rel = (diff / bmed * 100) if bmed > 0 else 0.0
    return {
        "Benchmark": display_name(benchmark, name_map),
        "Fuzzer Median": f"{fmed:.0f}",
        "Baseline Median": f"{bmed:.0f}",
        "Diff": f"{diff:+.0f}",
        r"Rel\%": f"{rel:+.1f}",
        "A12": f"{cov['a12']:.2f}",
        "Effect": cov["effect_size"],
        "Sig": sig_stars(cov["p_value"], cov["significant"]),
    }


def generate_coverage_table(args, fuzzer, baseline, coverage_df, name_map, prefix=None):
    """Generate per-benchmark coverage comparison table.

    If a name_map is provided, only benchmarks present in the name_map are included.
    """
    benchmarks = sorted(coverage_df["benchmark"].unique())
    rows = []

    for benchmark in benchmarks:
        if name_map and benchmark not in name_map:
            continue
        cov = compare_coverage(coverage_df, fuzzer, baseline, benchmark)
        if cov is None:
            continue
        rows.append(_coverage_row(benchmark, cov, name_map))

    if not rows:
        return

    df = pd.DataFrame(rows)
    name = table_name("coverage", fuzzer, baseline, prefix)
    define_latex_table(args, name, df)
    print(f"  Written: {name}.tex ({len(rows)} rows)")


def _bold_s(value):
    """Wrap a value in bold for use inside an siunitx S column."""
    return r"{\textbf{" + str(value) + r"}}"


def _format_p(p_value, bold_significant=True):
    """Format a p-value for display.

    Non-numeric values are wrapped in braces for siunitx S-column compatibility.
    If bold_significant is True, significant values (p < 0.05) are bolded.
    """
    if p_value is None:
        return "{---}"
    if p_value < 0.001:
        # Use \llap to right-align the < symbol without taking space
        text = r"{\llap{$<$}0.001}"
        return r"{\textbf" + text + r"}" if bold_significant else text
    if p_value < 0.05:
        text = f"{p_value:.3f}"
        return _bold_s(text) if bold_significant else text
    return f"{p_value:.3f}"


def _format_a12(a12, bold_nonnegligible=False):
    """Format an A12 effect size.

    If bold_nonnegligible is True, non-negligible effects (|A12 - 0.5| >= 0.06) are bolded.
    """
    formatted = f"{a12:.2f}"
    if bold_nonnegligible and abs(a12 - 0.5) >= 0.06:
        return _bold_s(formatted)
    return formatted


def _effect_size_level(a12, p_value):
    """Determine effect size level for shading.

    Returns:
        None if not significant or negligible effect
        ('better'|'worse', 1|2|3) for small/medium/large effect

    Vargha-Delaney thresholds:
        - small:  0.56 <= A12 < 0.64  (or 0.36 < A12 <= 0.44)
        - medium: 0.64 <= A12 < 0.71  (or 0.29 < A12 <= 0.36)
        - large:  A12 >= 0.71         (or A12 <= 0.29)
    """
    if p_value is None or p_value >= 0.05:
        return None

    # Distance from 0.5 determines effect magnitude
    dist = abs(a12 - 0.5)
    if dist < 0.06:  # negligible
        return None

    direction = 'better' if a12 > 0.5 else 'worse'

    if dist >= 0.21:  # large (A12 >= 0.71 or <= 0.29)
        level = 3
    elif dist >= 0.14:  # medium (A12 >= 0.64 or <= 0.36)
        level = 2
    else:  # small (A12 >= 0.56 or <= 0.44)
        level = 1

    return (direction, level)


def _shade_cell(value, effect=None):
    """Wrap a cell value in shadecell based on effect direction and magnitude.

    Args:
        value: The cell value
        effect: None (no shading), or (direction, level) tuple
    """
    if effect is None:
        return value
    direction, level = effect
    color = f"cellbetter{level}" if direction == 'better' else f"cellworse{level}"
    return r"\shadecell{" + color + "}{" + str(value) + "}"


def generate_combined_coverage_table(args, comparisons, group_names, coverage_df, name_map, prefix=None):
    """Generate a combined coverage table with column groups for each comparison.

    Each comparison becomes a column group with: Delta, Delta%, A12, p.
    Benchmarks are filtered by name_map if provided.
    If --shade-significant is set, cells for groups with p < 0.05 and non-negligible
    effect size are shaded.
    """
    bold_a12 = getattr(args, "bold_a12", False)
    bold_p = not getattr(args, "no_bold_p", False)
    shade_significant = getattr(args, "shade_significant", False)

    benchmarks = sorted(coverage_df["benchmark"].unique())
    comp_list = list(comparisons.items())
    rows = []

    for benchmark in benchmarks:
        if name_map and benchmark not in name_map:
            continue

        row = {"Benchmark": display_name(benchmark, name_map)}
        has_any = False

        for i, (fuzzer, baseline) in enumerate(comp_list):
            gname = group_names[i]
            cov = compare_coverage(coverage_df, fuzzer, baseline, benchmark)
            if cov is None:
                row[f"{gname} $\\Delta$"] = "{---}"
                row[f"{gname} $\\Delta$\\%"] = "{---}"
                row[f"{gname} $\\hat{{A}}_{{12}}$"] = "{---}"
                row[f"{gname} $p$"] = "{---}"
            else:
                has_any = True
                fmed = cov["fuzzer_median"]
                bmed = cov["baseline_median"]
                diff = fmed - bmed
                rel = (diff / bmed * 100) if bmed > 0 else 0.0

                # Determine if this group should be shaded (direction and level)
                effect = None
                if shade_significant:
                    effect = _effect_size_level(cov['a12'], cov['p_value'])

                delta_val = f"+{diff:.0f}" if diff > 0 else f"{diff:.0f}"
                rel_val = f"+{rel:.1f}" if rel > 0 else f"{rel:.1f}"
                a12_val = _format_a12(cov['a12'], bold_nonnegligible=bold_a12)
                p_val = _format_p(cov["p_value"], bold_significant=bold_p)

                row[f"{gname} $\\Delta$"] = _shade_cell(delta_val, effect)
                row[f"{gname} $\\Delta$\\%"] = _shade_cell(rel_val, effect)
                row[f"{gname} $\\hat{{A}}_{{12}}$"] = _shade_cell(a12_val, effect)
                row[f"{gname} $p$"] = _shade_cell(p_val, effect)

        if has_any:
            rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows)
    name = "coverage"
    if prefix:
        name = f"coverage-{prefix}"
    name = name.replace("_", "-")
    define_latex_table(args, name, df)
    print(f"  Written: {name}.tex ({len(rows)} rows)")


def _bug_stats(bugs_df, fuzzer, benchmark, bug_id):
    """Compute instance triggering rate and median TTF for a bug.

    Returns (instances_triggered, total_instances, median_ttf_minutes) or None.
    TTF is the median first cycle (in minutes) among instances that triggered.
    """
    MINUTES_PER_CYCLE = 15.0  # 15 min per cycle

    subset = bugs_df[
        (bugs_df["fuzzer"] == fuzzer)
        & (bugs_df["benchmark"] == benchmark)
        & (bugs_df["bug_id"] == bug_id)
    ]
    if subset.empty:
        return None

    total_instances = subset["instance"].nunique()
    # For each instance, find the first cycle where triggered > 0
    first_triggers = (
        subset[subset["triggered"] > 0]
        .groupby("instance")["cycle"]
        .min()
    )
    instances_triggered = len(first_triggers)
    median_ttf = first_triggers.median() * MINUTES_PER_CYCLE if instances_triggered > 0 else None
    return instances_triggered, total_instances, median_ttf


def _compute_bug_significance(bugs_df, fuzzer, baseline, name_map):
    """Compute Fisher's exact test significance for each bug.

    Returns a dict: (benchmark, bug_id) -> ('better'|'worse'|None)
    where 'better' means fuzzer triggers more, 'worse' means baseline triggers more.
    Only significant results (after BH correction) are included.
    """
    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests

    f_rates = _per_bug_triggering_rates(bugs_df, fuzzer, name_map)
    b_rates = _per_bug_triggering_rates(bugs_df, baseline, name_map)

    if f_rates.empty and b_rates.empty:
        return {}

    # Merge on (benchmark, bug_id)
    merged = pd.merge(
        f_rates, b_rates,
        on=["benchmark", "bug_id"], how="outer",
        suffixes=("_f", "_b"),
    ).fillna(0).astype({"triggered_f": int, "total_f": int, "triggered_b": int, "total_b": int})

    merged["n"] = merged[["total_f", "total_b"]].max(axis=1)
    merged = merged[(merged["triggered_f"] > 0) | (merged["triggered_b"] > 0)]

    if merged.empty:
        return {}

    # Fisher's exact test per bug
    p_values = []
    directions = []
    for _, row in merged.iterrows():
        n = int(row["n"])
        a = int(row["triggered_f"])
        b = int(row["triggered_b"])
        table = [[a, n - a], [b, n - b]]
        _, p = fisher_exact(table, alternative="two-sided")
        p_values.append(p)
        directions.append('better' if a > b else ('worse' if b > a else None))

    merged["p_value"] = p_values
    merged["direction"] = directions

    # Benjamini-Hochberg correction
    reject, _, _, _ = multipletests(merged["p_value"], alpha=0.05, method="fdr_bh")
    merged["significant"] = reject

    # Build result dict
    result = {}
    for _, row in merged[merged["significant"]].iterrows():
        key = (row["benchmark"], row["bug_id"])
        result[key] = row["direction"]

    return result


def generate_combined_bugs_table(args, comparisons, group_names, bugs_df, name_map, prefix=None):
    """Generate a combined bug triggering table with column groups.

    Each group shows: fuzzer triggered instances, baseline triggered instances,
    fuzzer median TTF (minutes), baseline median TTF (minutes).
    Rows are grouped by benchmark, one row per bug.
    Significant cells (Fisher's exact test, BH-corrected) are shaded.
    """
    if bugs_df is None or bugs_df.empty:
        return

    comp_list = list(comparisons.items())
    shade_significant = getattr(args, 'shade_significant', False)

    # Pre-compute significance for each comparison group
    significance_maps = {}
    if shade_significant:
        for fuzzer, baseline in comp_list:
            significance_maps[(fuzzer, baseline)] = _compute_bug_significance(
                bugs_df, fuzzer, baseline, name_map
            )

    # Only Magma benchmarks have bugs
    magma_benchmarks = sorted(bugs_df["benchmark"].unique())
    rows = []

    for benchmark in magma_benchmarks:
        if name_map and benchmark not in name_map:
            continue

        all_bug_ids = sorted(bugs_df[bugs_df["benchmark"] == benchmark]["bug_id"].unique())

        for bug_id in all_bug_ids:
            row = {
                "Benchmark": display_name(benchmark, name_map),
                "Bug": bug_id,
            }
            any_triggered = False

            for i, (fuzzer, baseline) in enumerate(comp_list):
                gname = group_names[i]
                fstats = _bug_stats(bugs_df, fuzzer, benchmark, bug_id)
                bstats = _bug_stats(bugs_df, baseline, benchmark, bug_id)

                if fstats is None and bstats is None:
                    row[f"{gname} $n$"] = "{---}"
                    row[f"{gname} $n_b$"] = "{---}"
                    row[f"{gname} TTF"] = "{---}"
                    row[f"{gname} $\\text{{TTF}}_b$"] = "{---}"
                    continue

                f_trig, f_total, f_ttf = fstats if fstats else (0, 0, None)
                b_trig, b_total, b_ttf = bstats if bstats else (0, 0, None)

                if f_trig > 0 or b_trig > 0:
                    any_triggered = True

                # Check if this bug is significant for this comparison
                sig_dir = None
                if shade_significant:
                    sig_map = significance_maps.get((fuzzer, baseline), {})
                    sig_dir = sig_map.get((benchmark, bug_id))

                # Format cell values with optional shading (only for trigger counts,
                # not TTF - Fisher's test only compares triggering rates)
                n_val = f"{f_trig}"
                nb_val = f"{b_trig}"
                ttf_val = f"{f_ttf:.0f}" if f_ttf is not None else "{---}"
                ttfb_val = f"{b_ttf:.0f}" if b_ttf is not None else "{---}"

                if sig_dir is not None:
                    # Use level 3 (strongest) for significant bugs
                    color = "cellbetter3" if sig_dir == 'better' else "cellworse3"
                    n_val = r"\shadecell{" + color + "}{" + n_val + "}"
                    nb_val = r"\shadecell{" + color + "}{" + nb_val + "}"

                row[f"{gname} $n$"] = n_val
                row[f"{gname} $n_b$"] = nb_val
                row[f"{gname} TTF"] = ttf_val
                row[f"{gname} $\\text{{TTF}}_b$"] = ttfb_val

            if any_triggered:
                rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows)
    name = "bugs"
    if prefix:
        name = f"bugs-{prefix}"
    name = name.replace("_", "-")
    define_latex_table(args, name, df, group_column=0)
    print(f"  Written: {name}.tex ({len(rows)} rows)")


def generate_bugs_table(args, fuzzer, baseline, bugs_df, name_map, prefix=None):
    """Generate per-bug Magma triggering comparison table."""
    if bugs_df is None or bugs_df.empty:
        return

    fuzzer_bugs = bugs_df[bugs_df["fuzzer"] == fuzzer]
    baseline_bugs = bugs_df[bugs_df["fuzzer"] == baseline]

    if fuzzer_bugs.empty and baseline_bugs.empty:
        return

    rows = []
    all_benchmarks = sorted(
        set(fuzzer_bugs["benchmark"].unique()) | set(baseline_bugs["benchmark"].unique())
    )

    for benchmark in all_benchmarks:
        if name_map and benchmark not in name_map:
            continue
        fb = fuzzer_bugs[fuzzer_bugs["benchmark"] == benchmark]
        bb = baseline_bugs[baseline_bugs["benchmark"] == benchmark]

        all_bug_ids = sorted(
            set(fb["bug_id"].unique()) | set(bb["bug_id"].unique())
        )

        for bug_id in all_bug_ids:
            fuz_trig = int(fb[fb["bug_id"] == bug_id]["triggered"].sum()) if len(fb) > 0 else 0
            base_trig = int(bb[bb["bug_id"] == bug_id]["triggered"].sum()) if len(bb) > 0 else 0
            delta = fuz_trig - base_trig

            if fuz_trig == 0 and base_trig == 0:
                continue

            rows.append({
                "Benchmark": display_name(benchmark, name_map),
                "Bug ID": bug_id,
                "Fuzzer Triggers": str(fuz_trig),
                "Baseline Triggers": str(base_trig),
                "Delta": f"{delta:+d}",
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    name = table_name("bugs", fuzzer, baseline, prefix)
    define_latex_table(args, name, df)
    print(f"  Written: {name}.tex ({len(rows)} rows)")


def _latex_cmd_name(group_name):
    """Convert a group name to a camelCase LaTeX command suffix.

    ``"VRP"`` -> ``"Vrp"``, ``"Call Unfolding"`` -> ``"CallUnfolding"``.
    """
    return "".join(w.capitalize() for w in group_name.split())


def _prefixed(base: str, prefix: str | None) -> str:
    """Return ``base-prefix`` if *prefix* is set, otherwise *base*.

    Underscores in *prefix* are replaced with hyphens for LaTeX-safe filenames.
    """
    if prefix:
        return f"{base}-{prefix.replace('_', '-')}"
    return base


def _define_prefix(base: str, prefix: str | None) -> str:
    r"""Return a LaTeX define name prefix incorporating *prefix*.

    ``_define_prefix("pspEnt", "fuzzbench")`` -> ``"pspEntFuzzbench"``
    ``_define_prefix("pspEnt", None)`` -> ``"pspEnt"``
    """
    if prefix:
        return base + _latex_cmd_name(prefix)
    return base


def _per_trial_bug_counts(bugs_df, fuzzer, name_map):
    """Count unique bugs triggered per trial instance for *fuzzer*.

    Returns a list of integer counts, one per instance, sorted by instance id.
    """
    df = bugs_df[bugs_df["fuzzer"] == fuzzer]
    if name_map:
        df = df[df["benchmark"].isin(name_map.keys())]
    if df.empty:
        return []

    all_instances = sorted(df["instance"].unique())

    triggered = df[df["triggered"] > 0].drop_duplicates(
        subset=["instance", "benchmark", "bug_id"]
    )
    if triggered.empty:
        return [0] * len(all_instances)

    counts = triggered.groupby("instance").size()
    return [int(counts.get(inst, 0)) for inst in all_instances]


def generate_aggregate_bug_defines(args, comparisons, group_names, bugs_df, name_map):
    """Compute per-trial aggregate bug counts and emit LaTeX defines.

    For each comparison group, writes:
      - ``\\pspBugs<Group>FuzzerMean``   mean bugs per trial (fuzzer)
      - ``\\pspBugs<Group>BaselineMean`` mean bugs per trial (baseline)
      - ``\\pspBugs<Group>Atwelve``      Vargha--Delaney A12
      - ``\\pspBugs<Group>Pvalue``       formatted Mann--Whitney U p-value
    """
    if bugs_df is None or bugs_df.empty:
        return

    comp_list = list(comparisons.items())

    for i, (fuzzer, baseline) in enumerate(comp_list):
        gname = group_names[i]
        cmd = _latex_cmd_name(gname)

        f_counts = _per_trial_bug_counts(bugs_df, fuzzer, name_map)
        b_counts = _per_trial_bug_counts(bugs_df, baseline, name_map)

        if not f_counts or not b_counts:
            continue

        a12 = vargha_delaney_a12(f_counts, b_counts)
        _, p_value, significant = mann_whitney_test(f_counts, b_counts)

        prefix = f"pspBugs{cmd}"
        define_latex_var(args, f"{prefix}FuzzerMean", round(float(np.mean(f_counts)), 1))
        define_latex_var(args, f"{prefix}BaselineMean", round(float(np.mean(b_counts)), 1))
        define_latex_var(args, f"{prefix}Atwelve", round(a12, 2))

        if p_value is not None and p_value < 0.001:
            p_str = r"$<$\num{0.001}"
        elif p_value is not None:
            p_str = f"\\num{{{p_value:.3f}}}"
        else:
            p_str = "---"
        define_latex_var(args, f"{prefix}Pvalue", p_str)

        f_med = float(np.median(f_counts))
        b_med = float(np.median(b_counts))
        print(f"  {gname}: mean {np.mean(f_counts):.1f} vs {np.mean(b_counts):.1f}, "
              f"median {f_med:.0f} vs {b_med:.0f}, A12={a12:.2f}, p={p_value:.3f}")


def _per_bug_triggering_rates(bugs_df, fuzzer, name_map):
    """Compute per-bug triggering rate for *fuzzer*.

    Returns a DataFrame with columns ``benchmark``, ``bug_id``,
    ``triggered`` (number of instances that triggered), ``total`` (total instances).
    """
    df = bugs_df[bugs_df["fuzzer"] == fuzzer]
    if name_map:
        df = df[df["benchmark"].isin(name_map.keys())]
    if df.empty:
        return pd.DataFrame(columns=["benchmark", "bug_id", "triggered", "total"])

    total = df.groupby(["benchmark", "bug_id"])["instance"].nunique().rename("total")
    trig = (
        df[df["triggered"] > 0]
        .drop_duplicates(subset=["instance", "benchmark", "bug_id"])
        .groupby(["benchmark", "bug_id"])["instance"]
        .nunique()
        .rename("triggered")
    )
    result = total.to_frame().join(trig, how="left").fillna(0).astype({"triggered": int}).reset_index()
    return result


def generate_per_bug_fisher_defines(args, comparisons, group_names, bugs_df, name_map):
    """Per-bug Fisher's exact test with Benjamini--Hochberg correction.

    For each comparison, tests whether PSP changes the triggering rate of
    individual bugs.  Emits defines summarising the number of bugs with
    significantly different rates and the number that favour the fuzzer vs
    the baseline.
    """
    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests

    if bugs_df is None or bugs_df.empty:
        return

    comp_list = list(comparisons.items())

    for i, (fuzzer, baseline) in enumerate(comp_list):
        gname = group_names[i]
        cmd = _latex_cmd_name(gname)

        f_rates = _per_bug_triggering_rates(bugs_df, fuzzer, name_map)
        b_rates = _per_bug_triggering_rates(bugs_df, baseline, name_map)

        if f_rates.empty and b_rates.empty:
            continue

        # Merge on (benchmark, bug_id) -- keep bugs seen in either config
        merged = pd.merge(
            f_rates, b_rates,
            on=["benchmark", "bug_id"], how="outer",
            suffixes=("_f", "_b"),
        ).fillna(0).astype({"triggered_f": int, "total_f": int, "triggered_b": int, "total_b": int})

        # Use the max total across configs as the number of trials
        # (should be 30 for both, but be safe)
        merged["n"] = merged[["total_f", "total_b"]].max(axis=1)

        # Skip bugs never triggered by either
        merged = merged[(merged["triggered_f"] > 0) | (merged["triggered_b"] > 0)]

        if merged.empty:
            continue

        # Fisher's exact test per bug
        p_values = []
        odds_dirs = []  # +1 = fuzzer better, -1 = baseline better
        for _, row in merged.iterrows():
            n = int(row["n"])
            a = int(row["triggered_f"])
            b = int(row["triggered_b"])
            table = [[a, n - a], [b, n - b]]
            _, p = fisher_exact(table, alternative="two-sided")
            p_values.append(p)
            odds_dirs.append(1 if a > b else (-1 if b > a else 0))

        merged["p_value"] = p_values
        merged["direction"] = odds_dirs

        # Benjamini-Hochberg correction
        reject, p_adj, _, _ = multipletests(merged["p_value"], alpha=0.05, method="fdr_bh")
        merged["p_adj"] = p_adj
        merged["significant"] = reject

        n_tested = len(merged)
        n_sig = int(merged["significant"].sum())
        sig_rows = merged[merged["significant"]]
        n_favour_fuzzer = int((sig_rows["direction"] > 0).sum())
        n_favour_baseline = int((sig_rows["direction"] < 0).sum())

        # Complementarity: bugs found exclusively by one config (across all trials)
        n_only_fuzzer = int(((merged["triggered_f"] > 0) & (merged["triggered_b"] == 0)).sum())
        n_only_baseline = int(((merged["triggered_b"] > 0) & (merged["triggered_f"] == 0)).sum())
        n_both = int(((merged["triggered_f"] > 0) & (merged["triggered_b"] > 0)).sum())

        prefix = f"pspFisher{cmd}"
        define_latex_var(args, f"{prefix}Tested", n_tested)
        define_latex_var(args, f"{prefix}Sig", n_sig)
        define_latex_var(args, f"{prefix}FavourFuzzer", n_favour_fuzzer)
        define_latex_var(args, f"{prefix}FavourBaseline", n_favour_baseline)
        define_latex_var(args, f"{prefix}OnlyFuzzer", n_only_fuzzer)
        define_latex_var(args, f"{prefix}OnlyBaseline", n_only_baseline)
        define_latex_var(args, f"{prefix}Both", n_both)

        print(f"  {gname}: {n_sig}/{n_tested} bugs with significantly different rates (BH-corrected)")
        print(f"    favour fuzzer: {n_favour_fuzzer}, favour baseline: {n_favour_baseline}")
        print(f"    exclusive to fuzzer: {n_only_fuzzer}, exclusive to baseline: {n_only_baseline}, shared: {n_both}")

        if n_sig > 0:
            for _, row in sig_rows.iterrows():
                bm = row["benchmark"]
                bug = row["bug_id"]
                d = "+" if row["direction"] > 0 else "-"
                print(f"    {d} {bm}/{bug}: {int(row['triggered_f'])}/{int(row['n'])} vs {int(row['triggered_b'])}/{int(row['n'])} (p_adj={row['p_adj']:.3f})")


# ---------------------------------------------------------------------------
# Analysis package loading
# ---------------------------------------------------------------------------

def load_analysis_package(package_path: str) -> tuple:
    """Load analysis results from a zstd-compressed tar archive.

    Reads individual ``results/{benchmark}/{fuzzer_instance}.json`` files
    (not all_results.json which is too large).

    Returns:
        (runs, metadata) where runs is a list of dicts (one per instance)
        and metadata is the top-level metadata dict.
    """
    import pyzstd

    runs = []
    metadata = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        with pyzstd.ZstdFile(package_path, 'r') as zf:
            with tarfile.open(fileobj=zf, mode='r|') as tf:
                for member in tf:
                    if member.name == "metadata.json":
                        f = tf.extractfile(member)
                        metadata = json.load(f)
                    elif member.name.startswith("results/") and member.name.endswith(".json"):
                        f = tf.extractfile(member)
                        if f is not None:
                            try:
                                run = json.load(f)
                                runs.append(run)
                            except json.JSONDecodeError:
                                continue

    print(f"Loaded {len(runs)} analysis results from {package_path}")
    if metadata:
        print(f"Experiment: {metadata.get('experiment', 'unknown')}")
        print(f"Benchmarks: {len(metadata.get('benchmarks', []))}")
        print(f"Fuzzers: {metadata.get('fuzzers', [])}")

    return runs, metadata


# ---------------------------------------------------------------------------
# Analysis DataFrames
# ---------------------------------------------------------------------------

def _analysis_to_df(runs, name_map, fuzzer_filter, extract_fn):
    """Build a per-benchmark median DataFrame from analysis runs.

    *extract_fn(run)* receives a full run dict and returns a dict of
    numeric fields, or None to skip the run.
    """
    rows = []
    for run in runs:
        if fuzzer_filter and run["fuzzer"] != fuzzer_filter:
            continue
        if name_map and run["benchmark"] not in name_map:
            continue
        fields = extract_fn(run)
        if fields is None:
            continue
        rows.append({
            "benchmark": run["benchmark"],
            "fuzzer": run["fuzzer"],
            "instance": run.get("instance", 0),
            **fields,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg_cols = [c for c in df.columns if c not in ("benchmark", "fuzzer", "instance")]
    return df.groupby(["benchmark", "fuzzer"])[agg_cols].median().reset_index()


def _extract_entanglement(run):
    """Extract entanglement fields from a run.

    The archive stores per-input classifications where:
      - psp_only_count:    inputs discovering only PSP edges
      - both_count:        inputs discovering both PSP and non-PSP edges
      - baseline_only_count: inputs discovering only non-PSP edges

    Entanglement rate = both / (both + psp_only), i.e. among inputs that
    discover at least one PSP edge, how many also discover a non-PSP edge.
    """
    ent = run.get("analyzer_results", {}).get("vrp_hit_entanglement", {}).get("summary")
    if ent is None:
        return None
    both = ent["both_count"]
    psp_only = ent["psp_only_count"]
    baseline_only = ent["baseline_only_count"]
    psp_denom = both + psp_only
    total_cov = both + psp_only + baseline_only
    return {
        "entanglement_rate": (both / psp_denom) if psp_denom > 0 else 0.0,
        "non_psp_frac": (baseline_only / total_cov) if total_cov > 0 else 0.0,
        "psp_only_frac": (psp_only / total_cov) if total_cov > 0 else 0.0,
        "both_frac": (both / total_cov) if total_cov > 0 else 0.0,
        "total_psp_edges": ent["total_psp_edges"],
        "total_baseline_edges": ent["total_baseline_edges"],
    }


def _extract_stepping_stone(run):
    """Extract stepping-stone fields from a run."""
    ss = run.get("analyzer_results", {}).get("stepping_stone", {}).get("summary")
    if ss is None:
        return None
    return {k: ss[k] for k in (
        "total_psp_only_inputs", "successful_stepping_stones", "dead_ends",
        "stepping_stone_success_rate", "avg_hops_to_baseline",
    )}


def _extract_partition_progression(run):
    """Extract partition-progression fields from a run."""
    pp = run.get("analyzer_results", {}).get("partition_progression", {}).get("summary")
    if pp is None:
        return None
    return {k: pp[k] for k in (
        "total_comparison_groups", "fully_reached_groups", "partially_reached_groups",
        "unreached_groups", "sequential_climbing_rate",
        "total_psp_edges_in_binary", "total_psp_edges_reached", "psp_reach_rate",
    )}


# ---------------------------------------------------------------------------
# Shared helpers for subcommand preambles
# ---------------------------------------------------------------------------

def _load_analysis(args):
    """Shared preamble for analysis subcommands."""
    name_map = load_name_map(args.name_map)
    Path(args.latex_output).mkdir(parents=True, exist_ok=True)
    runs, metadata = load_analysis_package(args.package)
    fuzzer_filter = getattr(args, "fuzzer", None)
    return runs, name_map, fuzzer_filter


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_edge_coverage(args):
    """Generate edge-coverage tables from showmap_edge_growth analysis data."""
    runs, name_map, fuzzer_filter = _load_analysis(args)

    comparisons = parse_comparisons(args.comparisons_args)
    if not comparisons:
        print("Error: No valid comparisons. Use 'fuzzer:baseline' format", file=sys.stderr)
        sys.exit(1)

    # Build a coverage DataFrame from showmap_edge_growth.final_edges_covered.
    # Shape: benchmark, fuzzer, cycle (always 0), edges_covered.
    # This is compatible with compare_coverage().
    cov_rows = []
    for run in runs:
        seg = run.get("analyzer_results", {}).get("showmap_edge_growth")
        if seg is None:
            continue
        cov_rows.append({
            "benchmark": run["benchmark"],
            "fuzzer": run["fuzzer"],
            "cycle": 0,
            "edges_covered": seg["final_edges_covered"],
        })

    if not cov_rows:
        print("No showmap_edge_growth data found.")
        return

    coverage_df = pd.DataFrame(cov_rows)
    print(f"Loaded showmap coverage for {coverage_df['benchmark'].nunique()} benchmarks, "
          f"{coverage_df['fuzzer'].nunique()} fuzzers, "
          f"{len(coverage_df)} instances")

    prefix = args.prefix

    if args.group_names:
        if len(args.group_names) != len(comparisons):
            print(f"Error: --group-names expects {len(comparisons)} names, "
                  f"got {len(args.group_names)}", file=sys.stderr)
            sys.exit(1)
        print("\nGenerating combined coverage table")
        generate_combined_coverage_table(
            args, comparisons, args.group_names, coverage_df, name_map, prefix
        )

    for fuzzer, baseline in comparisons.items():
        print(f"\nGenerating tables: {fuzzer} vs {baseline}")
        if not args.group_names:
            generate_coverage_table(args, fuzzer, baseline, coverage_df, name_map, prefix)

    print("\nDone.")


def cmd_bug_data(args):
    """Generate bug-data tables (combined bugs + aggregate defines + Fisher)."""
    comparisons = parse_comparisons(args.comparisons_args)
    if not comparisons:
        print("Error: No valid comparisons. Use 'fuzzer:baseline' format", file=sys.stderr)
        sys.exit(1)

    name_map = load_name_map(args.name_map)
    Path(args.latex_output).mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {args.package}")
    coverage_df, stats_df, bugs_df, plot_df, metadata = load_data_package(args.package)

    print(f"Experiment: {metadata['experiment']}")
    print(f"Benchmarks: {metadata['benchmarks']}")
    print(f"Fuzzers: {metadata['fuzzers']}")

    # Validate comparisons
    available = set(metadata["fuzzers"])
    for fuzzer, baseline in comparisons.items():
        if fuzzer not in available:
            print(f"Warning: fuzzer '{fuzzer}' not found in data")
        if baseline not in available:
            print(f"Warning: baseline '{baseline}' not found in data")

    prefix = args.prefix

    if args.group_names:
        if len(args.group_names) != len(comparisons):
            print(f"Error: --group-names expects {len(comparisons)} names, "
                  f"got {len(args.group_names)}", file=sys.stderr)
            sys.exit(1)
        print("\nGenerating combined bugs table")
        generate_combined_bugs_table(
            args, comparisons, args.group_names, bugs_df, name_map, prefix
        )
        print("\nGenerating aggregate bug defines")
        generate_aggregate_bug_defines(args, comparisons, args.group_names, bugs_df, name_map)
        print("\nPer-bug Fisher's exact test (BH-corrected)")
        generate_per_bug_fisher_defines(args, comparisons, args.group_names, bugs_df, name_map)

    for fuzzer, baseline in comparisons.items():
        print(f"\nGenerating tables: {fuzzer} vs {baseline}")
        generate_bugs_table(args, fuzzer, baseline, bugs_df, name_map, prefix)

    print("\nDone.")


def cmd_entanglement(args):
    """Generate entanglement table and defines from analysis package."""
    runs, name_map, fuzzer_filter = _load_analysis(args)

    fuzzers = args.fuzzers if args.fuzzers else None
    group_names = args.group_names

    if fuzzers and group_names:
        if len(group_names) != len(fuzzers):
            print(f"Error: --group-names expects {len(fuzzers)} names, "
                  f"got {len(group_names)}", file=sys.stderr)
            sys.exit(1)
        _cmd_entanglement_grouped(args, runs, name_map, fuzzers, group_names)
    else:
        # Single-fuzzer mode (use --fuzzer filter or show all)
        df = _analysis_to_df(runs, name_map, fuzzer_filter, _extract_entanglement)
        if df.empty:
            print("No entanglement data found.")
            return
        _cmd_entanglement_single(args, df, name_map)


def _cmd_entanglement_single(args, df, name_map):
    """Write a single-fuzzer entanglement table."""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Benchmark": display_name(r["benchmark"], name_map),
            r"$\mathcal{E}$\ (\%)": f"{r['entanglement_rate'] * 100:.1f}",
            r"Non-PSP (\%)": f"{r['non_psp_frac'] * 100:.1f}",
            "PSP Edges": f"{r['total_psp_edges']:.0f}",
            "Base Edges": f"{r['total_baseline_edges']:.0f}",
        })
    if not rows:
        print("No rows to write.")
        return

    prefix = args.prefix
    tbl_name = _prefixed("entanglement", prefix)
    def_pfx = _define_prefix("pspEnt", prefix)

    define_latex_table(args, tbl_name, pd.DataFrame(rows))
    print(f"  Written: {tbl_name}.tex ({len(rows)} rows)")

    median_ent_rate = df["entanglement_rate"].median() * 100
    n_benchmarks = df["benchmark"].nunique()
    define_latex_var(args, f"{def_pfx}MedianEntanglementRate", round(median_ent_rate, 1))
    define_latex_var(args, f"{def_pfx}BenchmarkCount", n_benchmarks)
    print(f"  Median entanglement rate: {median_ent_rate:.1f}%")
    print(f"  Benchmarks: {n_benchmarks}")
    print("\nDone.")


def _cmd_entanglement_grouped(args, runs, name_map, fuzzers, group_names):
    """Write a combined entanglement table with column groups per fuzzer."""
    # Build per-fuzzer DataFrames
    per_fuzzer = {}
    for fz in fuzzers:
        df = _analysis_to_df(runs, name_map, fz, _extract_entanglement)
        if not df.empty:
            per_fuzzer[fz] = df.set_index("benchmark")

    if not per_fuzzer:
        print("No entanglement data found.")
        return

    # Collect all benchmarks that appear in any fuzzer
    all_benchmarks = sorted(set().union(*(df.index for df in per_fuzzer.values())))

    rows = []
    for bm in all_benchmarks:
        if name_map and bm not in name_map:
            continue
        row = {"Benchmark": display_name(bm, name_map)}
        has_any = False
        base_edges = None
        for fz, gname in zip(fuzzers, group_names):
            df = per_fuzzer.get(fz)
            if df is not None and bm in df.index:
                r = df.loc[bm]
                has_any = True
                row[f"{gname} $\\mathcal{{E}}$\\ (\\%)"] = f"{r['entanglement_rate'] * 100:.1f}"
                row[f"{gname} Non-PSP (\\%)"] = f"{r['non_psp_frac'] * 100:.1f}"
                row[f"{gname} PSP-only (\\%)"] = f"{r['psp_only_frac'] * 100:.1f}"
                row[f"{gname} Both (\\%)"] = f"{r['both_frac'] * 100:.1f}"
            else:
                row[f"{gname} $\\mathcal{{E}}$\\ (\\%)"] = "{---}"
                row[f"{gname} Non-PSP (\\%)"] = "{---}"
                row[f"{gname} PSP-only (\\%)"] = "{---}"
                row[f"{gname} Both (\\%)"] = "{---}"
        if has_any:
            rows.append(row)

    if not rows:
        print("No rows to write.")
        return

    prefix = args.prefix
    tbl_name = _prefixed("entanglement", prefix)
    def_pfx = _define_prefix("pspEnt", prefix)

    define_latex_table(args, tbl_name, pd.DataFrame(rows))
    print(f"  Written: {tbl_name}.tex ({len(rows)} rows)")

    # Emit per-group defines
    for fz, gname in zip(fuzzers, group_names):
        df = per_fuzzer.get(fz)
        if df is None:
            continue
        suffix = _latex_cmd_name(gname)
        median_non_psp = df["non_psp_frac"].median() * 100
        median_psp_only = df["psp_only_frac"].median() * 100
        median_both = df["both_frac"].median() * 100
        median_ent = df["entanglement_rate"].median() * 100
        n_bm = len(df)
        define_latex_var(args, f"{def_pfx}{suffix}MedianNonPspFrac", round(median_non_psp, 1))
        define_latex_var(args, f"{def_pfx}{suffix}MedianPspOnlyFrac", round(median_psp_only, 1))
        define_latex_var(args, f"{def_pfx}{suffix}MedianBothFrac", round(median_both, 1))
        define_latex_var(args, f"{def_pfx}{suffix}MedianEntanglementRate", round(median_ent, 1))
        define_latex_var(args, f"{def_pfx}{suffix}BenchmarkCount", n_bm)
        print(f"  {gname}: non-PSP={median_non_psp:.1f}%, PSP-only={median_psp_only:.1f}%, both={median_both:.1f}%, E={median_ent:.1f}%, benchmarks={n_bm}")

    print("\nDone.")


def cmd_stepping_stones(args):
    """Generate stepping-stones table and defines from analysis package."""
    runs, name_map, fuzzer_filter = _load_analysis(args)

    df = _analysis_to_df(runs, name_map, fuzzer_filter, _extract_stepping_stone)
    if df.empty:
        print("No stepping-stone data found.")
        return

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Benchmark": display_name(r["benchmark"], name_map),
            "PSP-only": f"{r['total_psp_only_inputs']:.0f}",
            "Successful": f"{r['successful_stepping_stones']:.0f}",
            "Dead Ends": f"{r['dead_ends']:.0f}",
            r"Success Rate \%": f"{r['stepping_stone_success_rate'] * 100:.1f}",
            "Avg Hops": f"{r['avg_hops_to_baseline']:.1f}",
        })

    if not rows:
        print("No rows to write.")
        return

    prefix = args.prefix
    tbl_name = _prefixed("stepping-stones", prefix)
    def_pfx = _define_prefix("pspStep", prefix)

    table_df = pd.DataFrame(rows)
    define_latex_table(args, tbl_name, table_df)
    print(f"  Written: {tbl_name}.tex ({len(rows)} rows)")

    # Emit defines
    median_success_rate = df["stepping_stone_success_rate"].median() * 100
    median_avg_hops = df["avg_hops_to_baseline"].median()
    n_benchmarks = df["benchmark"].nunique()

    define_latex_var(args, f"{def_pfx}MedianSuccessRate", round(median_success_rate, 1))
    define_latex_var(args, f"{def_pfx}MedianAvgHops", round(float(median_avg_hops), 1))
    define_latex_var(args, f"{def_pfx}BenchmarkCount", n_benchmarks)

    print(f"  Median success rate: {median_success_rate:.1f}%")
    print(f"  Median avg hops: {median_avg_hops:.1f}")
    print(f"  Benchmarks: {n_benchmarks}")
    print("\nDone.")


def cmd_partition_progression(args):
    """Generate partition-progression table and defines from analysis package."""
    runs, name_map, fuzzer_filter = _load_analysis(args)

    df = _analysis_to_df(runs, name_map, fuzzer_filter, _extract_partition_progression)
    if df.empty:
        print("No partition-progression data found.")
        return

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Benchmark": display_name(r["benchmark"], name_map),
            "Groups": f"{r['total_comparison_groups']:.0f}",
            "Full": f"{r['fully_reached_groups']:.0f}",
            "Partial": f"{r['partially_reached_groups']:.0f}",
            "Unreached": f"{r['unreached_groups']:.0f}",
            r"Climbing \%": f"{r['sequential_climbing_rate'] * 100:.1f}",
            "PSP Edges": f"{r['total_psp_edges_in_binary']:.0f}",
            "Reached": f"{r['total_psp_edges_reached']:.0f}",
            r"Reach \%": f"{r['psp_reach_rate'] * 100:.1f}",
        })

    if not rows:
        print("No rows to write.")
        return

    prefix = args.prefix
    tbl_name = _prefixed("partition-progression", prefix)
    def_pfx = _define_prefix("pspPart", prefix)

    table_df = pd.DataFrame(rows)
    define_latex_table(args, tbl_name, table_df)
    print(f"  Written: {tbl_name}.tex ({len(rows)} rows)")

    # Emit defines
    median_climbing = df["sequential_climbing_rate"].median() * 100
    median_reach = df["psp_reach_rate"].median() * 100
    n_benchmarks = df["benchmark"].nunique()

    define_latex_var(args, f"{def_pfx}MedianClimbingRate", round(median_climbing, 1))
    define_latex_var(args, f"{def_pfx}MedianReachRate", round(median_reach, 1))
    define_latex_var(args, f"{def_pfx}BenchmarkCount", n_benchmarks)

    print(f"  Median climbing rate: {median_climbing:.1f}%")
    print(f"  Median reach rate: {median_reach:.1f}%")
    print(f"  Benchmarks: {n_benchmarks}")
    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI setup
# ---------------------------------------------------------------------------

def main():
    # -- Parent parsers (shared argument groups) --

    # Common args: --latex-output, --name-map, --prefix
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--latex-output", type=str, default="generated/psp",
        help="Output directory for LaTeX table files (default: generated/psp)",
    )
    common_parser.add_argument(
        "--name-map", type=str, default=None,
        help="Path to a JSON file mapping internal names to display names. "
             "When given, only benchmarks present in the map are included.",
    )
    common_parser.add_argument(
        "--prefix", type=str, default=None,
        help="Prefix for output file names (e.g., 'fuzzbench' -> 'entanglement-fuzzbench.tex').",
    )

    # Analysis args: package, --fuzzer
    analysis_parser = argparse.ArgumentParser(add_help=False)
    analysis_parser.add_argument("package", help="Path to analysis data package (.tar.zstd)")
    analysis_parser.add_argument(
        "--fuzzer", type=str, default=None,
        help="Filter results to a specific fuzzer name.",
    )

    # -- Top-level parser with subcommands --

    parser = argparse.ArgumentParser(
        description="Generate LaTeX table bodies from PSP data packages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  edge-coverage           Coverage comparison tables from experiment packages
  bug-data                Bug triggering tables from experiment packages
  entanglement            VRP hit/bucket entanglement from analysis packages
  stepping-stones         Stepping-stone analysis from analysis packages
  partition-progression   Partition progression from analysis packages

Examples:
  %(prog)s edge-coverage data/psp/pkg.tar.zst aflplusplus_vrp:aflplusplus --group-names VRP
  %(prog)s bug-data data/psp/pkg.tar.zst aflplusplus_vrp:aflplusplus --group-names VRP
  %(prog)s entanglement data/psp/analysis-data.tar.zstd --latex-output generated/psp
  %(prog)s stepping-stones data/psp/analysis-data.tar.zstd
  %(prog)s partition-progression data/psp/analysis-data.tar.zstd --fuzzer aflplusplus_vrp
        """,
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # edge-coverage
    sp_cov = subparsers.add_parser(
        "edge-coverage",
        parents=[common_parser, analysis_parser],
        help="Generate edge-coverage comparison tables from showmap analysis data",
    )
    sp_cov.add_argument(
        "comparisons_args", nargs="+", metavar="FUZZER:BASELINE",
        help="Comparisons in 'fuzzer:baseline' format",
    )
    sp_cov.add_argument(
        "--group-names", type=str, nargs="+", default=None,
        help="Display names for each comparison group (e.g., 'VRP' 'Unfold'). "
             "When given, generates a combined table with column groups.",
    )
    sp_cov.add_argument(
        "--no-bold-p", action="store_true", default=False,
        help="Do not bold significant p-values (default: bold them).",
    )
    sp_cov.add_argument(
        "--bold-a12", action="store_true", default=False,
        help="Bold non-negligible A12 effect sizes (|A12 - 0.5| >= 0.06). Default: False.",
    )
    sp_cov.add_argument(
        "--shade-significant", action="store_true", default=False,
        help="Shade cells for comparison groups with p < 0.05 and non-negligible effect size.",
    )
    sp_cov.set_defaults(func=cmd_edge_coverage)

    # bug-data
    sp_bug = subparsers.add_parser(
        "bug-data",
        parents=[common_parser],
        help="Generate bug triggering tables and defines",
    )
    sp_bug.add_argument("package", help="Path to experiment data package (.tar.zst)")
    sp_bug.add_argument(
        "comparisons_args", nargs="+", metavar="FUZZER:BASELINE",
        help="Comparisons in 'fuzzer:baseline' format",
    )
    sp_bug.add_argument(
        "--group-names", type=str, nargs="+", default=None,
        help="Display names for each comparison group (e.g., 'VRP' 'Unfold'). "
             "When given, generates a combined table with column groups.",
    )
    sp_bug.add_argument(
        "--shade-significant", action="store_true", default=False,
        help="Shade cells for bugs with significantly different triggering rates.",
    )
    sp_bug.set_defaults(func=cmd_bug_data)

    # entanglement
    sp_ent = subparsers.add_parser(
        "entanglement",
        parents=[common_parser, analysis_parser],
        help="Generate VRP entanglement table from analysis data",
    )
    sp_ent.add_argument(
        "fuzzers", nargs="*", default=None,
        help="Fuzzer names to include as column groups (e.g., aflplusplus_vrp aflplusplus_unfold_all).",
    )
    sp_ent.add_argument(
        "--group-names", type=str, nargs="+", default=None,
        help="Display names for each fuzzer group (e.g., 'VRP' 'Call Unfolding').",
    )
    sp_ent.set_defaults(func=cmd_entanglement)

    # stepping-stones
    sp_step = subparsers.add_parser(
        "stepping-stones",
        parents=[common_parser, analysis_parser],
        help="Generate stepping-stones table from analysis data",
    )
    sp_step.set_defaults(func=cmd_stepping_stones)

    # partition-progression
    sp_part = subparsers.add_parser(
        "partition-progression",
        parents=[common_parser, analysis_parser],
        help="Generate partition-progression table from analysis data",
    )
    sp_part.set_defaults(func=cmd_partition_progression)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
