#!/usr/bin/env python3
"""Compare fuzzer configurations against baselines.

Usage:
    ./compare_fuzzers.py -e <experiment> fuzzer1:baseline1 fuzzer2:baseline2 ...

Examples:
    ./compare_fuzzers.py -e vrp-unfold-eval02 aflplusplus_vrp_lto:aflplusplus_lto aflplusplus_vrp:aflplusplus
    ./compare_fuzzers.py -e vrp-unfold-24h -c config.yaml aflplusplus_vrp:aflplusplus
"""

import argparse
import json
import os
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as ss
import yaml

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        """Fallback when tqdm is not available."""
        return iterable


def get_db_url_from_config(config_path: str) -> str:
    """Extract database URL from config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("database_url")


def get_coverage_data(experiment: str, db_url: str) -> pd.DataFrame:
    """Get coverage data from database.

    Note: This function requires the fuzzbench database module to be available.
    """
    # Lazy import of database modules (not needed when using --from-package)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from database import utils as db_utils
    from database.models import GroupSnapshot

    if db_url:
        os.environ["SQL_DATABASE_URL"] = db_url

    with db_utils.session_scope() as session:
        query = (
            session.query(
                GroupSnapshot.fuzzer,
                GroupSnapshot.benchmark,
                GroupSnapshot.instance_group_num,
                GroupSnapshot.cycle,
                GroupSnapshot.edges_covered,
            )
            .filter(GroupSnapshot.experiment == experiment)
        )
        df = pd.read_sql_query(query.statement, db_utils.engine)

    return df


def parse_fuzzer_stat_value(value: str):
    """Parse a fuzzer stat value to appropriate type.

    Handles:
    - Percentage strings like "100.00%" -> 100.0
    - Integer strings like "12345" -> 12345
    - Float strings like "123.45" -> 123.45
    - Other strings remain as strings
    """
    if not isinstance(value, str):
        return value

    # Handle percentage values
    if value.endswith('%'):
        try:
            return float(value.rstrip('%'))
        except ValueError:
            return value

    # Try integer first
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    return value


def get_fuzzer_stats(experiment_folder: str, benchmark: str, fuzzer: str, latest_only: bool = False) -> list:
    """Get fuzzer stats from stats files.

    Args:
        experiment_folder: Path to experiment folders
        benchmark: Benchmark name
        fuzzer: Fuzzer name
        latest_only: If True, only return latest cycle per instance

    Returns list of dicts: [{instance, cycle, stat_name: value, ...}]
    All available fields from the fuzzer stats JSON are extracted dynamically.
    """
    folder = Path(experiment_folder) / f"{benchmark}-{fuzzer}"
    if not folder.exists():
        return []

    all_stats = []
    for trial_dir in folder.iterdir():
        # Support both trial-* and instance-* folder naming conventions
        if trial_dir.name.startswith("trial-"):
            instance_num = int(trial_dir.name.split("-")[1])
        elif trial_dir.name.startswith("instance-"):
            instance_num = int(trial_dir.name.split("-")[1])
        else:
            continue

        stats_dir = trial_dir / "stats"
        if not stats_dir.exists():
            continue

        # Get all stats files sorted by cycle
        stats_files = sorted(stats_dir.glob("stats-*.json*"))
        if not stats_files:
            continue

        # If latest_only, just process the last file
        files_to_process = [stats_files[-1]] if latest_only else stats_files

        for stats_file in files_to_process:
            try:
                # Extract cycle number from filename (stats-0001.json or stats-0001.json.gz)
                cycle_str = stats_file.name.split('-')[1].split('.')[0]
                cycle = int(cycle_str)

                with open(stats_file) as f:
                    data = json.load(f)
                if "fuzzer" in data and isinstance(data["fuzzer"], dict):
                    fuzzer_stats = data["fuzzer"]
                    # Dynamically extract all fields with proper type conversion
                    record = {
                        'instance': instance_num,
                        'cycle': cycle,
                    }
                    for key, value in fuzzer_stats.items():
                        # Skip non-numeric metadata fields that aren't useful for analysis
                        if key in ('afl_banner', 'command_line', 'target_mode', 'afl_version'):
                            continue
                        record[key] = parse_fuzzer_stat_value(value)
                    all_stats.append(record)
            except (json.JSONDecodeError, KeyError, ValueError, TypeError, AttributeError):
                continue

    return all_stats


def _get_fuzzer_stats_for_combo(args: tuple) -> list:
    """Get fuzzer stats for a single benchmark-fuzzer combo (for parallel execution).

    Args:
        args: Tuple of (experiment_folder, benchmark, fuzzer)

    Returns list of dicts with benchmark and fuzzer fields included.
    """
    experiment_folder, benchmark, fuzzer = args
    stats = get_fuzzer_stats(experiment_folder, benchmark, fuzzer)
    # Add benchmark and fuzzer to each record
    for record in stats:
        record['benchmark'] = benchmark
        record['fuzzer'] = fuzzer
    return stats


def get_fuzzer_stats_parallel(experiment_folder: str, benchmarks: list, fuzzers: list,
                               max_workers: int = None, show_progress: bool = True) -> list:
    """Collect fuzzer stats for all benchmark-fuzzer combinations in parallel.

    Returns list of dicts: [{benchmark, fuzzer, instance, cycle, ...stats}]
    """
    combos = [(experiment_folder, b, f) for b in benchmarks for f in fuzzers]

    if not combos:
        return []

    all_stats = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_get_fuzzer_stats_for_combo, combo): combo
                   for combo in combos}

        if show_progress and HAS_TQDM:
            iterator = tqdm(as_completed(futures), total=len(futures),
                           desc="Collecting fuzzer stats", unit="combo")
        else:
            iterator = as_completed(futures)

        for future in iterator:
            try:
                records = future.result()
                all_stats.extend(records)
            except Exception:
                pass

    return all_stats


# Column names from AFL++ plot_data header
PLOT_DATA_COLUMNS = [
    'relative_time', 'cycles_done', 'cur_item', 'corpus_count', 'pending_total',
    'pending_favs', 'map_size', 'saved_crashes', 'saved_hangs', 'max_depth',
    'execs_per_sec', 'total_execs', 'edges_found', 'total_crashes', 'servers_count'
]


def _extract_plot_data_from_archive(archive_info: tuple) -> list:
    """Extract plot_data from a single corpus archive.

    Args:
        archive_info: Tuple of (archive_path, benchmark, fuzzer, instance_num)

    Returns list of dicts with plot_data records.
    """
    archive_path, benchmark, fuzzer, instance_num = archive_info

    try:
        records = []
        with tarfile.open(archive_path, 'r:gz') as tar:
            # Find the plot_data file
            plot_data_member = None
            for member in tar.getmembers():
                if member.name.endswith('/plot_data'):
                    plot_data_member = member
                    break

            if not plot_data_member:
                return []

            f = tar.extractfile(plot_data_member)
            if not f:
                return []

            content = f.read().decode('utf-8')
            for line in content.strip().split('\n'):
                if line.startswith('#'):
                    continue
                values = [v.strip().rstrip('%') for v in line.split(',')]
                if len(values) != len(PLOT_DATA_COLUMNS):
                    continue

                record = {
                    'benchmark': benchmark,
                    'fuzzer': fuzzer,
                    'instance': instance_num,
                }
                for col, val in zip(PLOT_DATA_COLUMNS, values):
                    try:
                        if '.' in val:
                            record[col] = float(val)
                        else:
                            record[col] = int(val)
                    except ValueError:
                        record[col] = val
                records.append(record)

        return records

    except (tarfile.TarError, OSError, ValueError):
        return []


def collect_plot_data_archives(experiment_folder: str, benchmarks: list, fuzzers: list) -> list:
    """Collect corpus archive paths for plot_data extraction.

    Only collects the LAST archive per instance since plot_data is cumulative
    (each archive contains all data points from start to that cycle).

    Returns list of tuples: [(archive_path, benchmark, fuzzer, instance_num), ...]
    """
    archives = []
    for benchmark in benchmarks:
        for fuzzer in fuzzers:
            folder = Path(experiment_folder) / f"{benchmark}-{fuzzer}"
            if not folder.exists():
                continue

            for trial_dir in folder.iterdir():
                # Support both trial-* and instance-* folder naming conventions
                if trial_dir.name.startswith("trial-"):
                    instance_num = int(trial_dir.name.split("-")[1])
                elif trial_dir.name.startswith("instance-"):
                    instance_num = int(trial_dir.name.split("-")[1])
                else:
                    continue

                corpus_dir = trial_dir / "corpus"
                if not corpus_dir.exists():
                    continue

                # Only get the LAST archive (plot_data is cumulative)
                instance_archives = sorted(corpus_dir.glob("corpus-archive-*.tar.gz"))
                if instance_archives:
                    archives.append((instance_archives[-1], benchmark, fuzzer, instance_num))

    return archives


def get_plot_data_parallel(experiment_folder: str, benchmarks: list, fuzzers: list,
                           max_workers: int = None, show_progress: bool = True) -> list:
    """Extract plot_data from all corpus archives in parallel.

    Args:
        experiment_folder: Path to experiment folders
        benchmarks: List of benchmark names
        fuzzers: List of fuzzer names
        max_workers: Max parallel workers (default: min(32, cpu_count + 4))
        show_progress: Whether to show progress bar

    Returns list of dicts: [{benchmark, fuzzer, instance, cycle, relative_time, ...}]
    """
    # Collect all archive paths
    archives = collect_plot_data_archives(experiment_folder, benchmarks, fuzzers)

    if not archives:
        return []

    all_data = []

    # Process archives in parallel across multiple cores
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_extract_plot_data_from_archive, info): info
                   for info in archives}

        if show_progress and HAS_TQDM:
            iterator = tqdm(as_completed(futures), total=len(futures),
                           desc="Extracting plot_data", unit="archive")
        else:
            iterator = as_completed(futures)

        for future in iterator:
            try:
                records = future.result()
                all_data.extend(records)
            except Exception:
                pass

    return all_data


def get_plot_data(experiment_folder: str, benchmark: str, fuzzer: str) -> list:
    """Extract fine-grained plot_data from corpus archives (single benchmark-fuzzer).

    The plot_data file in corpus archives contains ~5 second granularity data
    (vs 15 min for stats JSON), useful for detailed coverage/performance curves.

    Returns list of dicts: [{instance, relative_time, corpus_count, edges_found, ...}]
    """
    # Use the parallel function with a single benchmark/fuzzer
    results = get_plot_data_parallel(experiment_folder, [benchmark], [fuzzer], show_progress=False)
    # Remove benchmark/fuzzer from results since caller adds them
    for r in results:
        r.pop('benchmark', None)
        r.pop('fuzzer', None)
    return results


def compare_fuzzer_stats(experiment_folder: str, fuzzer: str, baseline: str, benchmark: str) -> dict:
    """Compare fuzzer stats between fuzzer and baseline."""
    fuzzer_stats = get_fuzzer_stats(experiment_folder, benchmark, fuzzer, latest_only=True)
    baseline_stats = get_fuzzer_stats(experiment_folder, benchmark, baseline, latest_only=True)

    if not fuzzer_stats or not baseline_stats:
        return None

    # Stats to compare
    stat_names = ['execs_done', 'execs_per_sec', 'corpus_count', 'corpus_favored',
                  'edges_found', 'cycles_done', 'stability', 'bitmap_cvg']

    results = {}
    for stat_name in stat_names:
        fuzzer_values = [s[stat_name] for s in fuzzer_stats if stat_name in s]
        baseline_values = [s[stat_name] for s in baseline_stats if stat_name in s]

        if not fuzzer_values or not baseline_values:
            continue

        fuzzer_mean = np.mean(fuzzer_values)
        baseline_mean = np.mean(baseline_values)
        a12 = vargha_delaney_a12(fuzzer_values, baseline_values)
        stat, p_value, significant = mann_whitney_test(fuzzer_values, baseline_values)

        results[stat_name] = {
            'fuzzer_mean': fuzzer_mean,
            'baseline_mean': baseline_mean,
            'fuzzer_median': np.median(fuzzer_values),
            'baseline_median': np.median(baseline_values),
            'absolute_diff': fuzzer_mean - baseline_mean,
            'relative_diff_pct': ((fuzzer_mean - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0,
            'a12': a12,
            'effect_size': effect_size_interpretation(a12),
            'p_value': p_value,
            'significant': significant,
        }

    return results


def get_magma_bugs(experiment_folder: str, benchmark: str, fuzzer: str, latest_only: bool = False) -> list:
    """Get Magma bug data from stats files.

    Args:
        experiment_folder: Path to experiment folders
        benchmark: Benchmark name
        fuzzer: Fuzzer name
        latest_only: If True, only return latest cycle per instance (legacy behavior)

    Returns list of dicts: [{instance, cycle, bug_id, reached, triggered}]
    """
    folder = Path(experiment_folder) / f"{benchmark}-{fuzzer}"
    if not folder.exists():
        return []

    all_bugs = []
    for instance_dir in folder.iterdir():
        # Support both trial-* and instance-* folder naming conventions
        if instance_dir.name.startswith("trial-"):
            instance_num = int(instance_dir.name.split("-")[1])
        elif instance_dir.name.startswith("instance-"):
            instance_num = int(instance_dir.name.split("-")[1])
        else:
            continue

        stats_dir = instance_dir / "stats"
        if not stats_dir.exists():
            continue

        # Get all stats files sorted by cycle
        stats_files = sorted(stats_dir.glob("stats-*.json*"))
        if not stats_files:
            continue

        # If latest_only, just process the last file
        files_to_process = [stats_files[-1]] if latest_only else stats_files

        for stats_file in files_to_process:
            try:
                # Extract cycle number from filename (stats-0001.json or stats-0001.json.gz)
                cycle_str = stats_file.name.split('-')[1].split('.')[0]
                cycle = int(cycle_str)

                with open(stats_file) as f:
                    data = json.load(f)
                if "benchmark" in data and isinstance(data["benchmark"], dict):
                    for bug_id, counts in data["benchmark"].items():
                        all_bugs.append({
                            'instance': instance_num,
                            'cycle': cycle,
                            'bug_id': bug_id,
                            'reached': counts[0],
                            'triggered': counts[1],
                        })
            except (json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError):
                continue

    return all_bugs


def get_magma_bugs_by_instance(experiment_folder: str, benchmark: str, fuzzer: str) -> dict:
    """Get Magma bug data (latest cycle only) grouped by instance.

    Legacy helper for compare_magma_bugs - returns dict: {instance_num: {bug_id: (reached, triggered)}}
    """
    bugs_list = get_magma_bugs(experiment_folder, benchmark, fuzzer, latest_only=True)
    bugs_by_instance = {}
    for bug in bugs_list:
        instance = bug['instance']
        if instance not in bugs_by_instance:
            bugs_by_instance[instance] = {}
        bugs_by_instance[instance][bug['bug_id']] = (bug['reached'], bug['triggered'])
    return bugs_by_instance


def _get_magma_bugs_for_combo(args: tuple) -> list:
    """Get magma bugs for a single benchmark-fuzzer combo (for parallel execution).

    Args:
        args: Tuple of (experiment_folder, benchmark, fuzzer)

    Returns list of dicts with benchmark and fuzzer fields included.
    """
    experiment_folder, benchmark, fuzzer = args
    bugs = get_magma_bugs(experiment_folder, benchmark, fuzzer)
    # Add benchmark and fuzzer to each record
    for record in bugs:
        record['benchmark'] = benchmark
        record['fuzzer'] = fuzzer
    return bugs


def get_magma_bugs_parallel(experiment_folder: str, benchmarks: list, fuzzers: list,
                            max_workers: int = None, show_progress: bool = True) -> list:
    """Collect magma bugs for all benchmark-fuzzer combinations in parallel.

    Only processes benchmarks ending with '_magma'.

    Returns list of dicts: [{benchmark, fuzzer, instance, cycle, bug_id, reached, triggered}]
    """
    magma_benchmarks = [b for b in benchmarks if b.endswith('_magma')]
    combos = [(experiment_folder, b, f) for b in magma_benchmarks for f in fuzzers]

    if not combos:
        return []

    all_bugs = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_get_magma_bugs_for_combo, combo): combo
                   for combo in combos}

        if show_progress and HAS_TQDM:
            iterator = tqdm(as_completed(futures), total=len(futures),
                           desc="Collecting Magma bugs", unit="combo")
        else:
            iterator = as_completed(futures)

        for future in iterator:
            try:
                records = future.result()
                all_bugs.extend(records)
            except Exception:
                pass

    return all_bugs


def vargha_delaney_a12(x: list, y: list) -> float:
    """Compute Vargha-Delaney A12 effect size.

    A12 measures the probability that a randomly selected value from x
    is greater than a randomly selected value from y.

    A12 = 0.5 means no difference
    A12 > 0.5 means x tends to be larger
    A12 < 0.5 means y tends to be larger

    Effect size interpretation:
    - |A12 - 0.5| < 0.06: negligible
    - |A12 - 0.5| < 0.14: small
    - |A12 - 0.5| < 0.21: medium
    - |A12 - 0.5| >= 0.21: large
    """
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0.5

    r = ss.rankdata(x + y)
    r1 = sum(r[:m])

    # A12 = (R1/m - (m+1)/2) / n
    a12 = (r1 / m - (m + 1) / 2) / n
    return a12


def effect_size_interpretation(a12: float) -> str:
    """Interpret A12 effect size."""
    diff = abs(a12 - 0.5)
    if diff < 0.06:
        return "negligible"
    elif diff < 0.14:
        return "small"
    elif diff < 0.21:
        return "medium"
    else:
        return "large"


def mann_whitney_test(x: list, y: list) -> tuple:
    """Perform Mann-Whitney U test.

    Returns (statistic, p_value, significant)
    """
    if len(x) < 2 or len(y) < 2:
        return None, None, False

    try:
        stat, p = ss.mannwhitneyu(x, y, alternative='two-sided')
        return stat, p, p < 0.05
    except ValueError:
        return None, None, False


def compare_coverage(df: pd.DataFrame, fuzzer: str, baseline: str, benchmark: str) -> dict:
    """Compare coverage between fuzzer and baseline for a benchmark."""
    fuzzer_data = df[(df['fuzzer'] == fuzzer) & (df['benchmark'] == benchmark)]
    baseline_data = df[(df['fuzzer'] == baseline) & (df['benchmark'] == benchmark)]

    if fuzzer_data.empty or baseline_data.empty:
        return None

    # Get final coverage per instance
    max_cycle = df['cycle'].max()
    fuzzer_final = fuzzer_data[fuzzer_data['cycle'] == max_cycle]['edges_covered'].tolist()
    baseline_final = baseline_data[baseline_data['cycle'] == max_cycle]['edges_covered'].tolist()

    if not fuzzer_final or not baseline_final:
        return None

    # Statistics
    fuzzer_mean = np.mean(fuzzer_final)
    baseline_mean = np.mean(baseline_final)
    fuzzer_median = np.median(fuzzer_final)
    baseline_median = np.median(baseline_final)
    fuzzer_std = np.std(fuzzer_final)
    baseline_std = np.std(baseline_final)

    # Effect size and significance
    a12 = vargha_delaney_a12(fuzzer_final, baseline_final)
    stat, p_value, significant = mann_whitney_test(fuzzer_final, baseline_final)

    # Relative improvement
    if baseline_mean > 0:
        rel_improvement = ((fuzzer_mean - baseline_mean) / baseline_mean) * 100
    else:
        rel_improvement = 0

    return {
        'fuzzer_mean': fuzzer_mean,
        'baseline_mean': baseline_mean,
        'fuzzer_median': fuzzer_median,
        'baseline_median': baseline_median,
        'fuzzer_std': fuzzer_std,
        'baseline_std': baseline_std,
        'absolute_diff': fuzzer_mean - baseline_mean,
        'relative_improvement_pct': rel_improvement,
        'a12': a12,
        'effect_size': effect_size_interpretation(a12),
        'p_value': p_value,
        'significant': significant,
        'fuzzer_values': fuzzer_final,
        'baseline_values': baseline_final,
    }


def compare_coverage_over_time(df: pd.DataFrame, fuzzer: str, baseline: str, benchmark: str) -> dict:
    """Compare coverage progression over time."""
    fuzzer_data = df[(df['fuzzer'] == fuzzer) & (df['benchmark'] == benchmark)]
    baseline_data = df[(df['fuzzer'] == baseline) & (df['benchmark'] == benchmark)]

    if fuzzer_data.empty or baseline_data.empty:
        return None

    # Get mean coverage per cycle
    fuzzer_by_cycle = fuzzer_data.groupby('cycle')['edges_covered'].mean()
    baseline_by_cycle = baseline_data.groupby('cycle')['edges_covered'].mean()

    # Area under curve (approximate integral)
    fuzzer_auc = np.trapezoid(fuzzer_by_cycle.values, fuzzer_by_cycle.index)
    baseline_auc = np.trapezoid(baseline_by_cycle.values, baseline_by_cycle.index)

    # Time to reach baseline's final coverage
    baseline_final = baseline_by_cycle.iloc[-1] if len(baseline_by_cycle) > 0 else 0
    time_to_reach = None
    for cycle, cov in fuzzer_by_cycle.items():
        if cov >= baseline_final:
            time_to_reach = cycle
            break

    return {
        'fuzzer_auc': fuzzer_auc,
        'baseline_auc': baseline_auc,
        'auc_improvement_pct': ((fuzzer_auc - baseline_auc) / baseline_auc * 100) if baseline_auc > 0 else 0,
        'time_to_reach_baseline_final': time_to_reach,
        'baseline_final_coverage': baseline_final,
    }


def get_corpus_size(experiment_folder: str, benchmark: str, fuzzer: str, cycle: int = None) -> dict:
    """Get corpus size from archives.

    Returns dict: {instance_num: corpus_size}
    """
    folder = Path(experiment_folder) / f"{benchmark}-{fuzzer}"
    if not folder.exists():
        return {}

    corpus_sizes = {}
    for instance_dir in folder.iterdir():
        if not instance_dir.name.startswith("instance-"):
            continue
        instance_num = int(instance_dir.name.split("-")[1])

        corpus_dir = instance_dir / "corpus"
        if not corpus_dir.exists():
            continue

        # Find the archive for the specified cycle (or latest)
        archives = sorted(corpus_dir.glob("corpus-archive-*.tar.gz"))
        if not archives:
            continue

        if cycle is not None:
            # Find specific cycle
            target = corpus_dir / f"corpus-archive-{cycle:04d}.tar.gz"
            if target.exists():
                archive = target
            else:
                continue
        else:
            # Use latest
            archive = archives[-1]

        try:
            with tarfile.open(archive, 'r:gz') as tar:
                # Try to read corpus_file_list.txt
                try:
                    member = tar.getmember('corpus_file_list.txt')
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode('utf-8')
                        corpus_sizes[instance_num] = len(content.strip().split('\n'))
                except KeyError:
                    # Count files in archive
                    corpus_sizes[instance_num] = len([m for m in tar.getmembers() if m.isfile()])
        except (tarfile.TarError, OSError):
            continue

    return corpus_sizes


def compare_corpus_size(experiment_folder: str, fuzzer: str, baseline: str, benchmark: str, cycle: int = None) -> dict:
    """Compare corpus sizes between fuzzer and baseline."""
    fuzzer_sizes = get_corpus_size(experiment_folder, benchmark, fuzzer, cycle)
    baseline_sizes = get_corpus_size(experiment_folder, benchmark, baseline, cycle)

    if not fuzzer_sizes or not baseline_sizes:
        return None

    fuzzer_values = list(fuzzer_sizes.values())
    baseline_values = list(baseline_sizes.values())

    fuzzer_mean = np.mean(fuzzer_values)
    baseline_mean = np.mean(baseline_values)

    a12 = vargha_delaney_a12(fuzzer_values, baseline_values)
    stat, p_value, significant = mann_whitney_test(fuzzer_values, baseline_values)

    return {
        'fuzzer_mean': fuzzer_mean,
        'baseline_mean': baseline_mean,
        'fuzzer_median': np.median(fuzzer_values),
        'baseline_median': np.median(baseline_values),
        'absolute_diff': fuzzer_mean - baseline_mean,
        'relative_diff_pct': ((fuzzer_mean - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0,
        'a12': a12,
        'effect_size': effect_size_interpretation(a12),
        'p_value': p_value,
        'significant': significant,
        'fuzzer_values': fuzzer_values,
        'baseline_values': baseline_values,
    }


def compare_magma_bugs(experiment_folder: str, fuzzer: str, baseline: str, benchmark: str) -> dict:
    """Compare Magma bug triggering between fuzzer and baseline."""
    if not benchmark.endswith('_magma'):
        return None

    fuzzer_bugs = get_magma_bugs_by_instance(experiment_folder, benchmark, fuzzer)
    baseline_bugs = get_magma_bugs_by_instance(experiment_folder, benchmark, baseline)

    if not fuzzer_bugs or not baseline_bugs:
        return None

    # Aggregate bugs across instances
    all_bugs = set()
    for bugs in list(fuzzer_bugs.values()) + list(baseline_bugs.values()):
        all_bugs.update(bugs.keys())

    results = {}
    for bug_id in sorted(all_bugs):
        # Get triggered counts per instance
        fuzzer_triggered = [
            bugs.get(bug_id, (0, 0))[1]
            for bugs in fuzzer_bugs.values()
        ]
        baseline_triggered = [
            bugs.get(bug_id, (0, 0))[1]
            for bugs in baseline_bugs.values()
        ]

        # Get reached counts per instance
        fuzzer_reached = [
            bugs.get(bug_id, (0, 0))[0]
            for bugs in fuzzer_bugs.values()
        ]
        baseline_reached = [
            bugs.get(bug_id, (0, 0))[0]
            for bugs in baseline_bugs.values()
        ]

        # Count instances that triggered the bug
        fuzzer_instances_triggered = sum(1 for t in fuzzer_triggered if t > 0)
        baseline_instances_triggered = sum(1 for t in baseline_triggered if t > 0)

        results[bug_id] = {
            'fuzzer_triggered_total': sum(fuzzer_triggered),
            'baseline_triggered_total': sum(baseline_triggered),
            'fuzzer_reached_total': sum(fuzzer_reached),
            'baseline_reached_total': sum(baseline_reached),
            'fuzzer_instances_triggered': fuzzer_instances_triggered,
            'baseline_instances_triggered': baseline_instances_triggered,
            'fuzzer_instances': len(fuzzer_bugs),
            'baseline_instances': len(baseline_bugs),
        }

    return results


def export_data_package(experiment: str, coverage_df: pd.DataFrame, experiment_folder: str,
                        fuzzers: list, benchmarks: list, output_path: str,
                        include_plot_data: bool = False):
    """Export all data needed for offline analysis into a zip file.

    Creates a portable package containing:
    - coverage_data.parquet: Coverage over time from database
    - fuzzer_stats.parquet: AFL++ stats for all instances and cycles over time
    - magma_bugs.parquet: Magma bug reach/trigger counts for all instances and cycles
    - plot_data.parquet: Fine-grained (~5s) time series from corpus archives (optional)
    - metadata.json: Experiment info and fuzzer/benchmark lists

    Args:
        include_plot_data: If True, extract fine-grained plot_data from corpus archives.
                          This can significantly increase package size.
    """
    import zipfile
    import tempfile

    print(f"\nCreating data package: {output_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Save coverage data
        coverage_path = os.path.join(tmpdir, "coverage_data.parquet")
        coverage_df.to_parquet(coverage_path, index=False, compression='zstd')
        print(f"  - Coverage data: {len(coverage_df)} records")

        # 2. Collect all fuzzer stats (all cycles) - parallelized across cores
        all_stats = get_fuzzer_stats_parallel(experiment_folder, benchmarks, fuzzers)

        if all_stats:
            stats_df = pd.DataFrame(all_stats)
            stats_path = os.path.join(tmpdir, "fuzzer_stats.parquet")
            stats_df.to_parquet(stats_path, index=False, compression='zstd')
            print(f"  - Fuzzer stats: {len(stats_df)} records")
        else:
            stats_path = None

        # 3. Collect Magma bug data (all cycles) - parallelized across cores
        all_bugs = get_magma_bugs_parallel(experiment_folder, benchmarks, fuzzers)

        if all_bugs:
            bugs_df = pd.DataFrame(all_bugs)
            bugs_path = os.path.join(tmpdir, "magma_bugs.parquet")
            bugs_df.to_parquet(bugs_path, index=False, compression='zstd')
            print(f"  - Magma bugs: {len(bugs_df)} records")
        else:
            bugs_path = None

        # 4. Collect fine-grained plot_data from corpus archives (optional, parallelized)
        plot_data_path = None
        all_plot_data = []
        if include_plot_data:
            print("  - Extracting plot_data from corpus archives...")
            all_plot_data = get_plot_data_parallel(experiment_folder, benchmarks, fuzzers)

            if all_plot_data:
                plot_df = pd.DataFrame(all_plot_data)
                plot_data_path = os.path.join(tmpdir, "plot_data.parquet")
                plot_df.to_parquet(plot_data_path, index=False, compression='zstd')
                print(f"  - Plot data: {len(plot_df)} records")

        # 5. Save metadata
        metadata = {
            'experiment': experiment,
            'fuzzers': fuzzers,
            'benchmarks': benchmarks,
            'coverage_records': len(coverage_df),
            'fuzzer_stats_records': len(all_stats) if all_stats else 0,
            'magma_bug_records': len(all_bugs) if all_bugs else 0,
            'plot_data_records': len(all_plot_data) if all_plot_data else 0,
            'max_cycle': int(coverage_df['cycle'].max()) if len(coverage_df) > 0 else 0,
        }
        metadata_path = os.path.join(tmpdir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # 6. Create zip file (use ZIP_STORED for parquet since they're already zstd compressed)
        with zipfile.ZipFile(output_path, 'w') as zf:
            zf.write(coverage_path, "coverage_data.parquet", compress_type=zipfile.ZIP_STORED)
            zf.write(metadata_path, "metadata.json", compress_type=zipfile.ZIP_DEFLATED)
            if stats_path:
                zf.write(stats_path, "fuzzer_stats.parquet", compress_type=zipfile.ZIP_STORED)
            if bugs_path:
                zf.write(bugs_path, "magma_bugs.parquet", compress_type=zipfile.ZIP_STORED)
            if plot_data_path:
                zf.write(plot_data_path, "plot_data.parquet", compress_type=zipfile.ZIP_STORED)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Package created: {output_path} ({file_size:.1f} MB)")


def load_data_package(package_path: str) -> tuple:
    """Load data from a previously exported package.

    Returns (coverage_df, stats_df, bugs_df, plot_df, metadata)
    """
    import zipfile

    with zipfile.ZipFile(package_path, 'r') as zf:
        with zf.open("metadata.json") as f:
            metadata = json.load(f)

        with zf.open("coverage_data.parquet") as f:
            coverage_df = pd.read_parquet(f)

        stats_df = None
        if "fuzzer_stats.parquet" in zf.namelist():
            with zf.open("fuzzer_stats.parquet") as f:
                stats_df = pd.read_parquet(f)

        bugs_df = None
        if "magma_bugs.parquet" in zf.namelist():
            with zf.open("magma_bugs.parquet") as f:
                bugs_df = pd.read_parquet(f)

        plot_df = None
        if "plot_data.parquet" in zf.namelist():
            with zf.open("plot_data.parquet") as f:
                plot_df = pd.read_parquet(f)

    return coverage_df, stats_df, bugs_df, plot_df, metadata


def parse_comparisons(comparison_args: list) -> dict:
    """Parse comparison arguments into a dict.

    Args:
        comparison_args: List of "fuzzer:baseline" strings

    Returns:
        Dict mapping fuzzer name to baseline name
    """
    comparisons = {}
    for arg in comparison_args:
        if ':' in arg:
            fuzzer, baseline = arg.split(':', 1)
            comparisons[fuzzer.strip()] = baseline.strip()
        else:
            print(f"Warning: Invalid comparison format '{arg}', expected 'fuzzer:baseline'")
    return comparisons


def main():
    import tempfile

    parser = argparse.ArgumentParser(
        description="Compare fuzzer configurations against baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file (recommended - reads experiment name, DB URL, filestore from config):
  %(prog)s -c config-vrp-unfold-eval02/experiment.yaml aflplusplus_vrp:aflplusplus

  # Or specify experiment name explicitly:
  %(prog)s -e vrp-unfold-eval02 aflplusplus_vrp_lto:aflplusplus_lto

  # Export data for offline analysis:
  %(prog)s -c config.yaml --export-package data.zip aflplusplus_vrp:aflplusplus

  # Load from exported package (no database needed):
  %(prog)s --from-package data.zip aflplusplus_vrp:aflplusplus
        """
    )
    parser.add_argument("-e", "--experiment", help="Experiment name (can be read from config)")
    parser.add_argument("-c", "--config", help="Experiment config YAML file (provides experiment, db_url, filestore)")
    parser.add_argument("comparisons_args", nargs="+", metavar="FUZZER:BASELINE",
                        help="Comparisons in 'fuzzer:baseline' format")
    parser.add_argument("--db-url", help="Database URL (overrides config)")
    parser.add_argument("--filestore", help="Experiment filestore path (overrides config)")
    parser.add_argument("--output-dir", help="Output directory for CSV exports")
    parser.add_argument("--csv", action="store_true", help="Export results to CSV")
    parser.add_argument("--export-package", metavar="PATH",
                        help="Export all data to a portable zip file for offline analysis")
    parser.add_argument("--from-package", metavar="PATH",
                        help="Load data from a previously exported package (no database needed)")
    parser.add_argument("--include-plot-data", action="store_true",
                        help="Include fine-grained plot_data (~5s intervals) from corpus archives")

    args = parser.parse_args()

    # Parse comparisons
    comparisons = parse_comparisons(args.comparisons_args)

    if not comparisons:
        parser.error("No valid comparisons. Use 'fuzzer:baseline' format")

    # Determine package path (either provided or create temp)
    if args.from_package:
        package_path = args.from_package
        print(f"Loading data from package: {package_path}")
    else:
        # Online mode: read config first, then export to temp package and analyze

        # Get config values (read config early to get experiment name if not provided)
        db_url = args.db_url
        filestore = args.filestore
        experiment = args.experiment

        if args.config:
            with open(args.config) as f:
                config = yaml.safe_load(f)
            if not experiment:
                experiment = config.get("experiment")
            if not db_url:
                db_url = config.get("database_url")
            if not filestore:
                filestore = config.get("experiment_filestore")

        if not experiment:
            parser.error("--experiment or --config (with 'experiment' field) is required when not using --from-package")

        if not filestore:
            filestore = "/opt/fuzzbench-storage/fb-data/experiments"

        experiment_folder = os.path.join(filestore, experiment, "experiment-folders")

        print(f"Experiment: {experiment}")
        print(f"Comparisons: {comparisons}")
        print(f"Filestore: {filestore}")

        # Get coverage data from database
        print("\nLoading coverage data from database...")
        coverage_df = get_coverage_data(experiment, db_url)
        print(f"Loaded {len(coverage_df)} coverage records")
        print(f"Fuzzers: {sorted(coverage_df['fuzzer'].unique())}")
        print(f"Benchmarks: {sorted(coverage_df['benchmark'].unique())}")

        fuzzers = list(coverage_df['fuzzer'].unique())
        benchmarks = list(coverage_df['benchmark'].unique())

        # Export to temp package or user-specified path
        if args.export_package:
            package_path = args.export_package
        else:
            # Create temp file that persists for the duration
            tmp_file = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
            package_path = tmp_file.name
            tmp_file.close()

        export_data_package(experiment, coverage_df, experiment_folder,
                           fuzzers, benchmarks, package_path,
                           include_plot_data=args.include_plot_data)

    # Load from package and run analysis
    coverage_df, stats_df, bugs_df, plot_df, metadata = load_data_package(package_path)

    print(f"\nExperiment: {metadata['experiment']}")
    print(f"Coverage records: {len(coverage_df)}")
    print(f"Fuzzers: {metadata['fuzzers']}")
    print(f"Benchmarks: {metadata['benchmarks']}")

    # Validate comparisons
    available_fuzzers = set(metadata['fuzzers'])
    for fuzzer, baseline in comparisons.items():
        if fuzzer not in available_fuzzers:
            print(f"Warning: fuzzer '{fuzzer}' not found in data")
        if baseline not in available_fuzzers:
            print(f"Warning: baseline '{baseline}' not found in data")

    # Print report from package
    print_comparison_report_from_package(comparisons, coverage_df, stats_df, bugs_df, metadata)

    # Clean up temp file if we created one
    if not args.from_package and not args.export_package:
        os.unlink(package_path)

    # Export to CSV if requested
    if args.csv:
        output_dir = args.output_dir or "."
        # Re-export CSVs from the dataframes we have
        if len(coverage_df) > 0:
            coverage_df.to_csv(f"{output_dir}/coverage_data.csv", index=False)
            print(f"Saved: {output_dir}/coverage_data.csv")
        if stats_df is not None and len(stats_df) > 0:
            stats_df.to_csv(f"{output_dir}/fuzzer_stats.csv", index=False)
            print(f"Saved: {output_dir}/fuzzer_stats.csv")
        if bugs_df is not None and len(bugs_df) > 0:
            bugs_df.to_csv(f"{output_dir}/magma_bugs.csv", index=False)
            print(f"Saved: {output_dir}/magma_bugs.csv")
        if plot_df is not None and len(plot_df) > 0:
            plot_df.to_csv(f"{output_dir}/plot_data.csv", index=False)
            print(f"Saved: {output_dir}/plot_data.csv")


def print_comparison_report_from_package(comparisons: dict, coverage_df: pd.DataFrame,
                                         stats_df: pd.DataFrame, bugs_df: pd.DataFrame,
                                         metadata: dict):
    """Print comparison report using data from an exported package."""
    benchmarks = sorted(coverage_df['benchmark'].unique())

    for fuzzer, baseline in comparisons.items():
        print("\n" + "=" * 80)
        print(f"COMPARISON: {fuzzer} vs {baseline} (baseline)")
        print("=" * 80)

        # Coverage comparison (same as before)
        coverage_results = []
        for benchmark in benchmarks:
            cov = compare_coverage(coverage_df, fuzzer, baseline, benchmark)
            if cov:
                coverage_results.append({
                    'benchmark': benchmark,
                    'fuzzer_mean': cov['fuzzer_mean'],
                    'baseline_mean': cov['baseline_mean'],
                    'diff': cov['absolute_diff'],
                    'rel_pct': cov['relative_improvement_pct'],
                    'a12': cov['a12'],
                    'effect': cov['effect_size'],
                    'p_value': cov['p_value'],
                    'sig': '***' if cov['significant'] and cov['p_value'] < 0.001 else
                           '**' if cov['significant'] and cov['p_value'] < 0.01 else
                           '*' if cov['significant'] else '',
                })

        # Print coverage table
        print("\n### Final Coverage Comparison (edges_covered)")
        print("-" * 80)
        print(f"{'Benchmark':<35} {'Fuzzer':>8} {'Base':>8} {'Diff':>7} {'Rel%':>7} {'A12':>5} {'Effect':>10} {'Sig':>4}")
        print("-" * 80)

        wins, losses, ties = 0, 0, 0
        for r in coverage_results:
            print(f"{r['benchmark']:<35} {r['fuzzer_mean']:>8.0f} {r['baseline_mean']:>8.0f} "
                  f"{r['diff']:>+7.0f} {r['rel_pct']:>+6.1f}% {r['a12']:>5.2f} {r['effect']:>10} {r['sig']:>4}")
            if r['sig'] and r['a12'] > 0.5:
                wins += 1
            elif r['sig'] and r['a12'] < 0.5:
                losses += 1
            else:
                ties += 1

        print("-" * 80)
        print(f"Summary: {wins} wins, {losses} losses, {ties} ties (at p<0.05)")

        if coverage_results:
            avg_rel = np.mean([r['rel_pct'] for r in coverage_results])
            avg_a12 = np.mean([r['a12'] for r in coverage_results])
            print(f"Average relative improvement: {avg_rel:+.2f}%")
            print(f"Average A12: {avg_a12:.3f} ({effect_size_interpretation(avg_a12)})")

        # Fuzzer stats from package
        if stats_df is not None and len(stats_df) > 0:
            print("\n### Fuzzer Stats Comparison (from package)")
            print("-" * 100)

            fuzzer_stats_df = stats_df[stats_df['fuzzer'] == fuzzer]
            baseline_stats_df = stats_df[stats_df['fuzzer'] == baseline]

            if len(fuzzer_stats_df) > 0 and len(baseline_stats_df) > 0:
                stat_cols = ['execs_done', 'execs_per_sec', 'corpus_count', 'corpus_favored',
                            'edges_found', 'cycles_done', 'stability', 'bitmap_cvg']

                print(f"{'Metric':<18} {'Fuzzer Mean':>14} {'Base Mean':>14} {'Diff':>12} {'Rel%':>8} {'A12':>6} {'Effect':>10}")
                print("-" * 100)

                for col in stat_cols:
                    if col not in fuzzer_stats_df.columns:
                        continue
                    fuzzer_vals = fuzzer_stats_df[col].dropna().tolist()
                    baseline_vals = baseline_stats_df[col].dropna().tolist()

                    if not fuzzer_vals or not baseline_vals:
                        continue

                    fuzzer_mean = np.mean(fuzzer_vals)
                    baseline_mean = np.mean(baseline_vals)
                    a12 = vargha_delaney_a12(fuzzer_vals, baseline_vals)
                    diff = fuzzer_mean - baseline_mean
                    rel_pct = ((fuzzer_mean - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0

                    if col in ['execs_done']:
                        print(f"{col:<18} {fuzzer_mean:>14,.0f} {baseline_mean:>14,.0f} {diff:>+12,.0f} {rel_pct:>+7.1f}% {a12:>6.2f} {effect_size_interpretation(a12):>10}")
                    elif col in ['stability', 'bitmap_cvg']:
                        print(f"{col:<18} {fuzzer_mean:>13.1f}% {baseline_mean:>13.1f}% {diff:>+11.1f}% {rel_pct:>+7.1f}% {a12:>6.2f} {effect_size_interpretation(a12):>10}")
                    else:
                        print(f"{col:<18} {fuzzer_mean:>14,.1f} {baseline_mean:>14,.1f} {diff:>+12,.1f} {rel_pct:>+7.1f}% {a12:>6.2f} {effect_size_interpretation(a12):>10}")

                print("-" * 100)

        # Magma bugs from package
        if bugs_df is not None and len(bugs_df) > 0:
            print("\n### Magma Bug Triggering (from package)")
            print("-" * 80)

            fuzzer_bugs = bugs_df[bugs_df['fuzzer'] == fuzzer]
            baseline_bugs = bugs_df[bugs_df['fuzzer'] == baseline]

            if len(fuzzer_bugs) > 0 and len(baseline_bugs) > 0:
                print(f"{'Benchmark':<30} {'Bug':<8} {'Fuz Trig':>10} {'Base Trig':>10}")
                print("-" * 80)

                for benchmark in bugs_df['benchmark'].unique():
                    fb = fuzzer_bugs[fuzzer_bugs['benchmark'] == benchmark]
                    bb = baseline_bugs[baseline_bugs['benchmark'] == benchmark]

                    for bug_id in fb['bug_id'].unique():
                        fuz_trig = fb[fb['bug_id'] == bug_id]['triggered'].sum()
                        base_trig = bb[bb['bug_id'] == bug_id]['triggered'].sum() if len(bb[bb['bug_id'] == bug_id]) > 0 else 0
                        if fuz_trig > 0 or base_trig > 0:
                            print(f"{benchmark:<30} {bug_id:<8} {fuz_trig:>10} {base_trig:>10}")

                print("-" * 80)


if __name__ == "__main__":
    main()
