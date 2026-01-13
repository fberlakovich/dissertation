"""
Common constants and utilities for R2C benchmark analysis scripts.
"""

import json
import re
from pathlib import Path
from typing import Optional

# Full SPEC CPU 2017 benchmark names used in R2C evaluation
BENCHMARK_NAMES = [
    "perlbench",
    "gcc",
    "mcf",
    "lbm",
    "omnetpp",
    "xalancbmk",
    "x264",
    "deepsjeng",
    "imagick",
    "leela",
    "nab",
    "xz",
]

# Integer-only benchmarks (excluding floating point: lbm, imagick, nab)
BENCHMARK_NAMES_INT = [name for name in BENCHMARK_NAMES if name not in ["lbm", "imagick", "nab"]]

# SPEC benchmark full names (with number prefix)
SPEC_BENCHMARK_FULL_NAMES = [
    "600.perlbench_s",
    "602.gcc_s",
    "605.mcf_s",
    "619.lbm_s",
    "620.omnetpp_s",
    "623.xalancbmk_s",
    "625.x264_s",
    "631.deepsjeng_s",
    "638.imagick_s",
    "641.leela_s",
    "644.nab_s",
    "657.xz_s",
]

# Mapping from full SPEC names to short names
BENCH_FULL_TO_SHORT = {
    "600.perlbench_s": "perlbench",
    "602.gcc_s": "gcc",
    "605.mcf_s": "mcf",
    "619.lbm_s": "lbm",
    "620.omnetpp_s": "omnetpp",
    "623.xalancbmk_s": "xalancbmk",
    "625.x264_s": "x264",
    "631.deepsjeng_s": "deepsjeng",
    "638.imagick_s": "imagick",
    "641.leela_s": "leela",
    "644.nab_s": "nab",
    "657.xz_s": "xz",
}

# Reverse mapping
BENCH_SHORT_TO_FULL = {v: k for k, v in BENCH_FULL_TO_SHORT.items()}

# Standard geomean groupings
GEOMEAN_SETS = {
    "Geomean int": BENCHMARK_NAMES_INT,
    "Geomean all": BENCHMARK_NAMES,
}


def load_json_data(path: str | Path) -> dict:
    """Load JSON data from a file path."""
    with open(path) as f:
        return json.load(f)


def pct_change(value: float, baseline: float) -> Optional[float]:
    """Compute percentage change from baseline."""
    if baseline is None or baseline == 0 or value is None:
        return None
    return (value / baseline - 1) * 100


def ratio_to_pct(ratio: float) -> float:
    """Convert ratio (e.g., 1.15) to percentage overhead (e.g., 15.0)."""
    return (ratio - 1.0) * 100


def pct_to_ratio(pct: float) -> float:
    """Convert percentage overhead (e.g., 15.0) to ratio (e.g., 1.15)."""
    return 1.0 + pct / 100.0


def compute_geomean(values: list[float]) -> float:
    """Compute geometric mean of values."""
    from scipy.stats import gmean
    return gmean(values)


def compute_geomeans_for_groups(
    values_by_benchmark: dict[str, float],
    groups: dict[str, list[str]] = None,
) -> dict[str, float]:
    """
    Compute geometric means for benchmark groups.

    Args:
        values_by_benchmark: dict mapping benchmark short name to value
        groups: dict mapping group name to list of benchmark names (default: GEOMEAN_SETS)

    Returns:
        dict mapping group name to geomean value
    """
    from scipy.stats import gmean

    if groups is None:
        groups = GEOMEAN_SETS

    result = {}
    for group_name, benchmarks in groups.items():
        vals = [values_by_benchmark[b] for b in benchmarks if b in values_by_benchmark]
        if vals:
            result[group_name] = gmean(vals)
    return result


def natural_sort_key(s: str) -> list:
    """Natural sort key function for strings with embedded numbers."""
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', s)]


def extract_benchmarks_from_json(data: dict, pass_key: str = "1") -> list[str]:
    """Extract benchmark names from JSON data structure."""
    pass_data = data.get(pass_key, data)
    return list(pass_data.keys())


def extract_configs_from_json(data: dict, pass_key: str = "1") -> list[str]:
    """Extract config names from JSON data structure."""
    pass_data = data.get(pass_key, data)
    if not pass_data:
        return []
    first_bench = next(iter(pass_data.values()), {})
    return list(first_bench.keys())


def build_row_to_key_mapping(
    df_index: list[str],
    benchmark_names: list[str] = None,
) -> dict[str, str]:
    """
    Build mapping from display labels to DataFrame index keys.

    Args:
        df_index: list of index values from DataFrame
        benchmark_names: list of display names (default: BENCHMARK_NAMES)

    Returns:
        dict mapping display name to index key
    """
    if benchmark_names is None:
        benchmark_names = BENCHMARK_NAMES
    return dict(zip(benchmark_names, df_index))
