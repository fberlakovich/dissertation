import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np


def bench_from_name(name: str) -> str:
    """
    Extract benchmark key from a lit name like
    'test-suite :: External/SPEC/CINT2017speed/600.perlbench_s/600.perlbench_s.test'
    -> '600.perlbench_s'
    """
    parts = name.split("/")
    for part in reversed(parts):
        if part.endswith(".test"):
            return part.replace(".test", "")
        if part and part[0].isdigit() and "." in part:
            return part
    return name


def collect_baseline(input_dir: Path) -> dict:
    """
    Read all output*.json files in input_dir and aggregate exec_time per benchmark.
    Returns: {bench_key: {"runtime:all": [...], "runtime:median": float}}
    """
    result = {}
    files = sorted(input_dir.glob("output*.json"))
    if not files:
        raise FileNotFoundError(f"No output*.json files found in {input_dir}")

    for path in files:
        with path.open("r") as f:
            data = json.load(f)
        tests = data.get("tests", [])
        for t in tests:
            metrics = t.get("metrics", {})
            exec_time = metrics.get("exec_time")
            if exec_time is None:
                continue
            bench = bench_from_name(t.get("name", ""))
            entry = result.setdefault(bench, {"runtime:all": []})
            entry["runtime:all"].append(exec_time)

    for bench, entry in result.items():
        arr = np.asarray(entry["runtime:all"], dtype=float)
        entry["runtime:median"] = float(np.median(arr)) if arr.size else None

    return result


def main():
    parser = argparse.ArgumentParser(description="Convert baseline output*.json runs to full-r2c style JSON.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing baseline output*.json files (e.g., data/r2c/baseline/epyc)",
    )
    parser.add_argument(
        "--machine",
        type=str,
        required=True,
        help="Machine key to store in the output JSON (e.g., epyc)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the aggregated JSON (e.g., data/r2c/baseline-epyc.json)",
    )
    args = parser.parse_args()

    baseline = collect_baseline(args.input_dir)
    out_data = {args.machine: baseline}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(out_data, f, indent=2, sort_keys=True)
    print(f"Wrote {len(baseline)} benchmarks to {args.output}")


if __name__ == "__main__":
    main()
