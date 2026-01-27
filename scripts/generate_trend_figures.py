#!/usr/bin/env python3
"""
Generate publication trend figures and tables from DBLP cache or cluster JSON.

Supports two modes:
1. Keyword filtering: Filter papers by regex patterns (include/exclude)
2. Cluster extraction: Use pre-computed semantic clusters from dblp_stats.py

Generates:
1. A pgfplots bar chart showing publications per year
2. A longtable listing all matching papers by year

Usage:
    # Use a predefined category (keyword-based)
    python scripts/generate_trend_figures.py --category code-reuse

    # Extract from cluster JSON (semantic embedding-based)
    python scripts/generate_trend_figures.py --cluster-json data/dblp_trends.json --cluster-id 145

    # List clusters in a JSON file
    python scripts/generate_trend_figures.py --cluster-json data/dblp_trends.json --list-clusters

    # Custom keywords
    python scripts/generate_trend_figures.py --include "fuzzing,fuzz,fuzzer" --name fuzzing

    # With exclusions
    python scripts/generate_trend_figures.py --include "sgx,enclave" --exclude "cache" --name sgx

Available predefined categories:
    code-reuse   - ROP, CFI, ASLR, code reuse attacks/defenses
    fuzzing      - Fuzzing and fuzz testing
    sgx          - Intel SGX and enclaves
    spectre      - Speculative execution attacks
    blockchain   - Blockchain and smart contracts
"""

import argparse
import html
import json
import re
from collections import defaultdict
from pathlib import Path

# Predefined categories with include/exclude patterns
CATEGORIES = {
    "code-reuse": {
        "description": "Code reuse attacks and defenses (ROP, CFI, ASLR)",
        "include": [
            r"\brop\b",
            r"\bcfi\b",
            r"\baslr\b",
            r"code.?reuse",
            r"control.?flow.?(?:integrity|hijack|attack|protect)",
            r"gadget(?!.*(usb|driver|stack))",  # gadget but not USB gadget
            r"return.?oriented",
            r"shadow.?stack",
            r"code.?pointer.?integrity",
            r"return.?address.?(?:integrity|protect)",
            r"jump.?oriented",
            r"call.?oriented",
        ],
        "exclude": [
            r"cache.?randomization",
            r"differential.?privacy",
            r"multi.?user.*randomization",
            r"mac.?address.?random",
            r"ballot.?random",
            r"usb.?gadget",
            r"gadget.?stack",  # USB gadget stacks
            r"finality.?gadget",
            r"garbling.?gadget",
            r"deserialization.?gadget",
            r"prototype.?pollution.?gadget",
            r"script.?gadget",
            r"spectre.?gadget",
            r"transient.*gadget",
            r"machine.?learning.*control.?flow",
            r"neural.*control",
            r"dynamic.?control.?flow(?!.*(integrity|attack))",  # ML control flow
            r"control.?flow.?graph(?!.*(integrity|attack|protect))",  # Just CFG analysis
        ],
    },
    "fuzzing": {
        "description": "Fuzzing and fuzz testing",
        "include": [
            r"\bfuzz",
            r"fuzzer",
            r"fuzzing",
        ],
        "exclude": [
            r"differential.?fuzz",  # Often about ML
        ],
    },
    "sgx": {
        "description": "Intel SGX and trusted execution",
        "include": [
            r"\bsgx\b",
            r"\benclave",
            r"trusted.?execution",
        ],
        "exclude": [],
    },
    "spectre": {
        "description": "Speculative execution attacks",
        "include": [
            r"spectre",
            r"speculative.?execution",
            r"transient.?execution",
            r"meltdown",
        ],
        "exclude": [],
    },
    "blockchain": {
        "description": "Blockchain and smart contracts",
        "include": [
            r"blockchain",
            r"smart.?contract",
            r"\bethereum\b",
            r"\bbitcoin\b",
            r"\bdefi\b",
        ],
        "exclude": [],
    },
}


def load_dblp_cache(cache_dir: Path) -> list[dict]:
    """Load all papers from DBLP cache."""
    papers = []
    for cache_file in sorted(cache_dir.glob("conf_*.json")):
        # Extract venue from filename (conf_sp_2020.json -> SP)
        parts = cache_file.stem.split("_")
        venue = parts[1].upper() if len(parts) > 1 else "UNKNOWN"

        with open(cache_file) as f:
            data = json.load(f)

        for paper in data:
            paper["_venue"] = venue
            paper["_source_file"] = cache_file.name
            papers.append(paper)

    return papers


def load_cluster_json(json_path: Path) -> dict:
    """Load cluster data from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def list_clusters(cluster_data: dict):
    """Print summary of all clusters in JSON file."""
    print(f"\nClusters in file ({len(cluster_data['clusters'])} total):\n")
    print(f"{'ID':>4}  {'Size':>5}  {'Score':>7}  {'Label':<40}  Keywords")
    print("-" * 100)

    for c in sorted(cluster_data["clusters"], key=lambda x: -x["score"]):
        keywords = ", ".join(c["keywords"][:5])
        print(f"{c['id']:>4}  {c['size']:>5}  {c['score']:>7.4f}  {c['label']:<40}  {keywords}")


def extract_cluster_papers(cluster_data: dict, cluster_id: int) -> tuple[list[dict], str]:
    """Extract papers from a specific cluster."""
    for cluster in cluster_data["clusters"]:
        if cluster["id"] == cluster_id:
            papers = []
            for paper in cluster.get("papers", []):
                # Normalize venue names
                venue = paper.get("venue", "UNKNOWN").upper()
                papers.append({
                    "title": paper.get("title", ""),
                    "year": paper.get("year", 0),
                    "_venue": venue,
                })
            return papers, cluster["label"]

    raise ValueError(f"Cluster {cluster_id} not found")


def filter_papers(
    papers: list[dict],
    include_patterns: list[str],
    exclude_patterns: list[str] = None,
) -> list[dict]:
    """Filter papers by include/exclude regex patterns on title."""
    include_re = re.compile("|".join(include_patterns), re.IGNORECASE)
    exclude_re = re.compile("|".join(exclude_patterns), re.IGNORECASE) if exclude_patterns else None

    matched = []
    for paper in papers:
        title = paper.get("title", "")
        if include_re.search(title):
            if exclude_re and exclude_re.search(title):
                continue
            matched.append(paper)

    return matched


def group_by_year(papers: list[dict]) -> dict[int, list[dict]]:
    """Group papers by year."""
    by_year = defaultdict(list)
    for paper in papers:
        year = paper.get("year", 0)
        if year:
            by_year[year].append(paper)
    return dict(sorted(by_year.items()))


def generate_bar_chart(
    papers_by_year: dict[int, list[dict]],
    output_path: Path,
    description: str = "",
):
    """Generate pgfplots bar chart."""
    years = sorted(papers_by_year.keys())
    if not years:
        print("No papers found, skipping bar chart generation")
        return

    counts = [len(papers_by_year.get(y, [])) for y in years]
    max_count = max(counts) if counts else 10

    # Generate LaTeX
    lines = [
        f"% {description}",
        "% Data from DBLP cache",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    ybar,",
        r"    bar width=0.45cm,",
        r"    width=\textwidth,",
        r"    height=6cm,",
        r"    xlabel={Year},",
        r"    ylabel={Publications},",
        r"    ymin=0,",
        f"    ymax={int(max_count * 1.2)},",
        f"    xtick={{{','.join(map(str, years))}}},",
        r"    xticklabel style={rotate=45, anchor=east, font=\small, /pgf/number format/1000 sep={}},",
        r"    nodes near coords,",
        r"    nodes near coords style={font=\tiny, above},",
        r"    every node near coord/.append style={yshift=1pt},",
        r"    axis lines*=left,",
        r"    enlarge x limits=0.05,",
        r"]",
        r"\addplot[fill=gray!20, draw=black, line width=0.4pt] coordinates {",
    ]

    for y, c in zip(years, counts):
        lines.append(f"    ({y}, {c})")

    lines.extend([
        r"};",
        r"\end{axis}",
        r"\end{tikzpicture}",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Generated bar chart: {output_path}")


def generate_table(
    papers_by_year: dict[int, list[dict]],
    output_path: Path,
    description: str = "",
):
    """Generate longtable with papers by year."""
    # Normalize venue names - use unescaped forms, escaping happens after
    venue_map = {
        "SP": "IEEE S&P",
        "USS": "USENIX Sec.",
        "USENIX SECURITY": "USENIX Sec.",
        "EUROSP": "Euro S&P",
        "EURO S&P": "Euro S&P",
        "IEEE S&P": "IEEE S&P",
        "ASIACCS": "Asia CCS"
    }

    lines = [
        f"% {description}",
        "% Source: DBLP cache",
        "",
        r"\tiny",
        r"\begin{longtable}{@{}p{0.7cm}p{1.4cm}p{11cm}@{}}",
        r"\toprule",
        r"\textbf{Year} & \textbf{Venue} & \textbf{Title} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{3}{c}{\tablename\ \thetable{} -- continued from previous page} \\",
        r"\toprule",
        r"\textbf{Year} & \textbf{Venue} & \textbf{Title} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{3}{r}{Continued on next page} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
        "",
    ]

    for year in sorted(papers_by_year.keys()):
        papers = sorted(papers_by_year[year], key=lambda p: p["_venue"])
        lines.append(f"% === {year} ({len(papers)} papers) ===")

        first_in_year = True
        last_venue = None
        for paper in papers:
            venue = paper["_venue"]
            venue = venue_map.get(venue, venue)
            # Escape & in venue names (e.g., "IEEE S&P" -> "IEEE S\&P")
            venue = venue.replace("&", r"\&")

            # Clean and escape title
            title = html.unescape(paper.get("title", "")).rstrip(".")
            title = title.replace("&", r"\&").replace("_", r"\_")
            title = title.replace("#", r"\#").replace("%", r"\%")
            title = title.replace("$", r"\$")

            # No truncation - let LaTeX wrap naturally

            # Only show year on first row, only show venue on first row of each venue group
            year_col = str(year) if first_in_year else ""
            venue_col = venue if venue != last_venue else ""

            lines.append(f"{year_col} & {venue_col} & {title} \\\\")

            first_in_year = False
            last_venue = venue

        lines.append(r"\midrule")
        lines.append("")

    lines.append(r"\end{longtable}")
    lines.append(r"\normalsize")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Generated table: {output_path}")


def print_summary(papers_by_year: dict[int, list[dict]], name: str):
    """Print summary statistics."""
    years = sorted(papers_by_year.keys())
    total = sum(len(p) for p in papers_by_year.values())

    print(f"\n=== {name} ===")
    print(f"Total papers: {total}")
    print(f"Years: {min(years)} - {max(years)}")
    print("\nPer-year breakdown:")

    for year in years:
        count = len(papers_by_year[year])
        bar = "█" * count
        print(f"  {year}: {bar} ({count})")

    # Find peak
    peak_year = max(years, key=lambda y: len(papers_by_year[y]))
    peak_count = len(papers_by_year[peak_year])
    print(f"\nPeak: {peak_year} with {peak_count} papers")


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication trend figures from DBLP cache or cluster JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Cluster-based extraction
    parser.add_argument(
        "--cluster-json", "-j",
        type=Path,
        help="Path to cluster JSON file from dblp_stats.py discover",
    )
    parser.add_argument(
        "--cluster-id", "-id",
        type=int,
        help="Cluster ID to extract (use with --cluster-json)",
    )
    parser.add_argument(
        "--list-clusters",
        action="store_true",
        help="List all clusters in JSON file and exit",
    )

    # Keyword-based filtering
    parser.add_argument(
        "--category", "-c",
        choices=list(CATEGORIES.keys()),
        help="Use a predefined category",
    )
    parser.add_argument(
        "--include", "-i",
        help="Comma-separated include patterns (regex)",
    )
    parser.add_argument(
        "--exclude", "-e",
        help="Comma-separated exclude patterns (regex)",
    )

    # Output options
    parser.add_argument(
        "--name", "-n",
        default="trend",
        help="Name for output files (default: trend)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/.dblp_cache"),
        help="DBLP cache directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated/trends"),
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("figures/trends"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Skip table generation",
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Skip figure generation",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List available categories and exit",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2010,
        help="Start year for analysis (default: 2010)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="End year for analysis (default: 2024)",
    )

    args = parser.parse_args()

    # Handle --list-categories
    if args.list_categories:
        print("Available categories:")
        for name, cat in CATEGORIES.items():
            print(f"  {name:15} - {cat['description']}")
        return

    # Handle cluster-based extraction
    if args.cluster_json:
        print(f"Loading clusters from {args.cluster_json}...")
        cluster_data = load_cluster_json(args.cluster_json)

        if args.list_clusters:
            list_clusters(cluster_data)
            return

        if args.cluster_id is None:
            parser.error("--cluster-id is required when using --cluster-json")

        papers, description = extract_cluster_papers(cluster_data, args.cluster_id)
        name = args.name if args.name != "trend" else f"cluster-{args.cluster_id}"
        print(f"Extracted {len(papers)} papers from cluster {args.cluster_id}: {description}")

        # Filter by year range
        papers = [p for p in papers if args.start_year <= p.get("year", 0) <= args.end_year]

        # Group by year
        papers_by_year = group_by_year(papers)

    # Handle keyword-based filtering
    elif args.category or args.include:
        if args.category:
            cat = CATEGORIES[args.category]
            include_patterns = cat["include"]
            exclude_patterns = cat["exclude"]
            description = cat["description"]
            name = args.category
        else:
            include_patterns = [p.strip() for p in args.include.split(",")]
            exclude_patterns = [p.strip() for p in args.exclude.split(",")] if args.exclude else []
            description = f"Custom search: {args.include}"
            name = args.name

        # Load and filter papers
        print(f"Loading papers from {args.cache_dir}...")
        papers = load_dblp_cache(args.cache_dir)
        print(f"Loaded {len(papers)} papers")

        print(f"Filtering with {len(include_patterns)} include patterns...")
        matched = filter_papers(papers, include_patterns, exclude_patterns)
        print(f"Matched {len(matched)} papers")

        # Filter by year range
        matched = [p for p in matched if args.start_year <= p.get("year", 0) <= args.end_year]

        # Group by year
        papers_by_year = group_by_year(matched)

    else:
        parser.error("Either --category, --include, or --cluster-json is required")

    # Fill in missing years with empty lists
    for year in range(args.start_year, args.end_year + 1):
        if year not in papers_by_year:
            papers_by_year[year] = []

    # Print summary
    print_summary(papers_by_year, name)

    # Generate outputs
    if not args.no_figure:
        figure_path = args.figure_dir / f"{name}-trend.tex"
        generate_bar_chart(papers_by_year, figure_path, description)

    if not args.no_table:
        table_path = args.output_dir / f"{name}-papers-table.tex"
        generate_table(papers_by_year, table_path, description)


if __name__ == "__main__":
    main()
