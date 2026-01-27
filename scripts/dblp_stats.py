#!/usr/bin/env python3
"""
Query DBLP/OpenAlex for publication statistics in top security venues.

Generates year-by-year statistics showing publication trends for specified
keywords in IEEE S&P, USENIX Security, CCS, and NDSS.

Subcommands:
    discover       Automatically discover rising/declining research topics
    search         Search for specific keywords in paper titles
    report-missing Report papers that have no abstract in cache
    cache          Manage the abstract cache (clear, clear-misses, stats)

Usage Examples:
    # Discover trending topics with NLP (uses abstracts, years processed in parallel)
    python scripts/dblp_stats.py discover --use-abstracts --start-year 2010

    # Discover with LDA instead of NMF
    python scripts/dblp_stats.py discover --method lda --n-topics 30

    # Simple n-gram based discovery (faster, no abstracts needed)
    python scripts/dblp_stats.py discover --method ngram

    # Search for specific keywords
    python scripts/dblp_stats.py search --keywords "ROP,CFI,ASLR"

    # Use a preset keyword set
    python scripts/dblp_stats.py search --preset code-reuse

    # Search with a keywords file
    python scripts/dblp_stats.py search --keywords-file my_keywords.txt --start-year 2015

    # Report papers missing abstracts
    python scripts/dblp_stats.py report-missing --start-year 2010

    # Cache management
    python scripts/dblp_stats.py cache stats
    python scripts/dblp_stats.py cache clear-misses
    python scripts/dblp_stats.py cache clear

Abstract sources (tried in order):
    1. OpenAlex by DOI (best coverage)
    2. ACM Digital Library (for ACM papers like CCS)
    3. USENIX website (for USENIX Security papers)
    4. CrossRef (publisher-deposited)
    5. Semantic Scholar (rate limited, with retry on 429)
    6. Unpaywall (open access focused)
    7. OpenAlex by title (fallback for papers without DOI)

Abstracts are cached to data/.abstract_cache/ for faster subsequent runs.
Rate limiting (HTTP 429) is handled automatically with Retry-After support.
Years are processed in parallel by default for faster data collection.

Requirements for discover mode (NMF/LDA):
    pip install scikit-learn
"""

import argparse
import csv
import hashlib
import json
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ============================================================================
# Rate Limit Handling
# ============================================================================

# Global rate limit state (thread-safe)
_rate_limit_lock = threading.Lock()
_rate_limit_until: dict[str, float] = {}  # domain -> timestamp when we can retry


def get_domain(url: str) -> str:
    """Extract domain from URL for rate limit tracking."""
    from urllib.parse import urlparse
    return urlparse(url).netloc


def rate_limited_request(
    url: str,
    params: dict = None,
    headers: dict = None,
    timeout: int = 15,
    max_retries: int = 3,
    base_wait: float = 5.0,
) -> requests.Response | None:
    """
    Make an HTTP GET request with 429 rate limit handling.

    Respects Retry-After header when provided. Uses exponential backoff otherwise.
    Tracks rate limits per-domain to avoid hammering rate-limited APIs.

    Returns None if all retries exhausted or permanently rate limited.
    """
    domain = get_domain(url)

    # Check if we're still in cooldown for this domain
    with _rate_limit_lock:
        if domain in _rate_limit_until:
            wait_until = _rate_limit_until[domain]
            now = time.time()
            if now < wait_until:
                remaining = wait_until - now
                if remaining > 60:  # Don't wait more than 60s
                    return None
                time.sleep(remaining)
            del _rate_limit_until[domain]

    default_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    if headers:
        default_headers.update(headers)

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=default_headers, timeout=timeout)

            if response.status_code == 429:
                # Check Retry-After header
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait_time = int(retry_after)
                    except ValueError:
                        # Could be HTTP date format, use default
                        wait_time = base_wait * (2 ** attempt)
                else:
                    wait_time = base_wait * (2 ** attempt)

                # Cap wait time at 120 seconds
                wait_time = min(wait_time, 120)

                # Record rate limit for this domain
                with _rate_limit_lock:
                    _rate_limit_until[domain] = time.time() + wait_time

                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                else:
                    return None

            return response

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(base_wait)
                continue
            return None
        except requests.exceptions.RequestException:
            return None

    return None

# ============================================================================
# Abstract Cache (one file per abstract for incremental saving)
# ============================================================================

CACHE_DIR = Path(__file__).parent.parent / "data" / ".abstract_cache"


def get_cache_dir() -> Path:
    """Get the cache directory, creating it if needed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def get_cache_key(identifier: str) -> str:
    """Generate a safe filename from a DOI or title."""
    # Use hash for safe filenames
    key = identifier.lower().strip()
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def get_cache_file(identifier: str) -> Path:
    """Get the cache file path for an identifier."""
    return get_cache_dir() / f"{get_cache_key(identifier)}.txt"


def cache_abstract(identifier: str, abstract: str):
    """
    Cache an abstract immediately to disk.

    Saves even empty strings (to mark "no abstract available").
    """
    if not identifier:
        return
    cache_file = get_cache_file(identifier)
    # Write atomically: write to temp file, then rename
    tmp_file = cache_file.with_suffix('.tmp')
    try:
        with open(tmp_file, 'w') as f:
            f.write(abstract)
        tmp_file.rename(cache_file)
    except IOError:
        pass  # Ignore cache write errors


def get_cached_abstract(identifier: str) -> str | None:
    """
    Get a cached abstract, or None if not cached.

    Returns empty string if identifier was previously queried but had no abstract.
    Returns None if identifier was never queried.
    """
    if not identifier:
        return None
    cache_file = get_cache_file(identifier)
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                return f.read()  # May be empty string
        except IOError:
            return None
    return None


def clear_abstract_cache():
    """Clear the entire abstract cache."""
    cache_dir = get_cache_dir()
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)


def clear_cache_misses() -> int:
    """Clear only cache-miss entries (empty files). Returns count of deleted files."""
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return 0
    deleted = 0
    for f in cache_dir.glob("*.txt"):
        if f.stat().st_size == 0:
            f.unlink()
            deleted += 1
    return deleted


def report_missing_abstracts(start_year: int, end_year: int):
    """Report papers that have no abstract in cache."""
    print("Scanning for papers without abstracts...")
    print("(This requires querying DBLP - may take a while due to rate limits)")
    print()

    missing_with_doi = []
    missing_no_doi = []
    total_checked = 0

    for venue_name, venue_key in DEFAULT_VENUES.items():
        print(f"Checking {venue_name}...", end=" ", flush=True)
        venue_missing = 0

        for year in range(start_year, end_year + 1):
            papers = query_dblp_with_dois(venue_key, year)
            time.sleep(0.5)  # Rate limit

            for p in papers:
                total_checked += 1
                doi = p.get("doi")
                title = p.get("title", "")
                url = p.get("ee", "")

                # Skip proceedings front matter (titles starting with year or "Proceedings")
                if title and (title[0].isdigit() or title.lower().startswith("proceedings")):
                    continue

                cache_key = doi if doi else (title[:100] if title else None)
                if not cache_key:
                    continue

                cached = get_cached_abstract(cache_key)
                if cached == "":  # Empty = no abstract found
                    entry = {
                        "title": title,
                        "doi": doi,
                        "url": url,
                        "year": year,
                        "venue": venue_name,
                    }
                    if doi:
                        missing_with_doi.append(entry)
                    else:
                        missing_no_doi.append(entry)
                    venue_missing += 1

        print(f"{venue_missing} missing")

    print()
    print(f"Total papers checked: {total_checked}")
    print(f"Papers with no abstract: {len(missing_with_doi) + len(missing_no_doi)}")
    print()

    if missing_with_doi:
        print(f"=== {len(missing_with_doi)} papers WITH DOI but no abstract ===")
        print("(These might be retrievable with different methods)")
        for p in missing_with_doi[:20]:
            print(f"  [{p['year']}] {p['venue']}")
            print(f"    {p['title'][:70]}{'...' if len(p['title']) > 70 else ''}")
            print(f"    DOI: {p['doi']}")
            print()

    if missing_no_doi:
        print(f"=== {len(missing_no_doi)} papers WITHOUT DOI ===")
        print("(These rely on title search or URL scraping)")
        for p in missing_no_doi[:20]:
            print(f"  [{p['year']}] {p['venue']}")
            print(f"    {p['title'][:70]}{'...' if len(p['title']) > 70 else ''}")
            print(f"    URL: {p['url']}")
            print()


def get_cache_stats() -> tuple[int, int]:
    """Get cache statistics: (total entries, entries with abstracts)."""
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return 0, 0
    total = 0
    with_abstract = 0
    for f in cache_dir.glob("*.txt"):
        total += 1
        if f.stat().st_size > 0:
            with_abstract += 1
    return total, with_abstract


# Legacy compatibility - no longer needed but keep for now
def get_abstract_cache() -> dict[str, str]:
    """Legacy function - cache is now file-based."""
    return {}


def save_abstract_cache(cache: dict[str, str]):
    """Legacy function - cache is now saved per-abstract."""
    pass


# ============================================================================
# DBLP Paper Cache (caches paper metadata to avoid re-querying DBLP)
# ============================================================================

DBLP_CACHE_DIR = Path(__file__).parent.parent / "data" / ".dblp_cache"


def get_dblp_cache_dir() -> Path:
    """Get the DBLP cache directory, creating it if needed."""
    DBLP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return DBLP_CACHE_DIR


def get_dblp_cache_file(venue_key: str, year: int) -> Path:
    """Get cache file path for a venue/year."""
    # Sanitize venue key for filename
    safe_venue = venue_key.replace("/", "_")
    return get_dblp_cache_dir() / f"{safe_venue}_{year}.json"


def get_cached_dblp_papers(venue_key: str, year: int) -> list[dict] | None:
    """Get cached DBLP papers for a venue/year, or None if not cached."""
    cache_file = get_dblp_cache_file(venue_key, year)
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def cache_dblp_papers(venue_key: str, year: int, papers: list[dict]):
    """Cache DBLP papers for a venue/year."""
    cache_file = get_dblp_cache_file(venue_key, year)
    tmp_file = cache_file.with_suffix('.tmp')
    try:
        with open(tmp_file, 'w') as f:
            json.dump(papers, f)
        tmp_file.rename(cache_file)
    except IOError:
        pass


def clear_dblp_cache():
    """Clear the entire DBLP paper cache."""
    cache_dir = get_dblp_cache_dir()
    if cache_dir.exists():
        for f in cache_dir.glob("*.json"):
            f.unlink()


def get_dblp_cache_stats() -> tuple[int, int]:
    """Get DBLP cache statistics: (num_files, total_papers)."""
    cache_dir = get_dblp_cache_dir()
    if not cache_dir.exists():
        return 0, 0
    num_files = 0
    total_papers = 0
    for f in cache_dir.glob("*.json"):
        num_files += 1
        try:
            with open(f) as fp:
                papers = json.load(fp)
                total_papers += len(papers)
        except (json.JSONDecodeError, IOError):
            pass
    return num_files, total_papers


# Optional imports for topic modeling
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Optional imports for embedding-based clustering
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Security conferences - DBLP venue identifiers
DEFAULT_VENUES = {
    # Top 4
    "IEEE S&P": "conf/sp",
    "USENIX Security": "conf/uss",
    "CCS": "conf/ccs",
    "NDSS": "conf/ndss",
    # Additional
    "AsiaCCS": "conf/asiaccs",
    "Euro S&P": "conf/eurosp",
    "EuroSys": "conf/eurosys",
}

# Preset keyword sets for common topics
PRESETS = {
    "code-reuse": [
        "return-oriented", "ROP", "code-reuse", "code reuse",
        "control-flow integrity", "CFI", "ASLR", "address space layout randomization",
        "gadget", "JIT-ROP", "BROP", "stack pivoting", "shadow stack",
        "code pointer integrity", "CPI",
    ],
    "fuzzing": [
        "fuzzing", "fuzz testing", "fuzzer", "AFL", "libFuzzer",
        "coverage-guided", "mutation-based", "grammar-based fuzzing",
    ],
    "memory-safety": [
        "buffer overflow", "memory corruption", "memory safety",
        "use-after-free", "heap overflow", "stack overflow",
        "out-of-bounds", "spatial safety", "temporal safety",
    ],
}


def query_dblp_venue_year(venue_key: str, year: int) -> list[dict]:
    """
    Query DBLP for all papers in a venue for a specific year.

    Args:
        venue_key: DBLP venue key (e.g., 'conf/sp')
        year: Publication year

    Returns:
        List of paper entries
    """
    url = "https://dblp.org/search/publ/api"
    params = {
        "q": f"stream:streams/{venue_key}: year:{year}:",
        "format": "json",
        "h": 1000,  # max results per query
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        hits = data.get("result", {}).get("hits", {})
        total = int(hits.get("@total", 0))

        if total == 0:
            return []

        papers = hits.get("hit", [])
        if isinstance(papers, dict):  # Single result
            papers = [papers]

        return [p.get("info", {}) for p in papers]

    except requests.RequestException as e:
        print(f"  Warning: Failed to query {venue_key}/{year}: {e}")
        return []


def paper_matches_keywords(paper: dict, keywords: list[str]) -> bool:
    """Check if paper title contains any of the keywords (case-insensitive)."""
    title = paper.get("title", "").lower()
    return any(kw.lower() in title for kw in keywords)


def count_matching_papers(papers: list[dict], keywords: list[str]) -> int:
    """Count papers whose titles match any keyword."""
    return sum(1 for p in papers if paper_matches_keywords(p, keywords))


def get_matching_papers(papers: list[dict], keywords: list[str]) -> list[dict]:
    """Return papers whose titles match any keyword."""
    return [p for p in papers if paper_matches_keywords(p, keywords)]


def analyze_trends(
    keywords: list[str],
    start_year: int,
    end_year: int,
    venues: dict[str, str],
    verbose: bool = True
) -> dict:
    """
    Analyze publication trends across all venues and years.

    Uses cached DBLP data when available to avoid rate limiting.

    Returns:
        Dictionary with year -> venue -> {total, matches, percentage, papers}
    """
    results = defaultdict(lambda: defaultdict(dict))

    for year in range(start_year, end_year + 1):
        if verbose:
            print(f"\nYear {year}:")

        for venue_name, venue_key in venues.items():
            if verbose:
                print(f"  Querying {venue_name}...", end=" ", flush=True)

            # Use cached query (falls back to DBLP if not cached)
            papers = query_dblp_with_dois(venue_key, year, use_cache=True)
            total = len(papers)

            match_count = count_matching_papers(papers, keywords)
            matching_papers = get_matching_papers(papers, keywords)

            results[year][venue_name] = {
                "total": total,
                "matches": match_count,
                "percentage": (match_count / total * 100) if total > 0 else 0,
                "papers": [p.get("title", "") for p in matching_papers],
            }

            if verbose:
                print(f"{total} papers, {match_count} matches ({results[year][venue_name]['percentage']:.1f}%)")

    return dict(results)


def print_summary(results: dict, keywords: list[str]):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print(f"SUMMARY: Publications matching keywords")
    print(f"Keywords: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")
    print("=" * 70)

    print(f"\n{'Year':<6} {'Total':<8} {'Matches':<10} {'Percentage':<12}")
    print("-" * 40)

    yearly_data = []
    for year in sorted(results.keys()):
        total = sum(v["total"] for v in results[year].values())
        matches = sum(v["matches"] for v in results[year].values())
        pct = (matches / total * 100) if total > 0 else 0
        yearly_data.append((year, total, matches, pct))
        print(f"{year:<6} {total:<8} {matches:<10} {pct:<12.1f}%")

    # Peak year analysis
    if yearly_data:
        peak = max(yearly_data, key=lambda x: x[2])
        print(f"\nPeak year: {peak[0]} ({peak[2]} papers, {peak[3]:.1f}%)")

        # Trend analysis
        if len(yearly_data) >= 3:
            first_half = yearly_data[:len(yearly_data)//2]
            second_half = yearly_data[len(yearly_data)//2:]
            avg_first = sum(x[2] for x in first_half) / len(first_half)
            avg_second = sum(x[2] for x in second_half) / len(second_half)
            trend = "increasing" if avg_second > avg_first else "decreasing"
            print(f"Trend: {trend} (avg {avg_first:.1f} -> {avg_second:.1f} papers/year)")


def save_csv(results: dict, output_path: Path):
    """Save results to CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Year", "Venue", "Total Papers", "Matching Papers", "Percentage"])

        for year in sorted(results.keys()):
            for venue, data in results[year].items():
                writer.writerow([
                    year, venue, data["total"], data["matches"], f"{data['percentage']:.1f}"
                ])

    print(f"\nResults saved to {output_path}")


def save_papers_list(results: dict, output_path: Path):
    """Save list of matching papers to a text file."""
    with open(output_path, "w") as f:
        for year in sorted(results.keys()):
            f.write(f"\n{'='*60}\n")
            f.write(f"Year {year}\n")
            f.write(f"{'='*60}\n")

            for venue, data in results[year].items():
                if data["papers"]:
                    f.write(f"\n{venue}:\n")
                    for title in data["papers"]:
                        f.write(f"  - {title}\n")

    print(f"Paper list saved to {output_path}")


def load_keywords_file(path: Path) -> list[str]:
    """Load keywords from a file (one per line)."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


# ============================================================================
# Topic Discovery Mode
# ============================================================================

# Semantic Scholar venue mappings (backup)
SEMANTIC_SCHOLAR_VENUES = {
    "IEEE S&P": "IEEE Symposium on Security and Privacy",
    "USENIX Security": "USENIX Security Symposium",
    "CCS": "ACM Conference on Computer and Communications Security",
    "NDSS": "Network and Distributed System Security Symposium",
    "AsiaCCS": "ACM Asia Conference on Computer and Communications Security",
    "Euro S&P": "IEEE European Symposium on Security and Privacy",
    "EuroSys": "European Conference on Computer Systems",
}

# OpenAlex venue IDs (preferred - more reliable)
# Find IDs at: https://api.openalex.org/venues?search=venue_name
OPENALEX_VENUES = {
    "IEEE S&P": "V4210190690",       # IEEE Symposium on Security and Privacy
    "USENIX Security": "V4210219653", # USENIX Security Symposium
    "CCS": "V4210207506",            # ACM CCS
    "NDSS": "V4210233498",           # NDSS
    "AsiaCCS": "V4210192631",        # ACM AsiaCCS
    "Euro S&P": "V4210217498",       # IEEE Euro S&P
    "EuroSys": "V4210219095",        # EuroSys
}


def query_dblp_with_dois(venue_key: str, year: int, use_cache: bool = True) -> list[dict]:
    """
    Query DBLP for papers with DOIs (for abstract fetching).
    Results are cached to avoid hitting DBLP repeatedly.
    """
    # Check cache first
    if use_cache:
        cached = get_cached_dblp_papers(venue_key, year)
        if cached is not None:
            return cached

    url = "https://dblp.org/search/publ/api"
    params = {
        "q": f"stream:streams/{venue_key}: year:{year}:",
        "format": "json",
        "h": 1000,
    }

    try:
        response = rate_limited_request(url, params=params, timeout=30, base_wait=5.0)
        if response is None:
            print(f"  DBLP error: rate limited")
            return []
        response.raise_for_status()
        data = response.json()

        hits = data.get("result", {}).get("hits", {})
        papers = hits.get("hit", [])
        if isinstance(papers, dict):
            papers = [papers]

        results = []
        for p in papers:
            info = p.get("info", {})
            results.append({
                "title": info.get("title", ""),
                "doi": info.get("doi", ""),
                "ee": info.get("ee", ""),  # Electronic edition URL
                "year": int(info.get("year", 0)),
            })

        # Cache the results
        if use_cache and results:
            cache_dblp_papers(venue_key, year, results)

        return results

    except requests.RequestException as e:
        print(f"  DBLP error: {e}")
        return []


def fetch_abstract_crossref(doi: str) -> str:
    """
    Fetch abstract from CrossRef API using DOI.
    Note: CrossRef often lacks abstracts (publishers don't deposit them).
    """
    if not doi:
        return ""

    url = f"https://api.crossref.org/works/{doi}"
    headers = {
        "User-Agent": "AcademicResearchBot/1.0 (mailto:research@example.com)"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 404:
            return ""
        response.raise_for_status()

        data = response.json()
        message = data.get("message", {})

        # Abstract might be in different fields
        abstract = message.get("abstract", "")

        # Clean up HTML tags often present in CrossRef abstracts
        if abstract:
            abstract = re.sub(r'<[^>]+>', '', abstract)
            abstract = abstract.strip()

        return abstract

    except requests.RequestException:
        return ""


def fetch_abstract_openalex(doi: str) -> str:
    """
    Fetch abstract from OpenAlex API using DOI.
    OpenAlex has better abstract coverage than CrossRef.
    """
    if not doi:
        return ""

    url = f"https://api.openalex.org/works/doi:{doi}"
    params = {"mailto": "research@example.com"}  # Polite pool

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 404:
            return ""
        response.raise_for_status()

        data = response.json()
        abstract_inv = data.get("abstract_inverted_index")

        if abstract_inv:
            return invert_abstract_index(abstract_inv)

        return ""

    except requests.RequestException:
        return ""


def fetch_abstract_unpaywall(doi: str) -> str:
    """
    Fetch abstract from Unpaywall API using DOI.
    Unpaywall focuses on open access papers.
    """
    if not doi:
        return ""

    email = "research@example.com"
    url = f"https://api.unpaywall.org/v2/{doi}"
    params = {"email": email}

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 404:
            return ""
        response.raise_for_status()

        data = response.json()
        # Unpaywall doesn't always have abstracts, but sometimes does
        abstract = data.get("abstract", "")
        if abstract:
            return abstract.strip()

        return ""

    except requests.RequestException:
        return ""


def fetch_abstract_semantic_scholar(doi: str) -> str:
    """
    Fetch abstract from Semantic Scholar API using DOI.
    Has good coverage but stricter rate limits.
    """
    if not doi:
        return ""

    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    params = {"fields": "abstract"}

    response = rate_limited_request(url, params=params, timeout=10, base_wait=10.0)
    if response is None:
        return ""

    if response.status_code == 404:
        return ""
    if response.status_code != 200:
        return ""

    try:
        data = response.json()
        abstract = data.get("abstract", "")
        if abstract:
            return abstract.strip()
        return ""
    except (ValueError, KeyError):
        return ""


def fetch_abstract_openalex_by_title(title: str) -> str:
    """
    Fetch abstract from OpenAlex API using title search.
    Fallback for papers without DOI.
    """
    if not title:
        return ""

    url = "https://api.openalex.org/works"
    params = {
        "search": title,
        "select": "title,abstract_inverted_index",
        "per-page": 1,
        "mailto": "research@example.com",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 404:
            return ""
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])
        if not results:
            return ""

        # Check title matches reasonably
        result = results[0]
        result_title = (result.get("title") or "").lower()
        query_title = title.lower()

        # Simple similarity check - at least 70% of words match
        query_words = set(query_title.split())
        result_words = set(result_title.split())
        if query_words and len(query_words & result_words) / len(query_words) < 0.7:
            return ""  # Title mismatch

        abstract_inv = result.get("abstract_inverted_index")
        if abstract_inv:
            return invert_abstract_index(abstract_inv)

        return ""

    except requests.RequestException:
        return ""


def fetch_abstract_s2_by_title(title: str) -> str:
    """
    Fetch abstract from Semantic Scholar using title search.
    Fallback for papers without DOI. Has stricter rate limits.
    """
    if not title:
        return ""

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title,
        "fields": "title,abstract",
        "limit": 1,
    }

    response = rate_limited_request(url, params=params, timeout=15, base_wait=10.0)
    if response is None:
        return ""

    if response.status_code == 404:
        return ""
    if response.status_code != 200:
        return ""

    try:
        data = response.json().get("data", [])
        if not data:
            return ""

        result = data[0]
        result_title = (result.get("title") or "").lower()
        query_title = title.lower()

        # Check title matches reasonably
        query_words = set(query_title.split())
        result_words = set(result_title.split())
        if query_words and len(query_words & result_words) / len(query_words) < 0.7:
            return ""  # Title mismatch

        abstract = result.get("abstract", "")
        if abstract:
            return abstract.strip()

        return ""
    except (ValueError, KeyError):
        return ""


def fetch_abstract_usenix(url: str) -> str:
    """
    Fetch abstract from USENIX website directly.
    USENIX provides free access to all papers.
    """
    if not url or "usenix.org" not in url:
        return ""

    try:
        response = requests.get(url, timeout=20, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        response.raise_for_status()

        html = response.text

        # Pattern 1: field-name-field-paper-description (most common)
        # Matches: class="field field-name-field-paper-description field-type-text-long..."
        match = re.search(
            r'<div[^>]*field-name-field-paper-description[^>]*>(.*?)</div>\s*</div>',
            html, re.DOTALL | re.IGNORECASE
        )
        if match:
            abstract = re.sub(r'<[^>]+>', ' ', match.group(1))
            abstract = ' '.join(abstract.split())  # Normalize whitespace
            if len(abstract) > 50:
                return abstract.strip()

        # Pattern 2: field--name-field-paper-description (alternate naming)
        match = re.search(
            r'<div[^>]*field--name-field-paper-description[^>]*>(.*?)</div>',
            html, re.DOTALL | re.IGNORECASE
        )
        if match:
            abstract = re.sub(r'<[^>]+>', ' ', match.group(1))
            abstract = ' '.join(abstract.split())
            if len(abstract) > 50:
                return abstract.strip()

        # Pattern 3: Older USENIX pages with "Abstract" header
        match = re.search(
            r'<h[23][^>]*>\s*Abstract\s*</h[23]>\s*<p>(.+?)</p>',
            html, re.DOTALL | re.IGNORECASE
        )
        if match:
            abstract = re.sub(r'<[^>]+>', '', match.group(1))
            return abstract.strip()

        # Pattern 4: meta description as fallback
        match = re.search(
            r'<meta\s+name="description"\s+content="([^"]+)"',
            html, re.IGNORECASE
        )
        if match:
            abstract = match.group(1)
            if len(abstract) > 100:  # Only use if substantial
                return abstract.strip()

        return ""

    except requests.RequestException:
        return ""


def fetch_abstract_acm_dl(doi: str) -> str:
    """
    Fetch abstract from ACM Digital Library.
    Works for ACM papers (CCS, AsiaCCS, etc).
    """
    if not doi or "10.1145" not in doi:  # ACM DOIs start with 10.1145
        return ""

    url = f"https://dl.acm.org/doi/{doi}"

    try:
        # Use full browser-like headers to avoid blocking
        response = requests.get(url, timeout=20, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })

        if response.status_code == 403:
            return ""  # Blocked, skip

        response.raise_for_status()
        html = response.text

        # ACM DL abstract patterns
        patterns = [
            r'<div[^>]*class="[^"]*abstractSection[^"]*"[^>]*>.*?<p>(.+?)</p>',
            r'<section[^>]*class="[^"]*abstract[^"]*"[^>]*>.*?<p>(.+?)</p>',
            r'<div[^>]*role="doc-abstract"[^>]*>.*?<p>(.+?)</p>',
        ]

        for pattern in patterns:
            match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
            if match:
                abstract = re.sub(r'<[^>]+>', '', match.group(1))
                abstract = ' '.join(abstract.split())
                if len(abstract) > 50:
                    return abstract.strip()

        return ""

    except requests.RequestException:
        return ""


def fetch_abstract_embedded_metadata(url: str) -> str:
    """
    Extract abstract from embedded metadata in a webpage.
    Checks multiple formats like Zotero does:
    - Highwire Press tags (citation_abstract)
    - Dublin Core (DC.description)
    - Open Graph (og:description)
    - JSON-LD (schema.org)
    - Standard meta description
    """
    if not url:
        return ""

    try:
        response = requests.get(url, timeout=20, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        if response.status_code != 200:
            return ""

        html = response.text

        # 1. Highwire Press tags (most academic sites use this)
        match = re.search(
            r'<meta\s+name=["\']citation_abstract["\']\s+content=["\']([^"\']+)["\']',
            html, re.IGNORECASE
        )
        if not match:
            match = re.search(
                r'<meta\s+content=["\']([^"\']+)["\']\s+name=["\']citation_abstract["\']',
                html, re.IGNORECASE
            )
        if match:
            abstract = match.group(1)
            if len(abstract) > 50:
                return html_unescape(abstract.strip())

        # 2. Dublin Core
        match = re.search(
            r'<meta\s+name=["\']DC\.description["\']\s+content=["\']([^"\']+)["\']',
            html, re.IGNORECASE
        )
        if match:
            abstract = match.group(1)
            if len(abstract) > 50:
                return html_unescape(abstract.strip())

        # 3. JSON-LD (schema.org)
        jsonld_matches = re.findall(
            r'<script\s+type=["\']application/ld\+json["\']>(.*?)</script>',
            html, re.DOTALL | re.IGNORECASE
        )
        for jsonld_text in jsonld_matches:
            try:
                data = json.loads(jsonld_text)
                # Handle both single object and array
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if isinstance(item, dict):
                        abstract = item.get("abstract") or item.get("description", "")
                        if abstract and len(abstract) > 50:
                            return html_unescape(abstract.strip())
            except json.JSONDecodeError:
                continue

        # 4. Open Graph description (often used as fallback)
        match = re.search(
            r'<meta\s+property=["\']og:description["\']\s+content=["\']([^"\']+)["\']',
            html, re.IGNORECASE
        )
        if match:
            abstract = match.group(1)
            if len(abstract) > 100:  # OG descriptions can be short summaries
                return html_unescape(abstract.strip())

        # 5. Standard meta description (last resort)
        match = re.search(
            r'<meta\s+name=["\']description["\']\s+content=["\']([^"\']+)["\']',
            html, re.IGNORECASE
        )
        if match:
            abstract = match.group(1)
            if len(abstract) > 150:  # Only use if substantial
                return html_unescape(abstract.strip())

        return ""

    except requests.RequestException:
        return ""


def html_unescape(text: str) -> str:
    """Unescape HTML entities."""
    import html
    return html.unescape(text)


def fetch_abstract_doi_citeproc(doi: str) -> str:
    """
    Fetch abstract via DOI.org content negotiation (Citeproc JSON).
    This is how Zotero gets metadata when adding by DOI.
    """
    if not doi:
        return ""

    url = f"https://doi.org/{doi}"
    headers = {"Accept": "application/vnd.citationstyles.csl+json"}

    try:
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        if response.status_code != 200:
            return ""

        data = response.json()
        abstract = data.get("abstract", "")
        if abstract:
            # Clean up HTML that sometimes appears in abstracts
            abstract = re.sub(r'<[^>]+>', '', abstract)
            return abstract.strip()

        return ""

    except (requests.RequestException, ValueError):
        return ""


def fetch_abstract_multi_source(
    doi: str | None = None,
    title: str | None = None,
    url: str | None = None,
    verbose: bool = False
) -> tuple[str, str]:
    """
    Try to fetch abstract from multiple sources (with caching).

    Sources are tried in order of reliability/speed:
    1. Cache (fastest)
    2. OpenAlex by DOI (best coverage, fast)
    3. USENIX direct (for usenix.org URLs)
    4. ACM DL (for ACM DOIs)
    5. CrossRef (sometimes has abstracts)
    6. Semantic Scholar (good but rate limited)
    7. Unpaywall (open access focused)
    8. OpenAlex by title (fallback for no-DOI papers)

    Args:
        doi: Paper DOI (primary lookup key)
        title: Paper title (for title-based fallback)
        url: Paper URL (for direct scraping)

    Returns (abstract, source_name) tuple.
    """
    # Create cache key from DOI or title
    cache_key = doi if doi else (title[:100] if title else None)
    if not cache_key:
        return "", "no-id"

    # Check cache first
    cached = get_cached_abstract(cache_key)
    if cached is not None:
        return cached, "cache" if cached else "cache-miss"

    # Try DOI-based sources first (most reliable)
    if doi:
        doi_sources = [
            ("openalex", lambda: fetch_abstract_openalex(doi)),
            ("doi-citeproc", lambda: fetch_abstract_doi_citeproc(doi)),
            ("acm-dl", lambda: fetch_abstract_acm_dl(doi)),
            ("crossref", lambda: fetch_abstract_crossref(doi)),
            ("s2", lambda: fetch_abstract_semantic_scholar(doi)),
            ("unpaywall", lambda: fetch_abstract_unpaywall(doi)),
        ]

        for source_name, fetch_fn in doi_sources:
            try:
                abstract = fetch_fn()
                if abstract:
                    cache_abstract(cache_key, abstract)
                    return abstract, source_name
            except Exception:
                continue

    # Try URL-based scraping (skip PDFs - can't extract metadata from them)
    if url and not url.lower().endswith('.pdf'):
        # USENIX-specific scraping
        if "usenix.org" in url:
            try:
                abstract = fetch_abstract_usenix(url)
                if abstract:
                    cache_abstract(cache_key, abstract)
                    return abstract, "usenix"
            except Exception:
                pass

        # Generic embedded metadata extraction (like Zotero)
        # Works for many academic sites with Highwire Press tags, JSON-LD, etc.
        try:
            abstract = fetch_abstract_embedded_metadata(url)
            if abstract:
                cache_abstract(cache_key, abstract)
                return abstract, "embedded"
        except Exception:
            pass

    # Try title-based fallbacks
    if title:
        # OpenAlex by title
        try:
            abstract = fetch_abstract_openalex_by_title(title)
            if abstract:
                cache_abstract(cache_key, abstract)
                return abstract, "openalex-title"
        except Exception:
            pass

        # Semantic Scholar by title (rate limited, so try last)
        try:
            abstract = fetch_abstract_s2_by_title(title)
            if abstract:
                cache_abstract(cache_key, abstract)
                return abstract, "s2-title"
        except Exception:
            pass

    # Cache negative result
    cache_abstract(cache_key, "")
    return "", "none"


def _fetch_abstract_worker(paper: dict) -> tuple[dict, str, str]:
    """Worker function for parallel abstract fetching."""
    abstract, source = fetch_abstract_multi_source(
        doi=paper.get("doi"),
        title=paper.get("title"),
        url=paper.get("ee"),
    )
    return paper, abstract, source


def _process_year_venue(
    year: int,
    venue_name: str,
    venue_key: str,
    fetch_abstracts: bool,
    max_workers: int,
) -> tuple[int, str, list[dict], dict[str, int]]:
    """
    Process a single year/venue combination.
    Returns (year, venue_name, papers_list, source_stats).
    """
    # Get papers from DBLP (includes DOI and URL)
    papers = query_dblp_with_dois(venue_key, year)

    # Fetch abstracts in parallel
    source_stats = defaultdict(int)
    results = []

    if fetch_abstracts and papers:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_abstract_worker, p): p for p in papers}
            for future in as_completed(futures):
                paper, abstract, source = future.result()
                results.append((paper, abstract, source))
                source_stats[source] += 1
    else:
        results = [(p, "", "skip") for p in papers]

    # Build paper list (preserve order by title for consistency)
    paper_list = []
    for paper, abstract, source in sorted(results, key=lambda x: x[0].get("title", "")):
        paper_list.append({
            "title": paper["title"],
            "abstract": abstract,
            "doi": paper.get("doi", ""),
            "url": paper.get("ee", ""),
            "venue": venue_name,
        })

    return year, venue_name, paper_list, dict(source_stats)


def collect_papers_dblp_openalex(
    start_year: int,
    end_year: int,
    venues: dict[str, str],
    verbose: bool = True,
    fetch_abstracts: bool = True,
    max_workers: int = 10,
    parallel_years: bool = True,
) -> dict[int, list[dict]]:
    """
    Collect papers from DBLP with abstracts from multiple sources.

    Sources for abstracts (tried in order):
    - OpenAlex (by DOI)
    - ACM Digital Library (by DOI, for ACM papers)
    - USENIX website (by URL, for USENIX papers)
    - CrossRef (by DOI)
    - Semantic Scholar (by DOI)
    - Unpaywall (by DOI)
    - OpenAlex (by title search, fallback)

    Abstracts are cached to disk to speed up subsequent runs.
    Abstract fetching is parallelized for speed.
    Years are processed in parallel when parallel_years=True.
    """
    papers_by_year = defaultdict(list)
    initial_cache_total, initial_cache_abs = get_cache_stats()

    # Track source statistics
    source_stats = defaultdict(int)

    years = list(range(start_year, end_year + 1))
    total_tasks = len(years) * len(venues)

    if parallel_years and len(years) > 1:
        # Parallel processing of years
        if verbose:
            print(f"Processing {len(years)} years × {len(venues)} venues in parallel...")

        # Create all year/venue combinations
        tasks = [
            (year, venue_name, venue_key, fetch_abstracts, max_workers)
            for year in years
            for venue_name, venue_key in venues.items()
        ]

        completed = 0
        # Use fewer workers for year-level parallelism to avoid overwhelming APIs
        year_workers = min(4, len(years))

        with ThreadPoolExecutor(max_workers=year_workers) as executor:
            futures = {
                executor.submit(_process_year_venue, *task): task
                for task in tasks
            }

            for future in as_completed(futures):
                year, venue_name, paper_list, stats = future.result()
                papers_by_year[year].extend(paper_list)

                # Aggregate source stats
                for source, count in stats.items():
                    source_stats[source] += count

                completed += 1
                if verbose:
                    total_papers = len(paper_list)
                    with_abs = sum(1 for p in paper_list if p.get("abstract"))
                    print(f"  [{completed}/{total_tasks}] {year} {venue_name}: {total_papers} papers, {with_abs} abstracts")

        # Print year summaries
        if verbose:
            print(f"\n--- Year Summaries ---")
            for year in sorted(papers_by_year.keys()):
                total = len(papers_by_year[year])
                with_abs = sum(1 for p in papers_by_year[year] if p.get("abstract"))
                pct = (100 * with_abs / total) if total else 0
                print(f"  {year}: {total} papers, {with_abs} abstracts ({pct:.0f}%)")

    else:
        # Sequential processing (original behavior for single year or when disabled)
        for year in years:
            if verbose:
                print(f"\nYear {year}:")

            for venue_name, venue_key in venues.items():
                if verbose:
                    print(f"  {venue_name}...", end=" ", flush=True)

                # Get papers from DBLP (includes DOI and URL)
                papers = query_dblp_with_dois(venue_key, year)
                papers_with_doi = sum(1 for p in papers if p.get("doi"))

                if verbose:
                    print(f"{len(papers)} papers ({papers_with_doi} DOI)")

                # Fetch abstracts in parallel
                venue_stats = defaultdict(int)
                results = []

                if fetch_abstracts and papers:
                    if verbose:
                        print(f"    Fetching abstracts ({max_workers} workers)...", end=" ", flush=True)

                    completed = 0
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {executor.submit(_fetch_abstract_worker, p): p for p in papers}

                        for future in as_completed(futures):
                            paper, abstract, source = future.result()
                            results.append((paper, abstract, source))
                            venue_stats[source] += 1
                            source_stats[source] += 1

                            completed += 1
                            if verbose and completed % 10 == 0:
                                print(f"{completed}", end=" ", flush=True)

                    if verbose:
                        print(f"done")

                    # Show per-paper details if verbose
                    if verbose:
                        for paper, abstract, source in results:
                            short_title = paper.get("title", "")[:50]
                            if len(paper.get("title", "")) > 50:
                                short_title += "..."
                            mark = "✓" if abstract else ""
                            print(f"      {source:14} {mark:1} {short_title}")
                else:
                    results = [(p, "", "skip") for p in papers]

                # Add to results (preserve order by title for consistency)
                for paper, abstract, source in sorted(results, key=lambda x: x[0].get("title", "")):
                    papers_by_year[year].append({
                        "title": paper["title"],
                        "abstract": abstract,
                        "doi": paper.get("doi", ""),
                        "url": paper.get("ee", ""),
                        "venue": venue_name,
                    })

                if verbose and fetch_abstracts:
                    # Count abstracts from each source type
                    api_sources = ["openalex", "openalex-title", "doi-citeproc", "acm-dl",
                                   "usenix", "embedded", "crossref", "s2", "s2-title", "unpaywall"]
                    new_abstracts = sum(venue_stats.get(s, 0) for s in api_sources)
                    cached = venue_stats.get("cache", 0)
                    no_abstract = venue_stats.get("none", 0) + venue_stats.get("cache-miss", 0)
                    print(f"    Summary: {new_abstracts} new, {cached} cached, {no_abstract} n/a")

                time.sleep(0.1)  # Small delay between venues

            if verbose:
                total = len(papers_by_year[year])
                with_abs = sum(1 for p in papers_by_year[year] if p.get("abstract"))
                pct = (100 * with_abs / total) if total else 0
                print(f"  Year {year} total: {total} papers, {with_abs} abstracts ({pct:.0f}%)")

    # Print source statistics
    if verbose and fetch_abstracts:
        print(f"\n--- Abstract Source Statistics ---")
        all_sources = ["openalex", "openalex-title", "doi-citeproc", "acm-dl", "usenix",
                       "embedded", "crossref", "s2", "s2-title", "unpaywall", "cache", "cache-miss", "none", "no-id"]
        for source in all_sources:
            if source_stats[source] > 0:
                print(f"  {source:14}: {source_stats[source]:5}")

    # Show cache update stats
    final_cache_total, final_cache_abs = get_cache_stats()
    if verbose and final_cache_total > initial_cache_total:
        print(f"\nAbstract cache updated: {initial_cache_total} -> {final_cache_total} entries")
        print(f"Cache location: {get_cache_dir()}")

    return dict(papers_by_year)


def query_openalex(venue_id: str, year: int, per_page: int = 200) -> list[dict]:
    """
    Query OpenAlex for papers from a venue/year.

    OpenAlex is free, reliable, and has good abstract coverage.
    API docs: https://docs.openalex.org/
    """
    url = "https://api.openalex.org/works"

    papers = []
    cursor = "*"

    while True:
        params = {
            "filter": f"primary_location.source.id:{venue_id},publication_year:{year}",
            "select": "title,abstract_inverted_index,publication_year,primary_location",
            "per-page": per_page,
            "cursor": cursor,
            "mailto": "research@example.com",  # Polite pool (faster)
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                break

            for r in results:
                # Convert inverted index to plain text abstract
                abstract = ""
                if r.get("abstract_inverted_index"):
                    abstract = invert_abstract_index(r["abstract_inverted_index"])

                papers.append({
                    "title": r.get("title", ""),
                    "abstract": abstract,
                    "year": r.get("publication_year"),
                })

            # Pagination
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break

            time.sleep(0.1)  # Be nice to the API

        except requests.RequestException as e:
            print(f"  Warning: OpenAlex query failed: {e}")
            break

    return papers


def invert_abstract_index(inverted_index: dict) -> str:
    """Convert OpenAlex inverted index format to plain text."""
    if not inverted_index:
        return ""

    # inverted_index is {word: [positions]}
    words = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words.append((pos, word))

    words.sort(key=lambda x: x[0])
    return " ".join(w[1] for w in words)


def query_semantic_scholar(venue: str, year: int, limit: int = 100) -> list[dict]:
    """
    Query Semantic Scholar for papers from a venue/year.

    Returns papers with title and abstract.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    # Use venue name as search query (more reliable than venue: filter)
    # Add year to narrow results
    query = f'"{venue}" {year}'

    params = {
        "query": query,
        "fields": "title,abstract,year,venue",
        "limit": limit,
    }

    headers = {
        "User-Agent": "AcademicResearchScript/1.0"
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)

            if response.status_code == 429:
                wait_time = 30 * (attempt + 1)
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            if response.status_code >= 500:
                print(f"  Server error {response.status_code}, retrying...")
                time.sleep(10)
                continue

            response.raise_for_status()
            data = response.json()

            papers = data.get("data", [])
            # Filter to exact year
            papers = [p for p in papers if p.get("year") == year]

            return papers

        except requests.RequestException as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}...", end=" ")
                time.sleep(5)
            else:
                print(f"  Failed: {e}")
                return []

    return []


def collect_papers_with_abstracts(
    start_year: int,
    end_year: int,
    verbose: bool = True
) -> dict[int, list[dict]]:
    """
    Collect papers with abstracts from OpenAlex (preferred) or Semantic Scholar (fallback).

    Returns:
        Dictionary with year -> list of {title, abstract} dicts
    """
    papers_by_year = defaultdict(list)

    for year in range(start_year, end_year + 1):
        if verbose:
            print(f"\nYear {year}:")

        for venue_short in OPENALEX_VENUES.keys():
            if verbose:
                print(f"  {venue_short}...", end=" ", flush=True)

            # Try OpenAlex first (preferred)
            venue_id = OPENALEX_VENUES.get(venue_short)
            if venue_id:
                papers = query_openalex(venue_id, year)
            else:
                papers = []

            # Fallback to Semantic Scholar if OpenAlex fails
            if not papers and venue_short in SEMANTIC_SCHOLAR_VENUES:
                if verbose:
                    print("(trying S2)...", end=" ", flush=True)
                venue_full = SEMANTIC_SCHOLAR_VENUES[venue_short]
                papers = query_semantic_scholar(venue_full, year)
                time.sleep(2)  # Rate limit for S2

            for p in papers:
                if p.get("title"):
                    papers_by_year[year].append({
                        "title": p.get("title", ""),
                        "abstract": p.get("abstract") or "",
                        "venue": venue_short,
                    })

            if verbose:
                abstracts_count = sum(1 for p in papers if p.get("abstract"))
                print(f"{len(papers)} papers ({abstracts_count} with abstracts)")

        if verbose:
            total = len(papers_by_year[year])
            with_abs = sum(1 for p in papers_by_year[year] if p.get("abstract"))
            print(f"  Total: {total} papers, {with_abs} with abstracts ({100*with_abs/total:.0f}%)" if total else "  Total: 0")

    return dict(papers_by_year)


# Common stopwords to filter out
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "was", "are", "were", "been", "be", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "must", "shall", "can", "need", "dare", "ought", "used", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "what",
    "which", "who", "whom", "when", "where", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "just", "also", "now",
    "new", "using", "based", "via", "through", "towards", "toward", "against",
    "between", "into", "during", "before", "after", "above", "below", "under",
    "again", "further", "then", "once", "here", "there", "any", "about", "over",
}

# Generic academic terms to filter
ACADEMIC_STOPWORDS = {
    "analysis", "approach", "system", "systems", "study", "case", "research",
    "paper", "work", "method", "methods", "technique", "techniques", "efficient",
    "effective", "novel", "practical", "toward", "towards", "comprehensive",
    "understanding", "exploring", "beyond", "revisiting", "rethinking",
}


def extract_ngrams(title: str, n: int = 2) -> list[str]:
    """Extract n-grams from a title after cleaning."""
    # Clean and tokenize
    title = title.lower()
    title = re.sub(r'[^a-z0-9\s\-]', ' ', title)
    words = title.split()

    # Filter stopwords
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]

    # Generate n-grams
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i:i+n])
        # Skip if any word is a stopword or academic generic term
        ngram_words = ngram.split()
        if not any(w in ACADEMIC_STOPWORDS for w in ngram_words):
            ngrams.append(ngram)

    return ngrams


def collect_all_papers(
    start_year: int,
    end_year: int,
    venues: dict[str, str],
    verbose: bool = True
) -> dict[int, list[str]]:
    """
    Collect all paper titles organized by year.

    Returns:
        Dictionary with year -> list of titles
    """
    papers_by_year = defaultdict(list)

    for year in range(start_year, end_year + 1):
        if verbose:
            print(f"\nYear {year}:", end=" ", flush=True)

        for venue_name, venue_key in venues.items():
            papers = query_dblp_venue_year(venue_key, year)
            titles = [p.get("title", "") for p in papers if p.get("title")]
            papers_by_year[year].extend(titles)

            if verbose:
                print(f"{venue_name}:{len(papers)}", end=" ", flush=True)

            time.sleep(0.5)

        if verbose:
            print(f"= {len(papers_by_year[year])} total")

    return dict(papers_by_year)


def compute_ngram_frequencies(
    papers_by_year: dict[int, list[str]],
    ngram_sizes: list[int] = [2, 3]
) -> dict[str, dict[int, int]]:
    """
    Compute n-gram frequencies per year.

    Returns:
        Dictionary with ngram -> {year -> count}
    """
    ngram_counts = defaultdict(lambda: defaultdict(int))

    for year, titles in papers_by_year.items():
        for title in titles:
            for n in ngram_sizes:
                for ngram in extract_ngrams(title, n):
                    ngram_counts[ngram][year] += 1

    return dict(ngram_counts)


def compute_trend_score(
    counts_by_year: dict[int, int],
    years: list[int],
    totals_by_year: dict[int, int] | None = None,
) -> float:
    """
    Compute a trend score for a term.

    Positive = rising, Negative = declining.
    Uses linear regression slope normalized by mean frequency.

    If totals_by_year is provided, uses proportions (share of papers per year)
    instead of raw counts. This controls for overall publication growth.
    """
    if len(years) < 3:
        return 0.0

    # Get counts for each year (0 if not present)
    raw_values = [counts_by_year.get(y, 0) for y in years]

    if totals_by_year:
        # Use proportions instead of raw counts
        values = [
            raw_values[i] / totals_by_year.get(years[i], 1)
            for i in range(len(years))
        ]
    else:
        values = raw_values

    mean_val = sum(values) / len(values)

    if mean_val < 0.0001:  # Filter very rare terms (lower threshold for proportions)
        return 0.0

    # Simple linear regression slope
    n = len(years)
    x_mean = sum(years) / n
    y_mean = mean_val

    numerator = sum((years[i] - x_mean) * (values[i] - y_mean) for i in range(n))
    denominator = sum((years[i] - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return 0.0

    slope = numerator / denominator

    # Normalize by mean to get relative trend
    return slope / mean_val if mean_val > 0 else 0.0


def discover_trends(
    start_year: int,
    end_year: int,
    venues: dict[str, str],
    top_n: int = 30,
    min_total_count: int = 5,
    verbose: bool = True
) -> tuple[list, list, dict]:
    """
    Discover rising and declining topics using n-gram analysis.

    Returns:
        (rising_topics, declining_topics, all_ngram_data)
    """
    print("Collecting papers from all venues...")
    papers_by_year = collect_all_papers(start_year, end_year, venues, verbose)

    print("\nExtracting n-grams and computing frequencies...")
    ngram_freqs = compute_ngram_frequencies(papers_by_year)

    years = sorted(papers_by_year.keys())

    # Compute trend scores
    trends = []
    for ngram, counts in ngram_freqs.items():
        total = sum(counts.values())
        if total < min_total_count:
            continue

        score = compute_trend_score(counts, years)
        trends.append({
            "ngram": ngram,
            "score": score,
            "total": total,
            "counts": counts,
            "first_year": min(y for y, c in counts.items() if c > 0),
            "last_year": max(y for y, c in counts.items() if c > 0),
        })

    # Sort by trend score
    trends.sort(key=lambda x: x["score"], reverse=True)

    rising = trends[:top_n]
    declining = trends[-top_n:][::-1]  # Reverse to show most declining first

    return rising, declining, ngram_freqs


# ============================================================================
# NLP-based Topic Discovery (using scikit-learn)
# ============================================================================

def discover_topics_nlp(
    start_year: int,
    end_year: int,
    n_topics: int = 20,
    use_abstracts: bool = False,
    method: str = "nmf",
    verbose: bool = True,
    max_workers: int = 10,
) -> tuple[list, dict[int, list[dict]]]:
    """
    Discover topics using NLP (TF-IDF + NMF/LDA).

    Args:
        start_year: Start year
        end_year: End year
        n_topics: Number of topics to extract
        use_abstracts: If True, fetch abstracts from multiple sources
        method: 'nmf' or 'lda'
        verbose: Print progress
        max_workers: Number of parallel workers for abstract fetching

    Returns:
        (topics_with_trends, papers_by_year)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for NLP discovery. Install with: pip install scikit-learn")

    # Collect papers
    if use_abstracts:
        print("Collecting papers from DBLP + abstracts from multiple sources...")
        papers_by_year = collect_papers_dblp_openalex(
            start_year, end_year, DEFAULT_VENUES,
            verbose=verbose, fetch_abstracts=True, max_workers=max_workers
        )
    else:
        print("Collecting paper titles from DBLP...")
        titles_by_year = collect_all_papers(start_year, end_year, DEFAULT_VENUES, verbose)
        papers_by_year = {
            year: [{"title": t, "abstract": ""} for t in titles]
            for year, titles in titles_by_year.items()
        }

    # Prepare documents (combine title + abstract)
    all_docs = []
    doc_years = []
    for year in sorted(papers_by_year.keys()):
        for paper in papers_by_year[year]:
            text = paper["title"]
            if paper.get("abstract"):
                text += " " + paper["abstract"]
            all_docs.append(text)
            doc_years.append(year)

    if not all_docs:
        print("No documents found!")
        return [], papers_by_year

    print(f"\nAnalyzing {len(all_docs)} documents...")

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_df=0.8,           # Ignore terms in >80% of docs
        min_df=5,             # Ignore terms in <5 docs
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 3),   # Unigrams, bigrams, trigrams
        token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z-]+\b',  # Words with optional hyphen
    )

    tfidf_matrix = vectorizer.fit_transform(all_docs)
    feature_names = vectorizer.get_feature_names_out()

    print(f"Vocabulary size: {len(feature_names)} terms")

    # Topic modeling
    print(f"Extracting {n_topics} topics using {method.upper()}...")

    if method == "lda":
        # Convert to count matrix for LDA
        count_vectorizer = CountVectorizer(
            max_df=0.8, min_df=5, max_features=5000,
            stop_words='english', ngram_range=(1, 3),
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z-]+\b',
        )
        count_matrix = count_vectorizer.fit_transform(all_docs)
        feature_names = count_vectorizer.get_feature_names_out()

        model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20,
            learning_method='online',
        )
        doc_topics = model.fit_transform(count_matrix)
    else:
        # NMF (usually better for short texts)
        model = NMF(
            n_components=n_topics,
            random_state=42,
            max_iter=500,
            init='nndsvda',
        )
        doc_topics = model.fit_transform(tfidf_matrix)

    # Extract top words per topic
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[:-15:-1]  # Top 15 words
        top_words = [feature_names[i] for i in top_indices]
        topics.append({
            "id": topic_idx,
            "words": top_words,
            "label": " / ".join(top_words[:3]),  # Short label
        })

    # Compute topic prevalence per year
    years = sorted(set(doc_years))
    for topic in topics:
        topic["counts"] = {}
        topic_idx = topic["id"]

        for year in years:
            # Get documents from this year
            year_mask = [i for i, y in enumerate(doc_years) if y == year]
            if year_mask:
                # Average topic weight for this year's documents
                year_weights = [doc_topics[i][topic_idx] for i in year_mask]
                topic["counts"][year] = sum(year_weights)
            else:
                topic["counts"][year] = 0

        # Compute trend score
        topic["total"] = sum(topic["counts"].values())
        topic["score"] = compute_trend_score(topic["counts"], years)

    # Sort by trend score
    topics.sort(key=lambda x: x["score"], reverse=True)

    return topics, papers_by_year


def print_nlp_discovery_results(topics: list, years: list[int]):
    """Print NLP-discovered topics with trends."""
    print("\n" + "=" * 90)
    print("DISCOVERED TOPICS (sorted by trend: rising first, declining last)")
    print("=" * 90)

    # Split into rising and declining
    rising = [t for t in topics if t["score"] > 0.01]
    declining = [t for t in topics if t["score"] < -0.01]
    stable = [t for t in topics if -0.01 <= t["score"] <= 0.01]

    print(f"\n{'─'*90}")
    print("RISING TOPICS")
    print(f"{'─'*90}")
    print(f"{'#':<3} {'Trend':>7} {'Label':<40} {'Top Keywords':<40}")
    print("-" * 90)

    for i, t in enumerate(rising[:10], 1):
        label = t["label"][:38]
        keywords = ", ".join(t["words"][3:8])[:38]
        print(f"{i:<3} {t['score']:>+7.3f} {label:<40} {keywords:<40}")

    print(f"\n{'─'*90}")
    print("DECLINING TOPICS")
    print(f"{'─'*90}")
    print(f"{'#':<3} {'Trend':>7} {'Label':<40} {'Top Keywords':<40}")
    print("-" * 90)

    for i, t in enumerate(reversed(declining[-10:]), 1):
        label = t["label"][:38]
        keywords = ", ".join(t["words"][3:8])[:38]
        print(f"{i:<3} {t['score']:>+7.3f} {label:<40} {keywords:<40}")

    if stable:
        print(f"\n{'─'*90}")
        print(f"STABLE TOPICS ({len(stable)} topics with minimal trend)")
        print(f"{'─'*90}")
        for t in stable[:5]:
            print(f"  • {t['label']}")


def save_nlp_discovery_csv(topics: list, years: list[int], output_path: Path):
    """Save NLP discovery results to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = ["Topic_ID", "Label", "Trend_Score", "Keywords"] + [str(y) for y in years]
        writer.writerow(header)

        for t in topics:
            row = [
                t["id"],
                t["label"],
                f"{t['score']:.4f}",
                "; ".join(t["words"][:10]),
            ]
            row += [f"{t['counts'].get(y, 0):.2f}" for y in years]
            writer.writerow(row)

    print(f"\nNLP topic results saved to {output_path}")


# ============================================================================
# Embedding-based Topic Discovery
# ============================================================================

def discover_topics_embedding(
    start_year: int,
    end_year: int,
    n_clusters: int = 20,
    use_abstracts: bool = True,
    verbose: bool = True,
    max_workers: int = 10,
    min_cluster_size: int = 10,
    use_hdbscan: bool = True,
    model_name: str = "all-MiniLM-L6-v2",
) -> tuple[list, dict]:
    """
    Discover topics using sentence embeddings + clustering.

    This approach produces more semantically coherent clusters than NMF/LDA
    by using neural network embeddings that capture meaning.

    Args:
        start_year: First year to analyze
        end_year: Last year to analyze
        n_clusters: Number of clusters (ignored if use_hdbscan=True)
        use_abstracts: Whether to fetch/use abstracts
        verbose: Print progress
        max_workers: Parallel workers for abstract fetching
        min_cluster_size: Minimum papers per cluster (HDBSCAN)
        use_hdbscan: Use HDBSCAN (auto cluster count) vs KMeans
        model_name: Sentence transformer model name

    Returns:
        (topics_with_trends, papers_by_year)
    """
    if not EMBEDDINGS_AVAILABLE:
        raise ImportError(
            "sentence-transformers required for embedding-based discovery. "
            "Install with: pip install sentence-transformers"
        )
    if not NUMPY_AVAILABLE:
        raise ImportError("numpy required. Install with: pip install numpy")
    if use_hdbscan and not HDBSCAN_AVAILABLE:
        print("HDBSCAN not available, falling back to KMeans")
        use_hdbscan = False
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required. Install with: pip install scikit-learn")

    # Collect papers
    if use_abstracts:
        print("Collecting papers from DBLP + abstracts from multiple sources...")
        papers_by_year = collect_papers_dblp_openalex(
            start_year, end_year, DEFAULT_VENUES,
            verbose=verbose, fetch_abstracts=True, max_workers=max_workers
        )
    else:
        print("Collecting paper titles from DBLP...")
        titles_by_year = collect_all_papers(start_year, end_year, DEFAULT_VENUES, verbose)
        papers_by_year = {
            year: [{"title": t, "abstract": ""} for t in titles]
            for year, titles in titles_by_year.items()
        }

    # Prepare documents
    all_docs = []
    doc_years = []
    doc_papers = []  # Keep reference to original papers
    for year in sorted(papers_by_year.keys()):
        for paper in papers_by_year[year]:
            text = paper["title"]
            if paper.get("abstract"):
                text += " " + paper["abstract"]
            all_docs.append(text)
            doc_years.append(year)
            # Ensure paper has year field for later grouping
            paper_with_year = {**paper, "year": year}
            doc_papers.append(paper_with_year)

    if not all_docs:
        print("No documents found!")
        return [], papers_by_year

    print(f"\nEmbedding {len(all_docs)} documents with {model_name}...")

    # Load model and encode documents
    model = SentenceTransformer(model_name)
    embeddings = model.encode(all_docs, show_progress_bar=verbose, batch_size=64)
    embeddings = np.array(embeddings)

    print(f"Embedding shape: {embeddings.shape}")

    # Dimensionality reduction with UMAP (critical for HDBSCAN performance)
    cluster_embeddings = embeddings
    if use_hdbscan and UMAP_AVAILABLE:
        print("Reducing dimensions with UMAP...")
        reducer = umap.UMAP(
            n_components=15,  # Reduce to 15D for clustering
            n_neighbors=15,
            min_dist=0.0,
            metric='cosine',
            random_state=42,
        )
        cluster_embeddings = reducer.fit_transform(embeddings)
        print(f"Reduced to shape: {cluster_embeddings.shape}")
    elif use_hdbscan and not UMAP_AVAILABLE:
        print("Warning: UMAP not available, HDBSCAN may find fewer clusters")
        print("Install with: pip install umap-learn")

    # Cluster
    if use_hdbscan:
        print(f"Clustering with HDBSCAN (min_cluster_size={min_cluster_size})...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=5,
            metric='euclidean',
            cluster_selection_method='eom',
        )
        labels = clusterer.fit_predict(cluster_embeddings)
        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = sum(1 for l in labels if l == -1)
        print(f"Found {n_clusters_found} clusters, {n_noise} noise points")
    else:
        print(f"Clustering with KMeans (k={n_clusters})...")
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(embeddings)
        n_clusters_found = n_clusters

    # Build cluster info
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        if label >= 0:  # Skip noise points (-1)
            clusters[label].append(idx)

    # Extract keywords per cluster using TF-IDF
    print("Extracting cluster keywords...")
    vectorizer = TfidfVectorizer(
        max_df=0.8,
        min_df=2,
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 3),
        token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z-]+\b',
    )

    # Fit on all docs
    vectorizer.fit(all_docs)
    feature_names = vectorizer.get_feature_names_out()

    topics = []
    years = sorted(set(doc_years))

    for cluster_id in sorted(clusters.keys()):
        indices = clusters[cluster_id]
        cluster_docs = [all_docs[i] for i in indices]
        cluster_years = [doc_years[i] for i in indices]

        # Get TF-IDF for this cluster
        cluster_tfidf = vectorizer.transform(cluster_docs)
        # Average TF-IDF across cluster documents
        mean_tfidf = np.asarray(cluster_tfidf.mean(axis=0)).flatten()
        # Top keywords
        top_indices = mean_tfidf.argsort()[-15:][::-1]
        top_words = [feature_names[i] for i in top_indices]

        # Compute year distribution
        year_counts = Counter(cluster_years)

        # Build topic info
        topic = {
            "id": cluster_id,
            "words": top_words,
            "label": " / ".join(top_words[:3]),
            "size": len(indices),
            "counts": {y: year_counts.get(y, 0) for y in years},
            "papers": [doc_papers[i] for i in indices],  # All papers in cluster
        }

        # Compute trend score
        topic["total"] = sum(topic["counts"].values())
        topic["score"] = compute_trend_score(topic["counts"], years)

        topics.append(topic)

    # Sort by trend score
    topics.sort(key=lambda x: x["score"], reverse=True)

    return topics, papers_by_year


def make_sparkline(counts: dict, years: list[int]) -> str:
    """Create a simple ASCII sparkline showing trend over years (one char per year)."""
    values = [counts.get(y, 0) for y in years]
    if not values or max(values) == 0:
        return "─" * len(years)

    # Normalize to 0-8 range for block characters
    max_val = max(values)
    blocks = " ▁▂▃▄▅▆▇█"

    return "".join(blocks[min(8, int(v / max_val * 8))] if max_val > 0 else "─" for v in values)


def print_embedding_discovery_results(
    topics: list, years: list[int], top_n: int = 15, show_all: bool = False,
    show_years: bool = False, filter_keywords: list[str] | None = None
):
    """Print embedding-based discovered topics with trends."""

    # Filter topics if filter_keywords provided
    if filter_keywords:
        def matches_filter(t):
            text = (t["label"] + " " + " ".join(t["words"])).lower()
            return any(kw.lower() in text for kw in filter_keywords)

        topics = [t for t in topics if matches_filter(t)]
        print("\n" + "=" * 140)
        print(f"FILTERED TOPICS (matching: {', '.join(filter_keywords)})")
        print("=" * 140)

        # When filtering, show all matching topics with year details
        for i, t in enumerate(sorted(topics, key=lambda x: x["score"], reverse=True), 1):
            label = t["label"][:38]
            keywords = ", ".join(t["words"][:10])[:80]
            sparkline = make_sparkline(t["counts"], years)
            spark_padded = f"{sparkline:<15}"
            print(f"\n{i}. {t['label']}")
            print(f"   Trend: {t['score']:>+.3f}  Size: {t['size']}  Sparkline: {spark_padded}")
            print(f"   Keywords: {keywords}")

            # Always show year-by-year for filtered results
            year_str = "   Years:  " + "  ".join(f"{y}" for y in years)
            count_str = "   Count:  " + "  ".join(f"{t['counts'].get(y, 0):>4}" for y in years)
            print(year_str)
            print(count_str)

            # Show papers
            papers = t.get("papers", [])
            papers_by_year = {}
            for p in papers:
                y = p.get("year")
                if y:
                    # Normalize to int for consistent lookup
                    papers_by_year.setdefault(int(y), []).append(p)

            print(f"\n   Papers ({len(papers)} total):")
            for y in years:
                year_papers = papers_by_year.get(y, [])
                if year_papers:
                    print(f"     {y} ({len(year_papers)}):")
                    for p in year_papers:
                        title = p.get("title", "")
                        print(f"       • {title}")

        print(f"\n{'─'*140}")
        print(f"Found {len(topics)} matching clusters")
        return

    print("\n" + "=" * 140)
    print("DISCOVERED TOPICS (Embedding-based clustering, sorted by trend)")
    print("=" * 140)

    # Split into rising and declining
    rising = [t for t in topics if t["score"] > 0.01]
    declining = [t for t in topics if t["score"] < -0.01]
    stable = [t for t in topics if -0.01 <= t["score"] <= 0.01]

    # Determine how many to show
    n_rising = len(rising) if show_all else min(top_n, len(rising))
    n_declining = len(declining) if show_all else min(top_n, len(declining))
    n_stable = len(stable) if show_all else min(5, len(stable))

    def print_topic_row(i: int, t: dict, show_year_counts: bool):
        label = t["label"][:38]
        keywords = ", ".join(t["words"][3:10])[:60]
        sparkline = make_sparkline(t["counts"], years)
        # Sparkline width = number of years, pad to 12 for alignment
        spark_padded = f"{sparkline:<12}"
        print(f"{i:<3} {t['score']:>+7.3f} {t['size']:>5} {spark_padded} {label:<40} {keywords:<60}")

        if show_year_counts:
            # Show year-by-year counts
            year_str = "      " + " ".join(f"{y%100:>4}" for y in years)
            count_str = "      " + " ".join(f"{t['counts'].get(y, 0):>4}" for y in years)
            print(year_str)
            print(count_str)

            # Show papers grouped by year
            papers = t.get("papers", [])
            papers_by_year = {}
            for p in papers:
                y = p.get("year")
                if y:
                    papers_by_year.setdefault(y, []).append(p)

            print(f"\n      Papers ({len(papers)} total):")
            for y in years:
                year_papers = papers_by_year.get(y, [])
                if year_papers:
                    print(f"        {y} ({len(year_papers)}):")
                    for p in year_papers:
                        title = p.get("title", "")
                        print(f"          • {title}")
            print()

    width = 140
    print(f"\n{'─'*width}")
    print(f"RISING TOPICS" + (f" (showing {n_rising} of {len(rising)})" if n_rising < len(rising) else ""))
    print(f"{'─'*width}")
    header = f"{'#':<3} {'Trend':>7} {'Size':>5} {'Sparkline':<12} {'Label':<40} {'Top Keywords':<60}"
    print(header)
    print("-" * width)

    for i, t in enumerate(rising[:n_rising], 1):
        print_topic_row(i, t, show_years)

    if rising and not show_all and not show_years:
        # Show sample papers for top rising topic
        print(f"\n  Sample papers from top rising topic ({rising[0]['label'][:50]}):")
        for p in rising[0].get("papers", [])[:3]:
            print(f"    • {p['title'][:100]}...")

    print(f"\n{'─'*width}")
    print(f"DECLINING TOPICS" + (f" (showing {n_declining} of {len(declining)})" if n_declining < len(declining) else ""))
    print(f"{'─'*width}")
    print(header)
    print("-" * width)

    # Show most declining first
    declining_to_show = declining[-n_declining:] if n_declining else []
    for i, t in enumerate(reversed(declining_to_show), 1):
        print_topic_row(i, t, show_years)

    if stable:
        print(f"\n{'─'*width}")
        print(f"STABLE TOPICS ({len(stable)} topics with minimal trend)" + (f" - showing {n_stable}" if n_stable < len(stable) else ""))
        print(f"{'─'*width}")
        if show_years:
            # Show detailed view with papers
            print(header)
            print("-" * width)
            for i, t in enumerate(stable[:n_stable], 1):
                print_topic_row(i, t, show_years)
        else:
            for t in stable[:n_stable]:
                sparkline = make_sparkline(t["counts"], years)
                keywords = ", ".join(t["words"][3:8])[:50]
                print(f"  • {t['label']:<40} (n={t['size']:>4}) {sparkline:<12} {keywords}")

    # Summary stats
    total_papers = sum(t["size"] for t in topics)
    print(f"\n{'─'*width}")
    print(f"Summary: {len(topics)} clusters covering {total_papers} papers")
    print(f"Rising: {len(rising)}, Declining: {len(declining)}, Stable: {len(stable)}")
    print(f"Years: {years[0]}-{years[-1]} (sparklines show trend over time: ▁▂▃▄▅▆▇█)")


def save_embedding_discovery_csv(topics: list, years: list[int], output_path: Path):
    """Save embedding-based discovery results to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = ["Cluster_ID", "Label", "Trend_Score", "Size", "Keywords"] + [str(y) for y in years]
        writer.writerow(header)

        for t in topics:
            row = [
                t["id"],
                t["label"],
                f"{t['score']:.4f}",
                t["size"],
                "; ".join(t["words"][:10]),
            ]
            row += [t["counts"].get(y, 0) for y in years]
            writer.writerow(row)

    print(f"\nEmbedding topic results saved to {output_path}")


def save_clusters_json(topics: list, years: list[int], output_path: Path):
    """Save clusters with full paper lists to JSON."""
    def to_native(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        return obj

    output = {
        "years": [int(y) for y in years],
        "clusters": []
    }

    for t in topics:
        cluster = {
            "id": to_native(t["id"]),
            "label": t["label"],
            "score": to_native(t["score"]),
            "size": to_native(t["size"]),
            "keywords": t["words"][:20],
            "counts": {str(y): to_native(t["counts"].get(y, 0)) for y in years},
            "papers": []
        }

        # Add papers with year grouping
        papers_by_year = {}
        for paper in t.get("papers", []):
            year = to_native(paper.get("year", 0))
            if year not in papers_by_year:
                papers_by_year[year] = []
            papers_by_year[year].append({
                "title": paper.get("title", ""),
                "venue": paper.get("venue", ""),
                "year": year,
                "doi": paper.get("doi", ""),
            })

        cluster["papers_by_year"] = {
            str(y): papers_by_year.get(y, [])
            for y in sorted(papers_by_year.keys())
        }
        cluster["papers"] = [
            {"title": p.get("title", ""), "venue": p.get("venue", ""), "year": to_native(p.get("year", 0))}
            for p in t.get("papers", [])
        ]

        output["clusters"].append(cluster)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Cluster data with papers saved to {output_path}")


def print_discovery_results(
    rising: list,
    declining: list,
    years: list[int]
):
    """Print discovered trends."""
    print("\n" + "=" * 80)
    print("TOP RISING TOPICS (emerging)")
    print("=" * 80)
    print(f"{'Topic':<40} {'Score':>8} {'Total':>6} {'First':>6} {'Recent':>8}")
    print("-" * 70)

    for t in rising[:20]:
        recent = sum(t["counts"].get(y, 0) for y in years[-3:])
        print(f"{t['ngram']:<40} {t['score']:>8.3f} {t['total']:>6} {t['first_year']:>6} {recent:>8}")

    print("\n" + "=" * 80)
    print("TOP DECLINING TOPICS (fading)")
    print("=" * 80)
    print(f"{'Topic':<40} {'Score':>8} {'Total':>6} {'Peak':>6} {'Recent':>8}")
    print("-" * 70)

    for t in declining[:20]:
        # Find peak year
        peak_year = max(t["counts"].keys(), key=lambda y: t["counts"][y])
        recent = sum(t["counts"].get(y, 0) for y in years[-3:])
        print(f"{t['ngram']:<40} {t['score']:>8.3f} {t['total']:>6} {peak_year:>6} {recent:>8}")


def save_discovery_csv(
    rising: list,
    declining: list,
    years: list[int],
    output_path: Path
):
    """Save discovery results to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header with years
        header = ["Topic", "Trend", "Score", "Total"] + [str(y) for y in years]
        writer.writerow(header)

        for t in rising:
            row = [t["ngram"], "rising", f"{t['score']:.4f}", t["total"]]
            row += [t["counts"].get(y, 0) for y in years]
            writer.writerow(row)

        for t in declining:
            row = [t["ngram"], "declining", f"{t['score']:.4f}", t["total"]]
            row += [t["counts"].get(y, 0) for y in years]
            writer.writerow(row)

    print(f"\nDiscovery results saved to {output_path}")


def cmd_discover(args):
    """Handle 'discover' subcommand: automatic topic discovery."""
    years = list(range(args.start_year, args.end_year + 1))

    print(f"Analyzing publications from {args.start_year} to {args.end_year}")
    print(f"Venues: {', '.join(DEFAULT_VENUES.keys())}")

    # Show DBLP cache status
    dblp_files, dblp_papers = get_dblp_cache_stats()
    if dblp_files > 0:
        print(f"DBLP cache: {dblp_files} venue/year files ({dblp_papers} papers)")

    # Set default output path
    if args.output is None:
        args.output = Path("data/dblp_trends.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Show abstract cache status if using abstracts
    if args.use_abstracts:
        total, with_abs = get_cache_stats()
        print(f"Abstract cache: {total} entries ({with_abs} with abstracts)")

    if args.method == "ngram":
        # Simple n-gram based discovery
        print("Mode: Topic Discovery (n-gram analysis)")

        rising, declining, _ = discover_trends(
            args.start_year, args.end_year, DEFAULT_VENUES,
            top_n=args.top_n, min_total_count=args.min_count,
            verbose=not args.quiet
        )

        print_discovery_results(rising, declining, years)
        save_discovery_csv(rising, declining, years, args.output)

    elif args.method == "embedding":
        # Embedding-based clustering
        if not EMBEDDINGS_AVAILABLE:
            print("ERROR: sentence-transformers required for embedding-based discovery.")
            print("Install with: pip install sentence-transformers")
            sys.exit(1)

        print(f"Mode: Topic Discovery (Embedding-based clustering)")
        if args.use_abstracts:
            print("Fetching abstracts from multiple sources (cached to disk)")

        use_hdbscan = not args.kmeans
        topics, _ = discover_topics_embedding(
            args.start_year, args.end_year,
            n_clusters=args.n_topics,
            use_abstracts=args.use_abstracts,
            verbose=not args.quiet,
            max_workers=args.workers,
            min_cluster_size=args.min_cluster_size,
            use_hdbscan=use_hdbscan,
            model_name=args.embedding_model,
        )

        # Parse filter keywords if provided
        filter_kws = None
        if args.filter:
            filter_kws = [k.strip() for k in args.filter.split(",")]

        print_embedding_discovery_results(
            topics, years, top_n=args.top_n, show_all=args.show_all,
            show_years=args.show_years, filter_keywords=filter_kws
        )
        save_embedding_discovery_csv(topics, years, args.output)

        # Also save JSON with paper lists
        json_output = args.output.with_suffix(".json")
        save_clusters_json(topics, years, json_output)

    else:
        # NLP-based discovery (NMF or LDA)
        if not SKLEARN_AVAILABLE:
            print("ERROR: scikit-learn required for NLP discovery.")
            print("Install with: pip install scikit-learn")
            sys.exit(1)

        print(f"Mode: Topic Discovery (NLP with {args.method.upper()})")
        if args.use_abstracts:
            print("Fetching abstracts from multiple sources (cached to disk)")

        topics, _ = discover_topics_nlp(
            args.start_year, args.end_year,
            n_topics=args.n_topics,
            use_abstracts=args.use_abstracts,
            method=args.method,
            verbose=not args.quiet,
            max_workers=args.workers,
        )

        print_nlp_discovery_results(topics, years)
        save_nlp_discovery_csv(topics, years, args.output)


def cmd_search(args):
    """Handle 'search' subcommand: keyword-based search."""
    # Determine keywords
    if args.preset:
        keywords = PRESETS[args.preset]
    elif args.keywords_file:
        keywords = load_keywords_file(args.keywords_file)
    else:
        keywords = [k.strip() for k in args.keywords.split(",")]

    if not keywords:
        print("ERROR: No keywords specified")
        sys.exit(1)

    print(f"Analyzing publications from {args.start_year} to {args.end_year}")
    print(f"Venues: {', '.join(DEFAULT_VENUES.keys())}")
    print(f"Mode: Keyword Search")
    print(f"Keywords ({len(keywords)}): {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")

    # Set default output path
    if args.output is None:
        args.output = Path("data/dblp_stats.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    results = analyze_trends(
        keywords, args.start_year, args.end_year,
        DEFAULT_VENUES, verbose=not args.quiet
    )

    print_summary(results, keywords)
    save_csv(results, args.output)

    if args.papers_list:
        save_papers_list(results, args.papers_list)


def cmd_report_missing(args):
    """Handle 'report-missing' subcommand: find papers without abstracts."""
    report_missing_abstracts(args.start_year, args.end_year)


def cmd_cache(args):
    """Handle 'cache' subcommand: cache management."""
    if args.action == "clear":
        clear_abstract_cache()
        print("Abstract cache cleared.")
    elif args.action == "clear-misses":
        deleted = clear_cache_misses()
        print(f"Cleared {deleted} cache-miss entries (will retry these papers).")
    elif args.action == "clear-dblp":
        clear_dblp_cache()
        print("DBLP paper cache cleared.")
    elif args.action == "stats":
        # Abstract cache stats
        total, with_abs = get_cache_stats()
        miss_count = total - with_abs
        print(f"Abstract cache:")
        print(f"  Location: {get_cache_dir()}")
        print(f"  Total entries: {total}")
        print(f"  With abstracts: {with_abs}")
        print(f"  Cache misses: {miss_count}")
        if total > 0:
            print(f"  Hit rate: {100 * with_abs / total:.1f}%")

        # DBLP cache stats
        dblp_files, dblp_papers = get_dblp_cache_stats()
        print(f"\nDBLP paper cache:")
        print(f"  Location: {get_dblp_cache_dir()}")
        print(f"  Venue/year files: {dblp_files}")
        print(f"  Total papers: {dblp_papers}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze publication trends in top security venues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s discover --use-abstracts --start-year 2010
  %(prog)s discover --method ngram --start-year 2015
  %(prog)s search --preset code-reuse
  %(prog)s search --keywords "ROP,CFI,ASLR" --start-year 2015
  %(prog)s report-missing --start-year 2010
  %(prog)s cache stats
  %(prog)s cache clear-misses
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -------------------------------------------------------------------------
    # discover: automatic topic discovery
    # -------------------------------------------------------------------------
    discover_parser = subparsers.add_parser(
        "discover",
        help="Automatically discover rising/declining research topics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --use-abstracts --start-year 2010
  %(prog)s --method embedding --use-abstracts  (recommended)
  %(prog)s --method embedding --kmeans --n-topics 30
  %(prog)s --method nmf --n-topics 30
  %(prog)s --method ngram --min-count 10
        """
    )
    discover_parser.add_argument(
        "--start-year", type=int, default=2010,
        help="Start year for analysis (default: 2010)"
    )
    discover_parser.add_argument(
        "--end-year", type=int, default=2024,
        help="End year for analysis (default: 2024)"
    )
    discover_parser.add_argument(
        "--output", type=Path, default=None,
        help="Output CSV file path (default: data/dblp_trends.csv)"
    )
    discover_parser.add_argument(
        "--method", choices=["ngram", "nmf", "lda", "embedding"], default="embedding",
        help="Topic discovery method: embedding (recommended), nmf, lda, ngram (default: embedding)"
    )
    discover_parser.add_argument(
        "--n-topics", type=int, default=20,
        help="Number of topics/clusters (default: 20, ignored with HDBSCAN)"
    )
    discover_parser.add_argument(
        "--top-n", type=int, default=15,
        help="Number of top rising/declining topics to show (default: 15)"
    )
    discover_parser.add_argument(
        "--show-all", action="store_true",
        help="Show all discovered topics (overrides --top-n)"
    )
    discover_parser.add_argument(
        "--show-years", action="store_true",
        help="Show year-by-year paper counts for each topic"
    )
    discover_parser.add_argument(
        "--filter", type=str, default=None,
        help="Filter topics by keyword in label/keywords (comma-separated, e.g. 'cfi,memory,aslr')"
    )
    discover_parser.add_argument(
        "--min-count", type=int, default=5,
        help="Minimum total occurrences for a topic (default: 5, ngram only)"
    )
    discover_parser.add_argument(
        "--use-abstracts", action="store_true",
        help="Fetch abstracts from multiple sources (cached to disk)"
    )
    discover_parser.add_argument(
        "--workers", type=int, default=10,
        help="Number of parallel workers for abstract fetching (default: 10)"
    )
    discover_parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    # Embedding-specific options
    discover_parser.add_argument(
        "--min-cluster-size", type=int, default=15,
        help="Minimum papers per cluster (default: 15, embedding/HDBSCAN only)"
    )
    discover_parser.add_argument(
        "--kmeans", action="store_true",
        help="Use KMeans instead of HDBSCAN (embedding method only)"
    )
    discover_parser.add_argument(
        "--embedding-model", type=str, default="all-MiniLM-L6-v2",
        help="Sentence transformer model (default: all-MiniLM-L6-v2)"
    )
    discover_parser.set_defaults(func=cmd_discover)

    # -------------------------------------------------------------------------
    # search: keyword-based search
    # -------------------------------------------------------------------------
    search_parser = subparsers.add_parser(
        "search",
        help="Search for specific keywords in paper titles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Preset keyword sets:
  code-reuse    ROP, CFI, ASLR, gadgets, shadow stack, etc.
  fuzzing       fuzzing, AFL, coverage-guided, etc.
  memory-safety buffer overflow, use-after-free, etc.

Examples:
  %(prog)s --preset code-reuse
  %(prog)s --keywords "ROP,CFI,ASLR"
  %(prog)s --keywords-file my_keywords.txt --start-year 2015
        """
    )
    kw_group = search_parser.add_mutually_exclusive_group(required=True)
    kw_group.add_argument(
        "--keywords", type=str,
        help="Comma-separated list of keywords to search for"
    )
    kw_group.add_argument(
        "--keywords-file", type=Path,
        help="File containing keywords (one per line)"
    )
    kw_group.add_argument(
        "--preset", choices=list(PRESETS.keys()),
        help="Use a preset keyword set"
    )
    search_parser.add_argument(
        "--start-year", type=int, default=2010,
        help="Start year for analysis (default: 2010)"
    )
    search_parser.add_argument(
        "--end-year", type=int, default=2024,
        help="End year for analysis (default: 2024)"
    )
    search_parser.add_argument(
        "--output", type=Path, default=None,
        help="Output CSV file path (default: data/dblp_stats.csv)"
    )
    search_parser.add_argument(
        "--papers-list", type=Path, default=None,
        help="Output file for list of matching paper titles"
    )
    search_parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    search_parser.set_defaults(func=cmd_search)

    # -------------------------------------------------------------------------
    # report-missing: find papers without abstracts
    # -------------------------------------------------------------------------
    missing_parser = subparsers.add_parser(
        "report-missing",
        help="Report papers that have no abstract in cache"
    )
    missing_parser.add_argument(
        "--start-year", type=int, default=2010,
        help="Start year for analysis (default: 2010)"
    )
    missing_parser.add_argument(
        "--end-year", type=int, default=2024,
        help="End year for analysis (default: 2024)"
    )
    missing_parser.set_defaults(func=cmd_report_missing)

    # -------------------------------------------------------------------------
    # cache: cache management
    # -------------------------------------------------------------------------
    cache_parser = subparsers.add_parser(
        "cache",
        help="Manage the abstract cache"
    )
    cache_parser.add_argument(
        "action", choices=["clear", "clear-misses", "clear-dblp", "stats"],
        help="Cache action: clear (delete abstracts), clear-misses (retry failed), clear-dblp (re-fetch papers), stats"
    )
    cache_parser.set_defaults(func=cmd_cache)

    # Parse and dispatch
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
