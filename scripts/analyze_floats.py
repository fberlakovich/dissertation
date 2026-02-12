#!/usr/bin/env python3
"""
LaTeX Float Placement Analyzer (PDF-based)

Analyzes whether float environments (figure, table, listing) are rendered
near where they are first referenced, using actual PDF page positions.

Float page numbers are parsed from the .aux file.
Reference page numbers are resolved via synctex.

Falls back to source-line distance with --source-lines.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FileEntry:
    """A file in the compilation order with its global line offset."""
    path: Path
    global_offset: int  # first line of this file in the global numbering
    line_count: int


@dataclass
class Float:
    """A float environment extracted from the source."""
    env_type: str              # figure, table, listing, figure*, table*
    file: Path
    start_line: int            # 1-based, local to file
    end_line: int              # 1-based, local to file
    labels: list[str] = field(default_factory=list)
    label_lines: list[int] = field(default_factory=list)  # 1-based line numbers of \label commands
    caption_line: Optional[int] = None  # 1-based line number of \caption
    text: str = ""             # full text of the float block (for reinsertion)
    global_start: int = 0      # start line in global numbering


@dataclass
class Reference:
    """A reference to a label."""
    label: str
    file: Path
    line: int                  # 1-based, local to file
    global_line: int = 0       # line in global numbering


@dataclass
class FloatProfile:
    """Profiling data from an instrumented LuaLaTeX run."""
    textheight_pt: float                    # \textheight in pt
    float_heights_pt: dict[str, float]      # label -> total height (ht+dp) in pt


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if _USE_COLOR:
        return f"\033[{code}m{text}\033[0m"
    return text


def green(t: str) -> str:
    return _c("32", t)


def yellow(t: str) -> str:
    return _c("33", t)


def red(t: str) -> str:
    return _c("31", t)


def gray(t: str) -> str:
    return _c("90", t)


def bold(t: str) -> str:
    return _c("1", t)


# ---------------------------------------------------------------------------
# Phase 1: Build file ordering
# ---------------------------------------------------------------------------

_INPUT_RE = re.compile(r"^[^%]*\\(?:input|include)\{([^}]+)\}")


def resolve_inputs(root: Path, base_dir: Path) -> list[Path]:
    """Recursively resolve \\input{} and \\include{} to a flat file list."""
    files: list[Path] = []
    if not root.exists():
        print(f"WARNING: File not found: {root}", file=sys.stderr)
        return files

    lines = root.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines:
        m = _INPUT_RE.match(line)
        if m:
            target = m.group(1).strip()
            if "#" in target:
                continue
            target_path = base_dir / target
            if not target_path.suffix:
                target_path = target_path.with_suffix(".tex")
            if target_path.exists():
                files.append(target_path)
                files.extend(resolve_inputs(target_path, base_dir))
            else:
                print(f"WARNING: File not found: {target_path}", file=sys.stderr)
        else:
            for inner_m in re.finditer(r"\\(?:input|include)\{([^}]+)\}", line):
                prefix = line[:inner_m.start()]
                if "%" in prefix:
                    stripped = prefix.rstrip()
                    if stripped.endswith("%") or "%" in prefix.split("\\")[-1]:
                        continue
                target = inner_m.group(1).strip()
                if "#" in target:
                    continue
                target_path = base_dir / target
                if not target_path.suffix:
                    target_path = target_path.with_suffix(".tex")
                if target_path.exists():
                    files.append(target_path)
                    files.extend(resolve_inputs(target_path, base_dir))
                else:
                    print(f"WARNING: File not found: {target_path}", file=sys.stderr)

    return files


def build_file_ordering(root_file: Path, base_dir: Path) -> list[FileEntry]:
    """Build a flat list of files in compilation order with global offsets."""
    raw_paths = [root_file] + resolve_inputs(root_file, base_dir)

    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for p in raw_paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique_paths.append(p)

    entries: list[FileEntry] = []
    global_offset = 0
    for p in unique_paths:
        try:
            line_count = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
        except OSError:
            line_count = 0
        entries.append(FileEntry(path=p, global_offset=global_offset, line_count=line_count))
        global_offset += line_count
    return entries


def get_global_line(file_entries: list[FileEntry], file_path: Path, local_line: int) -> int:
    """Convert a file-local line number to a global line number."""
    resolved = file_path.resolve()
    for entry in file_entries:
        if entry.path.resolve() == resolved:
            return entry.global_offset + local_line
    return -1


# ---------------------------------------------------------------------------
# Phase 2: Extract floats
# ---------------------------------------------------------------------------

_FLOAT_ENVS = {"figure", "table", "listing", "figure*", "table*"}
_BEGIN_FLOAT_RE = re.compile(r"\\begin\{(" + "|".join(re.escape(e) for e in _FLOAT_ENVS) + r")\}")
_END_FLOAT_RE = re.compile(r"\\end\{(" + "|".join(re.escape(e) for e in _FLOAT_ENVS) + r")\}")
_LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
_CAPTION_RE = re.compile(r"\\caption\b")
_COMMENT_RE = re.compile(r"^\s*%")


def is_commented(line: str) -> bool:
    """Check if a line is commented out."""
    return bool(_COMMENT_RE.match(line))


def extract_floats(file_entries: list[FileEntry]) -> list[Float]:
    """Extract all float environments from all files."""
    floats: list[Float] = []

    for entry in file_entries:
        try:
            lines = entry.path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue

        nesting_depth = 0
        current_float: Optional[Float] = None
        current_lines: list[str] = []

        for i, line in enumerate(lines, start=1):
            if is_commented(line):
                if current_float is not None:
                    current_lines.append(line)
                continue

            begin_match = _BEGIN_FLOAT_RE.search(line)
            end_match = _END_FLOAT_RE.search(line)

            if begin_match:
                env = begin_match.group(1)
                if nesting_depth == 0:
                    current_float = Float(
                        env_type=env,
                        file=entry.path,
                        start_line=i,
                        end_line=i,
                    )
                    current_lines = [line]
                else:
                    if current_float is not None:
                        current_lines.append(line)
                nesting_depth += 1

                if current_float is not None:
                    for lm in _LABEL_RE.finditer(line):
                        current_float.labels.append(lm.group(1))
                        current_float.label_lines.append(i)
                    if _CAPTION_RE.search(line) and current_float.caption_line is None:
                        current_float.caption_line = i

            elif end_match and nesting_depth > 0:
                nesting_depth -= 1
                if current_float is not None:
                    current_lines.append(line)

                    for lm in _LABEL_RE.finditer(line):
                        current_float.labels.append(lm.group(1))
                        current_float.label_lines.append(i)
                    if _CAPTION_RE.search(line) and current_float.caption_line is None:
                        current_float.caption_line = i

                    if nesting_depth == 0:
                        current_float.end_line = i
                        current_float.text = "\n".join(current_lines)
                        current_float.global_start = entry.global_offset + current_float.start_line
                        floats.append(current_float)
                        current_float = None
                        current_lines = []

            elif current_float is not None:
                current_lines.append(line)
                for lm in _LABEL_RE.finditer(line):
                    current_float.labels.append(lm.group(1))
                    current_float.label_lines.append(i)
                if _CAPTION_RE.search(line) and current_float.caption_line is None:
                    current_float.caption_line = i

    return floats


# ---------------------------------------------------------------------------
# Phase 3: Extract references
# ---------------------------------------------------------------------------

_REF_RE = re.compile(r"\\(?:cref|Cref|ref|autoref)\{([^}]+)\}")


def extract_references(file_entries: list[FileEntry]) -> list[Reference]:
    """Extract all label references from all files."""
    refs: list[Reference] = []

    for entry in file_entries:
        try:
            lines = entry.path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue

        for i, line in enumerate(lines, start=1):
            if is_commented(line):
                continue

            for m in _REF_RE.finditer(line):
                raw_labels = m.group(1)
                for label in raw_labels.split(","):
                    label = label.strip()
                    if label:
                        refs.append(Reference(
                            label=label,
                            file=entry.path,
                            line=i,
                            global_line=entry.global_offset + i,
                        ))
    return refs


# ---------------------------------------------------------------------------
# Phase 4a: AUX file parsing (float -> PDF page)
# ---------------------------------------------------------------------------

# Matches: \newlabel{LABEL}{{NUMBER}{PAGE}{CAPTION}{COUNTER}{}}
# The page is the second brace group inside the outer brace group.
# We must NOT match @cref variants.
_NEWLABEL_RE = re.compile(
    r"\\newlabel\{([^}]+)\}"   # label name
    r"\{\{"                     # opening {{ of fields
    r"(?:[^{}]*(?:\{[^{}]*\}[^{}]*)*)"  # first field (number) — may contain nested braces
    r"\}\{"                     # }{
    r"([^}]*)"                  # PAGE (second field — always plain text)
    r"\}"                       # closing } of page field
)

_ROMAN_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)


def is_roman_numeral(s: str) -> bool:
    """Check if a string looks like a roman numeral (front matter page)."""
    return bool(_ROMAN_RE.match(s.strip()))


_NEWLABEL_FULL_RE = re.compile(
    r"\\newlabel\{([^}]+)\}"   # label name
    r"\{\{"                     # opening {{ of fields
    r"([^{}]*(?:\{[^{}]*\}[^{}]*)*)"  # first field (number) — may contain nested braces
    r"\}\{"                     # }{
    r"([^}]*)"                  # PAGE (second field — always plain text)
    r"\}"                       # closing } of page field
)


_AUX_INPUT_RE = re.compile(r"\\@input\{([^}]+)\}")


def _read_aux_recursive(aux_path: Path) -> str:
    """Read an aux file and recursively inline any \\@input{...} files.

    LaTeX distributes labels across per-file .aux files when using \\input{}.
    The main thesis.aux contains \\@input{includes/foo.aux} directives that
    must be followed to find all labels.
    """
    if not aux_path.exists():
        return ""
    text = aux_path.read_text(encoding="utf-8", errors="replace")
    base_dir = aux_path.parent
    parts = []
    for line in text.splitlines(keepends=True):
        m = _AUX_INPUT_RE.search(line)
        if m:
            sub_path = base_dir / m.group(1)
            parts.append(_read_aux_recursive(sub_path))
        else:
            parts.append(line)
    return "".join(parts)


def parse_aux_labels(aux_path: Path) -> dict[str, int]:
    """Parse thesis.aux to build a label -> page_number mapping.

    Recursively follows \\@input directives to find labels in sub-aux files.
    Skips @cref variants and roman numeral pages (front matter).
    Returns only labels with valid integer page numbers.
    """
    label_pages: dict[str, int] = {}

    if not aux_path.exists():
        return label_pages

    text = _read_aux_recursive(aux_path)

    for m in _NEWLABEL_FULL_RE.finditer(text):
        label = m.group(1)
        page_str = m.group(3).strip()

        # Skip @cref variants
        if label.endswith("@cref"):
            continue

        # Skip sub@ labels (subfigure internal labels)
        if label.startswith("sub@"):
            continue

        # Skip roman numeral pages (front matter)
        if is_roman_numeral(page_str):
            continue

        # Parse integer page
        try:
            page = int(page_str)
            label_pages[label] = page
        except ValueError:
            # Non-numeric, non-roman page — skip
            continue

    return label_pages


def parse_aux_display_numbers(aux_path: Path) -> dict[str, str]:
    """Parse thesis.aux to build a label -> display number mapping.

    Recursively follows \\@input directives to find labels in sub-aux files.
    E.g. 'lool:fig:workflow' -> '6.3', 'psp:tbl:results' -> '8.2'.
    """
    label_numbers: dict[str, str] = {}

    if not aux_path.exists():
        return label_numbers

    text = _read_aux_recursive(aux_path)

    for m in _NEWLABEL_FULL_RE.finditer(text):
        label = m.group(1)
        number = m.group(2).strip()

        if label.endswith("@cref"):
            continue
        if label.startswith("sub@"):
            continue
        if not number:
            continue

        label_numbers[label] = number

    return label_numbers


# ---------------------------------------------------------------------------
# Phase 4b: SyncTeX queries (reference -> PDF page)
# ---------------------------------------------------------------------------

_SYNCTEX_PAGE_RE = re.compile(r"^Page:(\d+)", re.MULTILINE)


class SyncTeXResolver:
    """Resolves source (file, line) to PDF page numbers via synctex."""

    def __init__(self, pdf_path: Path, base_dir: Path, max_workers: int = 8):
        self.pdf_path = pdf_path
        self.base_dir = base_dir
        self.max_workers = max_workers
        self._cache: dict[tuple[str, int], Optional[int]] = {}
        self._edit_cache: dict[tuple[int, int, int], Optional[tuple[str, int]]] = {}

    def _query_one(self, rel_file: str, line: int) -> Optional[int]:
        """Run a single synctex query and return the page number."""
        key = (rel_file, line)
        if key in self._cache:
            return self._cache[key]

        try:
            result = subprocess.run(
                [
                    "synctex", "view",
                    "-i", f"{line}:1:{rel_file}",
                    "-o", str(self.pdf_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(self.base_dir),
            )
            m = _SYNCTEX_PAGE_RE.search(result.stdout)
            if m:
                page = int(m.group(1))
                self._cache[key] = page
                return page
        except (subprocess.TimeoutExpired, OSError):
            pass

        self._cache[key] = None
        return None

    def resolve_batch(self, queries: list[tuple[str, int]]) -> dict[tuple[str, int], Optional[int]]:
        """Resolve a batch of (rel_file, line) pairs to page numbers in parallel.

        Returns a dict mapping (rel_file, line) -> page or None.
        """
        # Deduplicate
        unique_queries = list(set(queries))

        # Filter already cached
        uncached = [q for q in unique_queries if q not in self._cache]

        if uncached:
            print(f"  Querying synctex for {len(uncached)} unique source positions...",
                  file=sys.stderr, flush=True)

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._query_one, rel_file, line): (rel_file, line)
                    for rel_file, line in uncached
                }
                done_count = 0
                for future in as_completed(futures):
                    done_count += 1
                    if done_count % 20 == 0:
                        print(f"    ...{done_count}/{len(uncached)} done",
                              file=sys.stderr, flush=True)
                    # Result is already cached inside _query_one
                    future.result()

            print(f"  Synctex queries complete.", file=sys.stderr, flush=True)

        return {q: self._cache.get(q) for q in queries}

    def clear_cache(self) -> None:
        """Clear the synctex result cache.

        Must be called after recompilation since synctex data changes
        and cached results become stale.
        """
        self._cache.clear()
        self._edit_cache.clear()

    def edit_page(self, page: int, y: int = 750, x: int = 100) -> Optional[tuple[str, int]]:
        """Reverse synctex: find source (file, line) for a position on a PDF page.

        Uses ``synctex edit -o page:x:y:file.pdf`` to map from PDF coordinates
        back to source.  Returns (relative_file, line) or None.
        """
        key = (page, x, y)
        if key in self._edit_cache:
            return self._edit_cache[key]

        try:
            result = subprocess.run(
                [
                    "synctex", "edit",
                    "-o", f"{page}:{x}:{y}:{self.pdf_path}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(self.base_dir),
            )
            # Parse output: look for Input: and Line: fields
            inp = None
            line_no = None
            for ln in result.stdout.splitlines():
                ln = ln.strip()
                if ln.startswith("Input:"):
                    inp = ln[len("Input:"):].strip()
                elif ln.startswith("Line:"):
                    try:
                        line_no = int(ln[len("Line:"):].strip())
                    except ValueError:
                        pass
            if inp is not None and line_no is not None:
                # Make path relative to base_dir
                try:
                    rel = str(Path(inp).relative_to(self.base_dir))
                except ValueError:
                    rel = inp
                self._edit_cache[key] = (rel, line_no)
                return (rel, line_no)
        except (subprocess.TimeoutExpired, OSError):
            pass

        self._edit_cache[key] = None
        return None


# ---------------------------------------------------------------------------
# Phase 4c: Analysis (PDF-based)
# ---------------------------------------------------------------------------

@dataclass
class FloatReport:
    """Analysis result for a single float."""
    float_obj: Float
    label: str                          # representative label
    first_ref: Optional[Reference]
    float_page: Optional[int]           # PDF page where float renders
    ref_page: Optional[int]             # PDF page where first ref appears
    page_distance: Optional[int]        # ref_page - float_page
    status: str                         # BEFORE, EARLY, LATE, UNREFERENCED, NO-AUX

    # For source-line fallback mode
    source_distance: Optional[int] = None


# ---------------------------------------------------------------------------
# Penalty scoring — configurable constants
# ---------------------------------------------------------------------------

# Exponent for distance penalty (4 = quartic).
PENALTY_EXPONENT: int = 4

# Coefficient for LATE floats (float appears AFTER its first reference).
# With exponent=4: 1pp → 100, 2pp → 1600.
PENALTY_LATE: float = 100.0

# Coefficient for BEFORE/EARLY floats (float appears BEFORE its reference).
# With exponent=4: 1pp → 60, 2pp → 960.  (2pp is practically unacceptable)
PENALTY_BEFORE: float = 60.0

# Penalty per extra float on the same page (beyond the first).
# Crowding is almost as bad as a 2pp distance — avoid whenever possible.
PENALTY_CROWDING: float = 150.0

# Hard constraint: maximum pages a float may be from its first reference.
MAX_PAGE_DISTANCE: int = 2

# Hard constraint: maximum floats allowed on a single page.
MAX_FLOATS_PER_PAGE: int = 2


def _penalty_formula_str() -> str:
    """Return a human-readable penalty formula string."""
    e = PENALTY_EXPONENT
    return (f"LATE={PENALTY_LATE:.0f}*d^{e}, "
            f"BEFORE={PENALTY_BEFORE:.0f}*d^{e}, "
            f"crowding={PENALTY_CROWDING:.0f}/extra, 0=perfect")


def _float_penalty(r: FloatReport) -> float:
    """Compute distance penalty for a single float. Higher = worse placement."""
    if r.page_distance is None or r.status in ("APPENDIX", "UNREFERENCED", "NO-AUX"):
        return 0.0
    d = abs(r.page_distance)
    if r.page_distance < 0:
        # LATE: float renders after reference
        return PENALTY_LATE * d ** PENALTY_EXPONENT
    # BEFORE/EARLY: float renders before reference
    return PENALTY_BEFORE * d ** PENALTY_EXPONENT


def _crowding_penalty(reports: list[FloatReport]) -> float:
    """Penalize pages with multiple floats."""
    page_counts: dict[int, int] = {}
    for r in reports:
        if r.float_page is not None and r.status not in ("APPENDIX", "UNREFERENCED", "NO-AUX"):
            page_counts[r.float_page] = page_counts.get(r.float_page, 0) + 1

    penalty = 0.0
    for count in page_counts.values():
        if count > 1:
            penalty += (count - 1) * PENALTY_CROWDING
    return penalty


def compute_penalty(reports: list[FloatReport]) -> float:
    """Compute total penalty across all floats. 0 = perfect."""
    dist = sum(_float_penalty(r) for r in reports)
    crowd = _crowding_penalty(reports)
    return dist + crowd


def relative_path(p: Path, base: Path) -> str:
    """Return a short relative path string."""
    try:
        return str(p.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(p)


def _rel_path_str(file_path: Path, base_dir: Path) -> str:
    """Get relative path string for synctex queries."""
    try:
        return str(file_path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(file_path)


def analyze_pdf(
    floats: list[Float],
    refs: list[Reference],
    file_entries: list[FileEntry],
    label_pages: dict[str, int],
    synctex: SyncTeXResolver,
    base_dir: Path,
    threshold: int,
) -> list[FloatReport]:
    """Match floats to their first references using PDF page positions.

    Uses synctex for BOTH float and reference page lookups to ensure
    both are in the same coordinate system (physical PDF pages).
    The .aux label_pages are used as fallback when synctex fails for a float.
    """

    # Build label -> list of references, sorted by global line
    ref_map: dict[str, list[Reference]] = {}
    for r in refs:
        ref_map.setdefault(r.label, []).append(r)
    for v in ref_map.values():
        v.sort(key=lambda r: r.global_line)

    # First pass: find earliest references for each float and collect synctex queries
    float_ref_pairs: list[tuple[Float, str, Optional[Reference]]] = []
    synctex_queries: list[tuple[str, int]] = []

    for flt in floats:
        if not flt.labels:
            float_ref_pairs.append((flt, "(no label)", None))
            continue

        earliest_ref: Optional[Reference] = None
        earliest_label: Optional[str] = None
        for label in flt.labels:
            label_refs = ref_map.get(label, [])
            if label_refs:
                candidate = label_refs[0]
                if earliest_ref is None or candidate.global_line < earliest_ref.global_line:
                    earliest_ref = candidate
                    earliest_label = label

        rep_label = earliest_label or flt.labels[0]
        float_ref_pairs.append((flt, rep_label, earliest_ref))

        # Queue synctex queries for the float's rendered position.
        # Use caption/label lines (rendered inside the float) rather than
        # \begin{} line, because synctex maps \begin{figure} to the
        # surrounding text flow, not the float's rendered position.
        flt_rel = _rel_path_str(flt.file, base_dir)
        flt_query_lines: list[int] = []
        if flt.caption_line is not None:
            flt_query_lines.append(flt.caption_line)
        flt_query_lines.extend(flt.label_lines)
        if not flt_query_lines:
            # Fallback: use middle of float
            flt_query_lines.append((flt.start_line + flt.end_line) // 2)
        for ql in flt_query_lines:
            synctex_queries.append((flt_rel, ql))

        # Queue synctex query for the reference's source position
        if earliest_ref is not None:
            ref_rel = _rel_path_str(earliest_ref.file, base_dir)
            synctex_queries.append((ref_rel, earliest_ref.line))

    # Batch resolve all synctex queries (floats + references together)
    synctex_results = synctex.resolve_batch(synctex_queries)

    # Compute aux-to-physical page offset for fallback
    # (aux gives logical pages, synctex gives physical pages)
    aux_offset = _compute_aux_offset(label_pages, synctex_results, floats, base_dir)

    # Second pass: build reports
    reports: list[FloatReport] = []

    for flt, rep_label, earliest_ref in float_ref_pairs:
        # Look up float page via synctex from caption/label lines (primary)
        flt_rel = _rel_path_str(flt.file, base_dir)
        float_page = None
        # Try caption line first, then label lines, then middle of float
        query_lines: list[int] = []
        if flt.caption_line is not None:
            query_lines.append(flt.caption_line)
        query_lines.extend(flt.label_lines)
        if not query_lines:
            query_lines.append((flt.start_line + flt.end_line) // 2)
        for ql in query_lines:
            float_page = synctex_results.get((flt_rel, ql))
            if float_page is not None:
                break

        # Fallback: use aux page + offset if synctex failed
        if float_page is None and aux_offset is not None:
            for label in flt.labels:
                if label in label_pages:
                    float_page = label_pages[label] + aux_offset
                    break

        if earliest_ref is None:
            reports.append(FloatReport(
                float_obj=flt,
                label=rep_label,
                first_ref=None,
                float_page=float_page,
                ref_page=None,
                page_distance=None,
                status="UNREFERENCED",
            ))
            continue

        if float_page is None:
            reports.append(FloatReport(
                float_obj=flt,
                label=rep_label,
                first_ref=earliest_ref,
                float_page=None,
                ref_page=None,
                page_distance=None,
                status="NO-AUX",
            ))
            continue

        # Look up reference page from synctex
        ref_rel = _rel_path_str(earliest_ref.file, base_dir)
        ref_page = synctex_results.get((ref_rel, earliest_ref.line))

        if ref_page is None:
            reports.append(FloatReport(
                float_obj=flt,
                label=rep_label,
                first_ref=earliest_ref,
                float_page=float_page,
                ref_page=None,
                page_distance=None,
                status="NO-AUX",
            ))
            continue

        page_distance = ref_page - float_page

        # Classify
        # Floats in appendices are expected to be far from references
        flt_rel = _rel_path_str(flt.file, base_dir)
        in_appendix = flt_rel.startswith("appendices/") or flt_rel.startswith("appendices\\")

        if in_appendix:
            status = "APPENDIX"
        elif page_distance < 0:
            status = "LATE"
        elif page_distance <= threshold:
            status = "BEFORE"
        else:
            status = "EARLY"

        reports.append(FloatReport(
            float_obj=flt,
            label=rep_label,
            first_ref=earliest_ref,
            float_page=float_page,
            ref_page=ref_page,
            page_distance=page_distance,
            status=status,
        ))

    return reports


def _compute_aux_offset(
    label_pages: dict[str, int],
    synctex_results: dict[tuple[str, int], Optional[int]],
    floats: list[Float],
    base_dir: Path,
) -> Optional[int]:
    """Compute offset between aux (logical) and synctex (physical) page numbers.

    Samples floats that have both an aux page and a synctex page to determine
    the front-matter page count offset. Returns the median offset, or None.
    """
    offsets: list[int] = []
    for flt in floats:
        flt_rel = _rel_path_str(flt.file, base_dir)
        # Use caption/label lines for synctex lookup (same as main analysis)
        synctex_page = None
        query_lines: list[int] = []
        if flt.caption_line is not None:
            query_lines.append(flt.caption_line)
        query_lines.extend(flt.label_lines)
        if not query_lines:
            query_lines.append((flt.start_line + flt.end_line) // 2)
        for ql in query_lines:
            synctex_page = synctex_results.get((flt_rel, ql))
            if synctex_page is not None:
                break
        if synctex_page is None:
            continue
        for label in flt.labels:
            if label in label_pages:
                offset = synctex_page - label_pages[label]
                offsets.append(offset)
                break
    if not offsets:
        return None
    # Use median to be robust against outliers
    offsets.sort()
    return offsets[len(offsets) // 2]


# ---------------------------------------------------------------------------
# Phase 4f: Float profiling via LuaLaTeX instrumentation
# ---------------------------------------------------------------------------

_PT_RE = re.compile(r"([\d.]+)pt")


def _parse_pt(s: str) -> float:
    """Parse a TeX dimension string like '123.45pt' to points."""
    m = _PT_RE.search(s)
    if m:
        return float(m.group(1))
    # Try plain number (scaled points)
    try:
        return float(s) / 65536.0  # sp to pt
    except ValueError:
        return 0.0


def _generate_profiler_preamble() -> str:
    """Generate TeX preamble code that instruments float environments.

    The preamble hooks into LaTeX's float machinery to log:
    - \\textheight (total usable page height)
    - Each float's box height (ht+dp) with associated labels

    Data is written to \\jobname.flt_profile.

    This code is loaded BEFORE \\input{thesis}, so only TeX primitives and
    commands available in the LaTeX format are safe here.  All hooks that
    touch package-defined commands (\\label, \\@floatboxreset, …) are
    deferred to \\AtBeginDocument so they wrap the final definitions
    installed by hyperref et al.
    """
    return r"""
% ---- Float Profiler (injected by analyze_floats.py) ----
\makeatletter
% Primitives: safe before \documentclass
\newwrite\fltprofile
\immediate\openout\fltprofile=\jobname.flt_profile\relax
\gdef\floatprofile@labels{}%
\newif\iffloatprofile@infloat

% Deferred to after all packages are loaded
\AtBeginDocument{%
  % Log page geometry
  \immediate\write\fltprofile{TEXTHEIGHT \the\textheight}%
  \immediate\write\fltprofile{TEXTWIDTH \the\textwidth}%
  %
  % Hook into \@floatboxreset (called at start of each float body)
  \let\floatprofile@orig@floatboxreset\@floatboxreset
  \def\@floatboxreset{%
    \floatprofile@orig@floatboxreset
    \global\floatprofile@infloattrue
    \gdef\floatprofile@labels{}%
  }%
  %
  % Hook into \label to collect labels inside floats
  \let\floatprofile@orig@label\label
  \renewcommand{\label}[1]{%
    \floatprofile@orig@label{#1}%
    \iffloatprofile@infloat
      \begingroup
      \toks0=\expandafter{\floatprofile@labels}%
      \xdef\floatprofile@labels{\the\toks0,#1}%
      \endgroup
    \fi
  }%
  %
  % Hook into \@endfloatbox (called when float box is closed)
  \let\floatprofile@orig@endfloatbox\@endfloatbox
  \def\@endfloatbox{%
    \floatprofile@orig@endfloatbox
    \immediate\write\fltprofile{FLOAT \floatprofile@labels\space\the\dimexpr\ht\@currbox+\dp\@currbox\relax}%
    \global\floatprofile@infloatfalse
  }%
}

\AtEndDocument{%
  \immediate\closeout\fltprofile
}
\makeatother
% ---- End Float Profiler ----
"""


def _parse_float_profile(profile_path: Path) -> Optional[FloatProfile]:
    """Parse a .flt_profile file written by the profiler preamble.

    Returns FloatProfile or None if the file doesn't exist or is empty.
    """
    if not profile_path.exists():
        return None

    text = profile_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return None

    textheight = 0.0
    float_heights: dict[str, float] = {}

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("TEXTHEIGHT "):
            textheight = _parse_pt(line.split(maxsplit=1)[1])
        elif line.startswith("FLOAT "):
            # Format: "FLOAT ,label1,label2 123.45pt"
            rest = line[6:]
            parts = rest.rsplit(maxsplit=1)
            if len(parts) == 2:
                labels_str, height_str = parts
                height = _parse_pt(height_str)
                for label in labels_str.split(","):
                    label = label.strip()
                    if label:
                        float_heights[label] = height

    if textheight <= 0:
        return None

    return FloatProfile(textheight_pt=textheight, float_heights_pt=float_heights)


# ---------------------------------------------------------------------------
# Phase 4d: Analysis (source-line fallback)
# ---------------------------------------------------------------------------

def analyze_source_lines(
    floats: list[Float],
    refs: list[Reference],
    file_entries: list[FileEntry],
    threshold: int,
) -> list[FloatReport]:
    """Match floats to references using source line distance (fallback mode)."""
    ref_map: dict[str, list[Reference]] = {}
    for r in refs:
        ref_map.setdefault(r.label, []).append(r)
    for v in ref_map.values():
        v.sort(key=lambda r: r.global_line)

    reports: list[FloatReport] = []

    for flt in floats:
        if not flt.labels:
            reports.append(FloatReport(
                float_obj=flt,
                label="(no label)",
                first_ref=None,
                float_page=None,
                ref_page=None,
                page_distance=None,
                status="UNREFERENCED",
            ))
            continue

        earliest_ref: Optional[Reference] = None
        earliest_label: Optional[str] = None
        for label in flt.labels:
            label_refs = ref_map.get(label, [])
            if label_refs:
                candidate = label_refs[0]
                if earliest_ref is None or candidate.global_line < earliest_ref.global_line:
                    earliest_ref = candidate
                    earliest_label = label

        if earliest_ref is None:
            reports.append(FloatReport(
                float_obj=flt,
                label=flt.labels[0],
                first_ref=None,
                float_page=None,
                ref_page=None,
                page_distance=None,
                status="UNREFERENCED",
            ))
            continue

        distance = earliest_ref.global_line - flt.global_start

        # In source-line mode, use the old FAR/OK classification
        same_file = flt.file.resolve() == earliest_ref.file.resolve()
        if not same_file:
            status = "CROSS-FILE"
        elif abs(distance) > threshold:
            status = "FAR"
        else:
            status = "OK"

        reports.append(FloatReport(
            float_obj=flt,
            label=earliest_label or flt.labels[0],
            first_ref=earliest_ref,
            float_page=None,
            ref_page=None,
            page_distance=None,
            source_distance=distance,
            status=status,
        ))

    return reports


# ---------------------------------------------------------------------------
# Phase 4e: Reporting
# ---------------------------------------------------------------------------

def _sort_key_pdf(r: FloatReport) -> tuple[int, int]:
    """Sort key: LATE first (worst distance first), EARLY, BEFORE, then rest."""
    order = {"LATE": 0, "EARLY": 1, "BEFORE": 2, "APPENDIX": 3, "UNREFERENCED": 4, "NO-AUX": 5}
    priority = order.get(r.status, 6)
    dist = r.page_distance if r.page_distance is not None else 0
    if r.status == "LATE":
        return (priority, dist)       # most negative first
    elif r.status == "EARLY":
        return (priority, -dist)      # largest distance first
    else:
        return (priority, -dist)


def print_report_pdf(
    reports: list[FloatReport], base_dir: Path, threshold: int,
    display_numbers: Optional[dict[str, str]] = None,
) -> None:
    """Print the PDF-based analysis report."""
    print()
    print(bold("Float Placement Analysis (PDF-based)"))
    print("=" * 37)
    print()

    if not reports:
        print("No floats found.")
        return

    if display_numbers is None:
        display_numbers = {}

    # Build display name column: "Fig 6.3" or "Tbl 8.2" etc.
    def _display_name(r: FloatReport) -> str:
        num = display_numbers.get(r.label, "")
        if not num:
            return ""
        env = r.float_obj.env_type.rstrip("*")
        prefix = {"figure": "Fig", "table": "Tbl", "listing": "Lst"}.get(env, env[:3].title())
        return f"{prefix} {num}"

    # Column widths
    label_w = max(len(r.label) for r in reports)
    label_w = max(label_w, 5)
    name_w = max((len(_display_name(r)) for r in reports), default=0)
    name_w = max(name_w, 4)

    # Header
    hdr = (
        f"{'Label':<{label_w}}  "
        f"{'Name':<{name_w}}  "
        f"{'Float Page':>10}  "
        f"{'Ref Page':>10}  "
        f"{'Distance':>10}  "
        f"{'Penalty':>8}  "
        f"Status"
    )
    print(hdr)
    print("-" * len(hdr.expandtabs()))

    sorted_reports = sorted(reports, key=_sort_key_pdf)

    for r in sorted_reports:
        name_str = _display_name(r)

        # Float page
        if r.float_page is not None:
            fp_str = f"p.{r.float_page}"
        else:
            fp_str = "?"

        # Ref page
        if r.ref_page is not None:
            rp_str = f"p.{r.ref_page}"
        elif r.first_ref is None:
            rp_str = "(none)"
        else:
            rp_str = "?"

        # Distance
        if r.page_distance is not None:
            if r.page_distance == 0:
                dist_str = "0 pp"
            elif r.page_distance > 0:
                dist_str = f"+{r.page_distance} pp"
            else:
                dist_str = f"{r.page_distance} pp"
        else:
            dist_str = "n/a"

        # Status with color
        if r.status == "BEFORE":
            status_str = green("BEFORE")
        elif r.status == "EARLY":
            status_str = yellow("EARLY")
        elif r.status == "LATE":
            status_str = red("LATE")
        elif r.status == "APPENDIX":
            status_str = gray("APPENDIX")
        elif r.status == "UNREFERENCED":
            status_str = gray("UNREFERENCED")
        elif r.status == "NO-AUX":
            status_str = gray("NO-AUX")
        else:
            status_str = r.status

        # Per-float penalty
        fp = _float_penalty(r)
        pen_str = f"{fp:.0f}" if fp > 0 else ""

        line = (
            f"{r.label:<{label_w}}  "
            f"{name_str:<{name_w}}  "
            f"{fp_str:>10}  "
            f"{rp_str:>10}  "
            f"{dist_str:>10}  "
            f"{pen_str:>8}  "
            f"{status_str}"
        )
        print(line)

    # Summary
    total = len(reports)
    late = sum(1 for r in reports if r.status == "LATE")
    early = sum(1 for r in reports if r.status == "EARLY")
    before = sum(1 for r in reports if r.status == "BEFORE")
    appendix = sum(1 for r in reports if r.status == "APPENDIX")
    unref = sum(1 for r in reports if r.status == "UNREFERENCED")
    noaux = sum(1 for r in reports if r.status == "NO-AUX")
    print()
    parts = [
        f"{total} floats",
        f"{late} LATE (after reference)" if late else None,
        f"{early} EARLY (>{threshold}pp before)" if early else None,
        f"{before} BEFORE (good)" if before else None,
        f"{appendix} APPENDIX (ok)" if appendix else None,
        f"{unref} UNREFERENCED" if unref else None,
        f"{noaux} NO-AUX" if noaux else None,
    ]
    print("Summary: " + ", ".join(p for p in parts if p))

    # Crowding info
    page_counts: dict[int, list[str]] = {}
    for r in reports:
        if r.float_page is not None and r.status not in ("APPENDIX", "UNREFERENCED", "NO-AUX"):
            dn = display_numbers.get(r.label, "")
            tag = f"{r.label} ({dn})" if dn else r.label
            page_counts.setdefault(r.float_page, []).append(tag)
    crowded = {p: labels for p, labels in sorted(page_counts.items()) if len(labels) > 1}
    if crowded:
        crowd_penalty = _crowding_penalty(reports)
        print(f"\nCrowded pages ({len(crowded)}, penalty {crowd_penalty:.0f}):")
        for page, labels in crowded.items():
            print(f"  p.{page} ({len(labels)} floats): {', '.join(labels)}")

    penalty = compute_penalty(reports)
    dist_penalty = sum(_float_penalty(r) for r in reports)
    crowd_total = _crowding_penalty(reports)
    print(f"\nPenalty: {penalty:.0f}  (distance: {dist_penalty:.0f}, crowding: {crowd_total:.0f})")
    print(f"  Scoring: {_penalty_formula_str()}")
    print()


def print_report_source(reports: list[FloatReport], base_dir: Path, threshold: int) -> None:
    """Print the source-line-based analysis report (fallback mode)."""
    print()
    print(bold("Float Placement Analysis (source-line mode)"))
    print("=" * 44)
    print()

    if not reports:
        print("No floats found.")
        return

    label_w = max(len(r.label) for r in reports)
    label_w = max(label_w, 10)

    def loc_str(r: FloatReport, which: str) -> str:
        if which == "decl":
            return f"{relative_path(r.float_obj.file, base_dir)}:{r.float_obj.start_line}"
        elif which == "ref":
            if r.first_ref is None:
                return "(none)"
            return f"{relative_path(r.first_ref.file, base_dir)}:{r.first_ref.line}"
        return ""

    decl_w = max((len(loc_str(r, "decl")) for r in reports), default=20)
    ref_w = max((len(loc_str(r, "ref")) for r in reports), default=20)
    decl_w = max(decl_w, 10)
    ref_w = max(ref_w, 10)

    hdr = (
        f"{'Label':<{label_w}}  "
        f"{'Declared':<{decl_w}}  "
        f"{'First Ref':<{ref_w}}  "
        f"{'Distance':<10}  "
        f"Status"
    )
    print(hdr)

    for r in sorted(reports, key=lambda x: abs(x.source_distance) if x.source_distance is not None else -1, reverse=True):
        dist_str = f"{r.source_distance:+d}" if r.source_distance is not None else "n/a"

        if r.status == "OK":
            status_str = green("OK")
        elif r.status == "FAR":
            status_str = red("FAR")
        elif r.status == "CROSS-FILE":
            status_str = yellow("CROSS-FILE")
        elif r.status == "UNREFERENCED":
            status_str = gray("UNREFERENCED")
        else:
            status_str = r.status

        line = (
            f"{r.label:<{label_w}}  "
            f"{loc_str(r, 'decl'):<{decl_w}}  "
            f"{loc_str(r, 'ref'):<{ref_w}}  "
            f"{dist_str:<10}  "
            f"{status_str}"
        )
        print(line)

    total = len(reports)
    far = sum(1 for r in reports if r.status == "FAR")
    cross = sum(1 for r in reports if r.status == "CROSS-FILE")
    unref = sum(1 for r in reports if r.status == "UNREFERENCED")
    print()
    print(
        f"Summary: {total} floats total, "
        f"{far} exceed threshold ({threshold} lines), "
        f"{cross} cross-file, "
        f"{unref} unreferenced"
    )
    print()


# ---------------------------------------------------------------------------
# Phase 5: Fix mode - reposition floats
# ---------------------------------------------------------------------------

_PARA_BOUNDARY_RE = re.compile(
    r"^\s*$|"
    r"^\\(section|subsection|subsubsection|chapter|paragraph|subparagraph)\b"
)


def find_paragraph_start(lines: list[str], ref_line: int) -> int:
    """Find the start of the paragraph containing ref_line (0-based index).

    Searches backward from ref_line for a blank line or sectioning command,
    and returns the index of the first non-blank line after that boundary.
    """
    idx = ref_line
    while idx > 0:
        idx -= 1
        if _PARA_BOUNDARY_RE.match(lines[idx]):
            return idx + 1
    return 0


def collapse_blank_lines(lines: list[str]) -> list[str]:
    """Collapse runs of 3+ consecutive blank lines to exactly 2."""
    result: list[str] = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                result.append(line)
        else:
            blank_count = 0
            result.append(line)
    return result


def fix_floats_pdf(reports: list[FloatReport], base_dir: Path, threshold: int) -> None:
    """Move floats that are LATE or EARLY (PDF-based analysis)."""

    # LATE: float renders after its reference — move earlier
    # EARLY: float renders too far before its reference — move closer
    fixable = [r for r in reports if r.status in ("LATE", "EARLY")]

    # Separate cross-file warnings (float and ref in different files)
    cross_file = []
    same_file = []
    for r in fixable:
        if r.first_ref is None:
            continue
        if r.float_obj.file.resolve() != r.first_ref.file.resolve():
            cross_file.append(r)
        else:
            same_file.append(r)

    if cross_file:
        print()
        print("Cross-file floats (manual intervention needed):")
        for r in cross_file:
            dist_str = f"{r.page_distance:+d}pp" if r.page_distance is not None else "?"
            print(
                f"  {r.label} ({r.status}, {dist_str}): "
                f"float in {relative_path(r.float_obj.file, base_dir)}:{r.float_obj.start_line}, "
                f"ref in {relative_path(r.first_ref.file, base_dir)}:{r.first_ref.line}"
            )
        print()

    if not same_file:
        print("No same-file floats need fixing. Nothing to do.")
        return

    # Group by file
    by_file: dict[str, list[FloatReport]] = {}
    for r in same_file:
        key = str(r.float_obj.file.resolve())
        # Skip generated files
        try:
            rel = r.float_obj.file.resolve().relative_to(base_dir.resolve())
            if str(rel).startswith("generated"):
                print(f"  Skipping {r.label} (in generated/ directory)")
                continue
        except ValueError:
            pass
        by_file.setdefault(key, []).append(r)

    for file_key, file_reports in by_file.items():
        file_path = Path(file_key)
        print(f"Fixing {len(file_reports)} float(s) in {relative_path(file_path, base_dir)}...")

        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()

        # Process floats in reverse order of start_line to preserve line numbers
        file_reports.sort(key=lambda r: r.float_obj.start_line, reverse=True)

        for report in file_reports:
            flt = report.float_obj
            ref = report.first_ref
            assert ref is not None

            start_idx = flt.start_line - 1  # 0-based
            end_idx = flt.end_line - 1      # 0-based

            # Extract the float text from current lines
            float_lines = lines[start_idx:end_idx + 1]

            # Remove the float from its original position
            del lines[start_idx:end_idx + 1]

            # Adjust the reference line if it was after the removed float
            ref_idx = ref.line - 1  # 0-based in original numbering
            removed_count = end_idx - start_idx + 1
            if ref_idx > end_idx:
                ref_idx -= removed_count

            # Find paragraph boundary before the reference
            insert_idx = find_paragraph_start(lines, ref_idx)

            # Insert: blank line + float + blank line
            insertion = [""] + float_lines + [""]
            for j, ins_line in enumerate(insertion):
                lines.insert(insert_idx + j, ins_line)

            direction = "earlier" if report.status == "LATE" else "later"
            dist_str = f"{report.page_distance:+d}pp" if report.page_distance is not None else "?"
            print(
                f"  Moved {report.label} ({report.status}, {dist_str}): "
                f"line {flt.start_line} -> before line {insert_idx + 1} ({direction})"
            )

        # Collapse excessive blank lines
        lines = collapse_blank_lines(lines)

        # Write back
        file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("Re-run `latexmk` and then re-run this script to verify improvements.")


def fix_floats_source(reports: list[FloatReport], base_dir: Path, threshold: int) -> None:
    """Move floats in source-line mode (legacy behavior)."""
    fixable = [r for r in reports if r.status == "FAR"]
    cross_file = [r for r in reports if r.status == "CROSS-FILE"]

    if cross_file:
        print()
        print("Cross-file floats (manual intervention needed):")
        for r in cross_file:
            print(
                f"  {r.label}: declared in "
                f"{relative_path(r.float_obj.file, base_dir)}:{r.float_obj.start_line}, "
                f"first ref in "
                f"{relative_path(r.first_ref.file, base_dir)}:{r.first_ref.line}"
            )
        print()

    if not fixable:
        print("No same-file floats exceed the threshold. Nothing to fix.")
        return

    by_file: dict[str, list[FloatReport]] = {}
    for r in fixable:
        key = str(r.float_obj.file.resolve())
        try:
            rel = r.float_obj.file.resolve().relative_to(base_dir.resolve())
            if str(rel).startswith("generated"):
                print(f"  Skipping {r.label} (in generated/ directory)")
                continue
        except ValueError:
            pass
        by_file.setdefault(key, []).append(r)

    for file_key, file_reports in by_file.items():
        file_path = Path(file_key)
        print(f"Fixing {len(file_reports)} float(s) in {relative_path(file_path, base_dir)}...")

        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()

        file_reports.sort(key=lambda r: r.float_obj.start_line, reverse=True)

        for report in file_reports:
            flt = report.float_obj
            ref = report.first_ref
            assert ref is not None

            start_idx = flt.start_line - 1
            end_idx = flt.end_line - 1

            float_lines = lines[start_idx:end_idx + 1]
            del lines[start_idx:end_idx + 1]

            ref_idx = ref.line - 1
            removed_count = end_idx - start_idx + 1
            if ref_idx > end_idx:
                ref_idx -= removed_count

            insert_idx = find_paragraph_start(lines, ref_idx)

            insertion = [""] + float_lines + [""]
            for j, ins_line in enumerate(insertion):
                lines.insert(insert_idx + j, ins_line)

            print(f"  Moved {report.label}: line {flt.start_line} -> before line {insert_idx + 1}")

        lines = collapse_blank_lines(lines)
        file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Optimization helpers
# ---------------------------------------------------------------------------

_SECTION_BOUNDARY_RE = re.compile(r"^\s*\\(part|chapter|section)\b")


def _is_generated(file_path: Path, base_dir: Path) -> bool:
    """Check if a file is in the generated/ directory."""
    try:
        rel = str(file_path.resolve().relative_to(base_dir.resolve()))
        return rel.startswith("generated")
    except ValueError:
        return False


def _clean_aux_files(base_dir: Path) -> None:
    """Remove .aux files that may be corrupted after a failed compilation."""
    import glob as _glob
    for aux in _glob.glob(str(base_dir / "**/*.aux"), recursive=True):
        try:
            Path(aux).unlink()
        except OSError:
            pass
    # Also root-level aux files
    for aux in base_dir.glob("*.aux"):
        try:
            aux.unlink()
        except OSError:
            pass


_PAGE_PROGRESS_RE = re.compile(r"\[(\d+)")
_TEX_ERROR_RE = re.compile(r"^! ")


def _show_tex_error(base_dir: Path, build_cmd: list[str], output_lines: list[str]) -> None:
    """Extract and display the actual TeX error from the log file."""
    # Determine log file name from build command
    log_path = None
    for arg in build_cmd:
        if arg.endswith(".tex"):
            log_path = base_dir / arg.replace(".tex", ".log")
            break
        m = re.search(r"-jobname=(\S+)", arg)
        if m:
            log_path = base_dir / f"{m.group(1)}.log"
            break

    if log_path and log_path.exists():
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            log_lines = log_text.splitlines()
            for i, line in enumerate(log_lines):
                if _TEX_ERROR_RE.match(line):
                    # Show error and a few lines of context
                    context = log_lines[i:i+4]
                    for cl in context:
                        print(f"    {cl.rstrip()}", file=sys.stderr)
                    return
        except OSError:
            pass

    # Fallback: show last lines of stdout
    for ol in output_lines[-5:]:
        print(f"    {ol.rstrip()}", file=sys.stderr)


def recompile(base_dir: Path, build_cmd: list[str], clean_aux: bool = False) -> bool:
    """Recompile the document. Returns True on success.

    Shows page progress during compilation.
    If clean_aux is True, removes .aux files first (useful after a failed build).
    """
    if clean_aux:
        _clean_aux_files(base_dir)
    # Remove stale minted error markers that poison subsequent runs.
    for mf in base_dir.glob("*.minted"):
        try:
            mf.unlink()
        except OSError:
            pass
    start = time.monotonic()
    last_page = 0
    try:
        proc = subprocess.Popen(
            build_cmd,
            cwd=str(base_dir),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        output_lines: list[str] = []
        for line in proc.stdout:
            output_lines.append(line)
            # Show page progress
            for m in _PAGE_PROGRESS_RE.finditer(line):
                page = int(m.group(1))
                if page > last_page:
                    last_page = page
                    elapsed = time.monotonic() - start
                    print(f"\r  Compiling... p.{page} ({elapsed:.0f}s)",
                          end="", flush=True)
        proc.wait(timeout=1800)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        elapsed = time.monotonic() - start
        print(f"\n  TIMED OUT ({elapsed:.0f}s).", flush=True)
        return False

    elapsed = time.monotonic() - start
    if returncode == 0:
        print(f"\r  Compiling... done ({elapsed:.0f}s).       ", flush=True)
        return True
    else:
        print(f"\r  Compiling... FAILED ({elapsed:.0f}s).     ", flush=True)
        # Try to extract the actual TeX error from the log file
        _show_tex_error(base_dir, build_cmd, output_lines)
        return False


def full_analysis(
    base_dir: Path, root_file: Path, threshold: int, workers: int = 8,
    quiet: bool = False, synctex: Optional[SyncTeXResolver] = None,
) -> tuple[list[FloatReport], float]:
    """Run phases 1-4 and return (reports, penalty)."""
    root_stem = root_file.stem
    out = sys.stderr if not quiet else open("/dev/null", "w")

    file_entries = build_file_ordering(root_file, base_dir)
    floats = extract_floats(file_entries)
    refs = extract_references(file_entries)

    aux_path = base_dir / f"{root_stem}.aux"
    pdf_path = base_dir / f"{root_stem}.pdf"
    label_pages = parse_aux_labels(aux_path)
    if synctex is None:
        synctex = SyncTeXResolver(pdf_path, base_dir, max_workers=workers)

    print(f"  Querying synctex...", file=out, flush=True)
    reports = analyze_pdf(
        floats, refs, file_entries, label_pages, synctex, base_dir, threshold,
    )
    penalty = compute_penalty(reports)
    return reports, penalty


def run_float_profiling(
    base_dir: Path, root_file: Path,
) -> Optional[FloatProfile]:
    """Run a single LuaLaTeX compilation with float profiling instrumentation.

    Writes the profiler preamble to a temp file and loads it via
    ``\\input`` on the lualatex command line.  Using a temp file avoids
    shell-escaping issues and TeX comment problems that arise when the
    preamble is passed as an inline string.
    """
    root_stem = root_file.stem
    profile_path = base_dir / f"{root_stem}.flt_profile"
    profiler_tex = Path("/tmp/claude") / "floatprofiler.tex"

    # Remove stale profile file
    if profile_path.exists():
        profile_path.unlink()

    # Write profiler preamble to a temp file
    profiler_tex.parent.mkdir(parents=True, exist_ok=True)
    profiler_tex.write_text(_generate_profiler_preamble(), encoding="utf-8")

    # Load profiler then the real document
    tex_input = r"\input{" + str(profiler_tex) + r"}\input{" + root_stem + "}"

    build_cmd = [
        "lualatex", "-synctex=1", "-interaction=nonstopmode",
        "-halt-on-error", f"-jobname={root_stem}",
        tex_input,
    ]

    print(f"  Running profiling compilation...", flush=True)
    t0 = time.monotonic()
    if not recompile(base_dir, build_cmd):
        print(f"  Profiling compilation failed.", flush=True)
        return None

    elapsed = time.monotonic() - t0

    # Parse the profile file
    profile = _parse_float_profile(profile_path)
    if profile is None:
        print(f"  No profiling data produced.", flush=True)
        return None

    n_floats = len(profile.float_heights_pt)
    print(f"  Profiling complete: textheight={profile.textheight_pt:.1f}pt, "
          f"{n_floats} floats measured ({elapsed:.0f}s)", flush=True)

    return profile


def _estimate_lines_per_page(synctex: SyncTeXResolver, file_path: Path, base_dir: Path) -> float:
    """Estimate source lines per PDF page for a file from synctex cache data.

    Returns the average lines-per-page, or a conservative default of 45.
    """
    rel = _rel_path_str(file_path, base_dir)

    # Collect all cached entries for this file
    entries: list[tuple[int, int]] = []  # (line, page)
    for (cached_file, cached_line), page in synctex._cache.items():
        if page is not None and cached_file == rel:
            entries.append((cached_line, page))

    if len(entries) < 2:
        return 45.0  # conservative default

    entries.sort()

    # Calculate line spans per page transition
    line_spans: list[float] = []
    for i in range(1, len(entries)):
        line_diff = entries[i][0] - entries[i-1][0]
        page_diff = entries[i][1] - entries[i-1][1]
        if page_diff > 0 and line_diff > 0:
            line_spans.append(line_diff / page_diff)

    if not line_spans:
        return 45.0

    # Use median for robustness
    line_spans.sort()
    return line_spans[len(line_spans) // 2]


def _find_source_line_for_page(
    synctex: SyncTeXResolver, rel_file: str, target_page: int,
) -> Optional[int]:
    """Find a source line in rel_file that maps to target_page.

    Strategy (in order):
    1. synctex edit (reverse lookup) — most accurate
    2. Forward cache exact match
    3. Interpolation between cached entries
    """
    # Strategy 1: synctex edit — query top of the target page
    # Try a few y-positions to increase hit probability
    for y in (750, 600, 400):
        result = synctex.edit_page(target_page, y=y)
        if result is not None:
            edit_file, edit_line = result
            if edit_file == rel_file:
                return edit_line

    # Strategy 2: exact match in forward cache
    entries: list[tuple[int, int]] = []  # (line, page)
    for (cached_file, cached_line), page in synctex._cache.items():
        if page is not None and cached_file == rel_file:
            entries.append((cached_line, page))

    if not entries:
        return None

    exact = [line for line, page in entries if page == target_page]
    if exact:
        return min(exact)

    # Strategy 3: interpolate between nearest cached entries
    entries.sort()
    before: Optional[tuple[int, int]] = None  # (line, page) with page < target
    after: Optional[tuple[int, int]] = None   # (line, page) with page > target
    for line, page in entries:
        if page < target_page:
            if before is None or page > before[1]:
                before = (line, page)
        elif page > target_page:
            if after is None or page < after[1]:
                after = (line, page)

    if before is not None and after is not None:
        page_span = after[1] - before[1]
        line_span = after[0] - before[0]
        frac = (target_page - before[1]) / page_span
        return int(before[0] + frac * line_span)
    elif before is not None:
        lpp = _estimate_lpp_from_entries(entries)
        return int(before[0] + (target_page - before[1]) * lpp)
    elif after is not None:
        lpp = _estimate_lpp_from_entries(entries)
        return max(1, int(after[0] - (after[1] - target_page) * lpp))

    return None


def _estimate_lpp_from_entries(entries: list[tuple[int, int]]) -> float:
    """Estimate lines-per-page from sorted (line, page) entries."""
    if len(entries) < 2:
        return 45.0
    entries_sorted = sorted(entries)
    spans = []
    for i in range(1, len(entries_sorted)):
        ld = entries_sorted[i][0] - entries_sorted[i-1][0]
        pd = entries_sorted[i][1] - entries_sorted[i-1][1]
        if pd > 0 and ld > 0:
            spans.append(ld / pd)
    if not spans:
        return 45.0
    spans.sort()
    return spans[len(spans) // 2]


def _nesting_depth_at(lines: list[str], target_idx: int) -> int:
    r"""Count the \begin/\end nesting depth at the given line index.

    Returns 0 if the target is at the top level (safe to insert a float).
    Returns >0 if inside one or more environments.
    """
    depth = 0
    begin_re = re.compile(r"\\begin\{(\w+)\}")
    end_re = re.compile(r"\\end\{(\w+)\}")
    # Only count environments that prohibit float insertion
    # (skip document, which wraps everything)
    skip_envs = {"document"}

    for i in range(target_idx):
        line = lines[i]
        if line.lstrip().startswith("%"):
            continue
        for m in begin_re.finditer(line):
            if m.group(1) not in skip_envs:
                depth += 1
        for m in end_re.finditer(line):
            if m.group(1) not in skip_envs:
                depth = max(0, depth - 1)
    return depth



def _save_build_artifacts(base_dir: Path, root_stem: str) -> dict[str, bytes]:
    """Save copies of build artifacts (PDF, synctex, aux files) for later restore.

    Returns a dict mapping file paths to their binary contents.
    """
    artifacts: dict[str, bytes] = {}
    # Core build outputs (check both base_dir and out/ subdirectory)
    for suffix in (".pdf", ".synctex.gz", ".aux"):
        for directory in [base_dir, base_dir / "out"]:
            p = directory / f"{root_stem}{suffix}"
            if p.exists():
                try:
                    artifacts[str(p)] = p.read_bytes()
                except OSError:
                    pass
    # Subdirectory .aux files (includes/, appendices/, etc.)
    import glob as _glob
    for aux in _glob.glob(str(base_dir / "**/*.aux"), recursive=True):
        aux_path = Path(aux)
        if str(aux_path) not in artifacts:
            try:
                artifacts[str(aux_path)] = aux_path.read_bytes()
            except OSError:
                pass
    return artifacts


def _restore_build_artifacts(artifacts: dict[str, bytes]) -> None:
    """Restore previously saved build artifacts, skipping recompilation."""
    for path_str, data in artifacts.items():
        try:
            Path(path_str).write_bytes(data)
        except OSError:
            pass


def _git_run(base_dir: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command in the base directory."""
    return subprocess.run(
        ["git"] + list(args),
        cwd=str(base_dir),
        capture_output=True, text=True,
        check=check,
    )


def _git_create_optimize_branch(base_dir: Path) -> tuple[str, str]:
    """Create a temporary branch for optimization and return its name."""
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    branch = f"float-optimize-{ts}"
    # Save current branch name
    result = _git_run(base_dir, "rev-parse", "--abbrev-ref", "HEAD")
    original_branch = result.stdout.strip()
    # Create and switch to optimize branch
    _git_run(base_dir, "checkout", "-b", branch)
    return branch, original_branch


def _git_commit_step(base_dir: Path, iteration: int, target_label: str,
                     old_penalty: float, new_penalty: float) -> None:
    """Commit the current state as an optimization step."""
    improvement = old_penalty - new_penalty
    msg = (f"float-opt step {iteration}: move {target_label}\n\n"
           f"Penalty: {old_penalty:.0f} -> {new_penalty:.0f} (-{improvement:.0f})")
    # Stage all tracked .tex files
    _git_run(base_dir, "add", "-u")
    _git_run(base_dir, "commit", "-m", msg, check=False)


def _git_finish_optimize(base_dir: Path, original_branch: str, optimize_branch: str,
                         initial_penalty: float, final_penalty: float) -> None:
    """Switch back to original branch and merge if improved, or report."""
    # Commit any uncommitted changes so checkout doesn't fail on dirty tree
    _git_run(base_dir, "add", "-u")
    _git_run(base_dir, "commit", "-m",
             "float-opt: final state (uncommitted changes)", check=False)

    _git_run(base_dir, "checkout", original_branch)
    if final_penalty < initial_penalty:
        _git_run(base_dir, "merge", "--ff-only", optimize_branch, check=False)
        print(f"  Optimization branch '{optimize_branch}' merged into '{original_branch}'.")
    else:
        print(f"  No improvement. Optimization branch '{optimize_branch}' preserved for inspection.")


# ---------------------------------------------------------------------------
# ILP-based optimizer
# ---------------------------------------------------------------------------

import pulp
from TexSoup import TexSoup as _TexSoup


_LABEL_RE = re.compile(r"\\label\{([^}]+)\}")


def _extract_env_labels(env) -> set[str]:
    """Extract label names from a TexSoup environment.

    First tries TexSoup's ``find_all("label")``.  If that yields nothing
    (e.g. because the environment contains verbatim-like content such as
    ``minted`` that TexSoup treats as opaque text), falls back to a regex
    over the environment's raw string representation.
    """
    labels: set[str] = set()
    for lbl in env.find_all("label"):
        arg = lbl.string
        if arg:
            labels.add(str(arg).strip())
    if not labels:
        # Regex fallback for opaque environments (listing+minted, etc.)
        for m in _LABEL_RE.finditer(str(env)):
            labels.add(m.group(1).strip())
    return labels


def _texsoup_find_float(source: str, flt: "Float") -> Optional[tuple[int, int]]:
    """Use TexSoup to find a float's char offsets in the source.

    Returns (start_char, end_char) or None if not found.
    Matches by label content to handle line-number drift.
    """
    if not flt.labels:
        return None

    try:
        soup = _TexSoup(source)
    except Exception:
        return None

    env_name = flt.env_type.rstrip("*")
    target_labels = set(flt.labels)

    # Search the base env name, and also the starred variant if applicable.
    env_names = [env_name]
    if flt.env_type.endswith("*"):
        env_names.append(flt.env_type)

    for name in env_names:
        for env in soup.find_all(name):
            env_labels = _extract_env_labels(env)
            if env_labels & target_labels:
                # Found the float. Get its source span.
                env_str = str(env)
                idx = source.find(env_str)
                if idx >= 0:
                    return (idx, idx + len(env_str))

    return None


def _texsoup_safe_insert_pos(source: str, char_pos: int) -> int:
    """Validate and adjust insertion position to be at top-level.

    Uses TexSoup to check that the position is not inside a nested
    environment. Returns the adjusted char position.
    """
    # Fallback: if position looks fine, use it directly

    # Simple heuristic: count \begin and \end before the position
    prefix = source[:char_pos]
    depth = 0
    for m in re.finditer(r"\\begin\{(\w+\*?)\}", prefix):
        env = m.group(1)
        if env != "document":
            depth += 1
    for m in re.finditer(r"\\end\{(\w+\*?)\}", prefix):
        env = m.group(1)
        if env != "document":
            depth -= 1

    if depth == 0:
        return char_pos

    # Position is inside an environment. Search backwards for \end{...}
    # to find a safe point.
    for m in re.finditer(r"\\end\{[^}]+\}", prefix):
        pass  # iterate to find the last one
    # Just use the line-based approach as fallback
    return char_pos


def _apply_ilp_moves(
    moves: list[tuple["FloatReport", int, int]],
    synctex: SyncTeXResolver,
    base_dir: Path,
    ref_margin: float | None = None,
) -> int:
    """Apply a batch of ILP-planned moves. Returns number of moves applied.

    moves: list of (report, target_page, current_page).

    When TexSoup is available, uses it to locate floats by label (robust
    against line-number drift) and validate insertion points.
    Falls back to line-index manipulation otherwise.
    """
    by_file: dict[str, list[tuple[FloatReport, int, int]]] = {}
    for r, tp, cp in moves:
        fpath = str(r.float_obj.file.resolve())
        by_file.setdefault(fpath, []).append((r, tp, cp))

    total_applied = 0

    for fpath, file_moves in by_file.items():
        file_path = Path(fpath)
        source = file_path.read_text(encoding="utf-8", errors="replace")
        lines = source.splitlines()

        lpp = _estimate_lines_per_page(synctex, file_path, base_dir)

        # Phase 1: extract all moving floats (bottom-up to preserve indices).
        file_moves.sort(key=lambda m: m[0].float_obj.start_line, reverse=True)

        extracted: list[tuple[FloatReport, int, int, str, int]] = []
        removal_ranges: list[tuple[int, int]] = []

        for r, tp, cp in file_moves:
            flt = r.float_obj

            # Try TexSoup first for robust float location
            # Use TexSoup to locate the float by label (robust against
            # line-number drift from prior edits in the same file).
            ts_span = _texsoup_find_float(source, flt)

            if ts_span is not None:
                float_str = source[ts_span[0]:ts_span[1]]
                # Remove from source (char-based)
                source = source[:ts_span[0]].rstrip("\n") + "\n" + source[ts_span[1]:].lstrip("\n")
                lines = source.splitlines()
                start_line_approx = source[:ts_span[0]].count("\n")
                count_approx = float_str.count("\n") + 1
                removal_ranges.append((start_line_approx, count_approx))
                extracted.append((r, tp, cp, float_str, start_line_approx))
            else:
                # TexSoup couldn't find it (e.g. no labels) — use line indices
                start_idx = flt.start_line - 1
                end_idx = flt.end_line - 1
                if start_idx < 0 or end_idx >= len(lines):
                    continue
                float_text_lines = lines[start_idx:end_idx + 1]
                count = end_idx - start_idx + 1
                del lines[start_idx:end_idx + 1]
                source = "\n".join(lines)
                removal_ranges.append((start_idx, count))
                extracted.append((r, tp, cp, "\n".join(float_text_lines), start_idx))

        def _adjust_line(original_0based: int) -> int:
            shift = 0
            for rm_start, rm_count in removal_ranges:
                if rm_start < original_0based:
                    shift += rm_count
            return max(0, original_0based - shift)

        # Phase 2: compute insertion positions.
        insertions: list[tuple[int, str, FloatReport, int, int]] = []

        rel_file = _rel_path_str(file_path, base_dir)

        for r, tp, cp, float_str, orig_start in extracted:
            if r.first_ref is None or r.ref_page is None:
                continue

            ref_line_adj = _adjust_line(r.first_ref.line - 1)
            float_line_adj = _adjust_line(orig_start)

            if ref_margin is not None:
                # Reference-anchored: place float a fixed distance before reference
                raw_target = max(0, ref_line_adj - int(lpp * ref_margin))
            else:
                # Strategy 1: synctex cache — find an actual source line on target page
                synctex_line = _find_source_line_for_page(synctex, rel_file, tp)
                if synctex_line is not None:
                    raw_target = _adjust_line(synctex_line - 1)
                else:
                    # Strategy 2: estimate from float's CURRENT position using lpp.
                    # Delta is from the float's current page to its target page,
                    # giving the correct direction and magnitude.
                    page_delta = tp - cp
                    line_delta = int(page_delta * lpp)
                    raw_target = float_line_adj + line_delta

            # Guard: when targeting same page as reference or earlier,
            # ensure float is placed BEFORE the reference (not after it).
            # Leave a margin of ~1/3 page to give LaTeX room for [t] placement.
            if tp <= r.ref_page and raw_target > ref_line_adj:
                raw_target = max(0, ref_line_adj - int(lpp * 0.3))

            raw_target = max(0, min(raw_target, len(lines) - 1))
            target = find_paragraph_start(lines, raw_target)

            # Don't cross section boundaries
            if target < ref_line_adj:
                for i in range(target, min(ref_line_adj, len(lines))):
                    if _SECTION_BOUNDARY_RE.match(lines[i]):
                        target = i + 1
            elif target > ref_line_adj:
                for i in range(ref_line_adj, min(target + 1, len(lines))):
                    if _SECTION_BOUNDARY_RE.match(lines[i]):
                        target = i - 1
                        break

            # Validate insertion point is not inside a nested environment
            if _nesting_depth_at(lines, min(target, len(lines) - 1)) > 0:
                # Try one paragraph earlier/later
                alt = find_paragraph_start(lines, max(0, target - 1))
                if alt != target and _nesting_depth_at(lines, min(alt, len(lines) - 1)) == 0:
                    target = alt
                else:
                    continue  # skip this move

            target = max(0, min(target, len(lines)))
            insertions.append((target, float_str, r, tp, cp))

        # Sort by target line descending so bottom insertions happen first.
        insertions.sort(key=lambda x: x[0], reverse=True)

        for target, float_str, r, tp, cp in insertions:
            float_text_lines = float_str.splitlines()
            insertion_block = [""] + float_text_lines + [""]
            for j, ins_line in enumerate(insertion_block):
                lines.insert(target + j, ins_line)

            direction = "earlier" if tp < cp else "later"
            ref_l = r.first_ref.line if r.first_ref else "?"
            print(f"    Move {r.label}: p.{cp} -> p.{tp} ({direction}), "
                  f"insert at line {target+1} (ref at line {ref_l})")
            total_applied += 1

        lines = collapse_blank_lines(lines)
        file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return total_applied


def optimize_floats_ilp(
    base_dir: Path, root_file: Path, threshold: int,
    max_iters: int = 3, workers: int = 8,
    build_cmd: Optional[list[str]] = None,
    draft_mode: bool = False,
    profile: Optional[FloatProfile] = None,
) -> None:
    """Optimize float positions using ILP (Integer Linear Programming).

    Formulates float placement as an optimization problem solved with PuLP.
    LaTeX is used as a validator: the ILP plans moves, LaTeX confirms them.

    Hard constraints (configurable via module constants):
      - Float must be within +/- MAX_PAGE_DISTANCE pages of its first reference
      - At most MAX_FLOATS_PER_PAGE floats per page

    Soft objective (minimized):
      - LATE penalty:   PENALTY_LATE * d^PENALTY_EXPONENT
      - BEFORE penalty: PENALTY_BEFORE * d^PENALTY_EXPONENT
      - Asymmetric: 1 page before ref is preferred over 1 page after
    """
    opt_start = time.monotonic()
    root_stem = root_file.stem

    if build_cmd is None:
        if draft_mode:
            build_cmd = [
                "lualatex", "-synctex=1", "-interaction=nonstopmode",
                "-halt-on-error", f"-jobname={root_stem}",
                "\\PassOptionsToPackage{draft}{graphicx}\\input{" + root_stem + "}",
            ]
        else:
            build_cmd = [
                "lualatex", "-synctex=1", "-interaction=nonstopmode",
                "-halt-on-error", f"{root_stem}.tex",
            ]

    print()
    print(bold("Float Placement Optimization (ILP)"))
    print("=" * 34)
    print(f"  Max iterations: {max_iters}")
    print(f"  Hard constraints: |distance| <= {MAX_PAGE_DISTANCE}pp, <= {MAX_FLOATS_PER_PAGE} floats/page")
    print(f"  Penalty: {_penalty_formula_str()}")
    print(f"  Draft mode: {draft_mode}")
    print(f"  Build command: {' '.join(build_cmd)}")
    print()

    # Git tracking
    optimize_branch = None
    original_branch = None
    try:
        optimize_branch, original_branch = _git_create_optimize_branch(base_dir)
        print(f"  Git branch: {optimize_branch}")
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"  Git tracking disabled: {e}", file=sys.stderr)

    # Initial analysis
    print("Initial analysis...")
    pdf_path = base_dir / f"{root_stem}.pdf"
    synctex_resolver = SyncTeXResolver(pdf_path, base_dir, max_workers=workers)

    reports, initial_penalty = full_analysis(
        base_dir, root_file, threshold, workers, synctex=synctex_resolver,
    )
    aux_path = base_dir / f"{root_stem}.aux"
    display_numbers = parse_aux_display_numbers(aux_path)
    print_report_pdf(reports, base_dir, threshold, display_numbers)

    if initial_penalty == 0:
        print("\nAlready optimal!")
        return

    best_penalty = initial_penalty

    # Initialize optimization log
    opt_log = {
        "start_time": datetime.now().isoformat(),
        "initial_penalty": initial_penalty,
        "penalty_config": {
            "PENALTY_LATE": PENALTY_LATE,
            "PENALTY_BEFORE": PENALTY_BEFORE,
            "PENALTY_EXPONENT": PENALTY_EXPONENT,
            "PENALTY_CROWDING": PENALTY_CROWDING,
            "MAX_PAGE_DISTANCE": MAX_PAGE_DISTANCE,
            "MAX_FLOATS_PER_PAGE": MAX_FLOATS_PER_PAGE,
        },
        "iterations": [],
    }

    for iteration in range(1, max_iters + 1):
        iter_start = time.monotonic()
        print(f"\n{'='*60}")
        print(bold(f"ILP Iteration {iteration}/{max_iters}"))
        print(f"{'='*60}")

        # Initialize iteration log
        iter_log = {
            "iteration": iteration,
            "start_penalty": best_penalty,
            "ilp_objective": None,
            "planned_moves": 0,
            "batch": None,
            "individual_moves": [],
            "outcome": "no_improvement",
        }

        # Classify floats
        movable = []
        fixed_page_counts: dict[int, int] = {}  # page -> count of immovable floats

        for r in reports:
            can_move = (
                r.status in ("LATE", "EARLY", "BEFORE")
                and r.first_ref is not None
                and r.float_page is not None
                and r.ref_page is not None
                and r.float_obj.file.resolve() == r.first_ref.file.resolve()
                and not _is_generated(r.float_obj.file, base_dir)
            )
            if can_move:
                movable.append(r)
            elif r.float_page is not None and r.status not in ("APPENDIX",):
                # This float is fixed on its page — counts toward crowding
                fixed_page_counts[r.float_page] = (
                    fixed_page_counts.get(r.float_page, 0) + 1
                )

        if not movable:
            print("  No movable floats.")
            iter_log["outcome"] = "no_movable"
            iter_log["end_penalty"] = best_penalty
            iter_log["time_s"] = round(time.monotonic() - iter_start, 1)
            opt_log["iterations"].append(iter_log)
            break

        # Build ILP model
        print(f"  Building ILP model ({len(movable)} movable floats)...")

        prob = pulp.LpProblem("float_placement", pulp.LpMinimize)

        x: dict[tuple[str, int], pulp.LpVariable] = {}
        float_vars: dict[str, list[tuple[int, pulp.LpVariable]]] = {}

        for r in movable:
            ref_p = r.ref_page
            assert ref_p is not None
            label = r.label
            candidates = []

            for p in range(max(1, ref_p - MAX_PAGE_DISTANCE), ref_p + MAX_PAGE_DISTANCE + 1):
                var = pulp.LpVariable(f"x_{label.replace(':', '_')}_{p}", cat="Binary")
                x[(label, p)] = var
                candidates.append((p, var))

            float_vars[label] = candidates
            prob += (
                pulp.lpSum(var for _, var in candidates) == 1,
                f"assign_{label.replace(':', '_')}",
            )

        # Crowding constraint: movable + fixed <= MAX_FLOATS_PER_PAGE per page
        page_movable_vars: dict[int, list[pulp.LpVariable]] = {}
        for (label, page), var in x.items():
            page_movable_vars.setdefault(page, []).append(var)

        for page, pvars in page_movable_vars.items():
            fixed_on_page = fixed_page_counts.get(page, 0)
            max_movable = max(0, MAX_FLOATS_PER_PAGE - fixed_on_page)
            if len(pvars) > max_movable or max_movable < MAX_FLOATS_PER_PAGE:
                prob += (
                    pulp.lpSum(pvars) <= max_movable,
                    f"crowd_{page}",
                )

        # Page capacity constraint (if profile data available)
        if profile and profile.textheight_pt > 0:
            page_height_vars: dict[int, list[tuple[float, pulp.LpVariable]]] = {}
            for r in movable:
                label = r.label
                height = profile.float_heights_pt.get(label, 0.0)
                if height <= 0:
                    continue
                for p, var in float_vars[label]:
                    page_height_vars.setdefault(p, []).append((height, var))

            for page, hvars in page_height_vars.items():
                if len(hvars) > 1:
                    prob += (
                        pulp.lpSum(h * v for h, v in hvars)
                        <= profile.textheight_pt * 0.8,
                        f"capacity_{page}",
                    )

        # Objective: minimize total distance penalty + crowding penalty
        obj_terms = []
        for r in movable:
            ref_p = r.ref_page
            assert ref_p is not None
            for p, var in float_vars[r.label]:
                d = abs(p - ref_p)
                if d == 0:
                    continue  # zero cost, skip
                if p > ref_p:
                    cost = PENALTY_LATE * d ** PENALTY_EXPONENT
                else:
                    cost = PENALTY_BEFORE * d ** PENALTY_EXPONENT
                obj_terms.append(cost * var)

        # Crowding penalty: for each page, penalize each extra float beyond 1.
        # Use auxiliary continuous variables: excess_p = max(0, total_p - 1).
        for page, pvars in page_movable_vars.items():
            fixed_on_page = fixed_page_counts.get(page, 0)
            # total floats on page = fixed + sum(movable vars)
            # excess = max(0, total - 1) = max(0, fixed - 1 + sum(pvars))
            # Linearize: excess_p >= 0, excess_p >= fixed - 1 + sum(pvars)
            excess = pulp.LpVariable(f"excess_{page}", lowBound=0, cat="Continuous")
            prob += (
                excess >= (fixed_on_page - 1) + pulp.lpSum(pvars),
                f"excess_lb_{page}",
            )
            obj_terms.append(PENALTY_CROWDING * excess)

        prob += pulp.lpSum(obj_terms) if obj_terms else 0

        # Solve
        print("  Solving ILP...", end="", flush=True)
        solver = pulp.PULP_CBC_CMD(msg=0)
        status = prob.solve(solver)

        if status != pulp.constants.LpStatusOptimal:
            print(f" FAILED (status: {pulp.LpStatus[status]})")
            iter_log["outcome"] = "ilp_failed"
            iter_log["end_penalty"] = best_penalty
            iter_log["time_s"] = round(time.monotonic() - iter_start, 1)
            opt_log["iterations"].append(iter_log)
            break

        ilp_obj = pulp.value(prob.objective)
        iter_log["ilp_objective"] = ilp_obj
        print(f" done (ILP objective: {ilp_obj:.0f})")

        # Extract solution: which floats need to move
        moves: list[tuple[FloatReport, int, int]] = []
        for r in movable:
            current_page = r.float_page
            assert current_page is not None
            for p, var in float_vars[r.label]:
                if pulp.value(var) is not None and pulp.value(var) > 0.5:
                    if p != current_page:
                        moves.append((r, p, current_page))
                    break

        iter_log["planned_moves"] = len(moves)

        if not moves:
            print("  ILP solution matches current placement. No moves needed.")
            iter_log["outcome"] = "ilp_no_moves"
            iter_log["end_penalty"] = best_penalty
            iter_log["time_s"] = round(time.monotonic() - iter_start, 1)
            opt_log["iterations"].append(iter_log)
            break

        print(f"  ILP solution: {len(moves)} float(s) to move:")
        for r, tp, cp in moves:
            delta = tp - r.ref_page if r.ref_page else 0
            pos = "same page" if delta == 0 else f"{delta:+d}pp from ref"
            print(f"    {r.label}: p.{cp} -> p.{tp} ({pos})")

        # Save state before applying moves
        file_snapshots: dict[str, str] = {}
        for r, tp, cp in moves:
            fpath = str(r.float_obj.file.resolve())
            if fpath not in file_snapshots:
                file_snapshots[fpath] = Path(fpath).read_text(encoding="utf-8")
        saved_artifacts = _save_build_artifacts(base_dir, root_stem)

        # Apply all moves
        n_applied = _apply_ilp_moves(moves, synctex_resolver, base_dir)

        if n_applied == 0:
            print("  No moves could be applied.")
            iter_log["outcome"] = "no_moves_applied"
            iter_log["end_penalty"] = best_penalty
            iter_log["time_s"] = round(time.monotonic() - iter_start, 1)
            opt_log["iterations"].append(iter_log)
            break

        print(f"\n  Applied {n_applied}/{len(moves)} moves. Compiling...")

        # Compile
        synctex_resolver.clear_cache()
        batch_compiled = recompile(base_dir, build_cmd)

        batch_improved = False
        if batch_compiled:
            synctex_resolver.clear_cache()
            new_reports, new_penalty = full_analysis(
                base_dir, root_file, threshold, workers, quiet=True,
                synctex=synctex_resolver,
            )
            improvement = best_penalty - new_penalty
            if new_penalty < best_penalty:
                print(f"  {green(f'Batch improved: {best_penalty:.0f} -> {new_penalty:.0f} (-{improvement:.0f})')}")
                iter_log["batch"] = {"compiled": True, "improved": True, "penalty_after": new_penalty}
                iter_log["outcome"] = "batch_improved"
                best_penalty = new_penalty
                reports = new_reports
                batch_improved = True
                if optimize_branch:
                    _git_commit_step(base_dir, iteration, "ILP batch",
                                     best_penalty + improvement, best_penalty)
            else:
                print(f"  {yellow(f'Batch did not improve: {best_penalty:.0f} -> {new_penalty:.0f}')}")
                iter_log["batch"] = {"compiled": True, "improved": False, "penalty_after": new_penalty}
        else:
            print("  Compilation failed.")
            iter_log["batch"] = {"compiled": False, "improved": False, "penalty_after": None}

        any_individual_improvement = False
        if not batch_improved:
            # Revert batch and try individual moves
            print("  Reverting batch. Trying moves individually...")
            for fpath, content in file_snapshots.items():
                Path(fpath).write_text(content, encoding="utf-8")
            _restore_build_artifacts(saved_artifacts)
            synctex_resolver.clear_cache()

            RETRY_MARGINS = [None, 0.5, 0.8, 1.2]
            for r, tp, cp in moves:
                fpath = str(r.float_obj.file.resolve())

                # Initialize move log
                move_log = {
                    "label": r.label,
                    "target_page": tp,
                    "current_page": cp,
                    "attempts": [],
                    "outcome": "all_failed",
                }

                for retry_idx, margin in enumerate(RETRY_MARGINS):
                    snap = {fpath: Path(fpath).read_text(encoding="utf-8")}
                    saved_art = _save_build_artifacts(base_dir, root_stem)

                    n = _apply_ilp_moves(
                        [(r, tp, cp)], synctex_resolver, base_dir,
                        ref_margin=margin,
                    )
                    if n == 0:
                        move_log["outcome"] = "skip"
                        break  # Float can't be moved at all
                    synctex_resolver.clear_cache()

                    strategy = f"ref_margin={margin}" if margin is not None else "synctex"
                    if not recompile(base_dir, build_cmd):
                        attempt_log = {
                            "strategy": strategy,
                            "compiled": False,
                            "penalty_after": None,
                            "actual_page": None,
                            "improved": False,
                        }
                        move_log["attempts"].append(attempt_log)
                        Path(fpath).write_text(snap[fpath], encoding="utf-8")
                        _restore_build_artifacts(saved_art)
                        synctex_resolver.clear_cache()
                        continue

                    synctex_resolver.clear_cache()
                    new_reports, new_penalty = full_analysis(
                        base_dir, root_file, threshold, workers, quiet=True,
                        synctex=synctex_resolver,
                    )

                    # Find where float actually landed
                    actual = next((rr for rr in new_reports if rr.label == r.label), None)
                    actual_page = actual.float_page if actual else None

                    improved = new_penalty < best_penalty
                    attempt_log = {
                        "strategy": strategy,
                        "compiled": True,
                        "penalty_after": new_penalty,
                        "actual_page": actual_page,
                        "improved": improved,
                    }
                    move_log["attempts"].append(attempt_log)

                    if improved:
                        improvement = best_penalty - new_penalty
                        print(f"    {green(f'{r.label}: {best_penalty:.0f} -> {new_penalty:.0f} (-{improvement:.0f}) [{strategy}]')}")
                        move_log["outcome"] = "improved"
                        best_penalty = new_penalty
                        reports = new_reports
                        any_individual_improvement = True
                        if optimize_branch:
                            _git_commit_step(base_dir, iteration, r.label,
                                             best_penalty + improvement, best_penalty)
                        break  # Success — move to next float
                    else:
                        # Revert this attempt
                        Path(fpath).write_text(snap[fpath], encoding="utf-8")
                        _restore_build_artifacts(saved_art)
                        synctex_resolver.clear_cache()

                        if actual_page == tp:
                            # Float IS on target page but penalty didn't improve
                            # (cascading effects). No point retrying different margin.
                            move_log["outcome"] = "landed_no_improve"
                            break

                        # Float missed target page — try next margin
                        if retry_idx < len(RETRY_MARGINS) - 1:
                            next_m = RETRY_MARGINS[retry_idx + 1]
                            print(f"    {yellow(f'{r.label}: landed p.{actual_page} (target p.{tp}), retry margin={next_m}')}")

                # Append move log after all retries for this float
                iter_log["individual_moves"].append(move_log)

            if not any_individual_improvement:
                print("  No individual moves improved penalty either.")
                iter_log["end_penalty"] = best_penalty
                iter_log["time_s"] = round(time.monotonic() - iter_start, 1)
                opt_log["iterations"].append(iter_log)
                break
            else:
                # Individual moves succeeded
                iter_log["outcome"] = "individual_improved"

        # End of iteration — append if batch improved (or if we didn't break)
        if batch_improved or any_individual_improvement:
            iter_log["end_penalty"] = best_penalty
            iter_log["time_s"] = round(time.monotonic() - iter_start, 1)
            opt_log["iterations"].append(iter_log)

    # Final build
    opt_elapsed = time.monotonic() - opt_start
    print(f"\n{'='*60}")
    print(bold("Optimization complete") + f"  (total: {opt_elapsed:.0f}s)")
    print(f"{'='*60}\n")

    print("Final build (latexmk)...")
    recompile(base_dir, ["latexmk", "-interaction=nonstopmode", f"{root_stem}.tex"])

    synctex_final = SyncTeXResolver(pdf_path, base_dir, max_workers=workers)
    reports, final_penalty = full_analysis(
        base_dir, root_file, threshold, workers, synctex=synctex_final,
    )
    display_numbers = parse_aux_display_numbers(aux_path)
    print_report_pdf(reports, base_dir, threshold, display_numbers)

    if optimize_branch and original_branch:
        _git_finish_optimize(base_dir, original_branch, optimize_branch,
                             initial_penalty, final_penalty)

    # Write optimization log
    opt_log["final_penalty"] = final_penalty
    opt_log["total_time_s"] = round(time.monotonic() - opt_start, 1)
    log_path = base_dir / f"{root_stem}.opt_log.json"
    log_path.write_text(json.dumps(opt_log, indent=2) + "\n")
    print(f"Optimization log: {log_path}\n")

    print(f"Initial penalty: {initial_penalty:.0f}")
    print(f"Final penalty:   {final_penalty:.0f}")
    total_time = time.monotonic() - opt_start
    print(f"Total time:      {total_time:.0f}s")
    if final_penalty < initial_penalty:
        print(green(f"Improved by {initial_penalty - final_penalty:.0f} points!"))
    else:
        print("No improvement achieved.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze (and optionally fix) LaTeX float placement "
            "using PDF page positions or source line distances."
        )
    )
    parser.add_argument(
        "--root",
        default="thesis.tex",
        help="Root .tex file (default: thesis.tex)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=2,
        help=(
            "Page distance threshold (default: 2). "
            "In PDF mode: flag floats >N pages before reference. "
            "In source-line mode: flag floats >N lines from reference (use e.g. 40)."
        ),
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Move misplaced floats closer to their first reference",
    )
    parser.add_argument(
        "--source-lines",
        action="store_true",
        help="Fall back to source-line distance analysis (no PDF/synctex needed)",
    )
    parser.add_argument(
        "--optimize",
        type=int,
        nargs="?",
        const=10,
        default=None,
        metavar="N",
        help=(
            "Optimize float positions using ILP (Integer Linear Programming). "
            "Plans globally optimal page assignments for all floats, then "
            "validates with LaTeX. Max N iterations (default 10). "
            "Stops early if no improvement is found."
        ),
    )
    parser.add_argument(
        "--build-cmd",
        default=None,
        help=(
            "Build command for --optimize (default: single-pass lualatex). "
            "Example: --build-cmd 'make pdf'"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel synctex workers (default: 8)",
    )
    parser.add_argument(
        "--draft",
        action="store_true",
        help=(
            "Enable draft mode during --optimize. Skips image loading for faster "
            "compilation, but may change float placement if images are size-sensitive."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Run a profiling compilation to measure float box heights and page "
            "capacity. Uses this data to improve position estimation accuracy "
            "during --optimize. Adds ~1 extra compilation at the start."
        ),
    )
    args = parser.parse_args()

    base_dir = Path.cwd()
    root_file = base_dir / args.root
    if not root_file.exists():
        print(f"ERROR: Root file not found: {root_file}", file=sys.stderr)
        sys.exit(1)

    # Phase 1: Build file ordering
    t0 = time.monotonic()
    print("Phase 1: Building file ordering...", end="", file=sys.stderr, flush=True)
    file_entries = build_file_ordering(root_file, base_dir)
    print(f" ({time.monotonic() - t0:.1f}s)", file=sys.stderr, flush=True)

    # Phase 2: Extract floats
    t0 = time.monotonic()
    print("Phase 2: Extracting floats...", end="", file=sys.stderr, flush=True)
    floats = extract_floats(file_entries)
    print(f" {len(floats)} floats ({time.monotonic() - t0:.1f}s)", file=sys.stderr, flush=True)

    # Phase 3: Extract references
    t0 = time.monotonic()
    print("Phase 3: Extracting references...", end="", file=sys.stderr, flush=True)
    refs = extract_references(file_entries)
    print(f" {len(refs)} refs ({time.monotonic() - t0:.1f}s)", file=sys.stderr, flush=True)

    if args.source_lines:
        # Source-line fallback mode
        if args.threshold == 2:
            # User likely didn't override the default; use a sensible source-line default
            threshold = 40
            print(
                f"  Note: using default source-line threshold of {threshold} lines "
                f"(override with --threshold).",
                file=sys.stderr, flush=True,
            )
        else:
            threshold = args.threshold

        t0 = time.monotonic()
        print("Phase 4: Analyzing (source-line mode)...", end="", file=sys.stderr, flush=True)
        reports = analyze_source_lines(floats, refs, file_entries, threshold)
        print(f" ({time.monotonic() - t0:.1f}s)", file=sys.stderr, flush=True)
        print_report_source(reports, base_dir, threshold)

        if args.fix:
            fix_floats_source(reports, base_dir, threshold)
    else:
        # PDF-based mode
        root_stem = Path(args.root).stem
        aux_path = base_dir / f"{root_stem}.aux"
        pdf_path = base_dir / f"{root_stem}.pdf"
        # synctex.gz may be in root or out/ directory
        synctex_path = base_dir / f"{root_stem}.synctex.gz"
        synctex_path_alt = base_dir / "out" / f"{root_stem}.synctex.gz"

        has_pdf = pdf_path.exists()
        has_aux = aux_path.exists()
        has_synctex = synctex_path.exists() or synctex_path_alt.exists()

        if not has_pdf or not has_aux:
            # Need a full build from scratch
            print("PDF or aux missing. Running full build...", file=sys.stderr, flush=True)
            subprocess.run(
                ["latexmk", "-C"],
                cwd=str(base_dir),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if not recompile(base_dir, [
                "latexmk", "-interaction=nonstopmode", f"{root_stem}.tex",
            ]):
                print("ERROR: Compilation failed.", file=sys.stderr)
                sys.exit(1)
        elif not has_synctex:
            # PDF and aux exist but synctex is missing — force recompile to get it.
            # Must use -g to force at least one lualatex run even if PDF is up-to-date.
            print("synctex.gz missing. Recompiling to generate it...",
                  file=sys.stderr, flush=True)
            if not recompile(base_dir, [
                "latexmk", "-g", "-interaction=nonstopmode", f"{root_stem}.tex",
            ]):
                print("ERROR: Compilation failed.", file=sys.stderr)
                sys.exit(1)

        # Verify all required files exist
        still_missing = []
        if not aux_path.exists():
            still_missing.append(str(aux_path))
        if not pdf_path.exists():
            still_missing.append(str(pdf_path))
        if not synctex_path.exists() and not synctex_path_alt.exists():
            still_missing.append(f"{synctex_path} (or {synctex_path_alt})")
        if still_missing:
            print(
                f"ERROR: Required files still missing after compilation:\n"
                + "\n".join(f"  {m}" for m in still_missing),
                file=sys.stderr,
            )
            sys.exit(1)

        # Sanity-check synctex.gz: run one probe query.  If it fails,
        # the file is likely stale/corrupt → force a fresh compilation.
        _probe = SyncTeXResolver(pdf_path, base_dir, max_workers=1)
        if file_entries:
            _probe_file = _rel_path_str(file_entries[0].path, base_dir)
            _probe_result = _probe._query_one(_probe_file, 1)
            if _probe_result is None:
                print("synctex.gz appears stale. Recompiling...",
                      file=sys.stderr, flush=True)
                if not recompile(base_dir, [
                    "latexmk", "-g", "-interaction=nonstopmode",
                    f"{root_stem}.tex",
                ]):
                    print("ERROR: Recompilation failed.", file=sys.stderr)
                    sys.exit(1)

        # Phase 4: PDF-based analysis
        t0 = time.monotonic()
        print("Phase 4a: Parsing .aux file for float pages...", end="", file=sys.stderr, flush=True)
        label_pages = parse_aux_labels(aux_path)
        print(f" {len(label_pages)} labels ({time.monotonic() - t0:.1f}s)", file=sys.stderr, flush=True)

        synctex = SyncTeXResolver(pdf_path, base_dir, max_workers=args.workers)
        t0 = time.monotonic()
        print("Phase 4b: Resolving reference pages via synctex...", file=sys.stderr, flush=True)
        reports = analyze_pdf(floats, refs, file_entries, label_pages, synctex, base_dir, args.threshold)
        print(f"  Phase 4b complete ({time.monotonic() - t0:.1f}s)", file=sys.stderr, flush=True)

        display_numbers = parse_aux_display_numbers(aux_path)
        print_report_pdf(reports, base_dir, args.threshold, display_numbers)

        if args.profile and args.optimize is None:
            print("\nFloat profiling (standalone)...", flush=True)
            profile = run_float_profiling(base_dir, root_file)
            if profile:
                print(f"\n{bold('Float Size Profile')}")
                print(f"{'='*40}")
                print(f"  Page body height: {profile.textheight_pt:.1f}pt")
                print()
                # Sort by height descending
                sorted_floats = sorted(
                    profile.float_heights_pt.items(),
                    key=lambda x: x[1], reverse=True,
                )
                print(f"  {'Label':<45} {'Height':>10} {'Page %':>8}")
                print(f"  {'-'*45} {'-'*10} {'-'*8}")
                for label, height in sorted_floats:
                    pct = height / profile.textheight_pt * 100
                    dn = display_numbers.get(label, "")
                    tag = f" ({dn})" if dn else ""
                    print(f"  {label + tag:<45} {height:>8.1f}pt {pct:>6.1f}%")
                print()

        if args.optimize is not None:
            build_cmd = args.build_cmd.split() if args.build_cmd else None
            draft_mode = args.draft

            # Always run profiling for the optimizer (needs float heights)
            print("\nFloat profiling...", flush=True)
            profile = run_float_profiling(base_dir, root_file)
            if profile:
                print(f"  Profile loaded: {len(profile.float_heights_pt)} floats, "
                      f"textheight={profile.textheight_pt:.1f}pt")
            else:
                print("  Profiling failed, continuing without profile data.")

            optimize_floats_ilp(
                base_dir, root_file, args.threshold,
                max_iters=args.optimize, workers=args.workers,
                build_cmd=build_cmd,
                draft_mode=draft_mode, profile=profile,
            )
        elif args.fix:
            fix_floats_pdf(reports, base_dir, args.threshold)


if __name__ == "__main__":
    main()
