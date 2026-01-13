#!/usr/bin/env python3
"""
Export all pages from a .drawio file to PDFs with sanitized, lowercase filenames.

Usage:
  python scripts/export_drawio.py [-v|--verbose] DIAGRAM.drawio [OUTPUT_DIR]

Behavior mirrors scripts/export_drawio.sh:
  - DRAWIO executable is taken from $DRAWIO or defaults to 'draw.io'.
  - OUTPUT_DIR defaults to 'src/figures' if not provided.
  - Filenames are derived from page names by replacing non [a-zA-Z0-9_] with '-'
    and converting to lowercase.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import shlex
from pathlib import Path
from typing import Optional


def which_drawio() -> str:
    exe = os.environ.get("DRAWIO", "draw.io")
    # If env provided path is not absolute, rely on PATH lookup during subprocess
    return exe


def run(cmd: list[str], verbose: bool = False) -> subprocess.CompletedProcess:
    try:
        if verbose:
            print("+ " + " ".join(shlex.quote(part) for part in cmd))
        return subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except FileNotFoundError:
        raise SystemExit(
            f"Error: Command not found: {cmd[0]!r}. Ensure draw.io CLI is installed and available, or set $DRAWIO."
        )
    except subprocess.CalledProcessError as e:
        # Bubble up stderr for easier debugging
        raise SystemExit(
            f"Command failed with exit code {e.returncode}: {' '.join(cmd)}\n{e.stderr}"
        )


def sanitize(name: str) -> str:
    # Replace any char not in [a-zA-Z0-9_] with '-'
    s = re.sub(r"[^a-zA-Z0-9_]", "-", name)
    return s.lower()


def extract_page_names(drawio_exe: str, diagram_path: str, verbose: bool = False) -> list[str]:
    """Deprecated internal: kept for backward compatibility. Returns names only.
    New code should use extract_pages() to get (id, name) pairs.
    """
    pages = extract_pages(drawio_exe, diagram_path, verbose=verbose)
    return [p[1] for p in pages]


def extract_pages(drawio_exe: str, diagram_path: str, verbose: bool = False) -> list[tuple[Optional[str], str]]:
    # Prefer reading the .drawio file directly and extracting <diagram id="..." name="..."> entries.
    # Some draw.io CLI versions only emit a single page when exporting XML with --page-index 0,
    # which undercounts pages. Parsing the source file avoids that and lets us use --page-id.
    if verbose:
        print(f"Extracting pages from {diagram_path} ...")
    try:
        with open(diagram_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        content = ""

    id_name_pairs: list[tuple[Optional[str], str]] = []
    if content:
        # Match each <diagram ...> tag and capture id and name if present, regardless of attribute order
        # We'll iterate over each opening tag.
        tag_pattern = re.compile(r"<diagram\b([^>]*)>")
        attr_id = re.compile(r"\bid=\"([^\"]*)\"")
        attr_name = re.compile(r"\bname=\"([^\"]*)\"")
        for m in tag_pattern.finditer(content):
            attrs = m.group(1)
            pid = None
            pname = ""
            mid = attr_id.search(attrs)
            mname = attr_name.search(attrs)
            if mid:
                pid = mid.group(1)
            if mname:
                pname = mname.group(1)
            id_name_pairs.append((pid, pname))

    if not id_name_pairs:
        # Fallback to previous behavior: ask draw.io to export XML and parse names/ids from it
        if verbose:
            print(f"Falling back to draw.io XML export for page discovery using {drawio_exe} ...")
        proc = run(
            [
                drawio_exe,
                "--export",
                "--format",
                "xml",
                "--output",
                "/dev/stdout",
                "--page-index",
                "0",
                diagram_path,
            ],
            verbose=verbose,
        )
        xml = proc.stdout
        tag_pattern = re.compile(r"<diagram\b([^>]*)>")
        attr_id = re.compile(r"\bid=\"([^\"]*)\"")
        attr_name = re.compile(r"\bname=\"([^\"]*)\"")
        for m in tag_pattern.finditer(xml):
            attrs = m.group(1)
            mid = attr_id.search(attrs)
            mname = attr_name.search(attrs)
            pid = mid.group(1) if mid else None
            pname = mname.group(1) if mname else ""
            id_name_pairs.append((pid, pname))

        if not id_name_pairs:
            # Fallback: if still nothing, synthesize based on count
            count = len(re.findall(r"<diagram\b", xml))
            if count == 0:
                raise SystemExit("No pages found in the provided .drawio file.")
            id_name_pairs = [(None, f"page-{i+1}") for i in range(count)]

    # Normalize, sanitize, ensure uniqueness; synthesize names for empty ones
    used: set[str] = set()
    result: list[tuple[Optional[str], str]] = []
    for i, (pid, raw_name) in enumerate(id_name_pairs):
        base_raw = (raw_name or "").strip() or f"page-{i+1}"
        base = sanitize(base_raw) or f"page-{i+1}"
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}-{suffix}"
            suffix += 1
        used.add(candidate)
        result.append((pid, candidate))

    if verbose:
        preview = ", ".join(name for (_, name) in result)
        print(f"Found {len(result)} page(s): {preview}")
    return result


def export_pages(drawio_exe: str, diagram_path: str, output_dir: Path, pages: list[tuple[Optional[str], str]], verbose: bool = False) -> None:
    for idx, (pid, name) in enumerate(pages):
        out_pdf = output_dir / f"{name}.pdf"
        # Use --page-index (1-based according to help, despite documentation saying 0-based)
        # Place input file at the end after all flags
        page_index_1based = idx + 1
        cmd = [
            drawio_exe,
            "--export",
            "--format",
            "pdf",
            "--page-index",
            str(page_index_1based),
            "--crop",
            "--border",
            "0",
            "-t",
            "--output",
            str(out_pdf),
            str(diagram_path),
        ]
        label = f"index={page_index_1based}"
        if verbose:
            print(f"Exporting page {label} -> {out_pdf}")
        run(cmd, verbose=verbose)
        # Post-run sanity check to help diagnose sandbox/argument issues
        if verbose and not out_pdf.exists():
            print(
                f"Warning: draw.io reported success but output not found at {out_pdf}. "
                f"If this persists, try running the displayed command manually with absolute paths.",
                file=sys.stderr,
            )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Export all pages from a .drawio file to PDFs with sanitized names."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output to show progress and executed commands.",
    )
    parser.add_argument("diagram", help="Path to the .drawio file")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="src/figures",
        help="Target directory for exported PDFs (default: src/figures)",
    )
    args = parser.parse_args(argv)

    diagram_path = Path(args.diagram)
    if not diagram_path.suffix.lower() == ".drawio":
        print("Input file is not a .drawio file", file=sys.stderr)
        return 1
    if not diagram_path.exists():
        print(f"Diagram file not found: {diagram_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    if args.verbose:
        print(f"Ensuring output directory exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    drawio_exe = which_drawio()
    if args.verbose:
        print(f"Using draw.io executable: {drawio_exe}")

    # Optionally, warn if executable likely missing
    if os.path.sep not in drawio_exe and shutil.which(drawio_exe) is None:
        print(
            f"Warning: '{drawio_exe}' not found on PATH. If the next step fails, set $DRAWIO to the full path.",
            file=sys.stderr,
        )

    # Resolve absolute paths to avoid any sandbox/working-directory issues with the draw.io app
    try:
        diagram_path = diagram_path.resolve()
    except Exception:
        # Best effort; keep as-is if resolution fails
        pass
    try:
        output_dir = output_dir.resolve()
    except Exception:
        pass

    pages = extract_pages(drawio_exe, str(diagram_path), verbose=args.verbose)
    export_pages(drawio_exe, str(diagram_path), output_dir, pages, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
