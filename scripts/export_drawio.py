#!/usr/bin/env python3
"""
Export all pages from a .drawio file to PDFs with sanitized, lowercase filenames.
Optionally generates TikZ coordinate files (-defs.tex and -coords.tex) for each page.

Usage:
  python scripts/export_drawio.py [-v|--verbose] [--no-origins] DIAGRAM.drawio [OUTPUT_DIR]

Behavior mirrors scripts/export_drawio.sh:
  - DRAWIO executable is taken from $DRAWIO or defaults to 'draw.io'.
  - OUTPUT_DIR defaults to 'src/figures' if not provided.
  - Filenames are derived from page names by replacing non [a-zA-Z0-9_] with '-'
    and converting to lowercase.
  - By default, also generates TikZ coordinate files for overlay positioning.
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import shlex
import xml.etree.ElementTree as ET
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


def process_page_model(mx_graph_model, output_base: Path, ns_map: dict, page_name: str, verbose: bool = False) -> None:
    """
    Processes a single <mxGraphModel> to find bounds, extract coords,
    and write the -defs.tex and -coords.tex files.
    """
    if mx_graph_model is None:
        if verbose:
            print(f"Warning: Page '{page_name}' contains no <mxGraphModel>. Skipping origins.")
        return

    # Build a map of all cells for easy lookup
    all_cells = mx_graph_model.findall('.//mxCell[@id]', ns_map)
    cell_map = {cell.get('id'): cell for cell in all_cells}

    # Cache for memoizing absolute coordinates
    parent_coords_cache = {}
    parent_coords_cache["0"] = (0.0, 0.0)  # Root
    parent_coords_cache["1"] = (0.0, 0.0)  # Default layer

    def get_absolute_parent_coords(cell_id):
        """Recursively finds the absolute (x, y) coordinates of a cell's origin."""
        if cell_id in parent_coords_cache:
            return parent_coords_cache[cell_id]

        if cell_id not in cell_map:
            return (0.0, 0.0)

        cell = cell_map[cell_id]
        parent_id = cell.get('parent')
        parent_x, parent_y = get_absolute_parent_coords(parent_id)

        geo = cell.find('./mxGeometry', ns_map)
        rel_x, rel_y = 0.0, 0.0
        if geo is not None:
            try:
                rel_x = float(geo.get('x', 0))
                rel_y = float(geo.get('y', 0))
            except (ValueError, TypeError):
                pass

        abs_x = parent_x + rel_x
        abs_y = parent_y + rel_y
        parent_coords_cache[cell_id] = (abs_x, abs_y)
        return (abs_x, abs_y)

    # Find bounding box
    min_x, max_x_right = math.inf, -math.inf
    min_y_top, max_y_bottom = math.inf, -math.inf
    found_element = False

    # Process all vertices
    for cell in mx_graph_model.findall('.//mxCell[@vertex="1"]', ns_map):
        geo = cell.find('./mxGeometry', ns_map)
        if geo is not None:
            try:
                parent_id = cell.get('parent')
                parent_x, parent_y = get_absolute_parent_coords(parent_id)

                x = parent_x + float(geo.get('x', 0))
                y = parent_y + float(geo.get('y', 0))
                width = float(geo.get('width', 0))
                height = float(geo.get('height', 0))

                if x < min_x: min_x = x
                if x + width > max_x_right: max_x_right = x + width
                if y < min_y_top: min_y_top = y
                if y + height > max_y_bottom: max_y_bottom = y + height
                found_element = True
            except (ValueError, TypeError):
                continue

    # Process all edge points
    for cell in mx_graph_model.findall('.//mxCell[@edge="1"]', ns_map):
        geo = cell.find('./mxGeometry', ns_map)
        if geo is None:
            continue

        parent_id = cell.get('parent')
        parent_x, parent_y = get_absolute_parent_coords(parent_id)

        for point in geo.findall('.//mxPoint', ns_map):
            try:
                x = parent_x + float(point.get('x'))
                y = parent_y + float(point.get('y'))

                if x < min_x: min_x = x
                if x > max_x_right: max_x_right = x
                if y < min_y_top: min_y_top = y
                if y > max_y_bottom: max_y_bottom = y
                found_element = True
            except (ValueError, TypeError, KeyError):
                continue

    if not found_element:
        if verbose:
            print(f"Warning: No elements found on page '{page_name}'. Using (0,0) as origin.")
        min_x, max_x_right, min_y_top, max_y_bottom = 0, 0, 0, 0

    origin_x = min_x
    origin_y = max_y_bottom

    diagram_width_px = max_x_right - origin_x
    diagram_height_px = max_y_bottom - min_y_top

    # Extract coordinates
    coordinates = {}
    for obj in mx_graph_model.findall('.//object[@id]', ns_map):
        cell_id = obj.get('id')
        cell = obj.find('./mxCell', ns_map)
        if cell is None:
            continue

        geo = cell.find('./mxGeometry', ns_map)
        if geo is None:
            continue

        parent_id = cell.get('parent')
        parent_x, parent_y = get_absolute_parent_coords(parent_id)

        try:
            if cell.get('vertex') == '1':
                abs_x = parent_x + float(geo.get('x', 0))
                abs_y = parent_y + float(geo.get('y', 0))
                width = float(geo.get('width', 0))
                height = float(geo.get('height', 0))

                x_center = abs_x + width / 2
                y_center = abs_y + height / 2
                coordinates[cell_id] = (x_center - origin_x, origin_y - y_center)

            elif cell.get('edge') == '1':
                source_point = geo.find('./mxPoint[@as="sourcePoint"]', ns_map)
                target_point = geo.find('./mxPoint[@as="targetPoint"]', ns_map)

                if source_point is not None and target_point is not None:
                    src_x = parent_x + float(source_point.get('x'))
                    src_y = parent_y + float(source_point.get('y'))
                    tgt_x = parent_x + float(target_point.get('x'))
                    tgt_y = parent_y + float(target_point.get('y'))

                    mid_x = (src_x + tgt_x) / 2
                    mid_y = (src_y + tgt_y) / 2
                    coordinates[cell_id] = (mid_x - origin_x, origin_y - mid_y)

        except (ValueError, TypeError, KeyError):
            continue

    # Write output files
    output_defs_file = Path(str(output_base) + '-defs.tex')
    output_coords_file = Path(str(output_base) + '-coords.tex')

    try:
        with open(output_defs_file, 'w') as f:
            f.write(f"% Auto-generated definitions by export_drawio.py\n")
            f.write(f"% Page: {page_name}\n")
            f.write(f"\\def\\drawionativewidthpx{{{diagram_width_px:.4f}}}\n")
            f.write(f"\\def\\drawionativeheightpx{{{diagram_height_px:.4f}}}\n")
        if verbose:
            print(f"  -> Generated {output_defs_file}")
    except IOError as e:
        print(f"Error: Could not write to {output_defs_file}: {e}", file=sys.stderr)

    try:
        written_count = 0
        with open(output_coords_file, 'w') as f:
            f.write(f"% Auto-generated coordinates by export_drawio.py\n")
            f.write(f"% Page: {page_name}\n")

            if not coordinates:
                f.write(f"% WARNING: No objects with an 'id' attribute were found.\n")

            for name, (x, y) in coordinates.items():
                if not name.isdigit():
                    f.write(f"\\coordinate ({name}) at ({x:.2f}, {y:.2f});\n")
                    written_count += 1

        if verbose:
            print(f"  -> Generated {output_coords_file} ({written_count} coordinates)")

    except IOError as e:
        print(f"Error: Could not write to {output_coords_file}: {e}", file=sys.stderr)


def generate_origins(diagram_path: Path, output_dir: Path, pages: list[tuple[Optional[str], str]], verbose: bool = False) -> None:
    """
    Parses the .drawio file and generates TikZ coordinate files for each page.
    """
    try:
        tree = ET.parse(diagram_path)
        root = tree.getroot()
    except ET.ParseError:
        print(f"Error: Could not parse {diagram_path} for origins. Is it uncompressed XML?", file=sys.stderr)
        return
    except FileNotFoundError:
        print(f"Error: Input file not found at {diagram_path}", file=sys.stderr)
        return

    ns_map = {}
    if '}' in root.tag:
        ns_uri = root.tag.split('}')[0][1:]
        ns_map = {'': ns_uri}

    diagram_elements = root.findall('.//diagram', ns_map)

    if diagram_elements:
        # Multi-page file - match pages by index
        for idx, (pid, sanitized_name) in enumerate(pages):
            if idx < len(diagram_elements):
                page_element = diagram_elements[idx]
                page_name = page_element.get('name', sanitized_name)
                output_base = output_dir / sanitized_name
                mx_graph_model = page_element.find('./mxGraphModel', ns_map)
                process_page_model(mx_graph_model, output_base, ns_map, page_name, verbose)
    else:
        # Single-page file
        if pages:
            _, sanitized_name = pages[0]
            output_base = output_dir / sanitized_name
            mx_graph_model = root.find('.//mxGraphModel', ns_map)
            process_page_model(mx_graph_model, output_base, ns_map, "Default Page", verbose)


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
    parser.add_argument(
        "--origins",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate TikZ coordinate files (-defs.tex, -coords.tex) for each page (default: enabled)",
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

    if args.origins:
        if args.verbose:
            print("Generating TikZ coordinate files...")
        generate_origins(diagram_path, output_dir, pages, verbose=args.verbose)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
