import xml.etree.ElementTree as ET
import sys
import math
import argparse
import os
from pathlib import Path


def process_page_model(mx_graph_model, output_base, ns_map, page_name):
    """
    Processes a single <mxGraphModel> to find bounds, extract coords,
    and write the -defs.tex and -coords.tex files.
    """
    if mx_graph_model is None:
        print(f"Warning: Page '{page_name}' contains no <mxGraphModel>. Skipping.")
        return

    # --- (NEW) Parent Coordinate Resolution ---

    # Build a map of all cells for easy lookup
    all_cells = mx_graph_model.findall('.//mxCell[@id]', ns_map)
    cell_map = {cell.get('id'): cell for cell in all_cells}

    # Cache for memoizing absolute coordinates
    parent_coords_cache = {}
    parent_coords_cache["0"] = (0.0, 0.0) # Root
    parent_coords_cache["1"] = (0.0, 0.0) # Default layer

    def get_absolute_parent_coords(cell_id):
        """
        Recursively finds the absolute (x, y) coordinates of a cell's
        top-left origin by summing parent coordinates.
        """
        if cell_id in parent_coords_cache:
            return parent_coords_cache[cell_id]

        if cell_id not in cell_map:
            # Fallback for a parent ID that isn't 0 or 1 and isn't in the map
            return (0.0, 0.0)

        cell = cell_map[cell_id]
        parent_id = cell.get('parent')

        # Recurse to get parent's absolute coordinates
        parent_x, parent_y = get_absolute_parent_coords(parent_id)

        # Get this cell's relative coordinates
        geo = cell.find('./mxGeometry', ns_map)
        rel_x = 0.0
        rel_y = 0.0
        if geo is not None:
            try:
                rel_x = float(geo.get('x', 0))
                rel_y = float(geo.get('y', 0))
            except (ValueError, TypeError):
                pass # Keep 0,0

        abs_x = parent_x + rel_x
        abs_y = parent_y + rel_y

        parent_coords_cache[cell_id] = (abs_x, abs_y)
        return (abs_x, abs_y)

    # --- True Bounding Box and Origin Finding (MODIFIED) ---
    min_x, max_x_right = math.inf, -math.inf
    min_y_top, max_y_bottom = math.inf, -math.inf
    found_element = False

    # 1. Process all vertices
    for cell in mx_graph_model.findall('.//mxCell[@vertex="1"]', ns_map):
        geo = cell.find('./mxGeometry', ns_map)
        if geo is not None:
            try:
                # (MOD) Get absolute parent coords
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

    # 2. Process all edge points
    for cell in mx_graph_model.findall('.//mxCell[@edge="1"]', ns_map):
        geo = cell.find('./mxGeometry', ns_map)
        if geo is None: continue

        # (MOD) Get absolute parent coords
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
        print(f"Warning: No elements found on page '{page_name}'. Using (0,0) as origin.")
        min_x, max_x_right, min_y_top, max_y_bottom = 0, 0, 0, 0

    origin_x = min_x
    origin_y = max_y_bottom

    diagram_width_px = max_x_right - origin_x
    diagram_height_px = max_y_bottom - min_y_top

    # --- Coordinate Extraction (MODIFIED) ---
    coordinates = {}
    for obj in mx_graph_model.findall('.//object[@id]', ns_map):
        cell_id = obj.get('id')

        cell = obj.find('./mxCell', ns_map)
        if cell is None: continue

        geo = cell.find('./mxGeometry', ns_map)
        if geo is None: continue

        # (MOD) Get parent's absolute coordinates
        parent_id = cell.get('parent')
        parent_x, parent_y = get_absolute_parent_coords(parent_id)

        try:
            if cell.get('vertex') == '1':
                # (MOD) Calculate absolute position
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
                    # (MOD) Calculate absolute positions
                    src_x = parent_x + float(source_point.get('x'))
                    src_y = parent_y + float(source_point.get('y'))
                    tgt_x = parent_x + float(target_point.get('x'))
                    tgt_y = parent_y + float(target_point.get('y'))

                    mid_x = (src_x + tgt_x) / 2
                    mid_y = (src_y + tgt_y) / 2

                    coordinates[cell_id] = (mid_x - origin_x, origin_y - mid_y)

        except (ValueError, TypeError, KeyError):
            continue

    # --- File Writing ---
    output_defs_file = output_base + '-defs.tex'
    output_coords_file = output_base + '-coords.tex'

    try:
        with open(output_defs_file, 'w') as f:
            f.write(f"% Auto-generated definitions by parse_drawio.py\n")
            f.write(f"% Page: {page_name}\n")
            f.write(f"\\def\\drawionativewidthpx{{{diagram_width_px:.4f}}}\n")
            f.write(f"\\def\\drawionativeheightpx{{{diagram_height_px:.4f}}}\n")
        print(f"  -> Successfully generated {output_defs_file}")
    except IOError:
        print(f"Error: Could not write to {output_defs_file}")
        sys.exit(1)

    try:
        written_count = 0
        with open(output_coords_file, 'w') as f:
            f.write(f"% Auto-generated coordinates by parse_drawio.py\n")
            f.write(f"% Page: {page_name}\n")

            if not coordinates:
                f.write(f"% WARNING: No objects with an 'id' attribute were found.\n")

            for name, (x, y) in coordinates.items():
                if not name.isdigit():
                    f.write(f"\\coordinate ({name}) at ({x:.2f}, {y:.2f});\n")
                    written_count += 1

        print(f"  -> Successfully generated {output_coords_file} ({written_count} coordinates).")

    except IOError:
        print(f"Error: Could not write to {output_coords_file}")
        sys.exit(1)

def main_parser(input_file, output_dir):
    """
    Parses the main .drawio file and iterates over all pages.
    """
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
    except ET.ParseError:
        print(f"Error: Could not parse {input_file}. Is it an uncompressed .drawio XML file?")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        sys.exit(1)

    ns_map = {}
    if '}' in root.tag:
        ns_uri = root.tag.split('}')[0][1:]
        ns_map = {'': ns_uri}

    pages = root.findall('.//diagram', ns_map)

    if pages:
        # Multi-page file
        print(f"Found {len(pages)} pages in {input_file}. Processing...")
        for page_element in pages:
            page_name = page_element.get('name', 'Unnamed_Page')

            # Sanitize page name for use as a filename
            safe_page_name = page_name.replace(' ', '_').replace(os.path.sep, '-')

            print(f"Processing page: '{page_name}' (as '{safe_page_name}')...")

            name = str(Path(input_file).stem) + "-" + safe_page_name
            output_base = os.path.join(output_dir, name)
            mx_graph_model = page_element.find('./mxGraphModel', ns_map)
            process_page_model(mx_graph_model, output_base, ns_map, page_name)

    else:
        # Single-page file (no <diagram> tags)
        print(f"Single-page file detected: {input_file}. Processing...")

        # Use the input filename (minus extension) as the base name
        page_name = os.path.splitext(os.path.basename(input_file))[0]
        output_base = os.path.join(output_dir, page_name)

        mx_graph_model = root.find('.//mxGraphModel', ns_map)
        process_page_model(mx_graph_model, output_base, ns_map, "Default Page")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse a .drawio file (all pages) to generate scalable TikZ coordinates."
    )
    parser.add_argument(
        "input_file",
        help="Path to the input .drawio file (uncompressed XML)."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=".",
        help="Directory to save the output files. Page names (or the input filename) "
             "will be used as the base name. (Default: current directory)"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    if args.output_dir != "." and not os.path.exists(args.output_dir):
        print(f"Creating output directory: {args.output_dir}")
        os.makedirs(args.output_dir)

    main_parser(args.input_file, args.output_dir)
