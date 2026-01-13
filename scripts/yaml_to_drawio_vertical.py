#!/usr/bin/env python3
"""
Generate a VERTICAL .drawio diagram from the attack/defense YAML timeline.
Features:
- Vertical Spine (Years)
- Attacks Left, Defenses Right
- Wrapping for dense years (prevent too wide diagrams)
- Automatic height adjustment
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml
import math


def node_lane(node):
    lane = node.get("lane")
    if lane:
        return lane
    if node.get("type") == "attack":
        return "attack"
    return "defense"


def drawio_cell(root, cell_id, value="", style="", parent="1", vertex=False, edge=False, geometry=None):
    cell = ET.SubElement(root, "mxCell", {
        "id": str(cell_id),
        "value": value,
        "style": style,
        "parent": str(parent),
    })
    if vertex:
        cell.set("vertex", "1")
    if edge:
        cell.set("edge", "1")
    if geometry:
        geo = ET.SubElement(cell, "mxGeometry", geometry)
        geo.set("as", "geometry")
    return cell


def build_drawio_vertical(data, output_path, diagram_name="attack-defense-timeline-vert"):
    layout = data.get("layout", {})
    drawio_conf = layout.get("drawio", {})

    # Configuration
    MAX_COLS = 3          # Wrap after this many nodes horizontally
    base_year_gap = 60    # Minimum vertical gap between years
    center_x = 400        # X-coordinate of the Year Spine
    lane_offset = 100     # Distance from center spine to first column
    
    node_width = float(drawio_conf.get("node_width", 120))
    node_height = float(drawio_conf.get("node_height", 32))
    col_gap = 20          # Horizontal gap between nodes
    row_gap = 10          # Vertical gap between wrapped rows
    
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    years = sorted({n["year"] for n in nodes})
    
    # Group nodes
    by_year_lane = {y: {"attack": [], "defense": []} for y in years}
    for node in nodes:
        by_year_lane[node["year"]][node_lane(node)].append(node)

    # Calculate total height roughly (will refine during placement)
    # Just to set canvas size initially
    total_height = len(years) * 150 + 200
    width = 1000 

    mxfile = ET.Element("mxfile", {
        "host": "Electron",
        "agent": "Mozilla/5.0",
        "version": "24.0.4",
        "type": "device",
    })
    diagram = ET.SubElement(mxfile, "diagram", {"name": diagram_name, "id": "vertical_timeline"})
    model = ET.SubElement(diagram, "mxGraphModel", {
        "dx": str(width),
        "dy": str(total_height),
        "grid": "1",
        "gridSize": "10",
        "guides": "1",
        "tooltips": "1",
        "connect": "1",
        "arrows": "1",
        "fold": "1",
        "page": "1",
        "pageScale": "1",
        "pageWidth": "827",
        "pageHeight": "1169",
        "math": "0",
        "shadow": "0",
    })
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    next_id = 2
    node_id_map = {}

    # Layers
    layer_ids = {}
    for layer_name in ("background", "spine", "attacks", "defenses", "edges", "labels"):
        layer_id = str(next_id)
        next_id += 1
        layer_ids[layer_name] = layer_id
        drawio_cell(root, layer_id, value=layer_name, style="layer=1", parent="1")

    # 0. Headers
    header_y = 20
    header_style = "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=18;fontStyle=1"
    
    # Attack Header (Left)
    drawio_cell(root, next_id, value="Attacks", style=header_style, parent=layer_ids["labels"], vertex=True,
                geometry={"x": str(center_x - lane_offset - node_width), "y": str(header_y), "width": str(node_width), "height": "30"})
    next_id += 1
    
    # Defense Header (Right)
    drawio_cell(root, next_id, value="Defenses", style=header_style, parent=layer_ids["labels"], vertex=True,
                geometry={"x": str(center_x + lane_offset), "y": str(header_y), "width": str(node_width), "height": "30"})
    next_id += 1

    # Draw Timeline
    y_cursor = 80
    
    for i, year in enumerate(years):
        attacks = by_year_lane[year]["attack"]
        defenses = by_year_lane[year]["defense"]
        
        # Calculate rows needed for this year
        attack_rows = math.ceil(len(attacks) / MAX_COLS) if attacks else 0
        defense_rows = math.ceil(len(defenses) / MAX_COLS) if defenses else 0
        max_rows = max(1, attack_rows, defense_rows) # At least 1 row height for the year label itself
        
        year_block_height = max_rows * (node_height + row_gap)
        
        # 1. Draw Year Node (Spine) - Simple Text
        year_style = (
            "text;html=1;align=center;verticalAlign=middle;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;fontSize=14;fontStyle=1;fontColor=#666666;"
        )
        drawio_cell(
            root, next_id, value=str(year), style=year_style,
            parent=layer_ids["spine"], vertex=True,
            geometry={"x": str(center_x - 20), "y": str(y_cursor + (node_height/2) - 10), "width": "40", "height": "20"}
        )
        year_node_id = next_id
        next_id += 1
        
        # Add spine line to previous year?
        # Maybe imply spine by alignment, or dotted line? 
        # Let's add a subtle dotted line if not first
        if i > 0:
             # from previous year bottom roughly
             # y_cursor is top of current
             # previous y was... simpler: vertical line from y=80 down to bottom
             pass

        # 2. Draw Attacks (Left)
        # Fill from Right to Left (closest to spine first)
        for j, node in enumerate(attacks):
            # Row and Col
            row = j // MAX_COLS
            col = j % MAX_COLS
            
            # X Calculation (Left side)
            # col 0 is closest to spine: center - offset - width
            # col 1 is further left: center - offset - width - gap - width ...
            x = center_x - lane_offset - ((col + 1) * node_width) - (col * col_gap)
            
            # Y Calculation
            y = y_cursor + row * (node_height + row_gap)
            
            style = "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffcccc;strokeColor=#cc0000;glass=0;shadow=0;"
            drawio_cell(
                root, next_id, value=node["label"], style=style,
                parent=layer_ids["attacks"], vertex=True,
                geometry={"x": str(x), "y": str(y), "width": str(node_width), "height": str(node_height)}
            )
            node_id_map[node["id"]] = next_id
            next_id += 1

        # 3. Draw Defenses (Right)
        # Fill from Left to Right (closest to spine first)
        for j, node in enumerate(defenses):
            row = j // MAX_COLS
            col = j % MAX_COLS
            
            # X Calculation (Right side)
            # col 0 is closest to spine: center + offset
            # col 1 is further right: center + offset + width + gap
            x = center_x + lane_offset + (col * (node_width + col_gap))
            
            y = y_cursor + row * (node_height + row_gap)
            
            style = "rounded=1;whiteSpace=wrap;html=1;fillColor=#cce5ff;strokeColor=#0066cc;glass=0;shadow=0;"
            drawio_cell(
                root, next_id, value=node["label"], style=style,
                parent=layer_ids["defenses"], vertex=True,
                geometry={"x": str(x), "y": str(y), "width": str(node_width), "height": str(node_height)}
            )
            node_id_map[node["id"]] = next_id
            next_id += 1

        # Advance Cursor
        y_cursor += year_block_height + base_year_gap

    # Spine Line (Background)
    # Draw a line from top year to bottom year behind everything
    spine_style = "endArrow=none;html=1;strokeWidth=1;strokeColor=#CCCCCC;dashed=1;"
    spine_cell = drawio_cell(root, next_id, value="", style=spine_style, parent=layer_ids["background"], edge=True)
    ET.SubElement(spine_cell, "mxGeometry", {
        "relative": "1", "as": "geometry", 
        "x": "0", "y": "0", "width": "50", "height": "50"
    })
    # Set points manually from top to bottom
    geo = spine_cell.find("mxGeometry")
    # points = ET.SubElement(geo, "mxPoint") ... hard to set points on unconnected edge easily in xml
    # easier to just use a vertex line
    next_id += 1
    
    line_style = "rounded=0;whiteSpace=wrap;html=1;strokeColor=none;fillColor=#E0E0E0;"
    drawio_cell(
        root, next_id, value="", style=line_style, parent=layer_ids["background"], vertex=True,
        geometry={"x": str(center_x - 1), "y": "60", "width": "2", "height": str(y_cursor - 60)}
    )
    next_id += 1


    # Edges
    for edge in edges:
        if edge["from"] not in node_id_map or edge["to"] not in node_id_map:
            continue
        
        # Styles
        # primary: bold, solid
        # otherwise: thin, gray
        is_primary = edge.get("primary", False)
        color = "#000000" if is_primary else "#999999"
        width = "2" if is_primary else "1"
        style = f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;curved=1;strokeColor={color};strokeWidth={width};"
        
        edge_cell = drawio_cell(
            root, next_id, value="", style=style,
            parent=layer_ids["edges"], edge=True
        )
        edge_cell.set("source", str(node_id_map[edge["from"]]))
        edge_cell.set("target", str(node_id_map[edge["to"]]))
        ET.SubElement(edge_cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        next_id += 1

    output_path.write_text(ET.tostring(mxfile, encoding="unicode"))

def main():
    parser = argparse.ArgumentParser(description="Generate VERTICAL drawio from YAML.")
    parser.add_argument("yaml_path", help="Path to attack-defense-landscape.yaml")
    parser.add_argument("-o", "--output", default="figures/drawio/attack-defense-timeline-vert.drawio", help="Output .drawio file")
    args = parser.parse_args()

    data = yaml.safe_load(Path(args.yaml_path).read_text())
    build_drawio_vertical(data, Path(args.output))

if __name__ == "__main__":
    main()