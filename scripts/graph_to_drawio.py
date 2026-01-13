#!/usr/bin/env python3
"""
Generate a .drawio diagram from the attack/defense YAML timeline.

The output is an uncompressed XML .drawio file for easy manual editing.
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml


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


def build_drawio(data, output_path, diagram_name="attack-defense-timeline"):
    layout = data.get("layout", {})
    drawio = layout.get("drawio", {})

    column_width = float(drawio.get("column_width", 160))
    margin_left = float(drawio.get("margin_left", 80))
    margin_top = float(drawio.get("margin_top", 40))
    year_row_height = float(drawio.get("year_row_height", 28))
    year_between_lanes = bool(drawio.get("year_between_lanes", False))
    phase_height = float(drawio.get("phase_height", 20))
    phase_gap = float(drawio.get("phase_gap", 10))
    node_width = float(drawio.get("node_width", 120))
    node_height = float(drawio.get("node_height", 32))
    stack_gap = float(drawio.get("stack_gap", 8))
    lane_gap = float(drawio.get("lane_gap", 40))
    include_edges = bool(drawio.get("include_edges", False))

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    phases = layout.get("phases", [])

    years = sorted({n["year"] for n in nodes})
    year_index = {y: i for i, y in enumerate(years)}

    # Group nodes by year and lane
    by_year_lane = {y: {"attack": [], "defense": []} for y in years}
    for node in nodes:
        by_year_lane[node["year"]][node_lane(node)].append(node)

    max_attack_stack = max((len(by_year_lane[y]["attack"]) for y in years), default=0)
    max_defense_stack = max((len(by_year_lane[y]["defense"]) for y in years), default=0)

    attack_row_y = margin_top + year_row_height + phase_height + phase_gap
    attack_stack_height = max_attack_stack * (node_height + stack_gap)
    if year_between_lanes:
        gap_above_year = max((lane_gap - year_row_height) / 2, 6)
        year_row_y = attack_row_y + attack_stack_height + gap_above_year
        defense_row_y = year_row_y + year_row_height + gap_above_year
    else:
        year_row_y = margin_top
        defense_row_y = attack_row_y + attack_stack_height + lane_gap

    width = margin_left * 2 + len(years) * column_width
    height = defense_row_y + max_defense_stack * (node_height + stack_gap) + 80

    mxfile = ET.Element("mxfile", {
        "host": "app.diagrams.net",
        "modified": "2024-01-01T00:00:00.000Z",
        "agent": "yaml_to_drawio.py",
        "version": "20.8.10",
        "type": "device",
    })
    diagram = ET.SubElement(mxfile, "diagram", {"name": diagram_name})
    model = ET.SubElement(diagram, "mxGraphModel", {
        "dx": str(width),
        "dy": str(height),
        "grid": "1",
        "gridSize": "10",
        "guides": "1",
        "tooltips": "1",
        "connect": "1",
        "arrows": "1",
        "fold": "1",
        "page": "1",
        "pageScale": "1",
        "pageWidth": str(width),
        "pageHeight": str(height),
        "math": "0",
        "shadow": "0",
    })
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    next_id = 2

    # Layers
    layer_ids = {}
    for layer_name in ("phases", "labels", "attacks", "defenses", "edges"):
        layer_id = str(next_id)
        next_id += 1
        layer_ids[layer_name] = layer_id
        drawio_cell(root, layer_id, value=layer_name, style="layer=1", parent="1")

    # Phase bands
    for phase in phases:
        start = phase["start"]
        end = phase["end"]
        if start not in year_index or end not in year_index:
            continue
        start_idx = year_index[start]
        end_idx = year_index[end]
        x = margin_left + start_idx * column_width
        w = (end_idx - start_idx + 1) * column_width
        y = margin_top + year_row_height + 4 if not year_between_lanes else margin_top + 4
        label = phase.get("label", "")
        style = (
            "rounded=1;whiteSpace=wrap;html=1;"
            f"fillColor={phase.get('fill', '#f7f7f7')};"
            f"strokeColor={phase.get('border', '#e6e6e6')};"
            "opacity=35;strokeOpacity=35;"
        )
        drawio_cell(
            root,
            next_id,
            value=label,
            style=style,
            parent=layer_ids["phases"],
            vertex=True,
            geometry={
                "x": str(x),
                "y": str(y),
                "width": str(w),
                "height": str(phase.get("height", phase_height)),
            },
        )
        next_id += 1

    # Year labels
    for year in years:
        x = margin_left + year_index[year] * column_width + (column_width / 2) - 15
        y = year_row_y
        style = "text;html=1;align=center;verticalAlign=middle;resizable=0;"
        drawio_cell(
            root,
            next_id,
            value=str(year),
            style=style,
            parent=layer_ids["labels"],
            vertex=True,
            geometry={"x": str(x), "y": str(y), "width": "40", "height": str(year_row_height)},
        )
        next_id += 1

    # Lane labels
    label_x = 10
    attack_label_y = attack_row_y + (node_height / 2) - 8
    defense_label_y = defense_row_y + (node_height / 2) - 8
    for label, y in (("Attacks", attack_label_y), ("Defenses", defense_label_y)):
        style = "text;html=1;align=left;verticalAlign=middle;resizable=0;fontStyle=1;"
        drawio_cell(
            root,
            next_id,
            value=label,
            style=style,
            parent=layer_ids["labels"],
            vertex=True,
            geometry={"x": str(label_x), "y": str(y), "width": "70", "height": "20"},
        )
        next_id += 1

    # Nodes
    node_id_map = {}
    for year in years:
        x = margin_left + year_index[year] * column_width + (column_width - node_width) / 2

        for lane in ("attack", "defense"):
            items = by_year_lane[year][lane]
            for i, node in enumerate(items):
                y_base = attack_row_y if lane == "attack" else defense_row_y
                y = y_base + i * (node_height + stack_gap)
                style = (
                    "rounded=1;whiteSpace=wrap;html=1;"
                    "fillColor=#ffcccc;strokeColor=#cc0000;"
                    if lane == "attack"
                    else "rounded=1;whiteSpace=wrap;html=1;fillColor=#cce5ff;strokeColor=#0066cc;"
                )
                drawio_cell(
                    root,
                    next_id,
                    value=node["label"],
                    style=style,
                    parent=layer_ids["attacks"] if lane == "attack" else layer_ids["defenses"],
                    vertex=True,
                    geometry={
                        "x": str(x),
                        "y": str(y),
                        "width": str(node_width),
                        "height": str(node_height),
                    },
                )
                node_id_map[node["id"]] = next_id
                next_id += 1

    # Edges (optional)
    if include_edges:
        for edge in edges:
            if edge["from"] not in node_id_map or edge["to"] not in node_id_map:
                continue
            style = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
            edge_cell = drawio_cell(
                root,
                next_id,
                value="",
                style=style,
                parent=layer_ids["edges"],
                edge=True,
            )
            edge_cell.set("source", str(node_id_map[edge["from"]]))
            edge_cell.set("target", str(node_id_map[edge["to"]]))
            ET.SubElement(edge_cell, "mxGeometry", {"relative": "1", "as": "geometry"})
            next_id += 1

    tree = ET.ElementTree(mxfile)
    output_path.write_text(ET.tostring(mxfile, encoding="unicode"))


def main():
    parser = argparse.ArgumentParser(description="Generate a drawio diagram from YAML.")
    parser.add_argument("yaml_path", help="Path to attack-defense-landscape.yaml")
    parser.add_argument(
        "-o", "--output", default="scripts/attack_defense_timeline.drawio", help="Output .drawio file"
    )
    parser.add_argument("--name", default="attack-defense-timeline", help="Diagram name")
    args = parser.parse_args()

    data = yaml.safe_load(Path(args.yaml_path).read_text())
    build_drawio(data, Path(args.output), diagram_name=args.name)


if __name__ == "__main__":
    main()
