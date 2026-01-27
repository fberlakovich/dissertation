import argparse
from pathlib import Path
import math
import yaml

from pylatex import NoEscape
from pylatex.utils import escape_latex
from pylatex.tikz import (
    TikZ,
    TikZNode,
    TikZCoordinate,
    TikZDraw,
    TikZOptions,
    TikZPathList,
)

def format_label(raw_label):
    if isinstance(raw_label, list):
        lines = [str(line) for line in raw_label]
    else:
        raw = str(raw_label)
        if "\\\\" in raw or "\n" in raw:
            lines = raw.replace("\\\\", "\n").splitlines()
        else:
            lines = []
    if lines:
        escaped = [escape_latex(line.strip()) for line in lines]
        body = r"\\ ".join(escaped)
        return NoEscape(body)
    label = escape_latex(str(raw_label))
    return NoEscape(label)

def generate_tikz(nodes, output_path):
    years = sorted({n["year"] for n in nodes})
    by_year_lane = {y: {"attack": [], "defense": []} for y in years}
    for node in nodes:
        lane = node.get("lane")
        if not lane:
            lane = "attack" if node.get("type") == "attack" else "defense"
        by_year_lane[node["year"]][lane].append(node)
    
    # Configuration
    MAX_COLS = 2
    ROW_HEIGHT = 1.0      # Increased from 0.8 for better spacing when text wraps
    COL_WIDTH = 4.0
    SPINE_OFFSET_X = 2.0
    SPINE_PADDING_X = 0.6
    NODE_WIDTH = 3.6
    TEXT_WIDTH = 3.4      # Increased from 2.7 to fit longer labels
    LABEL_Y = 1.0

    tikz = TikZ(
        options=TikZOptions(
            r"year node/.style={font=\bfseries\sffamily\color{gray}, align=center, inner sep=2pt},"
            r"attack node/.style={draw=red!80, fill=red!5, rounded corners=2pt, font=\scriptsize\sffamily, align=center, "
            r"minimum width="
            + f"{NODE_WIDTH:.2f}"
            + r"cm, minimum height=0.5cm, inner sep=2pt, text width="
            + f"{TEXT_WIDTH:.2f}"
            + r"cm, execute at begin node=\setlength{\emergencystretch}{0pt}\tolerance 200\hyphenpenalty 10000\exhyphenpenalty 10000},"
            r"defense node/.style={draw=blue!80, fill=blue!5, rounded corners=2pt, font=\scriptsize\sffamily, align=center, "
            r"minimum width="
            + f"{NODE_WIDTH:.2f}"
            + r"cm, minimum height=0.5cm, inner sep=2pt, text width="
            + f"{TEXT_WIDTH:.2f}"
            + r"cm, execute at begin node=\setlength{\emergencystretch}{0pt}\tolerance 200\hyphenpenalty 10000\exhyphenpenalty 10000},"
            r"spine/.style={thick, gray!30, dashed}"
        )
    )

    lane_center_x = (SPINE_OFFSET_X + SPINE_PADDING_X + ((MAX_COLS - 1) * COL_WIDTH) + (NODE_WIDTH / 2)) / 2
    tikz.append(
        TikZNode(
            options=TikZOptions(r"font=\large\bfseries\sffamily"),
            at=TikZCoordinate(-lane_center_x, LABEL_Y),
            text="Attacks",
        )
    )
    tikz.append(
        TikZNode(
            options=TikZOptions(r"font=\large\bfseries\sffamily"),
            at=TikZCoordinate(lane_center_x, LABEL_Y),
            text="Defenses",
        )
    )
    
    y_offset = 0
    spine_start = 0.8
    max_y = LABEL_Y
    min_y = 0
    
    for year in years:
        attacks = by_year_lane[year]["attack"]
        defenses = by_year_lane[year]["defense"]
        
        a_rows = math.ceil(len(attacks) / MAX_COLS) if attacks else 0
        d_rows = math.ceil(len(defenses) / MAX_COLS) if defenses else 0
        max_rows = max(1, a_rows, d_rows)
        
        block_height = max_rows * ROW_HEIGHT
        
        # Center year label
        center_of_stack = y_offset - ((max_rows - 1) * ROW_HEIGHT) / 2
        
        tikz.append(NoEscape(f"  % Year {year}"))
        tikz.append(
            TikZNode(
                options=TikZOptions("year node"),
                at=TikZCoordinate(0, round(center_of_stack, 2)),
                text=str(year),
            )
        )
        max_y = max(max_y, center_of_stack)
        min_y = min(min_y, center_of_stack)
        
        attack_top = center_of_stack + ((a_rows - 1) * ROW_HEIGHT) / 2 if a_rows else center_of_stack
        defense_top = center_of_stack + ((d_rows - 1) * ROW_HEIGHT) / 2 if d_rows else center_of_stack

        for i, node in enumerate(attacks):
            row = i // MAX_COLS
            col = i % MAX_COLS
            x = -(SPINE_OFFSET_X + SPINE_PADDING_X + (col * COL_WIDTH))
            y = attack_top - (row * ROW_HEIGHT)

            # Use 'cite' field if present (legacy), otherwise use 'id' unless 'nocite' is set
            cite_key = node.get("cite") or (node.get("id") if not node.get("nocite") else None)
            cite_cmd = r" \allowbreak\cite{" + str(cite_key) + r"}" if cite_key else ""
            label_text = format_label(node["label"])
            node_text = NoEscape(f"{label_text}{cite_cmd}")

            tikz.append(
                TikZNode(
                    options=TikZOptions("attack node"),
                    at=TikZCoordinate(round(x, 2), round(y, 2)),
                    text=node_text,
                )
            )
            max_y = max(max_y, y)
            min_y = min(min_y, y)

        for i, node in enumerate(defenses):
            row = i // MAX_COLS
            col = i % MAX_COLS
            x = (SPINE_OFFSET_X + SPINE_PADDING_X + (col * COL_WIDTH))
            y = defense_top - (row * ROW_HEIGHT)

            # Use 'cite' field if present (legacy), otherwise use 'id' unless 'nocite' is set
            cite_key = node.get("cite") or (node.get("id") if not node.get("nocite") else None)
            cite_cmd = r" \allowbreak\cite{" + str(cite_key) + r"}" if cite_key else ""
            label_text = format_label(node["label"])
            node_text = NoEscape(f"{label_text}{cite_cmd}")

            tikz.append(
                TikZNode(
                    options=TikZOptions("defense node"),
                    at=TikZCoordinate(round(x, 2), round(y, 2)),
                    text=node_text,
                )
            )
            max_y = max(max_y, y)
            min_y = min(min_y, y)
        
        # Reduced gap between years
        y_offset -= (block_height + 0.2) 

    spine_end = y_offset + 0.2
    tikz.append(
        TikZDraw(
            path=TikZPathList(
                TikZCoordinate(0, spine_start),
                "--",
                TikZCoordinate(0, round(spine_end, 2)),
            ),
            options=TikZOptions("spine"),
        )
    )
    bbox_padding_y = ROW_HEIGHT
    bbox_min_y = min_y - bbox_padding_y
    bbox_max_y = max_y + bbox_padding_y
    bbox_max_x = SPINE_OFFSET_X + SPINE_PADDING_X + ((MAX_COLS - 1) * COL_WIDTH) + (NODE_WIDTH / 2)
    bbox_min_x = -bbox_max_x
    tikz.append(
        NoEscape(
            r"\path[use as bounding box] ("
            + f"{bbox_min_x:.2f}"
            + r","
            + f"{bbox_min_y:.2f}"
            + r") rectangle ("
            + f"{bbox_max_x:.2f}"
            + r","
            + f"{bbox_max_y:.2f}"
            + r");"
        )
    )

    # Output raw tikzpicture - let the including document handle sizing
    with open(output_path, "w") as f:
        f.write(tikz.dumps())

def epoch_output_paths(output_path, epochs):
    base = Path(output_path)
    if base.suffix:
        stem = base.stem
        suffix = base.suffix
        parent = base.parent
    else:
        stem = base.name
        suffix = ""
        parent = base.parent
    for epoch in epochs:
        epoch_id = epoch.get("id", "epoch")
        epoch_slug = epoch_id.removeprefix("epoch_").replace("_", "-")
        filename = f"{stem}-{epoch_slug}{suffix}"
        yield epoch, parent / filename

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path")
    parser.add_argument("-o", "--output")
    parser.add_argument("--split-epochs", action="store_true", help="write one figure per epoch (also writes combined file)")
    args = parser.parse_args()
    with open(args.yaml_path, "r") as f:
        data = yaml.safe_load(f)
    nodes = data.get("nodes", [])
    if args.split_epochs:
        epochs = sorted(data.get("epochs", []), key=lambda e: e.get("start", 0))
        for epoch, output_path in epoch_output_paths(args.output, epochs):
            start = epoch.get("start")
            end = epoch.get("end")
            if start is None or end is None:
                continue
            epoch_nodes = [n for n in nodes if start <= n.get("year", 0) <= end]
            generate_tikz(epoch_nodes, output_path)

        # Also generate a combined file with all epochs
        combined_path = Path(args.output)
        if combined_path.suffix:
            combined_file = combined_path.parent / f"{combined_path.stem}-all{combined_path.suffix}"
        else:
            combined_file = combined_path.parent / f"{combined_path.name}-all"
        generate_tikz(nodes, combined_file)
    else:
        generate_tikz(nodes, args.output)

if __name__ == "__main__":
    main()
