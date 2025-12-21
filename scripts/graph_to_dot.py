import os
import yaml
import textwrap
import graphviz
import argparse
import re
import subprocess
from PIL import Image


def load_data(filename):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)


def create_timeline(data, emit_dot_path=None, hide_edges=False, direction='TB'):
    # Initialize Digraph with specified layout direction
    dot = graphviz.Digraph(comment='Attack Defense Timeline', format='png')

    layout = data.get('layout', {})
    graph_cfg = layout.get('graph', {})
    edge_cfg = layout.get('edges', {})
    node_cfg = layout.get('nodes', {})
    year_cfg = layout.get('year_labels', {})
    lane_cfg = layout.get('lane_labels', {})
    lanes = layout.get('lanes', [
        {'id': 'attack', 'label': 'Attacks'},
        {'id': 'defense', 'label': 'Defenses'},
    ])
    lane_ids = [lane['id'] for lane in lanes]

    dot.attr(rankdir=direction,
             splines='ortho',
             nodesep=str(graph_cfg.get('nodesep', '0.6')),
             ranksep=str(graph_cfg.get('ranksep', '1.0')),
             fontname='Helvetica',
             bgcolor='white',
             pad=str(graph_cfg.get('pad', '0.5')),
             nslimit='5',
             mclimit='5')

    # Node defaults
    dot.attr('node', fontname='Fira Sans', fontsize='9')

    # Group nodes by year
    nodes_by_year = {}
    node_map = {}  # id -> node data
    for node in data['nodes']:
        year = node['year']
        if year not in nodes_by_year:
            nodes_by_year[year] = []
        nodes_by_year[year].append(node)
        node_map[node['id']] = node

    sorted_years = sorted(nodes_by_year.keys())

    # Pre-sort nodes within each year (lane order, then id)
    for year in sorted_years:
        nodes_by_year[year] = sorted(
            nodes_by_year[year],
            key=lambda x: (lane_ids.index(node_lane(x, lane_ids)) if node_lane(x, lane_ids) in lane_ids else 99, x['id'])
        )

    year_label_ids = []
    anchor_map = {lane_id: {} for lane_id in lane_ids}

    labeled_years = set()
    # Year label row
    with dot.subgraph(name='rank_years') as s:
        s.attr(rank='same')
        for year in sorted_years:
            year_id = f'year_{year}'
            year_label_ids.append(year_id)
            label_text = str(year) if is_year_labeled(year, sorted_years, year_cfg) else ''
            if label_text:
                labeled_years.add(year)
            s.node(year_id,
                   label=label_text,
                   shape='plaintext' if label_text else 'point',
                   style='invis' if not label_text else '',
                   group=f'year_{year}',
                   fontsize=str(year_cfg.get('fontsize', '12')),
                   fontname=year_cfg.get('fontname', 'Fira Sans Bold'),
                   width='0',
                   height='0')

    # Lane rows (anchors + labels)
    for lane in lanes:
        lane_id = lane['id']
        lane_label_id = f'lane_label_{lane_id}'
        with dot.subgraph(name=f'rank_lane_{lane_id}') as s:
            s.attr(rank='same')
            s.node(lane_label_id,
                   label=lane.get('label', lane_id),
                   shape='plaintext',
                   fontsize=str(lane_cfg.get('fontsize', '11')),
                   fontname=lane_cfg.get('fontname', 'Fira Sans Bold'),
                   width='0',
                   height='0')
            for year in sorted_years:
                anchor_id = f'anchor_{lane_id}_{year}'
                anchor_map[lane_id][year] = anchor_id
                s.node(anchor_id,
                       label='',
                       shape='point',
                       group=f'year_{year}',
                       width='0.01',
                       height='0.01',
                       style='invis')

    # Callouts row (optional)
    phases = layout.get('phases', [])
    phase_cfg = layout.get('phase_bands', {})

    # Optionally pin lane labels to the first year column
    label_year = sorted_years[0] if sorted_years else None
    if lane_cfg.get('align_to_first_year', False) and label_year is not None:
        for lane in lanes:
            lane_id = lane['id']
            dot.edge(f'lane_label_{lane_id}', anchor_map[lane_id][label_year],
                     style='invis',
                     weight='25',
                     minlen='1')

    # Render nodes per year/lane and tie them to anchors
    for year in sorted_years:
        nodes_in_year = nodes_by_year[year]

        # Align anchors and nodes per lane and year
        for lane_id in lane_ids:
            with dot.subgraph(name=f'rank_{lane_id}_{year}') as s:
                s.attr(rank='same')
                s.node(anchor_map[lane_id][year], label='', shape='point', width='0.01', height='0.01', style='invis')
                for node in nodes_in_year:
                    if node_lane(node, lane_ids) == lane_id:
                        s.node(node['id'])

        for node in nodes_in_year:
            clean_label = node['label'].replace('\n', ' ')
            wrap_width = int(node_cfg.get('wrap', 20))
            wrapped_text = "\\n".join(textwrap.wrap(clean_label, width=wrap_width))
            highlight = bool(node.get('highlight', False))
            node_width = str(node_cfg.get('width', '1.4'))
            node_height = str(node_cfg.get('height', '0.4'))
            node_margin = str(node_cfg.get('margin', '0.2,0.06'))
            node_fontsize = str(node_cfg.get('fontsize', '9'))

            if node['type'] == 'attack':
                fill = "#ffcccc"
                stroke = "#cc0000"
                penwidth = '1.6' if highlight else '1.0'
                dot.node(node['id'],
                         label=wrapped_text,
                         shape='box',
                         style='filled,rounded',
                         fillcolor=fill,
                         color=stroke,
                         penwidth=penwidth,
                         group=f'year_{year}',
                         width=node_width,
                         height=node_height,
                         fixedsize='false',
                         margin=node_margin,
                         fontsize=node_fontsize,
                         fontname=node_cfg.get('fontname', 'Fira Sans'))
            else:
                # Defense node - add category indicator(s)
                tags = node.get('tags', [])

                # Collect all matching category colors
                cat_colors = []
                if 'cfi' in tags:
                    cat_colors.append('#9370DB')  # purple for CFI
                if 'enforcement' in tags:
                    cat_colors.append('#FF8C00')  # orange for enforcement
                if 'randomization' in tags:
                    cat_colors.append('#20B2AA')  # teal for randomization
                if 'XOM' in tags:
                    cat_colors.append('#708090')  # gray for XOM

                penwidth = '1.6' if highlight else '1.0'
                if cat_colors:
                    # Build squares HTML for all categories
                    squares_html = ''
                    for cat_color in cat_colors:
                        squares_html += f'<TD VALIGN="MIDDLE"><TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="0"><TR><TD FIXEDSIZE="TRUE" WIDTH="10" HEIGHT="10" BGCOLOR="{cat_color}"></TD></TR></TABLE></TD><TD WIDTH="2"></TD>'

                    html_label = f'''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">
                        <TR>
                            {squares_html}
                            <TD WIDTH="2"></TD>
                            <TD VALIGN="MIDDLE">{wrapped_text.replace(chr(92) + "n", "<BR/>")}</TD>
                        </TR>
                    </TABLE>>'''
                    dot.node(node['id'],
                             label=html_label,
                             shape='box',
                             style='filled,rounded',
                             fillcolor='#cce5ff',
                             color='#0066cc',
                             penwidth=penwidth,
                             group=f'year_{year}',
                             width=node_width,
                             height=node_height,
                             fixedsize='false',
                             margin=node_margin,
                             fontsize=node_fontsize,
                             fontname=node_cfg.get('fontname', 'Fira Sans'))
                else:
                    dot.node(node['id'],
                             label=wrapped_text,
                             shape='box',
                             style='filled,rounded',
                             fillcolor='#cce5ff',
                             color='#0066cc',
                             penwidth=penwidth,
                             group=f'year_{year}',
                             width=node_width,
                             height=node_height,
                             fixedsize='false',
                             margin=node_margin,
                             fontsize=node_fontsize,
                             fontname=node_cfg.get('fontname', 'Fira Sans'))

            lane_id = node_lane(node, lane_ids)
            anchor_id = anchor_map.get(lane_id, {}).get(year)
            if anchor_id:
                dot.edge(anchor_id, node['id'],
                         style='invis',
                         weight='5',
                         minlen='1')

        # Align year labels with attack anchors (if present) or defense anchors
        primary_lane = 'attack' if 'attack' in lane_ids else lane_ids[0]
        dot.edge(f'year_{year}', anchor_map[primary_lane][year],
                 style='invis',
                 weight='10',
                 minlen=str(year_cfg.get('anchor_minlen', '1')))

        # Align lanes vertically per year
        for upper, lower in zip(lane_ids, lane_ids[1:]):
            dot.edge(anchor_map[upper][year], anchor_map[lower][year],
                     style='invis',
                     weight='10',
                     minlen='1')

    # Phase bands (single span per phase)
    if phases:
        with dot.subgraph(name='rank_phases') as s:
            s.attr(rank='same')
            for phase in phases:
                phase_years = [y for y in sorted_years if phase['start'] <= y <= phase['end']]
                if not phase_years:
                    continue
                phase_id = phase['id']
                cell_width = float(phase_cfg.get('cell_width', '1.0'))
                band_width = str(max(cell_width * len(phase_years), cell_width))
                band_height = str(phase.get('height', phase_cfg.get('height', '0.16')))
                band_font = phase.get('fontname', phase_cfg.get('fontname', 'Fira Sans Bold'))
                band_fontsize = str(phase.get('fontsize', phase_cfg.get('fontsize', '8')))
                phase_label = phase.get('label', '')

                band_id = f'phase_{phase_id}'
                start_year = phase_years[0]
                end_year = phase_years[-1]

                # Create spacer nodes to anchor band between start and end years
                start_id = f'phase_{phase_id}_start'
                end_id = f'phase_{phase_id}_end'
                s.node(start_id, label='', shape='point', width='0.01', style='invis', group=f'year_{start_year}')
                s.node(end_id, label='', shape='point', width='0.01', style='invis', group=f'year_{end_year}')

                s.node(band_id,
                       label=phase_label,
                       shape='box',
                       style='filled,rounded',
                       fillcolor=phase.get('fill', '#f7f7f7'),
                       color=phase.get('border', '#e6e6e6'),
                       fontname=band_font,
                       fontsize=band_fontsize,
                       width=band_width,
                       height=band_height,
                       fixedsize='true',
                       group=f'year_{start_year}')

                # Pull band between start/end anchors
                s.edge(start_id, band_id, style='invis', weight='10', minlen='1')
                s.edge(band_id, end_id, style='invis', weight='10', minlen='1')
                dot.edge(band_id, f'year_{start_year}', style='invis', weight='5', minlen='1')

    # Create invisible edges to establish ordering
    # Link years along primary axis
    for i in range(len(sorted_years) - 1):
        dot.edge(f'year_{sorted_years[i]}',
                 f'year_{sorted_years[i + 1]}',
                 style='invis',
                 weight='100')

    # Chain anchors per lane to enforce horizontal ordering
    for lane_id in lane_ids:
        lane_label_id = f'lane_label_{lane_id}'
        prev = lane_label_id
        for year in sorted_years:
            anchor_id = anchor_map[lane_id][year]
            dot.edge(prev, anchor_id,
                     style='invis',
                     weight='30',
                     minlen='1')
            prev = anchor_id

    # Optional gridlines per year
    grid_cfg = layout.get('gridlines', {})
    if grid_cfg.get('enabled', False):
        grid_color = grid_cfg.get('color', '#dddddd')
        grid_style = grid_cfg.get('style', 'dotted')
        grid_width = grid_cfg.get('penwidth', '0.6')
        only_labeled = grid_cfg.get('only_labeled', True)
        for year in sorted_years:
            if only_labeled and year not in labeled_years:
                continue
            top_anchor = f'year_{year}'
            bottom_anchor = anchor_map[lane_ids[-1]][year]
            dot.edge(top_anchor, bottom_anchor,
                     color=grid_color,
                     style=grid_style,
                     penwidth=str(grid_width),
                     arrowhead='none',
                     constraint='false')

    # Draw relationship edges with appropriate styling
    for edge in data['edges']:
        if hide_edges:
            continue

        visibility = edge_cfg.get('visibility', 'all')
        if visibility == 'none':
            continue
        if visibility == 'minimal':
            if not edge.get('primary', False):
                continue

        u = edge['from']
        v = edge['to']

        if u not in node_map or v not in node_map:
            continue

        u_year = node_map[u]['year']
        v_year = node_map[v]['year']

        relation = edge.get('label', '')
        # Use colors with transparency (alpha channel) to reduce visual clutter
        if relation == 'motivated':
            color = "#228B22CC"  # green with transparency
            style = "solid"
        elif relation == 'bypasses':
            color = "#DC143CCC"  # red with transparency
            style = "dashed"
        elif relation == 'extends':
            color = "#4169E1CC"  # blue with transparency
            style = "solid"
        elif relation == 'inspired':
            color = "#9370DBCC"  # purple with transparency
            style = "dotted"
        else:
            color = "#666666CC"
            style = "solid"

        # Let Graphviz handle routing - don't use ports to avoid clipping issues
        dot.edge(u, v,
                 color=color,
                 style=style,
                 arrowhead='normal',
                 arrowsize='0.5',
                 penwidth='0.8',
                 constraint='false')


    if emit_dot_path:
        try:
            dot.save(emit_dot_path)
        except Exception:
            pass

    return dot


def node_lane(node, lane_ids):
    lane = node.get('lane')
    if lane and lane in lane_ids:
        return lane
    if node.get('type') == 'attack':
        return 'attack'
    return 'defense'


def is_year_labeled(year, sorted_years, year_cfg):
    step = int(year_cfg.get('step', 1))
    show_first = year_cfg.get('show_first', True)
    show_last = year_cfg.get('show_last', True)
    if step <= 1:
        return True
    idx = sorted_years.index(year)
    if show_first and idx == 0:
        return True
    if show_last and idx == len(sorted_years) - 1:
        return True
    return (idx % step) == 0


def create_legend(show_edges=True):
    """Create a compact horizontal legend using the same styling as the graph."""
    leg = graphviz.Digraph(comment='Legend', format='png')

    # Global attributes
    leg.attr(rankdir='TB',
             fontname='Fira Sans',
             bgcolor='white',
             pad='0.1',
             margin='0',
             nodesep='0.1', # Reduced from 0.2 to decrease spacing between categories
             ranksep='0.4', # Spacing between ranks (rows)
             splines='line')
    leg.attr('node', fontname='Fira Sans', fontsize='10') # Slightly larger font for readability
    leg.attr('edge', fontname='Fira Sans', fontsize='10')

    # --- Row 1: Types (Attack, Defense, Categories) ---
    with leg.subgraph(name='group_types') as t:
        t.attr(rank='same')
        
        # Attack Node (Standard node for rounded corners)
        t.node('legend_attack', 'Attack', 
               shape='box', style='filled,rounded',
               fillcolor='#ffcccc', color='#cc0000',
               margin='0.2,0.05', # Generous margin to prevent cutoff
               height='0.3')

        # Defense Node (Standard node for rounded corners)
        t.node('legend_defense', 'Defense', 
               shape='box', style='filled,rounded',
               fillcolor='#cce5ff', color='#0066cc',
               margin='0.2,0.05', # Generous margin to prevent cutoff
               height='0.3')

        # Categories
        # merging square and text into single HTML-label nodes for tight packing
        categories = [
            ('legend_cfi', '#9370DB', 'CFI'),
            ('legend_enf', '#FF8C00', 'Enforcement'),
            ('legend_rand', '#20B2AA', 'Randomization'),
            ('legend_xom', '#708090', 'XOM'),
        ]
        
        previous_node = 'legend_defense'
        
        # Add space between Defense and first category
        # We do this by adding an invisible spacer node or just a longer edge
        
        for i, (name, color, text) in enumerate(categories):
            # HTML Label: Square cell + Text cell
            # CELLSPACING="4" puts them close together
            label = f'''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="4">
                <TR>
                    <TD BGCOLOR="{color}" WIDTH="12" HEIGHT="12" FIXEDSIZE="TRUE"></TD>
                    <TD VALIGN="MIDDLE">{text}</TD>
                </TR>
            </TABLE>>'''
            
            t.node(name, label=label, shape='plaintext')
            
            # Edge from previous to current
            # minlen='1.5' after Defense to give a bit more space than between categories
            minlen = '1'
            if previous_node == 'legend_defense':
                minlen = '1.5'
            
            t.edge(previous_node, name, style='invis', minlen=minlen)
            previous_node = name

    if show_edges:
        # --- Row 2: Relations ---
        relations = [
            ('mot', '#228B22CC', 'solid', 'motivated'),
            ('byp', '#DC143CCC', 'dashed', 'bypasses'),
            ('ext', '#4169E1CC', 'solid', 'extends'),
            ('insp', '#9370DBCC', 'dotted', 'inspired'),
        ]
        with leg.subgraph(name='group_rel') as r:
            r.attr(rank='same')
            prev_dst = None

            for prefix, color, style, text in relations:
                src = f'{prefix}_src'
                dst = f'{prefix}_dst'

                r.node(src, label='', shape='point', width='0.01', style='invis')
                r.node(dst, label='', shape='point', width='0.01', style='invis')

                r.edge(src, dst,
                       label=text,
                       color=color,
                       style=style,
                       arrowhead='normal',
                       arrowsize='0.6',
                       penwidth='1.0',
                       len='0.6',
                       constraint='false', # Don't affect layout rank
                       minlen='2') # Visual length

                if prev_dst:
                    r.edge(prev_dst, src, style='invis', minlen='1')

                r.edge(src, dst, style='invis')

                prev_dst = dst

        # Connect rows to maintain hierarchy
        leg.edge('legend_attack', 'mot_src', style='invis')

    return leg


def fuse_images(timeline_path, legend_path, output_path, padding=20, spacing=12):
    """Fuse timeline and legend images without letting legend affect layout."""
    timeline = Image.open(timeline_path).convert("RGBA")
    legend = Image.open(legend_path).convert("RGBA")

    width = max(timeline.width, legend.width) + padding * 2
    height = timeline.height + legend.height + padding * 3 + spacing

    result = Image.new("RGBA", (width, height), "white")

    legend_x = padding
    legend_y = padding
    timeline_x = padding
    timeline_y = legend_y + legend.height + spacing + padding

    result.paste(legend, (legend_x, legend_y), legend)
    result.paste(timeline, (timeline_x, timeline_y), timeline)

    result.save(output_path)
    return result


def merge_svgs_to_pdf(svg1_path, svg2_path, output_pdf_path, padding=20, spacing=12):
    """Merge two SVGs (legend on top, timeline below) and convert to PDF."""
    
    def parse_length(val: str) -> float:
        """Parse an SVG length string and return points (pt). Supports pt, px, in, cm, mm.
        Defaults to pt if unit missing."""
        val = val.strip()
        m = re.match(r'^([0-9]+(?:\.[0-9]+)?)\s*(pt|px|in|cm|mm)?$', val)
        if not m:
            # Fallback: try plain float
            try:
                return float(val)
            except Exception:
                return 0.0
        num = float(m.group(1))
        unit = (m.group(2) or 'pt').lower()
        if unit == 'pt':
            return num
        if unit == 'px':
            # Assume 96 px per inch, 72 pt per inch → 1 px = 0.75 pt
            return num * 0.75
        if unit == 'in':
            return num * 72.0
        if unit == 'cm':
            return num * 72.0 / 2.54
        if unit == 'mm':
            return num * 72.0 / 25.4
        return num

    def get_svg_dims(path):
        with open(path, 'r') as f:
            content = f.read()
        # Parse width and height from <svg ... width="..." height="...">
        w_match = re.search(r'<svg[^>]*\bwidth="([^"]+)"', content, re.IGNORECASE)
        h_match = re.search(r'<svg[^>]*\bheight="([^"]+)"', content, re.IGNORECASE)
        
        if not w_match or not h_match:
            # Fallback: try finding viewBox
            v_match = re.search(r'viewBox="([0-9.]+) ([0-9.]+) ([0-9.]+) ([0-9.]+)"', content)
            if v_match:
                return float(v_match.group(3)), float(v_match.group(4)), content
            raise ValueError(f"Could not parse dimensions from {path}")
        
        width_pt = parse_length(w_match.group(1))
        height_pt = parse_length(h_match.group(1))
        return width_pt, height_pt, content

    w1, h1, c1 = get_svg_dims(svg1_path) # Legend (top)
    w2, h2, c2 = get_svg_dims(svg2_path) # Timeline (bottom)

    total_width = max(w1, w2) + padding * 2
    total_height = h1 + h2 + padding * 3 + spacing

    def extract_content(svg_text: str) -> str:
        """Extract inner content of the root <svg> element, stripping XML prolog/doctype.
        We locate the first '<svg' and slice after its closing '>' up to the last '</svg>'."""
        # Remove XML prolog and DOCTYPE lines to simplify parsing
        # but we still search structurally for <svg ...>
        start_tag_idx = svg_text.lower().find('<svg')
        if start_tag_idx == -1:
            raise ValueError('No <svg> start tag found')
        # Find the end of the start tag '>'
        gt_idx = svg_text.find('>', start_tag_idx)
        if gt_idx == -1:
            raise ValueError('Malformed <svg> start tag (no closing ">")')
        end_idx = svg_text.lower().rfind('</svg>')
        if end_idx == -1 or end_idx <= gt_idx:
            raise ValueError('Malformed SVG: missing closing </svg> tag')
        return svg_text[gt_idx + 1:end_idx]

    inner1 = extract_content(c1)
    inner2 = extract_content(c2)

    merged_svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="{total_width}pt" height="{total_height}pt" viewBox="0.00 0.00 {total_width} {total_height}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="legend" transform="translate({padding},{padding})">
{inner1}
</g>
<g id="timeline" transform="translate({padding},{padding + h1 + spacing})">
{inner2}
</g>
</svg>
'''

    temp_svg = output_pdf_path.replace('.pdf', '.svg')
    with open(temp_svg, 'w') as f:
        f.write(merged_svg)

    try:
        subprocess.run(['rsvg-convert', '-f', 'pdf', '-o', output_pdf_path, temp_svg], check=True)
        print(f"Generated {output_pdf_path}")
    finally:
        if os.path.exists(temp_svg):
            os.remove(temp_svg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a timeline diagram from a YAML description, as PNG or PDF."
    )
    parser.add_argument(
        "yaml_path",
        nargs="?",
        default="attack-defense-landscape.yaml",
        help="Path to the input YAML file",
    )
    parser.add_argument(
        "-o", "--output",
        default="attack_defense_timeline",
        help="Output filename without extension",
    )
    parser.add_argument(
        "--emit-dot",
        action="store_true",
        help="Also save the DOT source (.gv file)",
    )
    parser.add_argument(
        "--hide-edges",
        action="store_true",
        help="Do not draw relationship edges between nodes.",
    )
    parser.add_argument(
        "--direction",
        choices=["TB", "LR"],
        default="TB",
        help="Layout direction: TB (Top-to-Bottom, vertical) or LR (Left-to-Right, horizontal). Default is TB.",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf"],
        default=None,
        help="Output format (png or pdf). If omitted, defaults to png unless --pdf is set.",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Deprecated: generate a vector PDF output (requires rsvg-convert). Prefer --format pdf.",
    )

    args = parser.parse_args()

    try:
        data = load_data(args.yaml_path)
        emit_dot_path = f"{args.output}.gv" if args.emit_dot else None

        # Resolve desired output format
        out_format = args.format
        if out_format is None:
            out_format = 'pdf' if args.pdf else 'png'

        # Generate timeline and legend graphs
        graph = create_timeline(
            data,
            emit_dot_path=emit_dot_path,
            hide_edges=args.hide_edges,
            direction=args.direction
        )
        edge_visibility = data.get('layout', {}).get('edges', {}).get('visibility', 'all')
        show_edges = not args.hide_edges and edge_visibility != 'none'
        legend_graph = create_legend(show_edges=show_edges)

        if out_format == 'png':
            # Render raster parts and compose PNG
            timeline_path = graph.render(f"{args.output}_timeline", cleanup=True)
            legend_path = legend_graph.render(f"{args.output}_legend", cleanup=True)

            fuse_images(timeline_path, legend_path, f"{args.output}.png")
            print(f"Generated {args.output}.png")

            # Cleanup intermediates
            try:
                for tmp in (timeline_path, legend_path):
                    if os.path.exists(tmp):
                        os.remove(tmp)
            except OSError:
                pass

        else:  # pdf
            # Render vector parts and compose PDF
            graph.format = 'svg'
            legend_graph.format = 'svg'

            timeline_svg = graph.render(f"{args.output}_timeline_vec", cleanup=True)
            legend_svg = legend_graph.render(f"{args.output}_legend_vec", cleanup=True)

            merge_svgs_to_pdf(legend_svg, timeline_svg, f"{args.output}.pdf")

            # Cleanup intermediate SVGs
            for tmp in (timeline_svg, legend_svg):
                if os.path.exists(tmp):
                    os.remove(tmp)

    except Exception as e:
        print(f"Error: {e}")
