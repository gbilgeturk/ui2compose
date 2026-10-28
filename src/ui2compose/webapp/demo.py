#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Streamlit demo: screenshot -> detections -> UI graph -> layout -> Jetpack Compose code.

Usage:
    streamlit run app/demo.py        (or the `ui2compose-demo` command)

The demo runs ui2compose.infer.pipeline.run_pipeline — the same function as
the `ui2compose` command — with the settings chosen in the sidebar (their
defaults are the pipeline defaults, DEMO_DEFAULTS), so its output equals the
command-line output for the same settings. It shows:

  * Detection: the screenshot, the kept detections with the boxes removed by
    de-duplication drawn dashed grey (the pipeline's 1_detections_viz.png),
    and the layout overlay: the layout tree the Kotlin was generated from
    (generator.last_layout_tree), coloured by Column / Row / Box / LazyColumn;
  * Graph: the UI graph's edges. They are an inspection export; code
    generation reads only the containment hierarchy;
  * Hierarchy: the containment forest and the layout tree with its rules;
  * Explain: the trace of one detection id through post-processing,
    containment and layout (pipeline.explain_detection, as `--explain`);
  * Compose Code and Compose Preview (an HTML approximation of the layout
    tree with one preview per code template).

Streamlit, OpenCV (through detect_components) and matplotlib are imported only
where they are used, so the helper functions can be tested without them
(tests/test_demo.py).
"""

import contextlib
import hashlib
import io
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from ui2compose.infer.build_ui_graph import containment_roots
from ui2compose.infer.generate_compose_code import (DEFAULT_FALLBACK, DEFAULT_SCREEN_WIDTH_DP,
                                                    FALLBACK_MODES, LAYOUT_MODES,
                                                    PROGRESS_SQUARE_MAX_ASPECT)
from ui2compose.infer.pipeline import CLI_DEFAULTS, explain_detection, run_pipeline


# ── Project root ────────────────────────────────────────────
# The demo reads the repository's data folders (runs/, examples/, configs/).
# Those are not part of the installed package, so locate them starting from the
# directory the demo was launched from — normally the repository root.
def _find_project_root() -> Path:
    """Finds the directory holding the runs/, examples/ and configs/ folders.

    Input:  none
    Output: project root Path; the current working directory if no marker is found
    """
    markers = ("configs", "runs", "examples")
    for candidate in (Path.cwd(), *Path(__file__).resolve().parents):
        if any((candidate / m).is_dir() for m in markers):
            return candidate
    return Path.cwd()


PROJECT_ROOT = _find_project_root()


# ── Constants ───────────────────────────────────────────────

#: Sidebar defaults; they equal the defaults of run_pipeline (checked by a test).
DEMO_DEFAULTS: Dict[str, Any] = {
    'conf_threshold': 0.3,
    'dedup': True,
    'layout': 'xycut',
    'fallback': DEFAULT_FALLBACK,
    'size_from_boxes': False,
}

#: Colours (RGB) of the layout containers in the overlay and the preview.
LAYOUT_COLORS: Dict[str, Tuple[int, int, int]] = {
    'Column': (255, 140, 0),      # orange
    'Row': (0, 160, 0),           # green
    'Box': (150, 70, 200),        # purple
    'LazyColumn': (30, 110, 230),  # blue
}
LAYOUT_LEGEND = ("🟠 Column  🟢 Row  🟣 Box (container widget + its contained children)  "
                 "🔵 LazyColumn (list container + one item per group)")

#: Overlay geometry, in pixels at LAYOUT_REF_WIDTH image width (scaled with the width):
#: frame line width, frame inflation per nesting level, label font size.
LAYOUT_REF_WIDTH = 1200
LAYOUT_LINE_WIDTH = 3
LAYOUT_PAD_STEP = 6
LAYOUT_FONT_SIZE = 26
#: Opacity (0-255) of the frame fill.
LAYOUT_FILL_ALPHA = 28

EDGE_COLORS: Dict[str, str] = {
    'parent_child': '#2ECC71', 'above': '#3498DB', 'below': '#9B59B6',
    'left_of': '#E74C3C', 'right_of': '#F39C12',
}
#: Cell values of the adjacency matrix (0 = no edge).
EDGE_CODES: Dict[str, int] = {'parent_child': 1, 'above': 2, 'below': 3, 'left_of': 4, 'right_of': 5}


# ── Helpers (no Streamlit) ──────────────────────────────────


def find_models() -> List[Path]:
    """Lists the trained model weights (best.pt) under runs/.

    Input:  none
    Output: sorted list of best.pt Paths; empty list if the runs/ folder does not exist
    """
    runs_dir = PROJECT_ROOT / "runs"
    if not runs_dir.exists():
        return []
    return sorted(runs_dir.glob("*/weights/best.pt"))


def class_colors_hex() -> Dict[str, str]:
    """Returns the class colours of the detection figures as hex strings.

    Imports detect_components (and therefore OpenCV) when called.

    Input:  none
    Output: dictionary class name -> '#RRGGBB' (detect_components.CLASS_COLORS_BGR)
    """
    from ui2compose.infer.detect_components import CLASS_COLORS_BGR
    return {cls: '#{:02X}{:02X}{:02X}'.format(r, g, b) for cls, (b, g, r) in CLASS_COLORS_BGR.items()}


def _load_detector(model_path: str, dataset_yaml: str) -> Any:
    """Loads the YOLO detector (imports ultralytics when called).

    Input:  model_path — weights file; dataset_yaml — dataset config (unused by the detector)
    Output: ComponentDetector instance
    """
    from ui2compose.infer.detect_components import ComponentDetector
    return ComponentDetector(model_path, dataset_yaml)


def run_demo_pipeline(image_path: str, model_path: str, dataset_yaml: str, detector: Any,
                      settings: Dict[str, Any], output_dir: str) -> Tuple[Dict[str, Any], str]:
    """Runs the command-line pipeline (run_pipeline) with the demo settings.

    The detection visualisation is written to output_dir/1_detections_viz.png
    by the pipeline itself, exactly as on the command line.

    Input:  image_path — screenshot; model_path — weights (recorded in 0_run_config.json);
            dataset_yaml — dataset config; detector — loaded ComponentDetector (or an
            object with the same detect/visualize methods); settings — run_pipeline
            keywords (keys of DEMO_DEFAULTS); output_dir — folder for the output files
    Output: (run_pipeline result dictionary, captured console log)
    """
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        res = run_pipeline(image_path, model_path, dataset_yaml, output_dir,
                           visualize=True, detector=detector, **settings)
    return res, log.getvalue()


def walk_layout(node: Dict, depth: int = 0) -> Iterator[Tuple[Dict, int]]:
    """Yields every node of a layout tree with its depth, depth-first.

    Input:  node — layout node (generate_compose_code.build_layout_tree); depth — its depth
    Output: iterator of (node, depth)
    """
    yield node, depth
    for child in node.get('children', []):
        yield from walk_layout(child, depth + 1)


def _subtree_height(node: Dict) -> int:
    """Returns the number of container levels below and including a node (0 for a leaf).

    Input:  node — layout node
    Output: height of the subtree in container levels
    """
    if node['type'] == 'leaf':
        return 0
    return 1 + max((_subtree_height(c) for c in node.get('children', [])), default=0)


def layout_container_counts(tree: Dict) -> Dict[str, int]:
    """Counts the containers of a layout tree by type.

    Input:  tree — root layout node
    Output: dictionary type -> count for Column / Row / Box / LazyColumn (zero counts omitted)
    """
    counts = Counter(node['type'] for node, _ in walk_layout(tree) if node['type'] != 'leaf')
    return {t: counts[t] for t in LAYOUT_COLORS if counts[t]}


def _font(size: int) -> ImageFont.ImageFont:
    """Returns PIL's default font at a given size (bitmap default font on Pillow < 10.1).

    Input:  size — font size in pixels
    Output: PIL font object
    """
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no size argument
        return ImageFont.load_default()


def draw_layout_overlay(image: Image.Image, tree: Dict) -> Image.Image:
    """Draws the containers of the layout tree the Kotlin was generated from on the screenshot.

    Every Column, Row, Box and LazyColumn is drawn as a frame around the union
    of its members' boxes (node['bbox']), in its LAYOUT_COLORS colour with a
    light fill and a label naming the type. A frame is inflated by
    LAYOUT_PAD_STEP pixels (scaled with the image width) per container level
    below it, so that nested containers with the same union box stay visible.
    Leaves are not drawn (the detection view shows them).

    Input:  image — PIL image of the screenshot; tree — root layout node
    Output: new RGB PIL image
    """
    base = image.convert('RGBA')
    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = base.size
    scale = width / LAYOUT_REF_WIDTH
    line = max(1, round(LAYOUT_LINE_WIDTH * scale))
    step = max(2, round(LAYOUT_PAD_STEP * scale))
    font = _font(max(10, round(LAYOUT_FONT_SIZE * scale)))

    frames = []
    for node, _ in walk_layout(tree):
        if node['type'] not in LAYOUT_COLORS:
            continue
        pad = step * _subtree_height(node)
        x1, y1, x2, y2 = node['bbox']
        rect = [max(0, x1 - pad), max(0, y1 - pad), min(width - 1, x2 + pad), min(height - 1, y2 + pad)]
        if rect[2] < rect[0] or rect[3] < rect[1]:
            continue  # box entirely outside the image
        frames.append((pad, rect, node['type']))
    frames.sort(key=lambda f: -f[0])  # outer frames first

    for _, rect, ctype in frames:
        color = LAYOUT_COLORS[ctype]
        draw.rectangle(rect, fill=color + (LAYOUT_FILL_ALPHA,), outline=color + (255,), width=line)
    for _, rect, ctype in frames:
        color = LAYOUT_COLORS[ctype]
        tx, ty = rect[0] + line, rect[1] + line
        box = draw.textbbox((tx, ty), ctype, font=font)
        draw.rectangle([box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2], fill=color + (230,))
        draw.text((tx, ty), ctype, fill=(255, 255, 255, 255), font=font)
    return Image.alpha_composite(base, overlay).convert('RGB')


def adjacency_matrix(graph: Dict) -> Tuple[List[List[int]], List[str]]:
    """Builds the adjacency matrix of the graph edges, indexed by node position (not by id).

    Detection ids can have gaps after de-duplication, so rows and columns
    follow the order of graph['nodes'] and ids are mapped to positions.

    Input:  graph — UI graph with 'nodes' and 'edges'
    Output: (matrix, labels) — matrix[i][j] = EDGE_CODES of the edge from node i to
            node j (0 = none); labels — '<id>:<class prefix>' per node
    """
    ids = [n['id'] for n in graph['nodes']]
    index = {node_id: k for k, node_id in enumerate(ids)}
    matrix = [[0] * len(ids) for _ in ids]
    for edge in graph.get('edges', []):
        s, t = index.get(edge['source']), index.get(edge['target'])
        if s is not None and t is not None:
            matrix[s][t] = EDGE_CODES.get(edge['relation'], 0)
    labels = [f"{n['id']}:{n['class_name'][:5]}" for n in graph['nodes']]
    return matrix, labels


def plot_graph(graph: Dict, img_w: int, img_h: int, colors: Dict[str, str]) -> Any:
    """Draws the graph nodes at their box centres and the edges as arrows (matplotlib).

    Input:  graph — UI graph; img_w, img_h — image size in pixels;
            colors — class name -> hex colour
    Output: matplotlib Figure
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Patch

    fig = Figure(figsize=(8, 8))
    ax = fig.subplots()
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_aspect("equal")
    ax.set_title("UI graph: nodes and edges (inspection export)", fontsize=12, fontweight="bold")

    centres = {}
    radius = min(img_w, img_h) * 0.018
    for node in graph['nodes']:
        x1, y1, x2, y2 = node['bbox']
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        centres[node['id']] = (cx, cy)
        ax.add_patch(Circle((cx, cy), radius, color=colors.get(node['class_name'], "#808080"),
                            ec="black", lw=1.5, zorder=3))
        ax.text(cx, cy, str(node['id']), ha="center", va="center", fontsize=7,
                fontweight="bold", color="white", zorder=4)

    for edge in graph.get('edges', []):
        if edge['source'] in centres and edge['target'] in centres:
            ax.annotate("", xy=centres[edge['target']], xytext=centres[edge['source']],
                        arrowprops=dict(arrowstyle="->", color=EDGE_COLORS.get(edge['relation'], "#808080"),
                                        lw=1.2), zorder=2)

    present = {e['relation'] for e in graph.get('edges', [])}
    handles = [Patch(color=c, label=r) for r, c in EDGE_COLORS.items() if r in present]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=7)
    ax.set_facecolor("#f5f5f5")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_adjacency(graph: Dict) -> Any:
    """Draws the adjacency matrix of the graph edges, coloured by relation (matplotlib).

    Input:  graph — UI graph with 'nodes' and 'edges'
    Output: matplotlib Figure ("No detections" text if the graph has no nodes)
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8, 7))
    ax = fig.subplots()
    matrix, labels = adjacency_matrix(graph)
    if not matrix:
        ax.text(0.5, 0.5, "No detections", ha="center", va="center")
        return fig
    cmap = ListedColormap(["#FFFFFF"] + [EDGE_COLORS[r] for r in EDGE_CODES])
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=len(EDGE_CODES))
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Adjacency matrix (row = source, column = target)", fontsize=12, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, ticks=list(range(len(EDGE_CODES) + 1)))
    cbar.ax.set_yticklabels(["none"] + list(EDGE_CODES))
    fig.tight_layout()
    return fig


def format_containment(graph: Dict) -> str:
    """Renders the containment forest of a UI graph as indented text (synthetic groups removed).

    Input:  graph — UI graph
    Output: multi-line string, one detection per line ('Class#id (conf)')
    """
    lines: List[str] = []

    def visit(node: Dict, depth: int):
        """Appends one hierarchy node and its children.

        Input:  node — hierarchy node; depth — indentation level
        Output: none (side effect of appending to lines)
        """
        conf = node.get('confidence')
        conf_txt = '' if conf is None else f" ({conf:.2f})"
        lines.append(f"{'  ' * depth}{'└─ ' if depth else ''}{node['class']}#{node['id']}{conf_txt}")
        for child in node.get('children', []):
            visit(child, depth + 1)

    for root in containment_roots(graph['hierarchy']):
        visit(root, 0)
    return "\n".join(lines)


def format_layout_tree(tree: Dict) -> str:
    """Renders a layout tree as indented text naming the rule of every container.

    Input:  tree — root layout node
    Output: multi-line string
    """
    lines = []
    for node, depth in walk_layout(tree):
        ind = '  ' * depth
        t = node['type']
        if t == 'leaf':
            info = node['node']
            conf = info.get('confidence')
            lines.append(f"{ind}leaf {info['class']}#{info['id']}"
                         + ('' if conf is None else f" ({conf:.2f})"))
        elif t in ('Box', 'LazyColumn'):
            c = node['container']
            extra = f", items split by {node.get('split_rule')}" if t == 'LazyColumn' else ''
            lines.append(f"{ind}{t} [rule={node['rule']}, container {c['class']}#{c['id']}{extra}]")
        else:
            axis = f", axis={node['axis']}" if node.get('axis') else ''
            lines.append(f"{ind}{t} [rule={node['rule']}{axis}, {len(node['children'])} children]")
    return "\n".join(lines)


def explain_options(raw: List[Dict], kept: List[Dict], removed: List[Dict]) -> List[Tuple[int, str]]:
    """Lists the detection ids that can be explained, with a short status label each.

    Input:  raw — raw detections; kept — detections after post-processing;
            removed — de-duplication removals
    Output: list of (id, label) in raw order, e.g. (11, '11 · TextView 0.48 · kept')
    """
    kept_ids = {d['id'] for d in kept}
    removed_by_id = {d['id']: d for d in removed}
    options = []
    for d in raw:
        if d['id'] in kept_ids:
            status = 'kept'
        elif d['id'] in removed_by_id:
            status = f"removed ({removed_by_id[d['id']].get('dedup_rule')})"
        else:
            status = 'removed (class_conf)'
        options.append((d['id'], f"{d['id']} · {d['class_name']} {d['confidence']:.2f} · {status}"))
    return options


def removed_rows(removed: List[Dict]) -> List[Dict[str, Any]]:
    """Formats the de-duplication removals as table rows.

    Input:  removed — 1_dedup_removed.json entries
    Output: list of row dictionaries (id, class, confidence, rule, overlap, removed by)
    """
    return [{'id': d['id'], 'class': d['class_name'], 'confidence': round(d['confidence'], 3),
             'rule': d.get('dedup_rule'), 'overlap': round(d.get('overlap', float('nan')), 3),
             'removed by id': d.get('removed_by')} for d in removed]


def _json_for_script(obj: Any) -> str:
    """Serialises data for embedding inside an HTML <script> element.

    Input:  obj — JSON-serialisable object
    Output: JSON text in which '</' is escaped, so it cannot close the script element
    """
    return json.dumps(obj).replace('</', '<\\/')


# ── Compose preview (HTML approximation of the layout tree) ──
# One leaf preview per code template of generate_compose_code (all 13 detector
# classes plus the 'Unknown' placeholder); Column / Row / Box / LazyColumn
# follow the layout tree. Texts and states are the templates' placeholders.
_PREVIEW_TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
  body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#16181d;background:transparent;display:flex;justify-content:center;}
  .phone{position:relative;background:#fff;border:9px solid #11141a;border-radius:24px;overflow:hidden;box-shadow:0 6px 22px rgba(0,0,0,.25);}
  .screen{position:relative;width:100%;height:100%;background:#fafbfc;overflow:auto;}
  .wrap{padding:8px 6px;}
  .c{display:flex;border-radius:8px;padding:8px 6px 6px;gap:6px;position:relative;margin-top:6px;box-sizing:border-box;}
  .col{flex-direction:column;align-items:center;border:1.4px dashed rgba(255,140,0,.75);background:rgba(255,140,0,.05);}
  .row{flex-direction:row;align-items:center;border:1.4px dashed rgba(0,160,0,.65);background:rgba(0,160,0,.05);}
  .box{display:block;border:1.4px dashed rgba(150,70,200,.75);background:rgba(150,70,200,.05);}
  .box>.over{position:absolute;top:8px;left:6px;right:6px;}
  .lazy{flex-direction:column;align-items:stretch;border:1.4px dashed rgba(30,110,230,.75);background:rgba(30,110,230,.05);overflow-y:auto;width:100%;}
  .item{border-bottom:1px solid rgba(30,110,230,.25);padding-bottom:4px;}
  .badge{position:absolute;top:-7px;left:6px;font-size:8px;font-weight:700;letter-spacing:.2px;padding:0 4px;border-radius:4px;background:#11141a;color:#fff;white-space:nowrap;}
  .cell{display:flex;align-items:center;justify-content:center;min-width:0;}
  .leaf{font-size:11px;box-sizing:border-box;}
  .l-text{color:#222;padding:4px 6px;}
  .l-img{background:#d3d3d3;display:flex;align-items:center;justify-content:center;color:#666;font-size:9px;text-align:center;}
  .l-btn{background:#6750a4;color:#fff;border:none;border-radius:20px;padding:7px 18px;font-size:11px;font-weight:600;}
  .l-outlined{background:#fff;color:#6750a4;border:1px solid #79747e;border-radius:20px;padding:6px 16px;font-size:11px;font-weight:600;}
  .l-field{border-bottom:1.5px solid #49454f;border-radius:4px 4px 0 0;padding:6px 8px;color:#49454f;font-size:10px;background:#e7e0ec;width:100%;}
  .l-toggle{display:flex;align-items:center;gap:6px;color:#222;}
  .l-switch{display:inline-block;width:30px;height:16px;border-radius:9px;background:#e7e0ec;border:2px solid #79747e;position:relative;}
  .l-switch:after{content:'';position:absolute;width:10px;height:10px;border-radius:5px;background:#79747e;top:3px;left:3px;}
  .l-slider{width:100%;height:4px;background:linear-gradient(90deg,#6750a4 50%,#e7e0ec 50%);border-radius:2px;position:relative;margin:10px 0;}
  .l-slider:after{content:'';position:absolute;left:calc(50% - 7px);top:-5px;width:14px;height:14px;border-radius:7px;background:#6750a4;}
  .l-linear{width:100%;height:4px;background:#e7e0ec;border-radius:2px;overflow:hidden;}
  .l-linear span{display:block;width:40%;height:100%;background:#6750a4;}
  .l-circular{width:28px;height:28px;border-radius:50%;border:3px solid #e7e0ec;border-top-color:#6750a4;}
  .l-web{width:100%;border:1px solid #b9c2cf;background:#f1f4f8;display:flex;align-items:center;justify-content:center;color:#6b7686;font-size:10px;}
  .l-list{width:100%;border:1px solid #b9c2cf;border-radius:4px;padding:6px;font-size:10px;color:#444;}
  .l-generic{border:1px dashed #b9c2cf;border-radius:6px;padding:8px;font-size:10.5px;text-align:center;color:#6b7686;background:#f1f4f8;}
</style></head><body>
<div class="phone" id="phone"><div class="screen" id="screen"></div></div>
<script>
const DATA = __DATA__;
const W = 260, H = Math.round(W / DATA.aspect);
const SX = W / DATA.img_w, SY = H / DATA.img_h, DP = W / DATA.screen_width_dp;
const p = document.getElementById("phone"); p.style.width = W + "px"; p.style.height = H + "px";
function esc(s){ return String(s).replace(/[&<>"']/g, function(c){ return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]; }); }
function px(v){ return Math.max(2, Math.round(v)) + 'px'; }
function leafHTML(info, flattened){
  const cls = info.class, b = info.bbox, w = b[2] - b[0], h = b[3] - b[1];
  switch(cls){
    case 'Button': return '<button class="leaf l-btn">Button</button>';
    case 'TextView': return '<div class="leaf l-text">Text Content</div>';
    case 'EditText': return '<div class="leaf l-field">Input</div>';
    case 'ImageView': {
      const s = DATA.size_from_boxes ? 'width:' + px(w * SX) + ';height:' + px(h * SY) : 'width:' + px(100 * DP) + ';height:' + px(100 * DP);
      return '<div class="leaf l-img" style="' + s + '">Image placeholder</div>';
    }
    case 'CheckBox': return '<label class="leaf l-toggle"><input type="checkbox" disabled> Checkbox</label>';
    case 'Switch': return '<span class="leaf l-switch"></span>';
    case 'RadioButton': return '<label class="leaf l-toggle"><input type="radio" disabled> RadioButton</label>';
    case 'SeekBar': return '<div class="leaf l-slider"></div>';
    case 'ProgressBar':
      return (Math.min(w, h) > 0 && Math.max(w, h) / Math.min(w, h) <= DATA.square_max_aspect)
        ? '<div class="leaf l-circular"></div>' : '<div class="leaf l-linear"><span></span></div>';
    case 'Spinner': return '<button class="leaf l-outlined">Select ▾</button>';
    case 'WebView': return '<div class="leaf l-web" style="height:' + px(h * SY) + '">WebView</div>';
    case 'ListView': case 'RecyclerView': {
      const note = (flattened && flattened.length)
        ? cls + ': ' + flattened.length + ' child detection(s) emitted separately (flat layout)'
        : cls + ': no items detected';
      return '<div class="leaf l-list" style="max-height:' + px(h * SY) + '">' + esc(note) + '</div>';
    }
    default: return '<div class="leaf l-generic">Unknown: ' + esc(cls) + '</div>';
  }
}
function badge(t){ return '<span class="badge">' + esc(t) + '</span>'; }
function render(node, inRow){
  const grow = inRow ? 'flex:1;min-width:0;' : '';
  if(node.type === 'leaf'){
    return '<div class="cell" style="' + grow + '">' + leafHTML(node.node, node.flattened_children) + '</div>';
  }
  const kids = node.children || [];
  if(node.type === 'Box'){
    const c = node.container;
    // As in the Kotlin: a Box in a Row has weight(1f), unless its size comes from its box.
    const boxStyle = DATA.size_from_boxes
      ? 'width:' + px((c.bbox[2] - c.bbox[0]) * SX) + ';height:' + px((c.bbox[3] - c.bbox[1]) * SY) + ';'
      : grow;
    return '<div class="c box" style="' + boxStyle + '">' + badge('Box · ' + c.class + '#' + c.id) + render(kids[0], false)
      + (kids.length > 1 ? '<div class="over">' + render(kids[1], false) + '</div>' : '') + '</div>';
  }
  if(node.type === 'LazyColumn'){
    const c = node.container;
    return '<div class="c lazy" style="' + grow + 'max-height:' + px((c.bbox[3] - c.bbox[1]) * SY) + '">'
      + badge('LazyColumn · ' + c.class + '#' + c.id)
      + kids.map(function(k){ return '<div class="item">' + render(k, false) + '</div>'; }).join('') + '</div>';
  }
  const isRow = node.type === 'Row';
  return '<div class="c ' + (isRow ? 'row' : 'col') + '" style="' + grow + (isRow ? '' : 'width:100%;') + '">'
    + badge(node.type + ' · ' + node.rule) + kids.map(function(k){ return render(k, isRow); }).join('') + '</div>';
}
document.getElementById("screen").innerHTML = '<div class="wrap">' + render(DATA.tree, false) + '</div>';
</script></body></html>"""


def build_layout_preview_html(tree: Dict, img_w: int, img_h: int, size_from_boxes: bool = False,
                              screen_width_dp: float = DEFAULT_SCREEN_WIDTH_DP) -> str:
    """Renders the layout tree as an HTML phone screen that approximates the compiled Compose UI.

    Column / Row / Box / LazyColumn follow the tree (a Box draws its contained
    children over the container widget at the top-left corner, as Compose's
    Box does); each leaf uses the preview of its code template. The preview
    is approximate (no Compose measurement); the Kotlin is authoritative.

    Input:  tree — root layout node (generator.last_layout_tree); img_w, img_h —
            screenshot size in px; size_from_boxes — whether the code was generated
            with box-derived sizes (images are then sized from their boxes);
            screen_width_dp — target screen width the code assumes
    Output: self-contained HTML string
    """
    data = {
        'tree': tree, 'aspect': round(img_w / img_h, 4), 'img_w': img_w, 'img_h': img_h,
        'screen_width_dp': screen_width_dp, 'size_from_boxes': bool(size_from_boxes),
        'square_max_aspect': PROGRESS_SQUARE_MAX_ASPECT,
    }
    return _PREVIEW_TEMPLATE.replace("__DATA__", _json_for_script(data))


_BBOX_TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
  body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#16181d;background:transparent;display:flex;justify-content:center;}
  .phone{position:relative;background:#fff;border:9px solid #11141a;border-radius:24px;overflow:hidden;box-shadow:0 6px 22px rgba(0,0,0,.25);}
  .screen{position:relative;width:100%;height:100%;background:#fafbfc;}
  .bb{position:absolute;border-radius:3px;border:1.5px solid rgba(0,0,0,.35);overflow:hidden;display:flex;align-items:flex-start;box-sizing:border-box;}
  .bb span{background:rgba(255,255,255,.72);padding:0 3px;font-size:8.5px;line-height:1.3;color:#11141a;}
</style></head><body>
<div class="phone" id="phone"><div class="screen" id="screen"></div></div>
<script>
const DATA = __DATA__;
const W = 260, H = Math.round(W / DATA.aspect);
const p = document.getElementById("phone"); p.style.width = W + "px"; p.style.height = H + "px";
function esc(s){ return String(s).replace(/[&<>"']/g, function(c){ return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]; }); }
document.getElementById("screen").innerHTML = DATA.boxes.map(function(b){
  const c = DATA.colors[b.cls] || '#bbbbbb';
  return '<div class="bb" style="left:' + (b.x * 100).toFixed(2) + '%;top:' + (b.y * 100).toFixed(2) + '%;width:'
    + (b.w * 100).toFixed(2) + '%;height:' + (b.h * 100).toFixed(2) + '%;background:' + c + '66;border-color:' + c
    + '"><span>' + esc(b.id + ' ' + b.cls) + '</span></div>';
}).join('');
</script></body></html>"""


def build_bbox_html(detections: List[Dict], img_w: int, img_h: int, colors: Dict[str, str]) -> str:
    """Renders the detected boxes at their positions on an HTML phone screen (no layout).

    Input:  detections — detection dictionaries with 'id', 'class_name', 'bbox';
            img_w, img_h — image size in px; colors — class name -> '#RRGGBB'
    Output: self-contained HTML string
    """
    boxes = [{'id': d['id'], 'cls': d['class_name'],
              'x': d['bbox'][0] / img_w, 'y': d['bbox'][1] / img_h,
              'w': (d['bbox'][2] - d['bbox'][0]) / img_w, 'h': (d['bbox'][3] - d['bbox'][1]) / img_h}
             for d in detections]
    data = {'boxes': boxes, 'aspect': round(img_w / img_h, 4), 'colors': colors}
    return _BBOX_TEMPLATE.replace("__DATA__", _json_for_script(data))


def _store_upload(data: bytes, suffix: str) -> str:
    """Writes an uploaded screenshot to a temporary file named by its content hash.

    The same upload therefore keeps the same path across Streamlit reruns,
    so cached results can be reused.

    Input:  data — file content; suffix — file extension including the dot
    Output: path of the stored file
    """
    digest = hashlib.sha1(data).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f"ui2compose_upload_{digest}{suffix}"
    if not path.exists():
        path.write_bytes(data)
    return str(path)


# ── Streamlit page ──────────────────────────────────────────


def main():
    """Renders the Streamlit demo. Entry point for `streamlit run app/demo.py`.

    Input:  none (user input comes from the Streamlit widgets)
    Output: none (side effect of rendering the page)
    """
    import streamlit as st
    import streamlit.components.v1 as components

    st.set_page_config(page_title="UI → Compose Demo", page_icon="📱", layout="wide")
    st.title("📱 UI Screenshot → Jetpack Compose")
    st.caption("Automatic Compose code generation from an Android screenshot "
               "(runs the same pipeline as the `ui2compose` command)")

    # ── Sidebar ─────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")
        models = find_models()
        if not models:
            st.error("No model found! Check `runs/*/weights/best.pt`.")
            st.stop()
        model_labels = [str(m.relative_to(PROJECT_ROOT)) for m in models]
        default_model = Path(CLI_DEFAULTS['model_path']).as_posix()   # the pipeline's default weights
        selected = st.selectbox("Model", range(len(models)), format_func=lambda i: model_labels[i],
                                index=next((i for i, lab in enumerate(model_labels)
                                            if Path(lab).as_posix() == default_model), 0))
        model_path = str(models[selected])

        settings = {
            'conf_threshold': st.slider("Confidence threshold", 0.1, 0.9,
                                        DEMO_DEFAULTS['conf_threshold'], 0.05),
            'dedup': st.checkbox("De-duplicate detections", value=DEMO_DEFAULTS['dedup'],
                                 help="postprocess.deduplicate_detections (thresholds selected on the "
                                      "validation split, results/eval_appsplit/selected_thresholds.json)"),
            'layout': st.radio("Layout", list(LAYOUT_MODES),
                               index=list(LAYOUT_MODES).index(DEMO_DEFAULTS['layout']),
                               help="xycut: recursive XY-cut over the containment forest; "
                                    "flat: baseline, one Column ordered by y-centre"),
            'fallback': st.selectbox("XY-cut fallback", list(FALLBACK_MODES),
                                     index=list(FALLBACK_MODES).index(DEMO_DEFAULTS['fallback']),
                                     help="used where no cut exists on either axis"),
            'size_from_boxes': st.checkbox("Sizes from boxes", value=DEMO_DEFAULTS['size_from_boxes']),
        }
        st.caption("Defaults are the pipeline defaults. Thresholds not exposed here keep "
                   "their defaults (see configs/pipeline_config.yaml).")
        st.divider()
        st.markdown("**Pipeline stages:**\n1. YOLO detection\n2. De-duplication\n"
                    "3. UI graph (containment)\n4. XY-cut layout + code templates")

    # ── Input image ─────────────────────────────────────────
    uploaded = st.file_uploader("Upload a screenshot", type=["png", "jpg", "jpeg"])
    examples = sorted((PROJECT_ROOT / "examples").glob("*.png"))
    image_path: Optional[str] = None
    if uploaded is not None:
        image_path = _store_upload(uploaded.getvalue(), Path(uploaded.name).suffix or ".png")
    elif examples:
        choice = st.selectbox("…or pick a bundled example (ReDraw screen)",
                              [p.name for p in examples])
        image_path = str(PROJECT_ROOT / "examples" / choice)
    if not image_path or not Path(image_path).exists():
        st.info("Upload a screenshot or pick one of the bundled examples to get started.")
        st.stop()

    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    run_btn = st.button("Run Pipeline", type="primary")
    if run_btn:
        st.session_state["ran"] = True
    if not st.session_state.get("ran"):
        st.stop()

    # ── Run (cached per image, model and settings) ──────────
    cache = st.session_state.setdefault("results", {})
    key = json.dumps([image_path, model_path, settings], sort_keys=True)
    if key not in cache:
        with st.status("Running the pipeline…", expanded=False) as status:
            detector = st.cache_resource(_load_detector)(
                model_path, str(PROJECT_ROOT / "configs" / "dataset.yaml"))
            out_dir = tempfile.mkdtemp(prefix="ui2compose_demo_")
            cache[key] = run_demo_pipeline(image_path, model_path,
                                           str(PROJECT_ROOT / "configs" / "dataset.yaml"),
                                           detector, settings, out_dir)
            status.update(label="Pipeline finished", state="complete")
    res, log = cache[key]
    raw, kept, removed = res['raw_detections'], res['detections'], res['removed']
    graph, tree, code = res['graph'], res['layout_tree'], res['code']
    n_lines = len(code.splitlines())

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw detections", len(raw))
    c2.metric("Removed by de-duplication", len(removed))
    c3.metric("Kept (in the code)", len(kept))
    c4.metric("Kotlin lines", n_lines)

    tab_det, tab_graph, tab_hier, tab_explain, tab_code, tab_preview = st.tabs([
        "Detection", "Graph", "Hierarchy", "Explain", "Compose Code", "Compose Preview"])

    # ── Detection ───────────────────────────────────────────
    with tab_det:
        col_orig, col_det, col_layout = st.columns(3)
        with col_orig:
            st.subheader("Original")
            st.image(img)
        with col_det:
            st.subheader("Detections")
            st.image(str(Path(res['output_dir']) / "1_detections_viz.png"))
            st.caption("Solid: kept detections. Dashed grey, tagged 'removed': removed by "
                       "de-duplication.")
        with col_layout:
            st.subheader("Layout")
            st.image(draw_layout_overlay(img, tree))
            counts = layout_container_counts(tree)
            st.caption(LAYOUT_LEGEND + ". The layout tree the Kotlin was generated from: "
                       + (", ".join(f"{n} {t}" for t, n in counts.items()) or "no containers") + ".")

        if removed:
            st.subheader(f"Removed by de-duplication ({len(removed)})")
            st.dataframe(removed_rows(removed))
        st.subheader("Class distribution (kept detections)")
        st.bar_chart(dict(Counter(d["class_name"] for d in kept).most_common()))
        with st.expander("Detection JSON (kept)"):
            st.json(kept)
        with st.expander("Pipeline log"):
            st.code(log, language=None)

    # ── Graph ───────────────────────────────────────────────
    with tab_graph:
        st.caption("The edges are an inspection export of the graph stage. Code generation "
                   "reads only the containment hierarchy and computes rows and columns by "
                   "XY-cut (see the Hierarchy tab).")
        colors = class_colors_hex()
        col_g, col_a = st.columns(2)
        with col_g:
            st.pyplot(plot_graph(graph, img_w, img_h, colors))
        with col_a:
            st.pyplot(plot_adjacency(graph))
        st.subheader("Edge distribution")
        st.bar_chart(dict(Counter(e["relation"] for e in graph["edges"]).most_common()))
        with st.expander("Graph JSON"):
            st.json(graph)

    # ── Hierarchy ───────────────────────────────────────────
    with tab_hier:
        col_c, col_l = st.columns(2)
        with col_c:
            st.subheader("Containment forest")
            st.caption("Each detection has at most one parent: the smallest box that covers "
                       f"≥ {graph['params']['containment_ratio']:g} of its area.")
            st.code(format_containment(graph) or "(empty)", language=None)
        with col_l:
            st.subheader("Layout tree")
            st.caption("The tree the Kotlin was generated from, with the rule of every container.")
            st.code(format_layout_tree(tree), language=None)
        with st.expander("Layout tree JSON"):
            st.json(tree)

    # ── Explain ─────────────────────────────────────────────
    with tab_explain:
        st.subheader("Why is a detection where it is?")
        options = explain_options(raw, kept, removed)
        if options:
            labels = dict(options)
            det_id = st.selectbox("Detection", [i for i, _ in options], format_func=lambda i: labels[i])
            lines = explain_detection(det_id, raw, kept, removed, graph, tree)
            st.code("\n".join(lines), language=None)
            st.caption(f"Same output as `ui2compose IMAGE --explain {det_id}`.")
        else:
            st.info("No detections.")

    # ── Code ────────────────────────────────────────────────
    with tab_code:
        st.subheader("Jetpack Compose code")
        st.code(code, language="kotlin")
        st.download_button("Download GeneratedScreen.kt", data=code,
                           file_name="GeneratedScreen.kt", mime="text/plain")

    # ── Preview ─────────────────────────────────────────────
    with tab_preview:
        st.subheader("Compose preview")
        st.caption("An HTML approximation of the layout tree: dashed frames are the "
                   "containers (" + LAYOUT_LEGEND + "), each leaf shows its code template "
                   "with the template's placeholder text. The Kotlin is authoritative.")
        ph_h = max(560, int(260 / (img_w / img_h)) + 60)
        col_orig, col_layout, col_code = st.columns([1, 1, 1.2])
        with col_orig:
            st.markdown("**Selected screenshot**")
            st.image(img)
        with col_layout:
            st.markdown("**Generated layout**")
            components.html(build_layout_preview_html(tree, img_w, img_h,
                                                      settings['size_from_boxes']), height=ph_h)
        with col_code:
            st.markdown("**GeneratedScreen.kt**")
            st.code(code, language="kotlin")
        with st.expander("Detected boxes at their positions (no layout)"):
            components.html(build_bbox_html(kept, img_w, img_h, class_colors_hex()), height=ph_h)


if __name__ == "__main__":
    main()
