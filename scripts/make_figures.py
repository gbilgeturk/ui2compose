#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Draws the figures of the SoftwareX article from the published v1.2.0 outputs.

Usage (from the repository root):
    python scripts/make_figures.py                 # every figure whose inputs exist
    python scripts/make_figures.py --only 2 3      # selected figures

Outputs (docs/figures/, PNG at 600 dpi at print size; see docs/figures/README.md):
  fig1_pipeline.png        the four stages, their modules and output files
  fig2_layout_example.png  cuts, containers and layout tree of the drivesafe example
  fig3_detections.png      detections on the same screen (single-column width)
  fig4_claude_comparison.png  input | ui2compose render | Claude run-1 render
  fig5_detector.png        per-class AP@50 and mAP@50 on the app-level split (test, val)

Every figure is drawn from files in the repository (results/, examples/). The values
of Fig. 5 come from results/eval_appsplit/detector_table2.json; the subset drawn is
also written to docs/figures/fig5_detector_data.json.
Fig. 4 is drawn only if every ui2compose render belongs to the current Kotlin file
(SHA-256 recorded with the render in results/compile_report.json).
The output is deterministic: no randomness, fonts are the
DejaVu fonts bundled with matplotlib, and PNG metadata carry no software
version. Every text is at least 8 pt at print size (checked before saving).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patches  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "docs" / "figures"
# (not in a folder named data/: .gitignore excludes every such folder)
FIG5_DATA = OUT_DIR / "fig5_detector_data.json"

# The worked example of Figs. 1-3: a test screen of the app-level split, i.e. an app the
# default weights (runs/oversample_5k_appsplit) never saw in training.
EXAMPLE = "westport.andrewirwin.com.drivesafe-screens_screenshot_3"
# Fig. 1 illustrates de-duplication with a 'same_class_nested' removal; the worked example
# has no removal, so it is taken from this screen (validation split of the app-level split).
DEDUP_EXAMPLE = "com.luigiedeveloper.battlesimulator-screens_screenshot_3"
EXAMPLES_DIR = REPO / "examples"
V120_DIR = REPO / "results" / "v1.2.0_examples"
MLLM_DIR = REPO / "results" / "mllm_claude"
# Fig. 4: the two examples that are held out for the default weights (app-level split:
# drivesafe test, battlesimulator val), and two of the screens chosen in
# results/mllm_claude/ANALYSIS.md, section 2 (foxfi: the typical case; mineworld: Claude's
# run 1 does not compile), which are in the app-level train split. The title says which.
FIG4_SCREENS = (
    ("westport.andrewirwin.com.drivesafe-screens_screenshot_3", "drivesafe #3 (test split)"),
    ("com.luigiedeveloper.battlesimulator-screens_screenshot_3", "battlesimulator #3 (val split)"),
    ("com.foxfi-screens_screenshot_1", "foxfi #1 (train split)"),
    ("com.mineworld.swordsaddon-screens_screenshot_2", "mineworld #2 (train split)"),
)

# --------------------------------------------------------------------------
# Style: print sizes (Elsevier), fonts, colour tokens
# --------------------------------------------------------------------------
DPI = 600
FULL_W = 7.48   # 190 mm, double-column width
COL_W = 3.54    # 90 mm, single-column width
MIN_PT = 8.0
FONT = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
PANEL = "#f4f4f1"
WHITE = "#ffffff"
HIGHLIGHT = "#d03b3b"   # status 'critical': marks the wrong groupings, always with a label

# Widget classes (Figs. 1, 3). Validated all-pairs (light): CVD dE >= 9.2, normal >= 16.3.
CLASS_COLORS = {"ImageView": "#2a78d6", "TextView": "#eb6834", "Button": "#1baf7a",
                "CheckBox": "#4a3aa7"}
CLASS_SHORT = {"ImageView": "Image", "TextView": "Text", "Button": "Button",
               "CheckBox": "Check"}
REMOVED_FILL = "#dcdbd5"
# Layout rules (Fig. 2). Validated all-pairs with HIGHLIGHT: CVD dE >= 9.1, normal >= 16.3.
RULE_COLORS = {"gap": "#2a78d6", "gap_tolerance": "#eda100", "containment": "#1baf7a",
               "list": "#4a3aa7", "fallback": MUTED}
# Training configurations (Fig. 5): the two ends of a one-hue ordinal ramp (blue 250 / 650,
# validated --ordinal) plus marker shape (square / circle), so print and CVD readers need not
# rely on colour.
CONFIGS = ("baseline", "oversample_5k")
CONFIG_LABEL = {"baseline": "baseline (no oversampling)", "oversample_5k": "oversampled to 5k (default)"}
CONFIG_COLOR = {"baseline": "#86b6ef", "oversample_5k": "#104281"}
CONFIG_MARKER = {"baseline": "s", "oversample_5k": "o"}
CONFIG_MS = {"baseline": 5.0, "oversample_5k": 5.6}
CLASS_NAMES = ("Button", "CheckBox", "EditText", "ImageView", "ListView", "ProgressBar",
               "RadioButton", "RecyclerView", "SeekBar", "Spinner", "Switch", "TextView", "WebView")
# src/ui2compose/preprocess/oversample_dataset.py RARE_CLASSES = {1, 5, 6, 7, 8, 9, 10, 12}
OVERSAMPLED = ("CheckBox", "ProgressBar", "RadioButton", "RecyclerView", "SeekBar", "Spinner",
               "Switch", "WebView")

plt.rcParams.update({
    "font.family": FONT, "font.size": 8, "mathtext.fontset": "dejavusans",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "axes.titlesize": 9, "axes.labelsize": 8, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "legend.fontsize": 8, "savefig.dpi": DPI,
    "figure.facecolor": WHITE, "axes.facecolor": WHITE, "hatch.linewidth": 0.6,
})


def rel(p: Path) -> str:
    """Path relative to the repository root (as written to the README and the data JSON)."""
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def sha256(path: Path) -> str:
    """SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_color_on(fill: str) -> str:
    """Ink or white, whichever has more contrast on a fill colour (WCAG relative luminance)."""
    rgb = [int(fill[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return WHITE if 1.05 / (lum + 0.05) >= (lum + 0.05) / 0.05 else INK


_MEASURE = None


def text_w(s: str, size: float = 8, weight: str = "normal", family: str = FONT,
           style: str = "normal") -> float:
    """Rendered width of a string in inches."""
    global _MEASURE
    if _MEASURE is None:
        _MEASURE = plt.figure(figsize=(1, 1), dpi=72)
    t = _MEASURE.text(0, 0, s, fontsize=size, weight=weight, family=family, style=style)
    w = t.get_window_extent(_MEASURE.canvas.get_renderer()).width / 72.0
    t.remove()
    return w


def canvas(w: float, h: float) -> Tuple[plt.Figure, plt.Axes]:
    """Figure of w x h inches with one axes in inch coordinates (origin bottom left)."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_axis_off()
    return fig, ax


def inch_axes(fig: plt.Figure, x: float, y: float, w: float, h: float) -> plt.Axes:
    """Axes placed at (x, y, w, h) inches of the figure."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh])


def save(fig: plt.Figure, name: str) -> Path:
    """Checks the font sizes and writes docs/figures/<name> (PNG, DPI, no software metadata)."""
    small = sorted({(t.get_text(), t.get_fontsize()) for t in fig.findobj(Text)
                    if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_PT - 1e-9})
    if small:
        raise SystemExit(f"{name}: text below {MIN_PT} pt: {small[:5]}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, facecolor=WHITE, metadata={"Software": None})
    plt.close(fig)
    w, h = Image.open(path).size
    print(f"wrote {rel(path)}  {w} x {h} px = {w / DPI:.2f} x {h / DPI:.2f} in at {DPI} dpi")
    return path


def load_rgb(path: Path) -> np.ndarray:
    """Image as RGB array, transparent pixels composited on white."""
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(bg, im).convert("RGB"))


def light_gray(path: Path, keep: float = 0.42) -> np.ndarray:
    """Screenshot as a light grey background for overlays (darkness scaled by `keep`)."""
    g = np.asarray(Image.fromarray(load_rgb(path)).convert("L"), dtype=np.float64)
    out = 255.0 - (255.0 - g) * keep
    return np.repeat(out.round().astype(np.uint8)[..., None], 3, axis=2)


def badge(ax: plt.Axes, x: float, y: float, label: str, color: str = HIGHLIGHT,
          r: float = 0.075, size: float = 8) -> None:
    """Filled circle with a centred label (inch coordinates of an equal-aspect axes)."""
    ax.add_patch(patches.Circle((x, y), r, fc=color, ec="none", zorder=20, clip_on=False))
    ax.text(x, y - 0.002, label, ha="center", va="center", fontsize=size, weight="bold",
            color=text_color_on(color), zorder=21, clip_on=False)


def badge_data(ax: plt.Axes, x: float, y: float, label: str, color: str = HIGHLIGHT) -> None:
    """Circle badge at data coordinates of any axes (drawn above that axes' content)."""
    ax.text(x, y, label, ha="center", va="center", fontsize=8, weight="bold", color=text_color_on(color),
            zorder=30, clip_on=False, bbox=dict(boxstyle="circle,pad=0.22", fc=color, ec="none"))


def chip(ax: plt.Axes, x: float, y: float, label: str, color: str, size: float = 8,
         family: str = MONO, ha: str = "left") -> float:
    """Rounded label on a coloured fill; returns its width in inches (inch-coordinate axes)."""
    pad = 0.035
    w = text_w(label, size, family=family) + 2 * pad
    h = size / 72 * 1.25
    x0 = x if ha == "left" else x - w
    ax.add_patch(patches.FancyBboxPatch((x0, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.03",
                                        fc=color, ec="none", zorder=5, clip_on=False))
    ax.text(x0 + w / 2, y - 0.004, label, ha="center", va="center", fontsize=size, family=family,
            color=text_color_on(color), zorder=6, clip_on=False)
    return w


# --------------------------------------------------------------------------
# Example data
# --------------------------------------------------------------------------
def load_example(stem: str = EXAMPLE) -> Dict[str, Any]:
    """Detections, removals and layout tree of one bundled example (results/v1.2.0_examples)."""
    d = V120_DIR / stem
    raw = json.loads((d / "1_detections_raw.json").read_text(encoding="utf-8"))
    kept = json.loads((d / "1_detections.json").read_text(encoding="utf-8"))
    removed = json.loads((d / "1_dedup_removed.json").read_text(encoding="utf-8"))
    tree = json.loads((d / "2_layout_tree.json").read_text(encoding="utf-8"))
    cfg = json.loads((d / "0_run_config.json").read_text(encoding="utf-8"))
    return {"dir": d, "raw": raw, "kept": kept, "removed": removed, "tree": tree, "config": cfg,
            "image": EXAMPLES_DIR / f"{stem}.png"}


def walk(node: Dict[str, Any], depth: int = 0, parent: Optional[int] = None,
         rows: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Pre-order list of layout-tree nodes with depth and parent row index."""
    rows = [] if rows is None else rows
    rows.append({"node": node, "depth": depth, "parent": parent})
    me = len(rows) - 1
    for c in node.get("children", []):
        walk(c, depth + 1, me, rows)
    return rows


def tolerance_cuts(node: Dict[str, Any]) -> List[Tuple[float, float]]:
    """Cuts of a node accepted only through the tolerance (gap interval [a, b] with a >= b)."""
    return [tuple(g) for g in node.get("gaps", []) if g[0] >= g[1]]


# --------------------------------------------------------------------------
# Fig. 1: pipeline
# --------------------------------------------------------------------------
def _ill_detect(ax: plt.Axes, x: float, y: float, w: float, h: float, ex: Dict[str, Any]) -> None:
    """The example screen with its raw detections (class colours)."""
    th = h - 0.04
    tw = th * 1200 / 1920
    ix, iy = x + 0.10, y + 0.02
    ax.imshow(light_gray(ex["image"], 0.5), extent=[ix, ix + tw, iy, iy + th], zorder=2,
              interpolation="lanczos")
    ax.add_patch(patches.Rectangle((ix, iy), tw, th, fill=False, ec=AXIS, lw=0.6, zorder=3))
    for d in ex["raw"]:
        x1, y1, x2, y2 = d["bbox"]
        ax.add_patch(patches.Rectangle((ix + x1 / 1200 * tw, iy + th - y2 / 1920 * th),
                                       (x2 - x1) / 1200 * tw, (y2 - y1) / 1920 * th, fill=False,
                                       ec=CLASS_COLORS.get(d["class_name"], INK2), lw=0.7, zorder=4))
    lx = ix + tw + 0.14
    n_cls = len({d["class_name"] for d in ex["raw"]})
    for k, (line, col, wt) in enumerate(((f"{len(ex['raw'])} boxes", INK, "bold"),
                                         (f"of {n_cls} classes", INK2, "normal"),
                                         ("(example)", INK2, "normal"))):
        ax.text(lx, y + h - 0.16 - k * 0.15, line, va="center", fontsize=8, color=col, weight=wt)


def _ill_dedup(ax: plt.Axes, x: float, y: float, w: float, h: float, ex: Dict[str, Any]) -> None:
    """A nested TextView pair (DEDUP_EXAMPLE): the looser, less confident box is removed."""
    src = load_example(DEDUP_EXAMPLE)
    rem = next((d for d in src["removed"] if d["dedup_rule"] == "same_class_nested"), None)
    if rem is None:
        raise SystemExit(f"Fig. 1: {DEDUP_EXAMPLE} has no 'same_class_nested' removal any more")
    kept = next(d for d in src["kept"] if d["id"] == rem["removed_by"])
    cx = x + w / 2
    ow, oh = w - 0.16, 0.30
    oy = y + h - 0.10 - oh
    ax.add_patch(patches.Rectangle((cx - ow / 2, oy), ow, oh, fill=False, ec=MUTED, lw=1.0,
                                   ls=(0, (3, 2)), zorder=3))
    iw, ih = 0.92, 0.20
    ax.add_patch(patches.Rectangle((cx - iw / 2, oy + (oh - ih) / 2), iw, ih, fc=WHITE,
                                   ec=CLASS_COLORS["TextView"], lw=1.2, zorder=4))
    ax.text(cx, oy + oh / 2, f"#{kept['id']} Text {kept['confidence']:.2f}", ha="center", va="center",
            fontsize=8, color=INK, zorder=5)
    ax.text(cx, oy - 0.10, f"kept: #{kept['id']} TextView {kept['confidence']:.2f}", ha="center",
            va="center", fontsize=8, color=INK)
    ax.text(cx, oy - 0.25, f"removed: #{rem['id']} TextView {rem['confidence']:.2f}", ha="center",
            va="center", fontsize=8, color=INK2)
    ax.text(cx, oy - 0.40, f"({rem['overlap']:.0%} nested; battlesimulator)", ha="center", va="center",
            fontsize=8, color=INK2)


def _ill_graph(ax: plt.Axes, x: float, y: float, w: float, h: float, ex: Dict[str, Any]) -> None:
    """A container box with two children and its containment tree; the spatial edge is export only."""
    c = CLASS_COLORS
    bx, by, bw, bh = x + 0.07, y + 0.08, 0.52, h - 0.16
    ax.add_patch(patches.Rectangle((bx, by), bw, bh, fill=False, ec=c["ImageView"], lw=1.2))
    ax.add_patch(patches.Rectangle((bx + 0.08, by + bh - 0.31), bw - 0.16, 0.21, fill=False,
                                   ec=c["CheckBox"], lw=1.2))
    ax.add_patch(patches.Rectangle((bx + 0.08, by + 0.10), bw - 0.16, 0.21, fill=False,
                                   ec=c["TextView"], lw=1.2))
    px, py = x + 0.93, y + h - 0.18
    c1, c2 = (px - 0.25, y + 0.40), (px + 0.25, y + 0.40)
    for cc in (c1, c2):
        ax.plot([px, cc[0]], [py, cc[1]], color=INK, lw=1.0, zorder=3, solid_capstyle="round")
    ax.plot([c1[0], c2[0]], [c1[1], c2[1]], color=MUTED, lw=1.0, ls=(0, (2.5, 2)), zorder=3)
    for (cx, cy), col in (((px, py), c["ImageView"]), (c1, c["CheckBox"]), (c2, c["TextView"])):
        ax.add_patch(patches.Circle((cx, cy), 0.065, fc=col, ec=WHITE, lw=1.5, zorder=4))
    ax.text(px + 0.10, py, "contains", va="center", fontsize=8, color=INK)
    ax.text(px, c1[1] - 0.11, "above", ha="center", va="top", fontsize=8, color=INK2)


def _ill_layout(ax: plt.Axes, x: float, y: float, w: float, h: float, ex: Dict[str, Any]) -> None:
    """XY-cut of a small screen (one horizontal cut, one vertical cut) and the generated nesting."""
    fx, fy, fw, fh = x + 0.07, y + 0.06, 0.62, h - 0.12
    ax.add_patch(patches.Rectangle((fx, fy), fw, fh, fill=False, ec=AXIS, lw=0.8))
    m = 0.06
    top_h = 0.24
    ty = fy + fh - m - top_h
    half = (fw - 3 * m) / 2
    ax.add_patch(patches.Rectangle((fx + m, ty), half, top_h, fill=False, ec=CLASS_COLORS["ImageView"], lw=1.0))
    ax.add_patch(patches.Rectangle((fx + 2 * m + half, ty), half, top_h, fill=False,
                                   ec=CLASS_COLORS["ImageView"], lw=1.0))
    ax.add_patch(patches.Rectangle((fx + m, fy + m), fw - 2 * m, ty - fy - 3 * m, fill=False,
                                   ec=CLASS_COLORS["TextView"], lw=1.0))
    cut_y = ty - 1.0 * m
    ax.plot([fx, fx + fw], [cut_y, cut_y], color=INK, lw=1.0, ls=(0, (4, 2)))
    ax.plot([fx + 1.5 * m + half] * 2, [ty - 0.02, fy + fh], color=INK, lw=1.0, ls=(0, (1, 1.5)))
    code = ["Column {", "  Row {", "    Image()", "    Image()", "  }", "  Text()", "}"]
    cx0 = fx + fw + 0.12
    ax.add_patch(patches.FancyBboxPatch((cx0, y + 0.02), x + w - 0.04 - cx0, h - 0.04,
                                        boxstyle="round,pad=0,rounding_size=0.04", fc=PANEL, ec="none"))
    for k, line in enumerate(code):
        ax.text(cx0 + 0.06, y + h - 0.12 - k * 0.128, line, va="center", fontsize=8, family=MONO,
                color=INK)


def fig1() -> Path:
    """Pipeline diagram: four stages, their modules (src/ui2compose/infer/) and output files."""
    ex = load_example()
    dd = ex["config"]["dedup_params"]
    W, H = FULL_W, 3.50
    fig, ax = canvas(W, H)
    widths = [1.70, 1.72, 1.62, 1.98]
    gap = 0.12
    x0 = (W - sum(widths) - gap * (len(widths) - 1)) / 2
    x1 = W - x0
    x = x0
    top, bot = H - 0.30, 0.78
    stages = [
        ("Detection", ["YOLOv8s, 13 classes:", "box, class, confidence",
                       f"(conf {ex['config']['conf_threshold']:g}, NMS IoU {ex['config']['iou_threshold']:g})"],
         "detect_components.py", ["1_detections_raw.json"], _ill_detect),
        ("De-duplication", [f"same class: IoU ≥ {dd['same_class_iou']:g}, or",
                            f"covers ≥ {dd['same_class_iomin']:.0%} if non-nesting;",
                            f"other class: IoU ≥ {dd['cross_class_iou']:g}"],
         "postprocess.py", ["1_detections.json", "1_dedup_removed.json"], _ill_dedup),
        ("Containment", ["parent: smallest box that",
                         f"holds ≥ {ex['config']['containment_ratio']:.0%} of the child;",
                         "spatial edges: export only"],
         "build_ui_graph.py", ["3_ui_graph.json"], _ill_graph),
        ("Layout and code", ["containers → Box, LazyColumn;",
                             "XY-cut → Column / Row;",
                             "rules logged; templates → .kt"],
         "generate_compose_code.py", ["2_layout_tree.json", "4_GeneratedScreen.kt"], _ill_layout),
    ]
    ill_h = 1.02
    for k, ((title, lines, module, outputs, draw), w) in enumerate(zip(stages, widths)):
        ax.add_patch(patches.FancyBboxPatch((x, bot), w, top - bot, boxstyle="round,pad=0,rounding_size=0.06",
                                            fc=WHITE, ec=AXIS, lw=0.8))
        badge(ax, x + 0.15, top - 0.16, str(k + 1), color=INK2, r=0.08, size=8)
        ax.text(x + 0.29, top - 0.16, title, va="center", fontsize=9, weight="bold", color=INK)
        draw(ax, x + 0.02, top - 0.32 - ill_h, w - 0.04, ill_h, ex)
        ty = top - 0.32 - ill_h - 0.14
        for j, line in enumerate(lines):
            ax.text(x + 0.09, ty - j * 0.135, line, va="center", fontsize=8, color=INK)
        ax.plot([x + 0.09, x + w - 0.09], [ty - 2 * 0.135 - 0.10] * 2, color=GRID, lw=0.6)
        my = ty - 2 * 0.135 - 0.21
        ax.text(x + 0.09, my, module, va="center", fontsize=8, family=MONO, color=INK)
        for j, o in enumerate(outputs):
            ax.text(x + 0.09, my - 0.15 - j * 0.13, "→ " + o, va="center", fontsize=8, family=MONO,
                    color=INK2)
        if k < len(stages) - 1:
            ay = top - 0.32 - ill_h / 2
            ax.annotate("", xy=(x + w + gap - 0.005, ay), xytext=(x + w + 0.005, ay),
                        arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.2", color=INK2,
                                        lw=1.0, shrinkA=0, shrinkB=0))
        x += w + gap
    ax.text(x0, H - 0.13, "In: one Android screenshot (PNG)", va="center", fontsize=8, color=INK2)
    ax.text(x1, H - 0.13, "Out: one compilable Jetpack Compose file (.kt)", va="center", ha="right",
            fontsize=8, color=INK2)
    # shared class legend
    ly = 0.62
    lx = x0 + 0.02
    ax.text(lx, ly, "Box colours:", va="center", fontsize=8, weight="bold", color=INK)
    lx += text_w("Box colours:", weight="bold") + 0.12
    for cls, col in CLASS_COLORS.items():
        ax.add_patch(patches.Rectangle((lx, ly - 0.05), 0.16, 0.10, fill=False, ec=col, lw=1.3))
        ax.text(lx + 0.22, ly, cls, va="center", fontsize=8, color=INK)
        lx += 0.22 + text_w(cls) + 0.20
    ax.text(lx, ly, f"(+ {len(CLASS_NAMES) - len(CLASS_COLORS)} classes not on the example)", va="center",
            fontsize=8, color=INK2)
    ax.add_patch(patches.FancyBboxPatch((x0, 0.05), x1 - x0, 0.42, boxstyle="round,pad=0,rounding_size=0.05",
                                        fc=PANEL, ec="none"))
    ax.text(x0 + 0.1, 0.36, "pipeline.py (command ui2compose) runs the four stages and logs every "
            "removal, parent choice and cut;", va="center", fontsize=8, color=INK)
    ax.text(x0 + 0.1, 0.16, "--explain <id> prints the path of one detection through all stages "
            "(2_explain_<id>.txt).  Modules: src/ui2compose/infer/", va="center", fontsize=8, color=INK)
    return save(fig, "fig1_pipeline.png")


# --------------------------------------------------------------------------
# Fig. 2: worked example, cuts and layout tree
# --------------------------------------------------------------------------
# What the leaves show on the drivesafe screen (for orientation only).
EXAMPLE_CAPTIONS = {0: "back arrow", 1: "“Menu”", 2: "menu drawer", 4: "“Auto Start”",
                    3: "check box", 6: "“Reply to Calls”", 5: "check box", 8: "“Reply to SMS”",
                    7: "check box"}


def _id_tag(ax, x, y, text, ha="left", va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=8, color=INK, zorder=12,
            bbox=dict(boxstyle="square,pad=0.12", fc=WHITE, ec="none", alpha=0.9))


def wrap_w(text: str, width: float, size: float = 8, weight: str = "normal") -> List[str]:
    """Greedy word wrap to a rendered width in inches."""
    lines: List[str] = []
    cur = ""
    for word in text.split():
        t = f"{cur} {word}".strip()
        if not cur or text_w(t, size, weight) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _draw_layout_overlay(a: plt.Axes, tree: Dict[str, Any], rows: List[Dict[str, Any]],
                         clip: bool = False) -> None:
    """Containers (rule colour), leaves, root cuts (dashed), inner cuts (dotted)."""
    for r in rows[1:]:
        n = r["node"]
        if n["type"] == "leaf":
            continue
        x1, y1, x2, y2 = n["bbox"]
        col = RULE_COLORS.get(n["rule"], MUTED)
        pad = 10
        for kw in (dict(fc=col, alpha=0.10, ec="none", zorder=3), dict(fill=False, ec=col, lw=1.2, zorder=4)):
            a.add_patch(patches.Rectangle((x1 - pad, y1 - pad), x2 - x1 + 2 * pad, y2 - y1 + 2 * pad,
                                          clip_on=clip, **kw))
    for r in rows:
        n = r["node"]
        if n["type"] == "leaf":
            x1, y1, x2, y2 = n["node"]["bbox"]
            a.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=INK2, lw=0.7, zorder=6))
    for g in tree["gaps"]:
        yy = (g[0] + g[1]) / 2
        a.plot([0, 1200], [yy, yy], color=INK, lw=0.9, ls=(0, (4, 2.5)), zorder=8)
    for r in rows[1:]:
        n = r["node"]
        if n["type"] in ("Row", "Column") and n.get("gaps"):
            x1, y1, x2, y2 = n["bbox"]
            for g in n["gaps"]:
                m = (g[0] + g[1]) / 2
                if n["axis"] == "x":
                    a.plot([m, m], [y1 - 4, y2 + 4], color=INK, lw=0.9, ls=(0, (1, 1.6)), zorder=8)
                else:
                    a.plot([x1 - 4, x2 + 4], [m, m], color=INK, lw=0.9, ls=(0, (1, 1.6)), zorder=8)


def _subtree_span(rows: List[Dict[str, Any]], i0: int) -> int:
    """Index of the last row of the subtree that starts at row i0 (pre-order)."""
    d0 = rows[i0]["depth"]
    i = i0 + 1
    while i < len(rows) and rows[i]["depth"] > d0:
        i += 1
    return i - 1


def fig2() -> Path:
    """Worked example: (a) cuts and containers, (b) layout tree with rules, (c) the detection error."""
    ex = load_example()
    tree = ex["tree"]
    rows = walk(tree)
    cfg = ex["config"]
    graph = json.loads((ex["dir"] / "3_ui_graph.json").read_text(encoding="utf-8"))
    # the error to point at: a Box whose container is an ImageView and that holds a nested layout
    err = next((i for i, r in enumerate(rows) if r["node"]["type"] == "Box"
                and r["node"]["container"]["class"] == "ImageView"
                and any(c["type"] != "leaf" for c in r["node"]["children"])), None)
    if err is None:
        raise SystemExit("Fig. 2: the example has no Box over an ImageView container any more")
    err_end = _subtree_span(rows, err)
    box = rows[err]["node"]
    cont = box["container"]
    covs = [e["coverage"] for e in graph["edges"]
            if e.get("relation") == "parent_child" and e.get("source") == cont["id"]]
    present = {("fallback" if r["node"]["rule"].startswith("fallback") else r["node"]["rule"])
               for r in rows if r["node"]["type"] != "leaf"}

    W = FULL_W
    aw = 2.40
    ah = aw * 1920 / 1200
    leg_h, title_h = 0.72, 0.30
    H = leg_h + ah + title_h
    fig, base = canvas(W, H)
    ax0, ay0 = 0.08, leg_h
    bx0, bw = ax0 + aw + 0.22, 2.86
    cx0 = bx0 + bw + 0.12
    cw = W - cx0 - 0.04
    ty = ay0 + ah + title_h / 2
    for x, t in ((ax0, "(a) Cuts and containers"), (bx0, "(b) Layout tree and rules"),
                 (cx0, "(c) Detection error")):
        base.text(x, ty, t, fontsize=9, weight="bold", color=INK, va="center")

    # ---- (a) screenshot -------------------------------------------------
    a = inch_axes(fig, ax0, ay0, aw, ah)
    a.imshow(light_gray(ex["image"]), extent=[0, 1200, 1920, 0], interpolation="lanczos")
    a.set_xlim(0, 1200)
    a.set_ylim(1920, 0)
    a.set_axis_off()
    a.add_patch(patches.Rectangle((0, 0), 1200, 1920, fill=False, ec=AXIS, lw=0.6))
    _draw_layout_overlay(a, tree, rows)
    for r in rows:
        n = r["node"]
        if n["type"] != "leaf":
            continue
        d = n["node"]
        x1, y1, x2, y2 = d["bbox"]
        if d["id"] == cont["id"]:
            _id_tag(a, x1 + 14, y2 - 14, f"#{d['id']}", va="bottom")
        elif x2 - x1 > 80 and y2 - y1 > 80:
            _id_tag(a, x1 + 6, y2 - 6, f"#{d['id']}", va="bottom")
        else:
            _id_tag(a, x2 + 12, (y1 + y2) / 2, f"#{d['id']}", ha="left")
    for r in rows[1:]:
        n = r["node"]
        if n["type"] in ("Box", "LazyColumn"):
            x1, y1, x2, y2 = n["bbox"]
            col = RULE_COLORS.get(n["rule"], MUTED)
            a.text(x2 - 2, y2 - 2, n["type"], ha="right", va="bottom", fontsize=8,
                   color=text_color_on(col), zorder=13, bbox=dict(boxstyle="square,pad=0.15", fc=col, ec="none"))
    x1, y1, x2, y2 = box["bbox"]
    a.add_patch(patches.FancyBboxPatch((x1 - 22, y1 - 22), x2 - x1 + 44, y2 - y1 + 44,
                                       boxstyle="round,pad=0,rounding_size=14", fill=False,
                                       ec=HIGHLIGHT, lw=1.6, zorder=14))
    badge_data(a, x2 + 60, (y1 + y2) / 2, "1")

    # ---- (b) layout tree --------------------------------------------------
    b = inch_axes(fig, bx0, ay0, bw, ah)
    b.set_xlim(0, bw)
    b.set_ylim(0, ah)
    b.set_axis_off()
    step = min(0.182, (ah - 0.12) / len(rows))
    indent, x_left = 0.20, 0.07
    ys = [ah - 0.08 - i * step for i in range(len(rows))]
    xs = [x_left + r["depth"] * indent for r in rows]
    b.add_patch(patches.Rectangle((0, ys[err_end] - step / 2), bw, ys[err] - ys[err_end] + step,
                                  fc=HIGHLIGHT, alpha=0.07, ec="none", zorder=0))
    b.add_patch(patches.Rectangle((0, ys[err_end] - step / 2), 0.025, ys[err] - ys[err_end] + step,
                                  fc=HIGHLIGHT, ec="none", zorder=1))
    badge(b, bw - 0.10, ys[err], "1")
    for i, r in enumerate(rows):
        if r["parent"] is None:
            continue
        p = r["parent"]
        px = xs[p] + 0.07
        b.plot([px, px, xs[i] - 0.03], [ys[p] - 0.07, ys[i], ys[i]], color=AXIS, lw=0.8, zorder=2)
    for i, r in enumerate(rows):
        n, x, y = r["node"], xs[i], ys[i]
        if n["type"] == "leaf":
            d = n["node"]
            label = f"#{d['id']} {d['class']} {d['confidence']:.2f}"
            b.text(x, y, label, va="center", fontsize=8, color=INK, zorder=4)
            cap = EXAMPLE_CAPTIONS.get(d["id"]) if ex["image"].stem == EXAMPLE else None
            if cap:
                b.text(x + text_w(label) + 0.07, y, cap, va="center", fontsize=8, color=INK2,
                       style="italic", zorder=4)
            continue
        b.text(x, y, n["type"], va="center", fontsize=8, weight="bold", color=INK, zorder=4)
        cx = x + text_w(n["type"], weight="bold") + 0.06
        rule = n["rule"]
        rkey = "fallback" if rule.startswith("fallback") or rule == "least_overlap" else rule
        cwid = chip(b, cx, y, rule, RULE_COLORS.get(rkey, MUTED))
        if n["type"] in ("Column", "Row"):
            k = len(n["gaps"])
            detail = f"{k} cut{'s' if k != 1 else ''}"
        else:
            c = n["container"]
            detail = f"{c['class']} #{c['id']}"
        b.text(cx + cwid + 0.06, y, detail, va="center", fontsize=8, color=INK2, zorder=4)

    # ---- (c) note ----------------------------------------------------------
    lh = 0.138
    y = ay0 + ah - 0.06
    badge(base, cx0 + 0.08, y, "1")
    base.text(cx0 + 0.20, y, "Container class", va="center", fontsize=8, weight="bold", color=INK)
    y -= 0.20
    inner = next(c for c in box["children"] if c["type"] != "leaf")
    n_rows = len(inner["children"])
    note = (f"The menu drawer is detected as {cont['class']} #{cont['id']} ({cont['confidence']:.2f}). "
            f"The {len(covs)} boxes of its {n_rows} rows lie inside it (coverage "
            + (f"{min(covs):.0%}" if min(covs) == max(covs) else f"{min(covs):.0%}–{max(covs):.0%}")
            + f", ≥ {cfg['containment_ratio']:.0%}), so the layout draws them as a Column in a Box over "
            "an Image. Had the drawer been detected as a list (ListView or RecyclerView), the "
            "same rule would have made it a LazyColumn.")
    for line in wrap_w(note, cw - 0.02):
        base.text(cx0, y, line, va="center", fontsize=8, color=INK)
        y -= lh
    y -= 0.16
    if ex["image"].stem == EXAMPLE:
        base.text(cx0, y, "Not detected", va="center", fontsize=8, weight="bold", color=INK)
        y -= 0.18
    note2 = ("Nothing behind the drawer's dimmed overlay gets a box (the DRIVE SAFE image, two "
             "icons, the “Call Reply ON” button), so the code has no element for it.")
    if ex["image"].stem == EXAMPLE:
        for line in wrap_w(note2, cw - 0.02):
            base.text(cx0, y, line, va="center", fontsize=8, color=INK)
            y -= lh
        y -= 0.16
    base.text(cx0, y, "Settings", va="center", fontsize=8, weight="bold", color=INK)
    y -= 0.18
    note3 = (f"cut tolerance {cfg['cut_tol_frac']:g} (cuts need a real gap), containment "
             f"{cfg['containment_ratio']:.0%}, fallback {cfg['fallback']}; "
             f"{len(ex['raw'])} detections, {len(ex['removed'])} removed by de-duplication.")
    for line in wrap_w(note3, cw - 0.02):
        base.text(cx0, y, line, va="center", fontsize=8, color=INK)
        y -= lh

    # ---- legend --------------------------------------------------------------
    y = 0.58
    x = ax0
    base.text(x, y, "Container rules:", va="center", fontsize=8, weight="bold", color=INK)
    x += text_w("Container rules:", weight="bold") + 0.08
    expl = (("gap", "cut through a gap"), ("containment", "Box over a container"),
            ("list", "list container"), ("fallback", "no cut possible"))
    for rule, txt in expl:
        if rule not in present:
            txt += " (none here)"
        need = text_w(rule, family=MONO) + 0.07 + 0.05 + text_w(txt) + 0.16
        if x + need > W - 0.04:
            y -= 0.20
            x = ax0
        cwid = chip(base, x, y, rule, RULE_COLORS[rule])
        base.text(x + cwid + 0.05, y, txt, va="center", fontsize=8, color=INK2)
        x += cwid + 0.05 + text_w(txt) + 0.16
    y -= 0.20
    x = ax0
    base.text(x, y, "Cuts in (a):", va="center", fontsize=8, weight="bold", color=INK)
    x += text_w("Cuts in (a):", weight="bold") + 0.08
    for style, col, txt in (((0, (4, 2.5)), INK, "root Column"),
                            ((0, (1, 1.6)), INK, "inside a Row or Column")):
        base.plot([x, x + 0.30], [y, y], color=col, ls=style, lw=1.1)
        base.text(x + 0.36, y, txt, va="center", fontsize=8, color=INK2)
        x += 0.36 + text_w(txt) + 0.20
    return save(fig, "fig2_layout_example.png")


# --------------------------------------------------------------------------
# Fig. 3: detections of the worked example (single-column width)
# --------------------------------------------------------------------------
def fig3() -> Path:
    """Detections on the example screen: id, class and confidence; removed boxes dashed grey."""
    ex = load_example()
    removed = {d["id"]: d for d in ex["removed"]}
    kept = {d["id"]: d for d in ex["kept"]}
    W = COL_W
    ih = W * 1920 / 1200
    classes = [c for c in CLASS_COLORS if any(d["class_name"] == c for d in ex["raw"])]
    leg_rows = [[(c, f"{CLASS_SHORT[c]} = {c}" if CLASS_SHORT[c] != c else c) for c in classes[k:k + 2]]
                for k in range(0, len(classes), 2)]
    if removed:
        leg_rows.append([(None, "removed by de-duplication")])
    top_m, leg_h = 0.17, 0.12 + 0.18 * len(leg_rows)
    H = top_m + ih + leg_h
    fig, base = canvas(W, H)
    a = inch_axes(fig, 0, leg_h, W, ih)
    a.imshow(light_gray(ex["image"], 0.45), extent=[0, 1200, 1920, 0], interpolation="lanczos")
    a.set_xlim(0, 1200)
    a.set_ylim(1920, 0)
    a.set_axis_off()
    a.add_patch(patches.Rectangle((0, 0), 1200, 1920, fill=False, ec=AXIS, lw=0.6))

    def place(d):
        """Label anchor and alignment for the drivesafe example (default: above the box)."""
        x1, y1, x2, y2 = d["bbox"]
        spec = {
            1: ((x2 + 8, (y1 + y2) / 2), "left", "center"),
            2: ((x1 + 10, y2 - 10), "left", "bottom"),
        }
        if ex["image"].stem != EXAMPLE:
            spec = {}
        return spec.get(d["id"], ((x1, y1 - 3), "left", "bottom"))
    order = sorted(list(kept.values()) + list(removed.values()), key=lambda d: d["id"])
    for d in order:
        x1, y1, x2, y2 = d["bbox"]
        gone = d["id"] in removed
        col = MUTED if gone else CLASS_COLORS.get(d["class_name"], INK2)
        a.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=col,
                                      lw=1.1 if gone else 1.3, ls=(0, (3, 2)) if gone else "-",
                                      zorder=5 if gone else 6))
    for d in order:
        gone = d["id"] in removed
        fill = REMOVED_FILL if gone else CLASS_COLORS.get(d["class_name"], INK2)
        (tx, ty), ha, va = place(d)
        label = f"#{d['id']} {CLASS_SHORT.get(d['class_name'], d['class_name'])} {d['confidence']:.2f}"
        a.text(tx, ty, label, ha=ha, va=va, fontsize=8, color=text_color_on(fill), zorder=9,
               clip_on=False, bbox=dict(boxstyle="square,pad=0.14", fc=fill, ec="none"))
    y = leg_h - 0.14
    for row in leg_rows:
        x = 0.04
        for cls, lab in row:
            if cls is None:
                base.add_patch(patches.Rectangle((x, y - 0.05), 0.16, 0.10, fill=False, ec=MUTED, lw=1.1,
                                                 ls=(0, (3, 2))))
            else:
                base.add_patch(patches.Rectangle((x, y - 0.05), 0.16, 0.10, fill=False,
                                                 ec=CLASS_COLORS[cls], lw=1.3))
            base.text(x + 0.22, y, lab, va="center", fontsize=8, color=INK)
            x += 1.62
        y -= 0.18
    return save(fig, "fig3_detections.png")


# --------------------------------------------------------------------------
# Fig. 4: input | ui2compose v1.2.0 | Claude run 1
# --------------------------------------------------------------------------
def check_renders() -> Dict[str, Any]:
    """Checks that each ui2compose render belongs to the current Kotlin (results/compile_report.json)."""
    report = json.loads((REPO / "results" / "compile_report.json").read_text(encoding="utf-8"))
    by_src = {f["source"]: f for f in report["files"]}
    claude = json.loads((MLLM_DIR / "compile_report.json").read_text(encoding="utf-8"))
    claude_by_src = {f["source"]: f for f in claude["files"]}
    out = {}
    for stem, _ in FIG4_SCREENS:
        kt = V120_DIR / stem / "4_GeneratedScreen.kt"
        f = by_src.get(rel(kt))
        if f is None or f["sha256"] != sha256(kt) or not f["rendered"] \
                or f["render"] != rel(kt.parent / "render.png"):
            raise SystemExit(f"{stem}: render.png does not belong to the current 4_GeneratedScreen.kt; "
                             "re-run harness/compile_and_render.sh")
        c = claude_by_src.get(rel(MLLM_DIR / stem / "run1" / "GeneratedScreen.kt"))
        out[stem] = {"render": kt.parent / "render.png", "render_sha256": sha256(kt.parent / "render.png"),
                     "kt_sha256": f["sha256"], "claude": c}
    return out


def fig4() -> Path:
    """Four example screens: input, ui2compose v1.2.0 render, Claude run-1 render (or its compile error)."""
    info = check_renders()
    W = FULL_W
    gap_b, gap_p = 0.20, 0.06
    bw = (W - 0.04 - gap_b) / 2
    pw = (bw - 2 * gap_p) / 3
    ph = pw * 1920 / 1200
    title_h, cap_h = 0.22, 0.20
    block_h = title_h + cap_h + ph
    row_gap = 0.14
    H = 2 * block_h + row_gap + 0.04
    fig, base = canvas(W, H)
    for k, (stem, title) in enumerate(FIG4_SCREENS):
        col, row = k % 2, k // 2
        bx = 0.02 + col * (bw + gap_b)
        by = H - 0.02 - (row + 1) * block_h - row * row_gap
        base.text(bx, by + block_h - title_h / 2, title, va="center", fontsize=9, weight="bold", color=INK)
        claude = info[stem]["claude"]
        panels = [("Input", EXAMPLES_DIR / f"{stem}.png"), ("ui2compose v1.2.0", info[stem]["render"]),
                  ("Claude, run 1", Path(REPO / claude["render"]) if claude and claude.get("rendered") else None)]
        for j, (cap, img) in enumerate(panels):
            px = bx + j * (pw + gap_p)
            base.text(px, by + ph + cap_h / 2, cap, va="center", fontsize=8, color=INK2)
            ax = inch_axes(fig, px, by, pw, ph)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            if img is not None:
                ax.imshow(load_rgb(img), extent=[0, 1, 0, 1], aspect="auto", interpolation="lanczos")
                for s in ax.spines.values():
                    s.set_color(AXIS)
                    s.set_linewidth(0.6)
                continue
            ax.add_patch(patches.Rectangle((0, 0), 1, 1, fc="#f2f1ed", hatch="////", edgecolor="#d6d5cf",
                                           lw=0))
            for s in ax.spines.values():
                s.set_color(HIGHLIGHT)
                s.set_linewidth(1.4)
            err = ((claude or {}).get("compile_errors") or [""])[0]
            line_no = err.split(":")[0] if err[:1].isdigit() else ""
            ax.text(0.5, 0.80, "Compile error", ha="center", va="center", fontsize=9, weight="bold",
                    color=HIGHLIGHT, bbox=dict(boxstyle="square,pad=0.2", fc=WHITE, ec="none"))
            detail = ["uses the internal", "Compose API", "‘weight’" + (f" (line {line_no})" if line_no else "")]
            ax.text(0.5, 0.56, "\n".join(detail), ha="center", va="center", fontsize=8, color=INK,
                    linespacing=1.4, bbox=dict(boxstyle="square,pad=0.25", fc=WHITE, ec="none"))
            ax.text(0.5, 0.25, "run 2 compiled\nand rendered", ha="center", va="center", fontsize=8,
                    color=INK2, linespacing=1.4, bbox=dict(boxstyle="square,pad=0.25", fc=WHITE, ec="none"))
    return save(fig, "fig4_claude_comparison.png")


# --------------------------------------------------------------------------
# Fig. 5: detection on the app-level split, baseline vs oversampling
# --------------------------------------------------------------------------
TABLE2 = REPO / "results" / "eval_appsplit" / "detector_table2.json"
FIG5_SETS = (("test_full", "Test"), ("val_full", "Validation"))
FIG5_SCHEME = "app_groups"
FIG5_PAIR = "oversample_5k - baseline"


def detector_data() -> Path:
    """Writes docs/figures/fig5_detector_data.json: the values Fig. 5 draws, taken from
    results/eval_appsplit/detector_table2.json (BCa intervals over the split's app groups)."""
    t2 = json.loads(TABLE2.read_text(encoding="utf-8"))
    out: Dict[str, Any] = {
        "description": "Fig. 5 data: app-level split, baseline vs oversample_5k, AP@50 with BCa 95% "
                       "intervals over app groups (paired differences use the same draws)",
        "source": {"file": rel(TABLE2), "sha256": sha256(TABLE2)},
        "settings": {k: t2["settings"][k] for k in ("bootstrap", "seed", "alpha", "primary_scheme")},
        "checkpoints": {m: {k: t2["checkpoints"][m][k] for k in ("label", "weights", "weights_md5")}
                        for m in CONFIGS},
        "sets": {}}
    for key, _ in FIG5_SETS:
        s = t2["sets"][key]
        sch = s["schemes"][FIG5_SCHEME]
        e = out["sets"][key] = {"split": s["split"], "n_images": s["n_images"], "n_units": sch["n_units"],
                                "support": s["support"], "models": {}, "diff": {}}
        for m in CONFIGS:
            mm = sch["models"][m]
            e["models"][m] = {"mAP50": {"point": mm["mAP50"]["point"], "bca": mm["mAP50"]["bca"]},
                              "per_class": {c: {"point": mm[f"AP50:{c}"]["point"], "bca": mm[f"AP50:{c}"]["bca"]}
                                            for c in CLASS_NAMES}}
        pp = sch["pairs"][FIG5_PAIR]
        e["diff"] = {"mAP50": {"point": pp["mAP50"]["point"], "bca": pp["mAP50"]["bca"],
                               "p_le_0": pp["mAP50"].get("p_le_0")},
                     "per_class": {c: {"point": pp[f"AP50:{c}"]["point"], "bca": pp[f"AP50:{c}"]["bca"]}
                                   for c in CLASS_NAMES}}
    FIG5_DATA.parent.mkdir(parents=True, exist_ok=True)
    FIG5_DATA.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {rel(FIG5_DATA)}")
    return FIG5_DATA


def _dot(ax: plt.Axes, x: float, y: float, cfg: str, **kw: Any) -> None:
    ax.plot([x], [y], marker=CONFIG_MARKER[cfg], ms=CONFIG_MS[cfg], color=CONFIG_COLOR[cfg], mec=WHITE,
            mew=0.8, ls="none", zorder=4, clip_on=False, **kw)


def _ci(ax: plt.Axes, lo: float, hi: float, y: float, cfg: str) -> None:
    ax.plot([lo, hi], [y, y], color=CONFIG_COLOR[cfg], lw=1.2, solid_capstyle="round", zorder=3)


def _dot_axes(ax: plt.Axes, xmin: float, xmax: float, step: float, xlabel: str) -> None:
    ax.patch.set_visible(False)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(np.arange(xmin, xmax + 0.1, step))
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel, fontsize=8, color=INK2)


def _pts(v: float) -> str:
    """Signed difference in percentage points, one decimal."""
    return f"{100 * v:+.1f}".replace("-", "−")


def fig5() -> Optional[Path]:
    """(a) per-class AP@50 on the app-split test, baseline vs 5k; (b) mAP@50 on test and validation."""
    if not TABLE2.is_file():
        print(f"skip Fig. 5: {rel(TABLE2)} missing")
        return None
    data = json.loads(detector_data().read_text(encoding="utf-8"))
    test = data["sets"]["test_full"]
    base_pc = test["models"]["baseline"]["per_class"]
    os_pc = test["models"]["oversample_5k"]["per_class"]
    diff = test["diff"]["per_class"]
    groups = [(f"Oversampled in training ({len(OVERSAMPLED)} classes)", [c for c in CLASS_NAMES if c in OVERSAMPLED]),
              (f"Not oversampled ({len(CLASS_NAMES) - len(OVERSAMPLED)} classes)",
               [c for c in CLASS_NAMES if c not in OVERSAMPLED])]
    for _, cl in groups:
        cl.sort(key=lambda c: -diff[c]["point"])
    W, H = FULL_W, 4.45
    fig, base = canvas(W, H)
    lx, ly = 0.08, H - 0.13
    for cfg in CONFIGS:
        base.plot([lx, lx + 0.22], [ly, ly], color=CONFIG_COLOR[cfg], lw=1.2, solid_capstyle="round")
        _dot(base, lx + 0.11, ly, cfg)
        base.text(lx + 0.30, ly, CONFIG_LABEL[cfg], va="center", fontsize=8, color=INK)
        lx += 0.30 + text_w(CONFIG_LABEL[cfg]) + 0.32
    base.text(lx, ly, "both retrained on the app-level split", va="center", fontsize=8, color=INK2)
    ax_y, ax_h = 0.50, H - 1.20
    title_y = H - 0.42
    # ---- (a) ---------------------------------------------------------------
    ax_x, ax_w = 1.30, 2.20
    base.text(0.08, title_y, f"(a) AP@50 per class, {test['n_images']:,} test screens of unseen apps",
              fontsize=9, weight="bold", color=INK, va="center")
    a = inch_axes(fig, ax_x, ax_y, ax_w, ax_h)
    ypos, labels, heads = [], [], []
    y = 0.0
    for gname, cl in groups:
        heads.append((gname, y))
        y += 1.0
        for c in cl:
            ypos.append(y)
            labels.append(c)
            y += 1.0
        y += 0.35
    y_top = -1.05
    a.set_ylim(y - 0.35 - 0.4, y_top)
    _dot_axes(a, 0, 100, 20, "AP@50 (%), 95% BCa CI")
    a.set_yticks(ypos)
    a.set_yticklabels([f"{c} ({test['support'][c]:,})" for c in labels], fontsize=8, color=INK)
    off = 0.20
    for yy, c in zip(ypos, labels):
        for cfg, pc, dy in (("baseline", base_pc, -off), ("oversample_5k", os_pc, off)):
            v = pc[c]
            _ci(a, 100 * v["bca"][0], 100 * v["bca"][1], yy + dy, cfg)
            _dot(a, 100 * v["point"], yy + dy, cfg)
    ytr = a.get_yaxis_transform()
    for gname, yy in heads:
        a.text(-ax_x / ax_w + 0.08 / ax_w, yy + 0.05, gname, transform=ytr, ha="left", va="center",
               fontsize=8, weight="bold", color=INK)
    a.text(-0.03, y_top + 0.15, "class (test instances)", transform=ytr, fontsize=8, color=INK2,
           va="center", ha="right")
    cx = 1 + 0.10 / ax_w
    a.text(cx, y_top + 0.15, "5k − baseline, points", transform=ytr, fontsize=8, weight="bold", color=INK,
           va="center")
    a.text(cx, y_top + 0.85, "(95% BCa CI)", transform=ytr, fontsize=8, color=INK2, va="center")
    for yy, c in zip(ypos, labels):
        d = diff[c]
        sig = d["bca"][0] > 0 or d["bca"][1] < 0
        a.text(cx, yy, f"{_pts(d['point'])} [{_pts(d['bca'][0])}, {_pts(d['bca'][1])}]", transform=ytr,
               fontsize=8, color=INK if sig else INK2, va="center", weight="bold" if sig else "normal")
    # ---- (b) ---------------------------------------------------------------
    bx = 5.25
    bw = W - bx - 0.14
    base.text(bx - 0.05, title_y, "(b) mAP@50 (13 classes)", fontsize=9, weight="bold", color=INK, va="center")
    b = inch_axes(fig, bx, ax_y, bw, ax_h)
    y = 0.0
    rows_b, heads_b = [], []
    for key, lab in FIG5_SETS:
        s = data["sets"][key]
        d = s["diff"]["mAP50"]
        heads_b.append((f"{lab} ({s['n_images']:,} screens)", f"{s['n_units']} app groups",
                        f"5k − baseline {_pts(d['point'])} [{_pts(d['bca'][0])}, {_pts(d['bca'][1])}]", y))
        y += 1.95
        for cfg in CONFIGS:
            rows_b.append((key, cfg, y))
            y += 0.70
        y += 0.80
    b.set_ylim(y - 0.80 - 0.2, -0.55)
    _dot_axes(b, 20, 60, 10, "mAP@50 (%), 95% BCa CI")
    b.set_yticks([])
    for key, cfg, yy in rows_b:
        s = data["sets"][key]["models"][cfg]["mAP50"]
        lo, hi = s["bca"]
        _ci(b, 100 * lo, 100 * hi, yy, cfg)
        _dot(b, 100 * s["point"], yy, cfg)
        b.text(100 * hi + 1.0, yy, f"{100 * s['point']:.1f}", fontsize=8, color=INK, va="center")
    for lab, note, dtxt, yy in heads_b:
        for k, (txt, col, wt) in enumerate(((lab, INK, "bold"), (note, INK2, "normal"), (dtxt, INK, "normal"))):
            b.text(20.3, yy + 0.58 * k, txt, fontsize=8, weight=wt, color=col, va="center",
                   bbox=dict(boxstyle="square,pad=0.05", fc=WHITE, ec="none"))
    return save(fig, "fig5_detector.png")


FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", type=int, nargs="+", choices=sorted(FIGS), help="figure numbers to draw")
    args = ap.parse_args(argv)
    os.chdir(REPO)
    for k in args.only or sorted(FIGS):
        if k == 4 and not (MLLM_DIR / "compile_report.json").is_file():
            # The Claude outputs of 20 October 2026 are not in the release
            # (results/mllm_claude/README.md); the committed Fig. 4 is kept.
            print("Fig. 4 skipped: the Claude renders are not in results/mllm_claude/")
            continue
        FIGS[k]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
