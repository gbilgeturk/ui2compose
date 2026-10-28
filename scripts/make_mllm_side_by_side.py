#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Draws the side-by-side figures of results/mllm_claude/ (see its ANALYSIS.md, section 2).

Usage (from the repository root):
    python scripts/make_mllm_side_by_side.py

Outputs:
  results/mllm_claude/side_by_side_all.png  all ten screens, one per row:
                                            input | ui2compose v1.2.0 render | Claude run-1 render
  results/mllm_claude/side_by_side.png      four screens (SCREENS_4) as columns, same three rows

The ui2compose panels are the current renders in results/v1.2.0_examples/ (default weights
and thresholds). Each is used only if results/compile_report.json records it as the render
of the current 4_GeneratedScreen.kt (same SHA-256), as in scripts/make_figures.py (Fig. 4).
A Claude run that did not compile is drawn as a hatched panel with a red frame and its first
compiler error. Fonts are the DejaVu fonts bundled with matplotlib; no model is called.
"""

from __future__ import annotations

import hashlib
import json
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
V120 = REPO / "results" / "v1.2.0_examples"
MLLM = REPO / "results" / "mllm_claude"
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"

PANEL_W, PANEL_H = 375, 600   # 1200 x 1920 inputs and 625 x 1000 renders, both 5:8
GAP = 18
FRAME = (195, 194, 183)
RED = (190, 30, 30)
INK = (11, 11, 11)
INK2 = (82, 81, 78)
GREEN = (30, 110, 40)

# Section 2 of ANALYSIS.md: the typical case, the held-out worked example of the paper,
# the closest case, and a Claude run-1 compile failure.
SCREENS_4 = (
    "com.foxfi-screens_screenshot_1",
    "westport.andrewirwin.com.drivesafe-screens_screenshot_3",
    "com.netbeast.yeti-screens_screenshot_1",
    "com.mineworld.swordsaddon-screens_screenshot_2",
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return str(p.resolve().relative_to(REPO))


def label(stem: str) -> str:
    app, shot = stem.split("-screens_screenshot_")
    return f"{app} #{shot}"


def stems() -> List[str]:
    return sorted(p.stem for p in EXAMPLES.glob("*.png"))


def ui2compose_renders() -> Dict[str, Path]:
    report = json.loads((REPO / "results" / "compile_report.json").read_text(encoding="utf-8"))
    by_src = {f["source"]: f for f in report["files"]}
    out = {}
    for stem in stems():
        kt = V120 / stem / "4_GeneratedScreen.kt"
        f = by_src.get(rel(kt))
        if f is None or f["sha256"] != sha256(kt) or not f["rendered"] or f["render"] != rel(kt.parent / "render.png"):
            raise SystemExit(f"{stem}: render.png does not belong to the current 4_GeneratedScreen.kt; "
                             "re-run harness/compile_and_render.sh")
        out[stem] = kt.parent / "render.png"
    return out


def claude_runs() -> Dict[Tuple[str, int], dict]:
    report = json.loads((MLLM / "compile_report.json").read_text(encoding="utf-8"))
    out = {}
    for f in report["files"]:
        parts = Path(f["source"]).parts
        out[(parts[-3], int(parts[-2][3:]))] = f
    return out


def panel_image(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("RGB").resize((PANEL_W, PANEL_H), Image.LANCZOS)


def error_panel(err: str) -> Image.Image:
    im = Image.new("RGB", (PANEL_W, PANEL_H), (238, 238, 236))
    d = ImageDraw.Draw(im)
    for x in range(-PANEL_H, PANEL_W, 22):
        d.line([(x, PANEL_H), (x + PANEL_H, 0)], fill=(214, 213, 207), width=1)
    d.rectangle([0, 0, PANEL_W - 1, PANEL_H - 1], outline=RED, width=6)
    y = 190
    for text, f, c in (("COMPILE ERROR", font(34, True), RED), ("no render", font(18), RED)):
        d.text((PANEL_W / 2, y), text, font=f, fill=c, anchor="mm")
        y += 36
    y += 20
    for line in textwrap.wrap(err, 30)[:5]:
        d.text((PANEL_W / 2, y), line, font=font(16), fill=INK, anchor="mm")
        y += 24
    y += 30
    for line in ("run 2 compiled", "and rendered"):
        d.text((PANEL_W / 2, y), line, font=font(18), fill=GREEN, anchor="mm")
        y += 26
    return im


def claude_panel(run: Optional[dict]) -> Image.Image:
    if run and run.get("rendered"):
        return panel_image(REPO / run["render"])
    return error_panel(((run or {}).get("compile_errors") or ["(no record)"])[0])


def framed(canvas: Image.Image, im: Image.Image, x: int, y: int) -> None:
    canvas.paste(im, (x, y))
    ImageDraw.Draw(canvas).rectangle([x - 1, y - 1, x + PANEL_W, y + PANEL_H], outline=FRAME, width=1)


def run_status(run: Optional[dict]) -> str:
    return "compiles" if run and run.get("compiled") else "compile error"


def draw_all(renders: Dict[str, Path], runs: Dict[Tuple[str, int], dict]) -> Path:
    cols = ("Input", "ui2compose v1.2.0", "Claude Sonnet 5.5, run 1")
    lab_h, head_h, left = 56, 50, 24
    rows = stems()
    W = left + 3 * PANEL_W + 2 * GAP + 24
    H = head_h + len(rows) * (lab_h + PANEL_H + GAP)
    c = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(c)
    for j, t in enumerate(cols):
        d.text((left + j * (PANEL_W + GAP) + PANEL_W / 2, head_h / 2), t, font=font(22, True), fill=INK, anchor="mm")
    for i, stem in enumerate(rows):
        y = head_h + i * (lab_h + PANEL_H + GAP)
        r1, r2 = runs.get((stem, 1)), runs.get((stem, 2))
        d.text((left, y + lab_h / 2), label(stem), font=font(20, True), fill=INK, anchor="lm")
        d.text((W - 24, y + lab_h / 2), f"Claude run 1: {run_status(r1)}; run 2: {run_status(r2)}",
               font=font(16), fill=INK2, anchor="rm")
        y += lab_h
        for j, im in enumerate((panel_image(EXAMPLES / f"{stem}.png"), panel_image(renders[stem]), claude_panel(r1))):
            framed(c, im, left + j * (PANEL_W + GAP), y)
    out = MLLM / "side_by_side_all.png"
    c.save(out, optimize=True)
    return out


def draw_four(renders: Dict[str, Path], runs: Dict[Tuple[str, int], dict]) -> Path:
    rows = ("Input screenshot", "ui2compose v1.2.0", "Claude Sonnet 5.5, run 1")
    left, head_h = 80, 90
    W = left + 4 * PANEL_W + 3 * GAP + 20
    H = head_h + 3 * PANEL_H + 2 * GAP + 20
    c = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(c)
    for k, stem in enumerate(SCREENS_4):
        x = left + k * (PANEL_W + GAP)
        app, shot = label(stem).split(" #")
        cut = app.rfind(".", 0, 26)   # long package names break before a dot
        lines = [app[:cut], f"{app[cut:]} #{shot}"] if len(app) > 24 and cut > 0 else [f"{app} #{shot}"]
        for n, line in enumerate(lines):
            d.text((x + PANEL_W / 2, head_h / 2 + (n - (len(lines) - 1) / 2) * 30), line,
                   font=font(24, True), fill=INK, anchor="mm")
        panels = (panel_image(EXAMPLES / f"{stem}.png"), panel_image(renders[stem]), claude_panel(runs.get((stem, 1))))
        for r, im in enumerate(panels):
            framed(c, im, x, head_h + r * (PANEL_H + GAP))
    for r, t in enumerate(rows):
        txt = Image.new("RGBA", (PANEL_H, 50), (255, 255, 255, 0))
        ImageDraw.Draw(txt).text((PANEL_H / 2, 25), t, font=font(26, True), fill=INK, anchor="mm")
        c.paste(txt.rotate(90, expand=True), (14, head_h + r * (PANEL_H + GAP)), txt.rotate(90, expand=True))
    out = MLLM / "side_by_side.png"
    c.save(out, optimize=True)
    return out


def main() -> int:
    renders, runs = ui2compose_renders(), claude_runs()
    for p in (draw_all(renders, runs), draw_four(renders, runs)):
        w, h = Image.open(p).size
        print(f"wrote {rel(p)}  {w} x {h} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
