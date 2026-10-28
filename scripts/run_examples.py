#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Runs the pipeline on the bundled examples and compares the output with v1.1.0.

Usage (from the repository root):
    python scripts/run_examples.py [--model runs/oversample_5k_appsplit/weights/best.pt]
                                   [--no-golden] [--no-docs-figure]
    python scripts/run_examples.py --compare-only

What it does:
  1. Runs ui2compose.infer.pipeline.run_pipeline with its default settings on
     every examples/*.png (one shared detector) and writes all outputs to
     results/v1.2.0_examples/<stem>/. Only the files the pipeline writes
     (PIPELINE_FILES and 2_explain_*.txt) are replaced; other files, such as
     the render.png of harness/compile_and_render.sh, are kept. When the
     Kotlin of a screen that has a render.png changes, the script names the
     screen: re-run the harness (harness/README.md), otherwise
     tests/test_harness.py reports the stale render. The detection
     visualisation (1_detections_viz.png) is written for the worked example
     only; for that screen the explain traces of EXPLAIN_IDS are also written
     (2_explain_<id>.txt, same format as `ui2compose --explain`).
  2. Copies 1_detections_raw.json, 2_layout_tree.json and 4_GeneratedScreen.kt
     of every screen to tests/golden/<stem>/ (tests/test_golden.py replays
     the pipeline from the raw detections and must reproduce the other two
     files exactly), and the worked example's visualisation to docs/example_detections.png.
  3. Writes results/examples_comparison.json and .md: per screen and in total,
     the v1.1.0 reference outputs (results/v1.1.0_examples) against the
     v1.2.0 outputs, with the metrics defined in METRIC_DEFINITIONS.

--compare-only skips steps 1 and 2 and recomputes step 3 from the files
already in results/v1.2.0_examples (no torch or ultralytics needed);
tests/test_golden.py uses the same functions to check that the published
comparison matches the published outputs.

Paths are passed to the pipeline relative to the repository root, so the
recorded 0_run_config.json does not contain machine-specific paths.
"""

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = Path('examples')
V110_DIR = Path('results/v1.1.0_examples')
V120_DIR = Path('results/v1.2.0_examples')
GOLDEN_DIR = Path('tests/golden')
COMPARISON_STEM = Path('results/examples_comparison')
DOCS_FIGURE = Path('docs/example_detections.png')
DEFAULT_MODEL = 'runs/oversample_5k_appsplit/weights/best.pt'
DATASET_YAML = 'configs/dataset.yaml'

#: Screen whose detection visualisation is kept (the paper's worked example, Figs. 2 and 3:
#: a test screen of the app-level split, so an app the default weights never saw).
EXAMPLE_STEM = 'westport.andrewirwin.com.drivesafe-screens_screenshot_3'
#: Detection ids of the worked example whose explain traces are written:
#: 2 = menu drawer detected as ImageView (becomes the container of a Box),
#: 3 = CheckBox of the first row, contained in ImageView 2,
#: 4 = TextView "Auto Start", in a Row with CheckBox 3.
EXPLAIN_IDS = (2, 3, 4)
#: Files copied to tests/golden/<stem>/.
GOLDEN_FILES = ('1_detections_raw.json', '2_layout_tree.json', '4_GeneratedScreen.kt')
#: Files run_pipeline writes into an output folder (plus 2_explain_<id>.txt); a
#: re-run replaces exactly these and keeps every other file of the folder.
PIPELINE_FILES = ('0_run_config.json', '1_detections_raw.json', '1_detections.json',
                  '1_dedup_removed.json', '1_detections_viz.png', '2_layout_tree.json',
                  '3_ui_graph.json', '4_GeneratedScreen.kt')
EXPLAIN_GLOB = '2_explain_*.txt'
#: File written next to the outputs by harness/compile_and_render.sh.
RENDER_FILE = 'render.png'

#: Layout rules counted as an XY-cut fallback (no cut existed on either axis).
FALLBACK_RULES = ('fallback_y_sorted', 'least_overlap')

METRIC_DEFINITIONS: Dict[str, str] = {
    'raw_detections': "detector output at the pipeline settings (conf 0.3, NMS IoU 0.7, the "
                      "checkpoint's imgsz 640)",
    'dedup_removed': "detections removed by postprocess.deduplicate_detections "
                     "(v1.1.0 had no de-duplication step: 0)",
    'emitted': "detections that appear in the generated Kotlin. v1.1.0: the leaves "
               "collected by the v1.1.0 generator (containers were dropped, a child of two "
               "parents was collected twice); v1.2.0: every leaf template plus every list "
               "container emitted as a LazyColumn (cross-checked against the trace comments "
               "of the generator)",
    'emitted_by_class': "the emitted detections counted by detector class",
    'duplicated_ids': "detection ids emitted more than once (id -> count)",
    'dropped_ids': "detections that reach code generation but are not emitted",
    'placeholders': "number of 'Unknown: <class>' placeholder texts in the Kotlin",
    'xycut_fallback_nodes': "layout nodes where XY-cut found no cut on either axis and the "
                            "fallback ordered the members (v1.1.0: zero-tolerance cuts; "
                            "v1.2.0: cuts with the default tolerance cut_tol_frac selected on the "
                            "validation split, 0.0)",
    'xycut_fallback_nodes_tol0': "the same count with cut_tol_frac=0 (the zero-tolerance cut "
                                 "rule of v1.1.0), recomputed from the v1.2.0 graph: separates "
                                 "the effect of containment and de-duplication from that of "
                                 "the tolerance (v1.1.0: equal to xycut_fallback_nodes)",
    'gap_tolerance_nodes': "v1.2.0 layout nodes with at least one cut that needed the "
                           "tolerance (touching or overlapping by less than cut_tol_frac of the "
                           "image extent)",
    'box_composites': "containers with detected children, emitted as Box { container "
                      "widget; children's layout } (v1.1.0: 'Box(' blocks in the Kotlin that "
                      "are not 'Unknown' placeholders)",
    'lazycolumn_composites': "list containers (ListView/RecyclerView) with detected children, "
                             "emitted as LazyColumn { item { ... } }; a list without children "
                             "is a leaf template and is not counted (v1.1.0: 'LazyColumn(' "
                             "occurrences in the Kotlin)",
    'parent_child_edges': "containment edges of the UI graph",
    'spatial_edges': "above/below/left_of/right_of edges of the UI graph (inspection export, "
                     "not read by code generation)",
    'kotlin_lines': "len(code.splitlines()) of 4_GeneratedScreen.kt",
}


# ── Metrics of one screen ───────────────────────────────────


def walk_layout(node: Dict) -> Iterator[Dict]:
    """Yields every node of a layout tree in depth-first order.

    Input:  node — layout node (see generate_compose_code.build_layout_tree)
    Output: iterator over the node and all its descendants
    """
    yield node
    for child in node.get('children', []):
        yield from walk_layout(child)


def emitted_detections(tree: Dict) -> List[Dict]:
    """Lists the detections a layout tree emits: every leaf and every LazyColumn list container.

    Input:  tree — root layout node
    Output: list of {'id', 'class', 'confidence'} in depth-first order (an id
            emitted twice would appear twice)
    """
    out = []
    for node in walk_layout(tree):
        if node['type'] == 'leaf':
            info = node['node']
        elif node['type'] == 'LazyColumn':
            info = node['container']
        else:
            continue
        out.append({'id': info['id'], 'class': info['class'], 'confidence': info['confidence']})
    return out


def emitted_from_trace_comments(code_with_trace: str) -> List[Dict]:
    """Recovers the emitted detections from Kotlin generated with trace_comments=True.

    The generator writes '// leaf <Class>#<id> (...)' before every leaf template
    and '// LazyColumn (rule=list, list container <Class>#<id> (...))' before
    every list container, so this is an independent count from the code text.

    Input:  code_with_trace — Kotlin source generated with trace comments
    Output: list of {'id', 'class'} in code order
    """
    pattern = re.compile(r'^\s*// (?:leaf|LazyColumn \(rule=list, list container) (\w+)#(\d+) ',
                         re.MULTILINE)
    return [{'id': int(m.group(2)), 'class': m.group(1)} for m in pattern.finditer(code_with_trace)]


def _short_counts(counter: Dict[str, int]) -> Dict[str, int]:
    """Returns a class-count dictionary sorted by class name, without zero entries.

    Input:  counter — mapping class name -> count
    Output: sorted plain dictionary
    """
    return {k: int(v) for k, v in sorted(counter.items()) if v}


def fallback_nodes(tree: Dict) -> int:
    """Counts the layout nodes (and LazyColumn item splits) decided by an XY-cut fallback.

    Input:  tree — root layout node
    Output: number of rules in FALLBACK_RULES
    """
    n = 0
    for node in walk_layout(tree):
        rules = [node.get('rule')]
        if node['type'] == 'LazyColumn':
            rules.append(node.get('split_rule'))
        n += sum(1 for r in rules if r in FALLBACK_RULES)
    return n


def fallback_nodes_at_zero_tolerance(graph: Dict, tree: Dict) -> Optional[int]:
    """Recomputes the layout of a graph with cut_tol_frac=0 and counts its fallback nodes.

    Input:  graph — 3_ui_graph.json (with 'hierarchy' and 'image_size');
            tree — 2_layout_tree.json (its 'meta' gives layout mode and fallback)
    Output: fallback count, or None if graph or tree lack the needed fields
    """
    from ui2compose.infer.generate_compose_code import build_layout_tree

    meta = tree.get('meta')
    if 'hierarchy' not in graph or not meta:
        return None
    zero = build_layout_tree(graph, image_size=meta['image_size'], layout=meta['layout'],
                             cut_tol_frac=0.0, fallback=meta['fallback'])
    return fallback_nodes(zero)


def v120_metrics(raw: List[Dict], kept: List[Dict], removed: List[Dict], graph: Dict,
                 tree: Dict, code: str) -> Dict[str, Any]:
    """Computes the comparison metrics of one screen from the v1.2.0 pipeline outputs.

    Input:  raw — 1_detections_raw.json; kept — 1_detections.json; removed —
            1_dedup_removed.json; graph — 3_ui_graph.json; tree — 2_layout_tree.json;
            code — 4_GeneratedScreen.kt
    Output: metrics dictionary (keys of METRIC_DEFINITIONS plus details)
    """
    emitted = emitted_detections(tree)
    ids = [e['id'] for e in emitted]
    id_counts = Counter(ids)
    kept_by_id = {d['id']: d for d in kept}
    dropped = [[i, kept_by_id[i]['class_name'], round(kept_by_id[i]['confidence'], 2)]
               for i in sorted(set(kept_by_id) - set(ids))]
    fallback = fallback_nodes(tree)
    gap_tol = 0
    for node in walk_layout(tree):
        rules = [node.get('rule')]
        if node['type'] == 'LazyColumn':
            rules.append(node.get('split_rule'))
        gap_tol += sum(1 for r in rules if r == 'gap_tolerance')
    placeholders = re.findall(r'Unknown: (\w+)', code)
    relations = Counter(e['relation'] for e in graph['edges'])
    return {
        'raw_detections': len(raw),
        'raw_by_class': _short_counts(Counter(d['class_name'] for d in raw)),
        'dedup_removed': len(removed),
        'dedup_removals': [[d['id'], d['class_name'], round(d['confidence'], 3), d['dedup_rule'],
                            round(d['overlap'], 3), d['removed_by']] for d in removed],
        'kept_detections': len(kept),
        'emitted': len(ids),
        'emitted_by_class': _short_counts(Counter(e['class'] for e in emitted)),
        'leaf_nodes': sum(1 for n in walk_layout(tree) if n['type'] == 'leaf'),
        'duplicated_ids': {str(i): c for i, c in sorted(id_counts.items()) if c > 1},
        'dropped_ids': dropped,
        'placeholders': len(placeholders),
        'placeholder_classes': sorted(set(placeholders)),
        'xycut_fallback_nodes': fallback,
        'xycut_fallback_nodes_tol0': fallback_nodes_at_zero_tolerance(graph, tree),
        'gap_tolerance_nodes': gap_tol,
        'box_composites': sum(1 for n in walk_layout(tree) if n['type'] == 'Box'),
        'lazycolumn_composites': sum(1 for n in walk_layout(tree) if n['type'] == 'LazyColumn'),
        'parent_child_edges': relations.get('parent_child', 0),
        'spatial_edges': sum(v for k, v in relations.items() if k != 'parent_child'),
        'kotlin_lines': len(code.splitlines()),
    }


def v110_metrics(summary_entry: Dict, graph: Dict, code: str) -> Dict[str, Any]:
    """Computes the comparison metrics of one screen from the v1.1.0 reference outputs.

    Input:  summary_entry — the screen's entry of results/v1.1.0_examples/summary.json
            (Phase-0 record of what the v1.1.0 generator emitted); graph — its
            3_ui_graph.json; code — its 4_GeneratedScreen.kt
    Output: metrics dictionary with the same keys as v120_metrics where defined
    """
    relations = Counter(e['relation'] for e in graph['edges'])
    placeholders = re.findall(r'Unknown: (\w+)', code)
    return {
        'raw_detections': summary_entry['raw_detections'],
        'raw_by_class': _short_counts(summary_entry['raw_by_class']),
        'dedup_removed': 0,
        'dedup_removals': [],
        'kept_detections': summary_entry['raw_detections'],
        'emitted': summary_entry['emitted_leaves'],
        'emitted_by_class': _short_counts(summary_entry['emitted_by_class']),
        'leaf_nodes': summary_entry['emitted_leaves'],
        'duplicated_ids': {str(k): v for k, v in summary_entry['duplicated_leaf_ids'].items()},
        'dropped_ids': [list(x) for x in summary_entry['dropped_container_ids']],
        'placeholders': len(placeholders),
        'placeholder_classes': sorted(set(placeholders)),
        'xycut_fallback_nodes': summary_entry['xycut_fallback_nodes'],
        'xycut_fallback_nodes_tol0': summary_entry['xycut_fallback_nodes'],
        'gap_tolerance_nodes': 0,
        'box_composites': code.count('Box(') - len(placeholders),
        'lazycolumn_composites': code.count('LazyColumn('),
        'parent_child_edges': relations.get('parent_child', 0),
        'spatial_edges': sum(v for k, v in relations.items() if k != 'parent_child'),
        'kotlin_lines': len(code.splitlines()),
    }


def raw_identical(raw_v110: List[Dict], raw_v120: List[Dict]) -> bool:
    """Checks that two raw detection lists agree in every field v1.1.0 wrote.

    v1.2.0 adds 'img_w'/'img_h'; all other fields must be equal exactly.

    Input:  raw_v110, raw_v120 — raw detection lists
    Output: True if both have the same length and every v1.1.0 field is equal
    """
    if len(raw_v110) != len(raw_v120):
        return False
    return all(all(b.get(k) == v for k, v in a.items()) for a, b in zip(raw_v110, raw_v120))


# ── Comparison over all screens ─────────────────────────────


def _read_json(path: Path) -> Any:
    """Reads a JSON file.

    Input:  path — file path
    Output: parsed object
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def screen_stems(examples_dir: Path) -> List[str]:
    """Lists the example screens.

    Input:  examples_dir — folder with the example PNGs
    Output: sorted list of file stems
    """
    return sorted(p.stem for p in Path(examples_dir).glob('*.png'))


def build_comparison(stems: Sequence[str], v110_dir: Path, v120_dir: Path) -> Dict[str, Any]:
    """Builds the v1.1.0 vs v1.2.0 comparison from the output folders.

    Input:  stems — screens to compare; v110_dir — results/v1.1.0_examples;
            v120_dir — results/v1.2.0_examples
    Output: dictionary {'definitions', 'settings', 'screens': {stem: {'v1.1.0', 'v1.2.0',
            'raw_identical'}}, 'totals': {'v1.1.0', 'v1.2.0'}}
    """
    summary = _read_json(Path(v110_dir) / 'summary.json')
    screens: Dict[str, Any] = {}
    settings = None
    for stem in stems:
        old, new = Path(v110_dir) / stem, Path(v120_dir) / stem
        m_old = v110_metrics(summary[stem], _read_json(old / '3_ui_graph.json'),
                             (old / '4_GeneratedScreen.kt').read_text(encoding='utf-8'))
        m_new = v120_metrics(_read_json(new / '1_detections_raw.json'),
                             _read_json(new / '1_detections.json'),
                             _read_json(new / '1_dedup_removed.json'),
                             _read_json(new / '3_ui_graph.json'),
                             _read_json(new / '2_layout_tree.json'),
                             (new / '4_GeneratedScreen.kt').read_text(encoding='utf-8'))
        screens[stem] = {
            'v1.1.0': m_old, 'v1.2.0': m_new,
            'raw_identical': raw_identical(_read_json(old / '1_detections.json'),
                                           _read_json(new / '1_detections_raw.json')),
        }
        if settings is None:
            run_config = _read_json(new / '0_run_config.json')
            settings = {k: v for k, v in run_config.items()
                        if k not in ('image_path', 'output_dir', 'explain_id', 'visualize')}
            # The layout parameters actually used (None in the run config = generator default).
            meta = _read_json(new / '2_layout_tree.json')['meta']
            settings['layout_tree_meta'] = {k: meta[k] for k in ('layout', 'fallback', 'cut_tol_frac')}
    totals = {version: _totals([s[version] for s in screens.values()])
              for version in ('v1.1.0', 'v1.2.0')}
    return {'definitions': METRIC_DEFINITIONS, 'settings': settings,
            'screens': screens, 'totals': totals}


def _totals(per_screen: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sums the per-screen metrics of one version.

    Input:  per_screen — list of metrics dictionaries
    Output: totals: integer metrics summed, class counts summed per class,
            duplicated/dropped ids and placeholders counted
    """
    out: Dict[str, Any] = {}
    for key in ('raw_detections', 'dedup_removed', 'kept_detections', 'emitted', 'leaf_nodes',
                'placeholders', 'xycut_fallback_nodes', 'xycut_fallback_nodes_tol0',
                'gap_tolerance_nodes', 'box_composites', 'lazycolumn_composites',
                'parent_child_edges', 'spatial_edges', 'kotlin_lines'):
        values = [m[key] for m in per_screen]
        out[key] = None if any(v is None for v in values) else sum(values)
    for key in ('raw_by_class', 'emitted_by_class'):
        total: Counter = Counter()
        for m in per_screen:
            total.update(m[key])
        out[key] = _short_counts(total)
    out['duplicated_ids'] = sum(len(m['duplicated_ids']) for m in per_screen)
    out['dropped_ids'] = sum(len(m['dropped_ids']) for m in per_screen)
    out['placeholder_classes'] = sorted({c for m in per_screen for c in m['placeholder_classes']})
    return out


# ── Markdown rendering ──────────────────────────────────────

_CLASS_ABBREV = {
    'Button': 'Btn', 'CheckBox': 'Check', 'EditText': 'Edit', 'ImageView': 'Img',
    'ListView': 'ListView', 'ProgressBar': 'Progress', 'RadioButton': 'Radio',
    'RecyclerView': 'RecyclerView', 'SeekBar': 'Seek', 'Spinner': 'Spinner',
    'Switch': 'Switch', 'TextView': 'Text', 'WebView': 'WebView',
}


def short_name(stem: str) -> str:
    """Shortens a screen stem for tables: '<package> #<n>'.

    Input:  stem — e.g. 'com.shazam.android-screens_screenshot_3'
    Output: e.g. 'com.shazam.android #3'
    """
    m = re.match(r'^(.*)-screens_screenshot_(\d+)$', stem)
    return f"{m.group(1)} #{m.group(2)}" if m else stem


def _fmt_classes(counts: Dict[str, int]) -> str:
    """Formats class counts compactly, e.g. 'Btn 2, Img 7, Text 9'.

    Input:  counts — class -> count
    Output: formatted string ('–' if empty)
    """
    if not counts:
        return '–'
    return ', '.join(f"{_CLASS_ABBREV.get(k, k)} {v}" for k, v in sorted(counts.items()))


def _pair(old: Any, new: Any) -> str:
    """Formats a v1.1.0 -> v1.2.0 pair; equal values are written once.

    Input:  old, new — values of the two versions
    Output: e.g. '209 → 159' or '16'
    """
    return f"{old}" if old == new else f"{old} → {new}"


def _count(value: Any) -> int:
    """Returns the number of entries of a list/dict metric, or the value of an integer metric.

    Input:  value — integer, list or dictionary
    Output: integer
    """
    return len(value) if isinstance(value, (list, dict)) else int(value)


def render_markdown(comparison: Dict[str, Any]) -> str:
    """Renders the comparison as a Markdown document (results/examples_comparison.md).

    Input:  comparison — output of build_comparison
    Output: Markdown text
    """
    cols = [('raw_detections', 'raw'), ('dedup_removed', 'dedup removed'),
            ('emitted', 'emitted'), ('duplicated_ids', 'duplicated ids'),
            ('dropped_ids', 'dropped'), ('placeholders', 'placeholders'),
            ('xycut_fallback_nodes', 'XY-cut fallback nodes'),
            ('xycut_fallback_nodes_tol0', 'fallback nodes at cut_tol_frac=0'),
            ('gap_tolerance_nodes', 'nodes with tolerance cuts'),
            ('box_composites', 'Box'), ('lazycolumn_composites', 'LazyColumn'),
            ('spatial_edges', 'spatial edges'), ('kotlin_lines', 'Kotlin lines')]
    lines = [
        "# Bundled examples: v1.1.0 vs v1.2.0",
        "",
        "Generated by `python scripts/run_examples.py` (do not edit by hand). "
        "v1.1.0 values come from `results/v1.1.0_examples/` (outputs and Phase-0 "
        "`summary.json` of the v1.1.0 code); v1.2.0 values from `results/v1.2.0_examples/` "
        "(pipeline defaults, default weights `runs/oversample_5k_appsplit`). A cell `a → b` gives "
        "v1.1.0 → v1.2.0; a single value means both versions agree. v1.1.0 ran the weights of "
        "the v1.1.0 release (`runs/oversample_5k`), so the raw detections differ as well as the "
        "code. The ten screens are test screens of the v1.1.0 screenshot-level split; on the "
        "app-level split the default weights are trained on, 8 of them are training screens, "
        "battlesimulator is a validation screen and drivesafe a test screen. They illustrate the "
        "behaviour of the pipeline and were not used to choose any threshold.",
        "",
        "## Per screen",
        "",
        "| screen | " + " | ".join(label for _, label in cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for stem, s in comparison['screens'].items():
        old, new = s['v1.1.0'], s['v1.2.0']
        cells = [_pair(_count(old[k]), _count(new[k])) for k, _ in cols]
        lines.append(f"| {short_name(stem)} | " + " | ".join(cells) + " |")
    t_old, t_new = comparison['totals']['v1.1.0'], comparison['totals']['v1.2.0']
    cells = [_pair(_count(t_old[k]), _count(t_new[k])) for k, _ in cols]
    lines.append("| **total** | " + " | ".join(f"**{c}**" for c in cells) + " |")

    n_identical = sum(1 for s in comparison['screens'].values() if s['raw_identical'])
    lines += ["", f"Raw detections identical to v1.1.0 (every v1.1.0 field equal): "
              f"{n_identical} of {len(comparison['screens'])} screens.", ""]

    lines += ["## Emitted detections by class", "",
              "| screen | raw v1.2.0 | emitted v1.1.0 | emitted v1.2.0 | placeholders v1.1.0 → v1.2.0 |",
              "|---|---|---|---|---|"]
    for stem, s in comparison['screens'].items():
        old, new = s['v1.1.0'], s['v1.2.0']
        ph_old = ', '.join(old['placeholder_classes']) or '–'
        ph_new = ', '.join(new['placeholder_classes']) or '–'
        lines.append(f"| {short_name(stem)} | {_fmt_classes(new['raw_by_class'])} | "
                     f"{_fmt_classes(old['emitted_by_class'])} | {_fmt_classes(new['emitted_by_class'])} | "
                     f"{old['placeholders']} ({ph_old}) → {new['placeholders']} ({ph_new}) |")
    lines.append(f"| **total** | {_fmt_classes(t_new['raw_by_class'])} | "
                 f"{_fmt_classes(t_old['emitted_by_class'])} | {_fmt_classes(t_new['emitted_by_class'])} | "
                 f"{t_old['placeholders']} → {t_new['placeholders']} |")

    lines += ["", "## Detections dropped or duplicated by v1.1.0", "",
              "| screen | dropped (id, class, conf) | duplicated ids (id: count) |", "|---|---|---|"]
    for stem, s in comparison['screens'].items():
        old = s['v1.1.0']
        if old['dropped_ids'] or old['duplicated_ids']:
            dropped = '; '.join(f"{i} {c} {conf:.2f}" for i, c, conf in old['dropped_ids']) or '–'
            dup = ', '.join(f"{k}: {v}" for k, v in old['duplicated_ids'].items()) or '–'
            lines.append(f"| {short_name(stem)} | {dropped} | {dup} |")
    lines += ["", "In v1.2.0 no detection is dropped or duplicated on any screen "
              f"(dropped: {t_new['dropped_ids']}, duplicated: {t_new['duplicated_ids']}).", ""]

    lines += ["## De-duplication removals (v1.2.0)", "",
              "| screen | removed id | class | conf | rule | overlap | removed by |",
              "|---|---|---|---|---|---|---|"]
    for stem, s in comparison['screens'].items():
        for i, cls, conf, rule, overlap, by in s['v1.2.0']['dedup_removals']:
            lines.append(f"| {short_name(stem)} | {i} | {cls} | {conf:.3f} | {rule} | "
                         f"{overlap:.3f} | {by} |")
    lines += ["", "The de-duplication thresholds were selected on the validation split "
              "(`results/eval_appsplit/selected_thresholds.json`); see "
              "`src/ui2compose/infer/postprocess.py`.", ""]

    lines += ["## Settings of the v1.2.0 run", "", "```json",
              json.dumps(comparison['settings'], indent=2), "```", "",
              "## Metric definitions", ""]
    for key, text in comparison['definitions'].items():
        lines.append(f"- `{key}`: {text}")
    lines.append("")
    return "\n".join(lines)


def write_comparison(comparison: Dict[str, Any], stem: Path):
    """Writes the comparison as <stem>.json and <stem>.md.

    Input:  comparison — output of build_comparison; stem — path without suffix
    Output: none (side effect of writing two files)
    """
    with open(stem.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
        f.write('\n')
    stem.with_suffix('.md').write_text(render_markdown(comparison), encoding='utf-8')


# ── Running the pipeline ────────────────────────────────────


def run_screen(stem: str, detector: Any, model_path: str, out_root: Path) -> Dict[str, Any]:
    """Runs the pipeline (default settings) on one example screen.

    Input:  stem — example stem; detector — shared ComponentDetector; model_path —
            weights path recorded in 0_run_config.json; out_root — results/v1.2.0_examples
    Output: run_pipeline result dictionary (the pipeline's console output is suppressed)
    """
    # Imported here so that --compare-only works without the pipeline's dependencies.
    from ui2compose.infer.generate_compose_code import ComposeCodeGenerator
    from ui2compose.infer.pipeline import explain_detection, run_pipeline

    out_dir = out_root / stem
    old_code = clear_pipeline_outputs(out_dir)
    is_example = stem == EXAMPLE_STEM
    with contextlib.redirect_stdout(io.StringIO()):
        res = run_pipeline(str(EXAMPLES_DIR / f"{stem}.png"), model_path, DATASET_YAML,
                           str(out_dir), visualize=is_example,
                           explain_id=EXPLAIN_IDS[0] if is_example else None,
                           detector=detector)
    if is_example:
        for det_id in EXPLAIN_IDS[1:]:
            lines = explain_detection(det_id, res['raw_detections'], res['detections'],
                                      res['removed'], res['graph'], res['layout_tree'])
            (out_dir / f"2_explain_{det_id}.txt").write_text("\n".join(lines) + "\n",
                                                             encoding='utf-8')

    # Independent check: the ids emitted according to the layout tree are exactly
    # the ids named by the generator's trace comments in the code text.
    traced = ComposeCodeGenerator(trace_comments=True)
    image_size = res['graph']['image_size']
    traced_code = traced.generate(res['graph'], image_size=image_size)
    from_tree = sorted((e['id'], e['class']) for e in emitted_detections(res['layout_tree']))
    from_code = sorted((e['id'], e['class']) for e in emitted_from_trace_comments(traced_code))
    if from_tree != from_code:
        raise RuntimeError(f"{stem}: layout tree and trace comments disagree: {from_tree} vs {from_code}")
    # Without its trace comments the traced code must be the published code.
    stripped = [line for i, line in enumerate(traced_code.split("\n"))
                if i == 0 or not line.lstrip().startswith("// ")]
    if "\n".join(stripped) != res['code']:
        raise RuntimeError(f"{stem}: code with trace comments differs from the published code")
    res['kotlin_changed'] = old_code is not None and old_code != res['code']
    return res


def clear_pipeline_outputs(out_dir: Path) -> Optional[str]:
    """Deletes the files a pipeline run writes into a folder, keeping every other file.

    Input:  out_dir — output folder of one screen (may not exist)
    Output: the previous 4_GeneratedScreen.kt text (None if there was none)
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return None
    kt = out_dir / '4_GeneratedScreen.kt'
    old_code = kt.read_text(encoding='utf-8') if kt.is_file() else None
    for name in PIPELINE_FILES:
        (out_dir / name).unlink(missing_ok=True)
    for f in out_dir.glob(EXPLAIN_GLOB):
        f.unlink()
    return old_code


def refresh_golden(stems: Sequence[str], out_root: Path, golden_root: Path):
    """Copies the golden files of every screen from the results folder.

    Input:  stems — screens; out_root — results/v1.2.0_examples; golden_root — tests/golden
    Output: none (side effect of replacing tests/golden/<stem>/)
    """
    for stem in stems:
        dst = golden_root / stem
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for name in GOLDEN_FILES:
            shutil.copyfile(out_root / stem / name, dst / name)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point (see the module docstring).

    Input:  argv — argument list (None = sys.argv[1:])
    Output: exit status 0
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--model', default=DEFAULT_MODEL, help=f"weights (default {DEFAULT_MODEL})")
    parser.add_argument('--compare-only', action='store_true',
                        help="only recompute the comparison from results/v1.2.0_examples")
    parser.add_argument('--no-golden', action='store_true', help="do not refresh tests/golden")
    parser.add_argument('--no-docs-figure', action='store_true',
                        help=f"do not refresh {DOCS_FIGURE}")
    args = parser.parse_args(argv)

    # All paths are relative to the repository root.
    os.chdir(REPO_ROOT)
    stems = screen_stems(EXAMPLES_DIR)

    if not args.compare_only:
        from ui2compose.infer.detect_components import ComponentDetector
        with contextlib.redirect_stdout(io.StringIO()):
            detector = ComponentDetector(args.model, DATASET_YAML)
        V120_DIR.mkdir(parents=True, exist_ok=True)
        stale_renders = []
        for stem in stems:
            res = run_screen(stem, detector, args.model, V120_DIR)
            print(f"{short_name(stem):45s} raw={len(res['raw_detections']):3d} "
                  f"removed={len(res['removed']):2d} kotlin_lines={len(res['code'].splitlines())}")
            if res['kotlin_changed'] and (V120_DIR / stem / RENDER_FILE).exists():
                stale_renders.append(stem)
        if stale_renders:
            print(f"Kotlin changed on {len(stale_renders)} screen(s) with a {RENDER_FILE}: "
                  f"{', '.join(short_name(s) for s in stale_renders)}; re-run "
                  f"harness/compile_and_render.sh {V120_DIR} tests/fixtures/codegen "
                  f"(see harness/README.md)")
        if not args.no_golden:
            refresh_golden(stems, V120_DIR, GOLDEN_DIR)
            print(f"golden files refreshed in {GOLDEN_DIR}/")
        if not args.no_docs_figure:
            shutil.copyfile(V120_DIR / EXAMPLE_STEM / '1_detections_viz.png', DOCS_FIGURE)
            print(f"figure refreshed: {DOCS_FIGURE}")

    comparison = build_comparison(stems, V110_DIR, V120_DIR)
    write_comparison(comparison, COMPARISON_STEM)
    print(f"comparison written: {COMPARISON_STEM}.json / .md")
    return 0


if __name__ == '__main__':
    sys.exit(main())
