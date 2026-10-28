# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Golden tests: replaying the saved raw detections must reproduce the published layout and Kotlin exactly.

tests/golden/<stem>/ holds, for each of the ten bundled examples, the raw
detections of the default weights (runs/oversample_5k_appsplit; 1_detections_raw.json) and the layout
tree (2_layout_tree.json) and Kotlin (4_GeneratedScreen.kt) that the pipeline
produced from them with its default settings; scripts/run_examples.py writes
them. The replay calls ui2compose.infer.pipeline.run_pipeline with a detector
that returns the saved raw detections, so per-class filtering,
de-duplication, graph construction, layout and code generation run exactly
as in a real run, without torch or ultralytics.

The file also checks that the published results (results/v1.2.0_examples and
results/examples_comparison.*) agree with the golden files and with the
metric code of scripts/run_examples.py, and, when ultralytics and the weights
are available, that the detector still produces the golden raw detections.
"""

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import pytest

from _reference_data import require_reference_data
from ui2compose.infer.generate_compose_code import ComposeCodeGenerator
from ui2compose.infer.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / 'tests' / 'golden'
EXAMPLES_DIR = REPO_ROOT / 'examples'
V110_DIR = REPO_ROOT / 'results' / 'v1.1.0_examples'
V120_DIR = REPO_ROOT / 'results' / 'v1.2.0_examples'
COMPARISON_JSON = REPO_ROOT / 'results' / 'examples_comparison.json'
COMPARISON_MD = REPO_ROOT / 'results' / 'examples_comparison.md'
WEIGHTS = REPO_ROOT / 'runs' / 'oversample_5k_appsplit' / 'weights' / 'best.pt'  # the pipeline default

GOLDEN_STEMS = sorted(p.name for p in GOLDEN_DIR.iterdir() if p.is_dir())


def _load_run_examples() -> ModuleType:
    """Imports scripts/run_examples.py as a module (it is a script, not part of the package).

    Input:  none
    Output: the loaded module
    """
    spec = importlib.util.spec_from_file_location('run_examples', REPO_ROOT / 'scripts' / 'run_examples.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN_EXAMPLES = _load_run_examples()


def _read_json(path: Path) -> Any:
    """Reads a JSON file.

    Input:  path — file path
    Output: parsed object
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _roundtrip(obj: Any) -> Any:
    """Returns obj as it reads back after JSON serialisation (tuples become lists).

    Input:  obj — JSON-serialisable object
    Output: the object after json.dumps / json.loads
    """
    return json.loads(json.dumps(obj))


class ReplayDetector:
    """Stands in for ComponentDetector: returns saved raw detections and records the call."""

    def __init__(self, detections: List[Dict]):
        """Stores the detections to return.

        Input:  detections — raw detection list (1_detections_raw.json)
        Output: none
        """
        self.detections = detections
        self.calls: List[Dict] = []

    def detect(self, image_path: str, **kwargs) -> List[Dict]:
        """Returns a deep copy of the saved detections.

        Input:  image_path — ignored; kwargs — detector settings (recorded)
        Output: list of detection dictionaries
        """
        self.calls.append(dict(kwargs))
        return copy.deepcopy(self.detections)

    def visualize(self, image_path: str, detections: List[Dict], output_path: str = None,
                  removed: List[Dict] = None):
        """Does nothing (the replay runs with visualize=False).

        Input:  as ComponentDetector.visualize
        Output: None
        """
        return None


def replay(stem: str, out_dir: Path) -> Dict[str, Any]:
    """Runs the pipeline with default settings on the golden raw detections of one screen.

    Input:  stem — example stem; out_dir — output directory
    Output: run_pipeline result dictionary
    """
    raw = _read_json(GOLDEN_DIR / stem / '1_detections_raw.json')
    detector = ReplayDetector(raw)
    res = run_pipeline(str(EXAMPLES_DIR / f"{stem}.png"), 'unused-weights.pt',
                       str(REPO_ROOT / 'configs' / 'dataset.yaml'), str(out_dir),
                       visualize=False, detector=detector)
    assert len(detector.calls) == 1
    return res


# ── Golden replay ───────────────────────────────────────────


def test_golden_set_covers_every_example():
    """There is one golden folder per bundled example screen, each with the three golden files."""
    assert GOLDEN_STEMS == sorted(p.stem for p in EXAMPLES_DIR.glob('*.png'))
    assert len(GOLDEN_STEMS) == 10
    for stem in GOLDEN_STEMS:
        assert sorted(p.name for p in (GOLDEN_DIR / stem).iterdir()) == sorted(RUN_EXAMPLES.GOLDEN_FILES)


@pytest.mark.parametrize('stem', GOLDEN_STEMS)
def test_replay_reproduces_golden_layout_and_kotlin(stem, tmp_path):
    """raw detections -> dedup -> graph -> codegen gives exactly the golden layout tree and Kotlin."""
    res = replay(stem, tmp_path)
    golden_tree = _read_json(GOLDEN_DIR / stem / '2_layout_tree.json')
    golden_code = (GOLDEN_DIR / stem / '4_GeneratedScreen.kt').read_text(encoding='utf-8')

    assert _roundtrip(res['layout_tree']) == golden_tree
    assert res['code'] == golden_code
    # The files the pipeline writes hold the same content.
    assert _read_json(tmp_path / '2_layout_tree.json') == golden_tree
    assert (tmp_path / '4_GeneratedScreen.kt').read_text(encoding='utf-8') == golden_code


@pytest.mark.parametrize('stem', GOLDEN_STEMS)
def test_golden_outputs_emit_every_kept_detection_once(stem, tmp_path):
    """In every golden screen each kept detection is emitted exactly once and no placeholder is written."""
    res = replay(stem, tmp_path)
    emitted = RUN_EXAMPLES.emitted_detections(res['layout_tree'])
    assert sorted(e['id'] for e in emitted) == sorted(d['id'] for d in res['detections'])
    assert 'Unknown: ' not in res['code']
    # The ids named by the generator's trace comments are the same detections.
    traced = ComposeCodeGenerator(trace_comments=True).generate(res['graph'])
    from_code = sorted((e['id'], e['class']) for e in RUN_EXAMPLES.emitted_from_trace_comments(traced))
    assert from_code == sorted((e['id'], e['class']) for e in emitted)


def test_worked_example_golden_matches_the_paper_counts(tmp_path):
    """Worked example (drivesafe, Figs. 2-3): 9 raw detections, none removed, each emitted once;
    the drawer detected as ImageView #2 becomes the container of the only Box."""
    stem = RUN_EXAMPLES.EXAMPLE_STEM
    res = replay(stem, tmp_path)
    assert len(res['raw_detections']) == 9 and res['removed'] == []
    metrics = RUN_EXAMPLES.v120_metrics(res['raw_detections'], res['detections'], res['removed'],
                                        res['graph'], res['layout_tree'], res['code'])
    assert metrics['emitted_by_class'] == {'Button': 1, 'CheckBox': 3, 'ImageView': 1, 'TextView': 4}
    assert metrics['duplicated_ids'] == {} and metrics['dropped_ids'] == []
    assert metrics['box_composites'] == 1 and metrics['lazycolumn_composites'] == 0
    box = res['layout_tree']['children'][1]
    assert box['type'] == 'Box' and box['container']['id'] == 2 and box['container']['class'] == 'ImageView'


# ── Published results ───────────────────────────────────────


@pytest.mark.parametrize('stem', GOLDEN_STEMS)
def test_published_results_equal_golden_files(stem):
    """results/v1.2.0_examples/<stem> holds byte-identical copies of the golden files."""
    require_reference_data(V120_DIR)
    for name in RUN_EXAMPLES.GOLDEN_FILES:
        assert (V120_DIR / stem / name).read_bytes() == (GOLDEN_DIR / stem / name).read_bytes(), name


def test_published_comparison_is_reproducible():
    """results/examples_comparison.{json,md} equal what scripts/run_examples.py computes from the outputs."""
    require_reference_data(V110_DIR, V120_DIR, COMPARISON_JSON, COMPARISON_MD)
    comparison = RUN_EXAMPLES.build_comparison(GOLDEN_STEMS, V110_DIR, V120_DIR)
    assert _roundtrip(comparison) == _read_json(COMPARISON_JSON)
    assert RUN_EXAMPLES.render_markdown(comparison) == COMPARISON_MD.read_text(encoding='utf-8')
    totals = comparison['totals']['v1.2.0']
    assert totals['dropped_ids'] == 0 and totals['duplicated_ids'] == 0
    assert totals['emitted'] == totals['kept_detections']
    # v1.1.0 ran the v1.1.0 weights, v1.2.0 the app-split weights: the raw detections differ
    # (identity with v1.1.0 under the v1.1.0 weights is checked in tests/test_dedup.py).


# ── Metric helpers of scripts/run_examples.py ───────────────


def _leaf(det_id: int, cls: str) -> Dict:
    """Builds a minimal layout leaf.

    Input:  det_id — detection id; cls — class name
    Output: layout node of type 'leaf'
    """
    return {'type': 'leaf', 'rule': 'leaf', 'children': [],
            'node': {'id': det_id, 'class': cls, 'confidence': 0.5, 'bbox': [0, 0, 1, 1]}}


def test_v120_metrics_count_duplicates_drops_fallbacks_and_composites():
    """The metric code counts duplicated and dropped ids, fallback / tolerance nodes and composites."""
    tree = {'type': 'Column', 'rule': 'gap_tolerance', 'children': [
        _leaf(1, 'TextView'), _leaf(2, 'TextView'),
        {'type': 'Column', 'rule': 'fallback_y_sorted', 'children': [_leaf(2, 'TextView')]},
        {'type': 'Box', 'rule': 'containment', 'container': {'id': 5, 'class': 'ImageView'},
         'children': [_leaf(5, 'ImageView'), _leaf(6, 'TextView')]},
        {'type': 'LazyColumn', 'rule': 'list', 'split_rule': 'least_overlap',
         'container': {'id': 3, 'class': 'ListView', 'confidence': 0.9}, 'children': [_leaf(7, 'Button')]},
    ]}
    kept = [{'id': i, 'class_name': c, 'confidence': 0.5}
            for i, c in [(1, 'TextView'), (2, 'TextView'), (3, 'ListView'), (4, 'Button'),
                         (5, 'ImageView'), (6, 'TextView'), (7, 'Button')]]
    removed = [{'id': 8, 'class_name': 'TextView', 'confidence': 0.4, 'dedup_rule': 'same_class_iou',
                'overlap': 0.7, 'removed_by': 1}]
    graph = {'edges': [{'relation': 'parent_child'}, {'relation': 'above'}, {'relation': 'left_of'}]}
    code = 'Text("Unknown: Foo")\nline\nText("Unknown: Foo")'
    m = RUN_EXAMPLES.v120_metrics(kept + removed, kept, removed, graph, tree, code)

    assert m['raw_detections'] == 8 and m['dedup_removed'] == 1 and m['kept_detections'] == 7
    assert m['emitted'] == 7                                  # 1, 2, 2, 5, 6, 3, 7
    assert m['emitted_by_class'] == {'Button': 1, 'ImageView': 1, 'ListView': 1, 'TextView': 4}
    assert m['duplicated_ids'] == {'2': 2}
    assert m['dropped_ids'] == [[4, 'Button', 0.5]]
    assert m['xycut_fallback_nodes'] == 2                     # fallback_y_sorted + LazyColumn split
    assert m['gap_tolerance_nodes'] == 1
    assert m['box_composites'] == 1 and m['lazycolumn_composites'] == 1
    assert m['parent_child_edges'] == 1 and m['spatial_edges'] == 2
    assert m['placeholders'] == 2 and m['placeholder_classes'] == ['Foo']
    assert m['kotlin_lines'] == 3
    assert m['xycut_fallback_nodes_tol0'] is None          # no hierarchy / meta to recompute from


def test_fallback_count_at_zero_tolerance_separates_the_tolerance_effect():
    """Rows overlapping by 2 px are cut with a tolerance (0 fallbacks) but need the fallback at cut_tol_frac=0."""
    from ui2compose.infer.build_ui_graph import UIGraphBuilder
    from ui2compose.infer.generate_compose_code import build_layout_tree
    boxes = [{'id': 0, 'class_name': 'TextView', 'confidence': 0.9, 'bbox': [0, 0, 600, 100]},
             {'id': 1, 'class_name': 'TextView', 'confidence': 0.9, 'bbox': [100, 98, 700, 200]}]
    graph = UIGraphBuilder().build_graph(boxes, image_size=(1200, 1920))
    tree = build_layout_tree(graph, image_size=(1200, 1920), cut_tol_frac=0.0025)
    assert tree['rule'] == 'gap_tolerance' and RUN_EXAMPLES.fallback_nodes(tree) == 0
    assert RUN_EXAMPLES.fallback_nodes_at_zero_tolerance(graph, tree) == 1


def test_run_screen_replaces_only_pipeline_files(tmp_path, monkeypatch):
    """A re-run keeps the harness's render.png, removes stale explain files and reports changed Kotlin."""
    monkeypatch.chdir(REPO_ROOT)                             # run_screen uses repository-relative paths
    stem = RUN_EXAMPLES.EXAMPLE_STEM
    out_dir = tmp_path / stem
    out_dir.mkdir()
    (out_dir / RUN_EXAMPLES.RENDER_FILE).write_bytes(b'render')
    (out_dir / '2_explain_99.txt').write_text('stale')
    (out_dir / '4_GeneratedScreen.kt').write_text('// old code')
    detector = ReplayDetector(_read_json(GOLDEN_DIR / stem / '1_detections_raw.json'))
    res = RUN_EXAMPLES.run_screen(stem, detector, 'unused-weights.pt', tmp_path)
    names = sorted(p.name for p in out_dir.iterdir())
    assert names == sorted([RUN_EXAMPLES.RENDER_FILE] + [f'2_explain_{i}.txt' for i in RUN_EXAMPLES.EXPLAIN_IDS] + [n for n in RUN_EXAMPLES.PIPELINE_FILES
                                                   if n != '1_detections_viz.png'])
    assert (out_dir / RUN_EXAMPLES.RENDER_FILE).read_bytes() == b'render'
    assert res['kotlin_changed'] is True
    res = RUN_EXAMPLES.run_screen(stem, detector, 'unused-weights.pt', tmp_path)
    assert res['kotlin_changed'] is False                    # same code again
    assert (out_dir / '4_GeneratedScreen.kt').read_text(encoding='utf-8') == \
        (GOLDEN_DIR / stem / '4_GeneratedScreen.kt').read_text(encoding='utf-8')


def test_v110_metrics_and_raw_identity():
    """v1.1.0 metrics come from the Phase-0 summary and the files; raw identity ignores the new img_w/img_h."""
    entry = {'raw_detections': 3, 'raw_by_class': {'TextView': 2, 'ListView': 1},
             'emitted_leaves': 3, 'emitted_by_class': {'TextView': 3},
             'duplicated_leaf_ids': {'1': 2}, 'dropped_container_ids': [[0, 'ListView', 0.9]],
             'xycut_fallback_nodes': 2}
    graph = {'edges': [{'relation': 'parent_child'}, {'relation': 'below'}]}
    code = 'Box(\n) {\n    Text("Unknown: RecyclerView")\n}\nBox(\n)'
    m = RUN_EXAMPLES.v110_metrics(entry, graph, code)
    assert m['dedup_removed'] == 0 and m['emitted'] == 3
    assert m['duplicated_ids'] == {'1': 2} and m['dropped_ids'] == [[0, 'ListView', 0.9]]
    assert m['placeholders'] == 1 and m['box_composites'] == 1 and m['lazycolumn_composites'] == 0
    assert m['spatial_edges'] == 1 and m['kotlin_lines'] == 6

    old = [{'id': 0, 'bbox': [1.0, 2.0, 3.0, 4.0]}]
    assert RUN_EXAMPLES.raw_identical(old, [{'id': 0, 'bbox': [1.0, 2.0, 3.0, 4.0], 'img_w': 9}])
    assert not RUN_EXAMPLES.raw_identical(old, [{'id': 0, 'bbox': [1.0, 2.0, 3.0, 4.5]}])
    assert not RUN_EXAMPLES.raw_identical(old, [])


# ── Real detector (optional) ────────────────────────────────


def test_default_weights_reproduce_golden_raw_detections():
    """The default weights, at the pipeline defaults, still produce the golden raw detections."""
    pytest.importorskip('ultralytics')
    require_reference_data(WEIGHTS)
    from ui2compose.infer.detect_components import ComponentDetector

    detector = ComponentDetector(str(WEIGHTS), str(REPO_ROOT / 'configs' / 'dataset.yaml'))
    for stem in GOLDEN_STEMS:
        golden = _read_json(GOLDEN_DIR / stem / '1_detections_raw.json')
        got = detector.detect(str(EXAMPLES_DIR / f"{stem}.png"))
        assert [(d['id'], d['class_name']) for d in got] == \
            [(d['id'], d['class_name']) for d in golden], stem
        for a, b in zip(got, golden):
            assert a['confidence'] == pytest.approx(b['confidence'], abs=1e-4), stem
            assert a['bbox'] == pytest.approx(b['bbox'], abs=1e-2), stem
            assert (a['img_w'], a['img_h']) == (b['img_w'], b['img_h'])
