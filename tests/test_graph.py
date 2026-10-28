# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Tests for the UI graph builder (ui2compose.infer.build_ui_graph).

Sections:
  1. Containment: ratio rule, single (smallest) parent, identical boxes, the
     forest property (every id exactly once) on fixed and random layouts.
  2. Resolution independence: the same layout at 0.5x / 1x / 2x gives the
     same edges and hierarchy; a fixed pixel tolerance (v1.1.0) does not.
  3. Output schema: synthetic flag, confidence, image_size, params, input
     validation, plain box lists, deprecated keyword arguments.
  4. Spatial-edge rule and the containment explanation.
  5. Edge ablation: the generated code does not depend on the edges or on the
     synthetic layout groups (imports the code generator).

The Shazam fixture below holds the 19 raw v1.1.0 detections of
examples/com.shazam.android-screens_screenshot_3.png (1200 x 1920 px), rounded
to 0.1 px, so the tests need neither the model nor the results/ folder.
"""

import copy
import inspect
import json
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

from _reference_data import require_reference_data
from ui2compose.infer.build_ui_graph import (DEFAULT_ALIGN_TOL_FRAC, DEFAULT_CONTAINMENT_RATIO,
                                             DEFAULT_IMAGE_SIZE, RelationType, UIGraphBuilder,
                                             containment_roots, coverage, explain_containment,
                                             resolve_image_size)

REPO_ROOT = Path(__file__).resolve().parents[1]

# (id, class, confidence, bbox) — raw v1.1.0 detections of the Shazam example
SHAZAM: List[Tuple[int, str, float, List[float]]] = [
    (0, 'ImageView', 0.54, [427.3, 61.6, 483.0, 117.3]),
    (1, 'ImageView', 0.62, [134.5, 62.0, 188.2, 119.4]),
    (2, 'ImageView', 0.55, [1011.7, 64.3, 1062.5, 114.8]),
    (3, 'ImageView', 0.68, [722.3, 65.2, 769.7, 114.5]),
    (4, 'TextView', 0.76, [685.4, 117.6, 805.8, 156.9]),
    (5, 'TextView', 0.71, [375.4, 118.5, 531.6, 157.9]),
    (6, 'TextView', 0.70, [993.2, 119.5, 1083.3, 157.9]),
    (7, 'TextView', 0.62, [121.4, 120.0, 202.6, 158.3]),
    (8, 'ImageView', 0.38, [70.4, 482.4, 953.7, 1086.6]),
    (9, 'ImageView', 0.31, [322.3, 537.9, 867.8, 1107.4]),
    (10, 'ImageView', 0.51, [465.3, 561.8, 739.9, 845.2]),
    (11, 'TextView', 0.48, [700.1, 861.6, 774.0, 936.3]),
    (12, 'TextView', 0.76, [313.1, 1164.5, 886.5, 1241.0]),
    (13, 'TextView', 0.55, [389.9, 1254.5, 806.8, 1306.1]),
    (14, 'TextView', 0.41, [32.6, 1254.8, 1170.7, 1307.6]),
    (15, 'Button', 0.77, [421.7, 1363.0, 778.9, 1458.9]),
    (16, 'TextView', 0.41, [19.7, 1450.4, 1182.1, 1757.6]),
    (17, 'RecyclerView', 0.74, [12.3, 1470.3, 1189.0, 1920.0]),
    (18, 'Button', 0.46, [5.7, 1731.5, 204.7, 1826.9]),
]
SHAZAM_SIZE = (1200, 1920)


def det(det_id: int, class_name: str, confidence: float, bbox: List[float], **extra) -> Dict:
    """Builds a detection dictionary in the detector's output format.

    Input:  det_id — id; class_name — class; confidence — score; bbox — [x1, y1, x2, y2];
            extra — additional fields (e.g. img_w, img_h)
    Output: detection dictionary with id, class_name, confidence, bbox, center
    """
    d = {'id': det_id, 'class_name': class_name, 'confidence': confidence,
         'bbox': [float(v) for v in bbox],
         'center': [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]}
    d.update(extra)
    return d


def shazam(scale: float = 1.0) -> List[Dict]:
    """Returns the Shazam fixture as detection dictionaries, optionally scaled.

    Input:  scale — factor applied to all coordinates
    Output: list of detection dictionaries
    """
    return [det(i, c, p, [v * scale for v in b]) for i, c, p, b in SHAZAM]


def all_ids(graph: Dict) -> List[int]:
    """Lists every node id in the hierarchy, depth-first (synthetic ids included).

    Input:  graph — output of build_graph
    Output: list of ids in traversal order
    """
    out: List[int] = []

    def walk(node: Dict):
        """Appends the node id and recurses into the children.

        Input:  node — hierarchy node
        Output: none (side effect of appending to out)
        """
        out.append(node['id'])
        for child in node['children']:
            walk(child)

    for root in graph['hierarchy']['roots']:
        walk(root)
    return out


def detection_ids(graph: Dict) -> List[int]:
    """Lists the ids of the detected (non-synthetic) nodes in the hierarchy.

    Input:  graph — output of build_graph
    Output: list of ids (with repetitions, if any)
    """
    return [i for i in all_ids(graph) if i >= 0]


def parent_map(graph: Dict) -> Dict[int, Optional[int]]:
    """Extracts {child id: containment parent id or None} from the hierarchy, skipping synthetic nodes.

    Input:  graph — output of build_graph
    Output: mapping of every detected node to its detected parent (None at the top level)
    """
    out: Dict[int, Optional[int]] = {}

    def walk(node: Dict, parent: Optional[int]):
        """Records the node's detected parent and recurses.

        Input:  node — hierarchy node; parent — id of the nearest detected ancestor
        Output: none (side effect of writing to out)
        """
        if node['synthetic']:
            for child in node['children']:
                walk(child, parent)
            return
        out[node['id']] = parent
        for child in node['children']:
            walk(child, node['id'])

    for root in graph['hierarchy']['roots']:
        walk(root, None)
    return out


def structure(node: Dict):
    """Reduces a hierarchy node to its class/id structure (no coordinates).

    Input:  node — hierarchy node
    Output: nested tuple (id, class, synthetic, children structures)
    """
    return (node['id'], node['class'], node['synthetic'],
            tuple(structure(c) for c in node['children']))


def edge_set(graph: Dict) -> List[Tuple[int, int, str]]:
    """Returns the graph edges as sorted (source, target, relation) triples.

    Input:  graph — output of build_graph
    Output: sorted list of triples
    """
    return sorted((e['source'], e['target'], e['relation']) for e in graph['edges'])


# ── 1. Containment ─────────────────────────────────────────


def test_single_parent_is_smallest_of_two_overlapping_parents():
    """A child inside two partly overlapping boxes appears once, under the smaller box."""
    dets = [det(0, 'ImageView', 0.9, [0, 0, 100, 100]),      # area 10000
            det(1, 'ImageView', 0.5, [50, 0, 140, 100]),     # area 9000, overlaps 0
            det(2, 'TextView', 0.8, [60, 10, 90, 90])]       # inside both
    graph = UIGraphBuilder().build_graph(dets, image_size=(200, 200))
    assert sorted(detection_ids(graph)) == [0, 1, 2]
    assert parent_map(graph) == {0: None, 1: None, 2: 1}
    pc = [e for e in graph['edges'] if e['relation'] == 'parent_child']
    assert [(e['source'], e['target']) for e in pc] == [(1, 2)]
    assert pc[0]['coverage'] == 1.0


def test_shazam_children_of_two_overlapping_parents_appear_once():
    """R2.4: ids 10 and 11 (inside both ImageView 8 and 9) appear exactly once."""
    strict = UIGraphBuilder(containment_ratio=1.0).build_graph(shazam(), image_size=SHAZAM_SIZE)
    assert sorted(detection_ids(strict)) == list(range(19))
    # 9 protrudes below 8, so with the strict test both are top-level and the
    # children go to the smaller one (9) only
    assert parent_map(strict)[10] == 9 and parent_map(strict)[11] == 9
    assert parent_map(strict)[9] is None

    at95 = UIGraphBuilder(containment_ratio=0.95).build_graph(shazam(), image_size=SHAZAM_SIZE)
    assert sorted(detection_ids(at95)) == list(range(19))
    pm = parent_map(at95)
    # 96.4% of 9 lies in 8 -> contained at ratio 0.95 -> chain 8 > 9 > {10, 11}
    assert pm[9] == 8 and pm[10] == 9 and pm[11] == 9
    assert pm[18] == 17          # Button inside the RecyclerView (96.7%)
    assert pm[16] is None        # TextView 16 has only 93.5% inside 17

    # Default ratio (0.9, selected on the validation split): the same chain, and
    # TextView 16 (93.5% inside 17) is now contained in the RecyclerView as well.
    assert DEFAULT_CONTAINMENT_RATIO == 0.9
    default = UIGraphBuilder().build_graph(shazam(), image_size=SHAZAM_SIZE)
    assert sorted(detection_ids(default)) == list(range(19))
    pm = parent_map(default)
    assert pm[9] == 8 and pm[10] == 9 and pm[11] == 9
    assert pm[18] == 17 and pm[16] == 17


def test_identical_boxes_no_cycle_no_loss():
    """Identical boxes: higher confidence is the parent; equal confidence -> lower id; nobody is lost."""
    box = [10, 10, 110, 60]
    graph = UIGraphBuilder().build_graph(
        [det(0, 'TextView', 0.4, box), det(1, 'TextView', 0.9, box)], image_size=(200, 200))
    assert sorted(detection_ids(graph)) == [0, 1]
    assert parent_map(graph) == {1: None, 0: 1}

    graph = UIGraphBuilder().build_graph(
        [det(0, 'TextView', 0.5, box), det(1, 'Button', 0.5, box), det(2, 'TextView', 0.5, box)],
        image_size=(200, 200))
    assert sorted(detection_ids(graph)) == [0, 1, 2]
    # 0 ranks highest (lower id); 2 is contained in 0 and 1, which tie on area and
    # confidence, so the lower id (0) is its parent
    assert parent_map(graph) == {0: None, 1: 0, 2: 0}


def test_equal_area_mutual_overlap_has_one_parent():
    """Two equal-area boxes covering >= ratio of each other: exactly one is the parent."""
    dets = [det(0, 'TextView', 0.3, [0, 0, 100, 50]), det(1, 'TextView', 0.6, [2, 0, 102, 50])]
    graph = UIGraphBuilder().build_graph(dets, image_size=(200, 200))
    assert parent_map(graph) == {1: None, 0: 1}


def test_ratio_containment_slightly_protruding_child():
    """A child protruding by 2% of its area is contained at 0.95 and at the default 0.9, but not at 1.0 (strict v1.1.0 test)."""
    parent = det(0, 'ImageView', 0.9, [0, 0, 200, 200])
    child = det(1, 'TextView', 0.8, [100, 10, 202, 30])      # 2 of 102 px outside -> 98.0%
    assert coverage(child['bbox'], parent['bbox']) == pytest.approx(100 / 102)
    assert parent_map(UIGraphBuilder().build_graph([parent, child], (400, 400)))[1] == 0
    assert parent_map(UIGraphBuilder(containment_ratio=0.95).build_graph(
        [parent, child], (400, 400)))[1] == 0
    assert parent_map(UIGraphBuilder(containment_ratio=1.0).build_graph(
        [parent, child], (400, 400)))[1] is None
    # coverage just below the ratio -> not contained
    loose = det(2, 'TextView', 0.8, [100, 10, 210, 30])       # 100/110 = 90.9%
    assert parent_map(UIGraphBuilder(containment_ratio=0.95).build_graph(
        [parent, loose], (400, 400)))[2] is None
    assert parent_map(UIGraphBuilder().build_graph([parent, loose], (400, 400)))[2] == 0
    looser = det(3, 'TextView', 0.8, [100, 10, 212, 30])      # 100/112 = 89.3% < 0.9
    assert parent_map(UIGraphBuilder().build_graph([parent, looser], (400, 400)))[3] is None


def test_strict_ratio_accepts_shared_borders():
    """containment_ratio=1.0 is the closed-box test of v1.1.0: a child touching the border is inside."""
    dets = [det(0, 'ImageView', 0.9, [0, 0, 100, 100]), det(1, 'TextView', 0.9, [0, 0, 100, 20])]
    graph = UIGraphBuilder(containment_ratio=1.0).build_graph(dets, image_size=(100, 100))
    assert parent_map(graph) == {0: None, 1: 0}


def test_zero_area_box_is_contained_if_inside():
    """A degenerate (zero-area) box counts as contained only when it lies inside the container."""
    assert coverage([5, 5, 5, 9], [0, 0, 10, 10]) == 1.0
    assert coverage([5, 5, 5, 19], [0, 0, 10, 10]) == 0.0
    dets = [det(0, 'ImageView', 0.9, [0, 0, 10, 10]), det(1, 'TextView', 0.9, [5, 5, 5, 9])]
    assert parent_map(UIGraphBuilder().build_graph(dets, (10, 10)))[1] == 0


def _v110_shape(node: Dict):
    """Reduces a hierarchy node to its synthetic-group structure (detected nodes as bare ids).

    Input:  node — hierarchy node of a v1.1.0 or v1.2.0 graph (synthetic = negative id)
    Output: nested tuple of synthetic classes with the ids of the grouped top-level nodes
    """
    if node['id'] < 0:
        return (node['class'], tuple(_v110_shape(c) for c in node['children']))
    return node['id']


def test_strict_ratio_reproduces_v110_containment_except_duplicates():
    """With containment_ratio=1.0, every parent is a v1.1.0 parent; only multi-parent children change.

    Also the synthetic layout groups around the top-level nodes are the same as in v1.1.0.
    Uses the v1.1.0 reference outputs in results/v1.1.0_examples (skipped if absent).
    """
    root = REPO_ROOT / "results/v1.1.0_examples"
    require_reference_data(root)
    cases = sorted(p for p in root.iterdir() if p.is_dir())
    multi_parent_children = 0
    for case in cases:
        dets = json.loads((case / "1_detections.json").read_text())
        old = json.loads((case / "3_ui_graph.json").read_text())
        old_parents: Dict[int, set] = {}
        for e in old['edges']:
            if e['relation'] == 'parent_child':
                old_parents.setdefault(e['target'], set()).add(e['source'])

        graph = UIGraphBuilder(containment_ratio=1.0).build_graph(dets)
        assert sorted(detection_ids(graph)) == sorted(d['id'] for d in dets), case.name
        for child, parent in parent_map(graph).items():
            before = old_parents.get(child, set())
            if parent is None:
                assert not before, (case.name, child)
            else:
                assert parent in before, (case.name, child, parent, before)
            multi_parent_children += len(before) > 1
        assert [_v110_shape(r) for r in graph['hierarchy']['roots']] == \
            [_v110_shape(r) for r in old['hierarchy']['roots']], case.name
    assert multi_parent_children >= 3     # Shazam ids 10 and 11, drivesafe id 5


@pytest.mark.parametrize("seed", range(25))
def test_random_layouts_form_a_forest(seed):
    """On random overlapping boxes every id appears exactly once and parents always contain children."""
    rng = random.Random(seed)
    dets = []
    for i in range(rng.randint(1, 40)):
        x1, y1 = rng.uniform(0, 1100), rng.uniform(0, 1800)
        w, h = rng.choice([rng.uniform(5, 100), rng.uniform(100, 1200)]), rng.uniform(5, 600)
        dets.append(det(i, rng.choice(['TextView', 'ImageView', 'Button']),
                        rng.choice([0.3, 0.5, 0.9]), [x1, y1, x1 + w, y1 + h]))
    # add exact duplicates to exercise the tie-breaks
    for j in range(rng.randint(0, 3)):
        src = rng.choice(dets)
        dets.append(det(len(dets), src['class_name'], src['confidence'], src['bbox']))

    ratio = rng.choice([0.8, 0.95, 1.0])
    graph = UIGraphBuilder(containment_ratio=ratio).build_graph(dets, image_size=SHAZAM_SIZE)
    assert sorted(detection_ids(graph)) == list(range(len(dets)))
    synthetic = [i for i in all_ids(graph) if i < 0]
    assert len(synthetic) == len(set(synthetic))
    by_id = {d['id']: d for d in dets}
    for child, parent in parent_map(graph).items():
        if parent is not None:
            assert coverage(by_id[child]['bbox'], by_id[parent]['bbox']) >= ratio
    # parent_child edges are exactly the hierarchy links
    pc = {(e['source'], e['target']) for e in graph['edges'] if e['relation'] == 'parent_child'}
    assert pc == {(p, c) for c, p in parent_map(graph).items() if p is not None}


# ── 2. Resolution independence ─────────────────────────────


def grid_layout(scale: float) -> List[Dict]:
    """Builds a 3 x 2 grid of boxes whose centres jitter by 15 px at 1200 px width.

    Input:  scale — factor applied to all coordinates (image width 1200 * scale)
    Output: list of detection dictionaries
    """
    dets = []
    for r in range(2):
        for c in range(3):
            jitter = 15 if (r + c) % 2 else 0
            x1 = 100 + 350 * c + jitter
            y1 = 300 + 250 * r + jitter
            dets.append(det(len(dets), 'TextView', 0.8,
                            [v * scale for v in (x1, y1, x1 + 200, y1 + 100)]))
    return dets


@pytest.mark.parametrize("layout", ["shazam", "grid"])
def test_same_layout_at_other_resolutions_gives_same_graph(layout):
    """R2.8: at 0.5x and 2x resolution the edges, hierarchy and layout groups are unchanged."""
    def build(scale: float) -> Dict:
        """Builds the graph of the layout at the given scale.

        Input:  scale — coordinate and image-size factor
        Output: graph dictionary
        """
        dets = shazam(scale) if layout == "shazam" else grid_layout(scale)
        size = (SHAZAM_SIZE[0] * scale, SHAZAM_SIZE[1] * scale)
        return UIGraphBuilder().build_graph(dets, image_size=size)

    ref = build(1.0)
    assert any(e['relation'] != 'parent_child' for e in ref['edges'])
    for scale in (0.5, 2.0):
        g = build(scale)
        assert edge_set(g) == edge_set(ref)
        assert [structure(r) for r in g['hierarchy']['roots']] == \
            [structure(r) for r in ref['hierarchy']['roots']]
        assert g['params']['tolerance_px'] == ref['params']['tolerance_px'] * scale


def test_fixed_pixel_tolerance_is_not_resolution_independent():
    """The v1.1.0 behaviour (fixed 20 px) loses the grid's alignment edges at 2x; the relative one keeps them."""
    def spatial(builder: UIGraphBuilder, scale: float) -> List[Tuple[int, int, str]]:
        """Returns the spatial edges of the grid layout at a scale.

        Input:  builder — graph builder; scale — coordinate factor
        Output: sorted spatial edge triples
        """
        g = builder.build_graph(grid_layout(scale), image_size=(1200 * scale, 1920 * scale))
        return [e for e in edge_set(g) if e[2] != 'parent_child']

    fixed = UIGraphBuilder(align_tol_px=20)
    assert spatial(fixed, 1.0) != spatial(fixed, 2.0)
    relative = UIGraphBuilder()
    assert spatial(relative, 1.0) == spatial(relative, 2.0)
    assert spatial(fixed, 1.0) == spatial(relative, 1.0)


def test_default_tolerance_equals_v110_20px_at_redraw_width():
    """align_tol_frac defaults to 20/1200, i.e. exactly v1.1.0's 20 px at 1200 px width."""
    builder = UIGraphBuilder()
    assert DEFAULT_ALIGN_TOL_FRAC == 20 / 1200
    assert builder.alignment_tolerance_px(1200) == 20.0
    assert UIGraphBuilder(align_tol_px=7).alignment_tolerance_px(1200) == 7


# ── 3. Output schema and input handling ────────────────────


def test_synthetic_nodes_are_flagged():
    """Synthetic layout nodes have negative ids, Synthetic* classes, synthetic=True and no confidence."""
    graph = UIGraphBuilder().build_graph(shazam(), image_size=SHAZAM_SIZE)
    conf = {i: p for i, _, p, _ in SHAZAM}
    seen_synthetic = 0

    def walk(node: Dict):
        """Checks the flags of the node and recurses.

        Input:  node — hierarchy node
        Output: none (assertions)
        """
        nonlocal seen_synthetic
        if node['id'] < 0:
            seen_synthetic += 1
            assert node['synthetic'] is True
            assert node['class'] in ('SyntheticRow', 'SyntheticColumn')
            assert node['confidence'] is None
        else:
            assert node['synthetic'] is False
            assert node['confidence'] == conf[node['id']]
        for child in node['children']:
            walk(child)

    for root in graph['hierarchy']['roots']:
        walk(root)
    assert seen_synthetic > 0


def test_containment_roots_removes_synthetic_groups():
    """containment_roots returns exactly the top-level detected nodes."""
    graph = UIGraphBuilder().build_graph(shazam(), image_size=SHAZAM_SIZE)
    roots = containment_roots(graph['hierarchy'])
    assert all(not r['synthetic'] for r in roots)
    top_level = {c for c, p in parent_map(graph).items() if p is None}
    assert sorted(r['id'] for r in roots) == sorted(top_level)


def test_params_and_image_size_in_output():
    """build_graph reports the image size and the thresholds it actually used."""
    graph = UIGraphBuilder(containment_ratio=0.9).build_graph(shazam(0.5), image_size=(600, 960))
    assert graph['image_size'] == [600, 960]
    assert graph['params'] == {
        'containment_ratio': 0.9, 'align_tol_frac': 20 / 1200, 'align_tol_px': None,
        'tolerance_px': 10.0, 'layout_row_tol': 0.6, 'layout_col_tol': 1.5,
        'image_size_source': 'argument'}
    json.dumps(graph)  # the whole graph is JSON-serialisable


def test_image_size_resolution_order():
    """image_size argument > img_w/img_h > bbox vs bbox_norm > ReDraw default."""
    plain = [det(0, 'TextView', 0.9, [100, 200, 300, 260])]
    assert resolve_image_size(plain, (640, 480)) == ((640, 480), 'argument')
    with_size = [det(0, 'TextView', 0.9, [100, 200, 300, 260], img_w=720, img_h=1280)]
    assert resolve_image_size(with_size) == ((720, 1280), 'detections')
    with_norm = [det(0, 'TextView', 0.9, [100, 200, 300, 260],
                     bbox_norm=[200 / 720, 230 / 1280, 200 / 720, 60 / 1280])]
    assert resolve_image_size(with_norm) == ((720, 1280), 'bbox_norm')
    assert resolve_image_size(plain) == (DEFAULT_IMAGE_SIZE, 'default')
    assert resolve_image_size([]) == (DEFAULT_IMAGE_SIZE, 'default')

    graph = UIGraphBuilder().build_graph(with_size)
    assert graph['image_size'] == [720, 1280]
    assert graph['params']['tolerance_px'] == pytest.approx(20 / 1200 * 720)

    with pytest.raises(ValueError):
        resolve_image_size(with_size + [det(1, 'TextView', 0.9, [0, 0, 1, 1], img_w=1, img_h=1)])
    with pytest.raises(ValueError):
        resolve_image_size(plain, (0, 100))


def test_image_size_from_v110_detection_file():
    """A v1.1.0 detection file (no img_w/img_h) yields 1200 x 1920 via bbox_norm."""
    path = REPO_ROOT / "results/v1.1.0_examples/com.shazam.android-screens_screenshot_3/1_detections.json"
    require_reference_data(path)
    dets = json.loads(path.read_text())
    graph = UIGraphBuilder().build_graph(dets)
    assert graph['image_size'] == [1200, 1920]
    assert graph['params']['image_size_source'] == 'bbox_norm'


def test_plain_box_lists_are_accepted_and_input_is_not_modified():
    """Plain boxes (bbox + class_name only) get ids by position and confidence 1.0; input stays unchanged."""
    boxes = [{'class_name': 'ImageView', 'bbox': [0, 0, 100, 100]},
             {'class_name': 'TextView', 'bbox': [10, 10, 50, 30]}]
    before = copy.deepcopy(boxes)
    graph = UIGraphBuilder().build_graph(boxes, image_size=(100, 100))
    assert boxes == before
    assert [n['id'] for n in graph['nodes']] == [0, 1]
    assert [n['confidence'] for n in graph['nodes']] == [1.0, 1.0]
    assert parent_map(graph) == {0: None, 1: 0}

    dets = shazam()
    before = copy.deepcopy(dets)
    UIGraphBuilder().build_graph(dets, image_size=SHAZAM_SIZE)
    assert dets == before


def test_invalid_inputs_are_rejected():
    """Missing fields, duplicate or negative ids and out-of-range thresholds raise errors."""
    builder = UIGraphBuilder()
    with pytest.raises(ValueError):
        builder.build_graph([{'bbox': [0, 0, 1, 1]}])
    with pytest.raises(ValueError):
        builder.build_graph([det(3, 'TextView', 0.5, [0, 0, 1, 1]),
                             det(3, 'TextView', 0.5, [2, 2, 3, 3])])
    with pytest.raises(ValueError):
        builder.build_graph([det(-1, 'TextView', 0.5, [0, 0, 1, 1])])
    with pytest.raises(ValueError):
        UIGraphBuilder(containment_ratio=0.0)
    with pytest.raises(ValueError):
        UIGraphBuilder(containment_ratio=1.5)
    with pytest.raises(ValueError):
        UIGraphBuilder(align_tol_frac=-0.1)
    with pytest.raises(TypeError):
        UIGraphBuilder(iou_threshold=0.8)       # removed: it was never used in v1.1.0
    with pytest.raises(TypeError):
        UIGraphBuilder(0.8, 20, 20)             # v1.1.0 positional call


def test_deprecated_pixel_thresholds_map_to_align_tol_px():
    """vertical_threshold / horizontal_threshold still work, with a DeprecationWarning."""
    with pytest.warns(DeprecationWarning):
        builder = UIGraphBuilder(vertical_threshold=30, horizontal_threshold=30)
    assert builder.align_tol_px == 30
    assert builder.alignment_tolerance_px(2400) == 30
    with pytest.warns(DeprecationWarning):
        assert UIGraphBuilder(horizontal_threshold=12).align_tol_px == 12
    with pytest.raises(ValueError):
        UIGraphBuilder(vertical_threshold=10, horizontal_threshold=20)
    with pytest.raises(ValueError):
        UIGraphBuilder(vertical_threshold=10, align_tol_px=10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        UIGraphBuilder()                        # no warning without deprecated keywords


# ── 4. Spatial edges and explanation ───────────────────────


def test_spatial_edge_rule_between_siblings_only():
    """Aligned centres give above/left_of edges; non-siblings and unaligned pairs give none."""
    dets = [det(0, 'TextView', 0.9, [100, 100, 300, 150]),
            det(1, 'TextView', 0.9, [110, 400, 310, 450]),    # x-centre +10 -> 0 above 1
            det(2, 'TextView', 0.9, [600, 105, 800, 155]),    # y-centre +5  -> 0 left_of 2
            det(3, 'ImageView', 0.9, [500, 600, 1100, 1000]),
            det(4, 'TextView', 0.9, [600, 700, 800, 750])]    # child of 3
    graph = UIGraphBuilder().build_graph(dets, image_size=(1200, 1920))
    spatial = {(s, t): r for s, t, r in edge_set(graph) if r != 'parent_child'}
    # 4 has no sibling; 3 is not aligned with anything (centre offsets >= 100 px)
    assert spatial == {(0, 1): RelationType.ABOVE.value, (0, 2): RelationType.LEFT_OF.value}
    assert (3, 4, 'parent_child') in edge_set(graph)

    # pairs are visited in input order (i < j): reversing 0 and 1 gives 'below'
    reverse = UIGraphBuilder().build_graph([dets[1], dets[0], dets[2]], image_size=(1200, 1920))
    rev = {(s, t): r for s, t, r in edge_set(reverse) if r != 'parent_child'}
    assert rev == {(1, 0): RelationType.BELOW.value, (0, 2): RelationType.LEFT_OF.value}


def test_right_of_relation():
    """A box whose centre is right of an aligned sibling's gets a right_of edge."""
    dets = [det(0, 'TextView', 0.9, [600, 100, 800, 150]), det(1, 'TextView', 0.9, [100, 102, 300, 152])]
    graph = UIGraphBuilder().build_graph(dets, image_size=(1200, 1920))
    assert edge_set(graph) == [(0, 1, 'right_of')]


def test_explain_containment_names_the_rule_and_candidates():
    """The explanation lists all containers, the chosen parent and near misses."""
    graph = UIGraphBuilder().build_graph(shazam(), image_size=SHAZAM_SIZE)
    text = "\n".join(explain_containment(graph, 11))
    assert "contained in: 9 ImageView" in text and "8 ImageView" in text
    assert "parent: 9 ImageView" in text and "smallest-area" in text
    text = "\n".join(explain_containment(graph, 16))
    assert "contained in: 17 RecyclerView (coverage 0.935" in text
    assert "parent: 17 RecyclerView — only container" in text
    at95 = UIGraphBuilder(containment_ratio=0.95).build_graph(shazam(), image_size=SHAZAM_SIZE)
    text = "\n".join(explain_containment(at95, 16))
    assert "top-level" in text and "near miss: 17 RecyclerView covers 0.935" in text
    text = "\n".join(explain_containment(graph, 9))
    assert "parent: 8 ImageView — only container" in text
    assert "children: 10 ImageView, 11 TextView" in text
    assert "not a node" in explain_containment(graph, 99)[0]


def test_explain_containment_agrees_with_hierarchy():
    """For every node, the parent named by the explanation is the parent in the hierarchy."""
    graph = UIGraphBuilder().build_graph(shazam(), image_size=SHAZAM_SIZE)
    for node_id, parent in parent_map(graph).items():
        lines = explain_containment(graph, node_id)
        named = [line for line in lines if line.strip().startswith("parent:")]
        if parent is None:
            assert not named
        else:
            assert named and named[0].strip().startswith(f"parent: {parent} ")


# ── 5. Edge ablation: code generation ignores edges and synthetic groups ──


def _generator():
    """Imports and instantiates the code generator (another module under revision).

    Input:  none
    Output: ComposeCodeGenerator instance
    """
    from ui2compose.infer.generate_compose_code import ComposeCodeGenerator
    return ComposeCodeGenerator()


def _layout_modes() -> List[Dict]:
    """Lists generate() keyword sets to check: the default call plus every layout mode it supports.

    Input:  none
    Output: list of keyword dictionaries
    """
    from ui2compose.infer.generate_compose_code import ComposeCodeGenerator
    params = inspect.signature(ComposeCodeGenerator.generate).parameters
    modes: List[Dict] = [{}]
    if 'layout' in params:
        modes += [{'layout': 'xycut', 'image_size': SHAZAM_SIZE},
                  {'layout': 'flat', 'image_size': SHAZAM_SIZE}]
    return modes


@pytest.mark.parametrize("ratio", [0.95, 1.0])
def test_generated_code_does_not_depend_on_edges(ratio):
    """R1.3: removing, shuffling or corrupting graph['edges'] leaves the generated code unchanged."""
    graph = UIGraphBuilder(containment_ratio=ratio).build_graph(shazam(), image_size=SHAZAM_SIZE)
    assert graph['edges']
    for kwargs in _layout_modes():
        reference = _generator().generate(copy.deepcopy(graph), **kwargs)

        no_edges = copy.deepcopy(graph)
        no_edges['edges'] = []
        assert _generator().generate(no_edges, **kwargs) == reference

        shuffled = copy.deepcopy(graph)
        random.Random(0).shuffle(shuffled['edges'])
        assert _generator().generate(shuffled, **kwargs) == reference

        corrupted = copy.deepcopy(graph)
        relations = [r.value for r in RelationType]
        for i, e in enumerate(corrupted['edges']):
            e['relation'] = relations[i % len(relations)]
            e['source'], e['target'] = e['target'], e['source']
        assert _generator().generate(corrupted, **kwargs) == reference


@pytest.mark.parametrize("ratio", [0.95, 1.0])
def test_generated_code_does_not_depend_on_synthetic_groups(ratio):
    """Replacing the synthetic Row/Column wrappers by the plain containment forest leaves the code unchanged."""
    graph = UIGraphBuilder(containment_ratio=ratio).build_graph(shazam(), image_size=SHAZAM_SIZE)
    for kwargs in _layout_modes():
        reference = _generator().generate(copy.deepcopy(graph), **kwargs)
        stripped = copy.deepcopy(graph)
        stripped['hierarchy'] = {'roots': containment_roots(stripped['hierarchy'])}
        assert _generator().generate(stripped, **kwargs) == reference
