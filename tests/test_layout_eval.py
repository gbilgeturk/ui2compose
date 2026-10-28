# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Tests of the layout evaluation (ui2compose.evaluation.layout_eval).

Sections:
  1. Reference trees: the label replay equals parse_redraw.extract_labels,
     the parse settings equal the parse_redraw CLI defaults, label lines are
     aligned to XML nodes (sequence / per line / near-exact / unmatched),
     pruning and single-child collapsing, composites stay internal.
  2. Canonical form: orientation labels from geometry, reading order,
     root wrapping, independence of the input child order.
  3. Distances: TEDS identities, known edits with exact values and
     operations, agreement with zss (skipped without apted / zss).
  4. Grouping F1 on hand-made cases and the Hungarian matching (skipped
     without scipy).
  5. Generated trees: v1.2.0 composites (LazyColumn, Box), the flat
     baseline, the v1.1.0 duplicates / lost containers / fallback tag
     (v1.1.0 files from 'git show v1.1.0:...'; skipped without git or the tag).
  6. Bootstrap: constant values, group resampling, replicate medians.
  7. End to end on a synthetic package: exclusions, files, aggregate,
     determinism, test-split guard, CLI.

The synthetic fixtures are written to tmp_path; nothing outside it is read
except the v1.1.0 sources of the repository's git history.
"""

import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pytest
from PIL import Image

from ui2compose.evaluation import layout_eval as LE
from ui2compose.preprocess import parse_redraw as pr

REPO = Path(__file__).resolve().parents[1]
NAMES = list(LE.DETECTOR_CLASSES)
NAME2ID = {n: i for i, n in enumerate(NAMES)}
W, H = 120, 192          # small screens keep the fixtures fast (area filter: 1.15 px^2)


# ── helpers ────────────────────────────────────────────────

def xml_node(cls: str, bounds: Optional[Sequence[int]] = None, children: Sequence[str] = (),
             **attrs) -> str:
    """Builds the XML text of one uiautomator node.

    Input:  cls — class attribute; bounds — (x1, y1, x2, y2) or None (no bounds
            attribute); children — XML texts of the child nodes; attrs — more attributes
    Output: XML text
    """
    a = f' class="{cls}"'
    if bounds is not None:
        x1, y1, x2, y2 = bounds
        a += f' bounds="[{x1},{y1}][{x2},{y2}]"'
    for k, v in attrs.items():
        a += f' {k}="{v}"'
    if not children:
        return f'<node{a} />'
    return f'<node{a}>' + ''.join(children) + '</node>'


def hierarchy(*nodes: str) -> str:
    """Wraps nodes into a uiautomator dump.

    Input:  nodes — XML texts of the top-level nodes
    Output: XML document text
    """
    return "<?xml version='1.0' encoding='UTF-8' standalone='yes' ?><hierarchy>" + \
        ''.join(nodes) + '</hierarchy>'


def write_labels_like_parse_redraw(xml_path: Path, label_path: Path, w: int = W, h: int = H) -> List[str]:
    """Writes the label file of a hierarchy with parse_redraw.extract_labels (CLI settings).

    Input:  xml_path — hierarchy file; label_path — target .txt; w, h — image size
    Output: written lines
    """
    labels = pr.extract_labels(xml_path, w, h, NAME2ID, drop_container=LE.PARSE_DROP_CONTAINER,
                               min_area=LE.PARSE_MIN_AREA, dedup_iou=LE.PARSE_DEDUP_IOU)
    lines = [f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" for cid, (xc, yc, bw, bh) in labels]
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(''.join(line + '\n' for line in lines), encoding='utf-8')
    return lines


#: A screen with a row (ImageView + TextView), a single-child chain around a
#: ListView with two items, an aliased Button, a container-named node, a
#: too-small node, a same-class duplicate and an x/y/width/height node.
MAIN_XML = hierarchy(xml_node('android.widget.FrameLayout', (0, 0, 120, 192), [
    xml_node('android.widget.LinearLayout', (0, 0, 120, 40), [
        xml_node('android.widget.ImageView', (0, 0, 30, 40)),
        xml_node('android.widget.TextView', (30, 0, 120, 40), [
            xml_node('android.widget.TextView', (30, 0, 120, 40)),          # duplicate: dropped
        ]),
    ]),
    xml_node('android.widget.LinearLayout', (0, 50, 120, 150), [
        xml_node('android.widget.FrameLayout', (0, 50, 120, 150), [
            xml_node('android.widget.ListView', (0, 50, 120, 150), [
                xml_node('android.widget.LinearLayout', (0, 50, 120, 90), [
                    xml_node('android.widget.TextView', (0, 50, 60, 90)),
                    xml_node('android.support.v7.widget.AppCompatButton', (60, 50, 120, 90)),
                ]),
                xml_node('android.widget.TextView', (0, 100, 120, 140)),
            ]),
        ]),
    ]),
    xml_node('android.widget.View', (0, 150, 1, 151)),                         # not a class
    xml_node('android.widget.TextView', (5, 160, 6, 161)),                     # too small
    xml_node('android.support.v7.widget.ActionBarContainer', (0, 160, 120, 192), [
        xml_node('android.widget.CheckBox', None, x=10, y=165, width=20, height=20),
    ]),
]))


def write_main_screen(root: Path, xml_text: str = MAIN_XML) -> Dict[str, Path]:
    """Writes the hierarchy, its label file (parse_redraw) and a blank screenshot.

    Input:  root — directory; xml_text — hierarchy text
    Output: {'xml', 'label', 'image'} paths
    """
    root.mkdir(parents=True, exist_ok=True)
    paths = {'xml': root / 'hierarchy_1.xml', 'label': root / 'screen.txt',
             'image': root / 'screen.png'}
    paths['xml'].write_text(xml_text, encoding='utf-8')
    Image.new('RGB', (W, H), 'white').save(paths['image'])
    write_labels_like_parse_redraw(paths['xml'], paths['label'])
    return paths


def build(paths: Dict[str, Path]):
    """Builds the reference of a written screen.

    Input:  paths — write_main_screen output
    Output: (record, canonical tree, gt boxes)
    """
    return LE.build_reference('screen', paths['xml'], paths['label'], paths['image'], NAMES)


def det(cls: str, i: int, box: Sequence[float], children=None) -> LE.TreeNode:
    """Shorthand for a detection node.

    Input:  cls — class; i — id; box — [x1, y1, x2, y2]; children — child nodes
    Output: TreeNode
    """
    return LE.detection(cls, i, box, 'test', children)


def grp(*children: LE.TreeNode) -> LE.TreeNode:
    """Shorthand for a container node.

    Input:  children — child nodes
    Output: TreeNode
    """
    return LE.container('test', list(children))


# ── 1. reference trees ─────────────────────────────────────

def test_parse_settings_are_the_parse_redraw_cli_defaults(monkeypatch):
    """The label replay uses the settings the dataset's label files were written with."""
    monkeypatch.setattr(sys, 'argv', ['parse_redraw'])
    args = pr.parse_args()
    assert args.drop_container is LE.PARSE_DROP_CONTAINER
    assert args.min_area == LE.PARSE_MIN_AREA
    assert args.dedup_iou == LE.PARSE_DEDUP_IOU


def test_trace_labels_replays_parse_redraw(tmp_path):
    """trace_labels yields the labels of parse_redraw.extract_labels, in the same order, with their nodes."""
    paths = write_main_screen(tmp_path)
    expected = pr.extract_labels(paths['xml'], W, H, NAME2ID, drop_container=True,
                                 min_area=LE.PARSE_MIN_AREA, dedup_iou=LE.PARSE_DEDUP_IOU)
    import xml.etree.ElementTree as ET
    traced = LE.trace_labels(ET.parse(paths['xml']).getroot(), W, H, NAME2ID)
    assert [(cid, box) for _, cid, box in traced] == expected
    classes = [NAMES[cid] for _, cid, _ in traced]
    assert classes == ['ImageView', 'TextView', 'ListView', 'TextView', 'Button', 'TextView',
                       'CheckBox']
    assert [LE._node_class(e).split('.')[-1] for e, _, _ in traced][4] == 'AppCompatButton'


def test_reference_leaf_set_equals_the_label_boxes(tmp_path):
    """Every label line becomes exactly one detection node; pruning and collapsing as specified."""
    rec, tree, gt = build(write_main_screen(tmp_path))
    assert rec['status'] == 'ok'
    assert rec['alignment']['method'] == 'sequence'
    assert rec['alignment']['n_exact'] == rec['n_gt'] == len(gt) == 7
    ids = [n.det_id for n, _, _ in LE.iter_preorder(tree) if n.cls is not None]
    assert sorted(ids) == list(range(7))
    # root (FrameLayout) -> Row container, ListView composite (chain collapsed), CheckBox
    # (the container-named parent has one child and collapses); the too-small node, the
    # View and the duplicate TextView are pruned.
    assert LE.bracket(tree) == ('{Column{Row{ImageView#0}{TextView#1}}'
                                '{ListView#2{Row{TextView#3}{Button#4}}{TextView#5}}'
                                '{CheckBox#6}}')
    stats = LE.tree_stats(tree)
    assert stats == {'nodes': 10, 'depth': 4, 'containers': 3, 'detections': 7, 'composites': 1}
    assert rec['flat'] is False
    # Ground-truth boxes are snapped back to the integer XML bounds.
    assert gt[0]['bbox'] == [0.0, 0.0, 30.0, 40.0]
    assert gt[6]['bbox'] == [10.0, 165.0, 30.0, 185.0]


def test_alignment_per_line_near_and_unmatched(tmp_path):
    """Reordered lines match exactly, a sub-pixel change matches near-exactly, a 5 px change is excluded."""
    paths = write_main_screen(tmp_path)
    lines = LE.read_label_lines(paths['label'])
    paths['label'].write_text('\n'.join(reversed(lines)) + '\n', encoding='utf-8')
    rec, tree, _ = build(paths)
    assert rec['status'] == 'ok'
    assert rec['alignment']['method'] == 'per_line'
    assert rec['alignment']['n_exact'] == 7 and rec['alignment']['n_unmatched'] == 0

    def shifted(line: str, dx_px: float) -> str:
        """Moves a label line horizontally.

        Input:  line — label line; dx_px — shift in pixels
        Output: label line
        """
        cid, xc, yc, w, h = line.split()
        return f"{cid} {float(xc) + dx_px / W:.6f} {yc} {w} {h}"

    near = list(lines)
    near[0] = shifted(lines[0], 0.5)
    paths['label'].write_text('\n'.join(near) + '\n', encoding='utf-8')
    rec, tree, _ = build(paths)
    assert rec['status'] == 'ok'
    assert (rec['alignment']['n_exact'], rec['alignment']['n_near']) == (6, 1)
    # The node keeps its place; the (label) box now overlaps its neighbour by 0.5 px -> Box.
    assert LE.bracket(tree).startswith('{Column{Box{ImageView#0}{TextView#1}}')

    far = list(lines)
    far[0] = shifted(lines[0], 5.0)
    paths['label'].write_text('\n'.join(far) + '\n', encoding='utf-8')
    rec, tree, _ = build(paths)
    assert rec['status'] == 'excluded:alignment_incomplete'
    assert tree is None
    assert rec['alignment']['unmatched_lines'] == [0]
    assert rec['alignment']['n_traced_unused'] == 1


def test_reference_exclusions(tmp_path):
    """Missing mapping, label file, empty labels and a broken XML are excluded with their reason."""
    paths = write_main_screen(tmp_path)
    assert LE.build_reference('s', None, paths['label'], paths['image'], NAMES)[0]['status'] == \
        'excluded:no_mapping'
    assert LE.build_reference('s', paths['xml'], tmp_path / 'none.txt', paths['image'],
                              NAMES)[0]['status'] == 'excluded:no_label_file'
    empty = tmp_path / 'empty.txt'
    empty.write_text('\n', encoding='utf-8')
    assert LE.build_reference('s', paths['xml'], empty, paths['image'], NAMES)[0]['status'] == \
        'excluded:empty_labels'
    broken = tmp_path / 'broken.xml'
    broken.write_text('<hierarchy><node', encoding='utf-8')
    assert LE.build_reference('s', broken, paths['label'], paths['image'], NAMES)[0]['status'] == \
        'excluded:xml_parse_error'


def test_flat_dump_is_depth_two(tmp_path):
    """A dump whose labels all hang below one chain of single-child containers is flat (depth 2)."""
    xml = hierarchy(xml_node('android.widget.FrameLayout', (0, 0, 120, 192), [
        xml_node('android.widget.LinearLayout', (0, 0, 120, 192), [
            xml_node('android.widget.TextView', (0, 0, 120, 40)),
            xml_node('android.widget.Button', (0, 50, 120, 90)),
            xml_node('android.widget.ImageView', (0, 100, 120, 190)),
        ])]))
    rec, tree, _ = build(write_main_screen(tmp_path, xml))
    assert rec['ref_depth'] == 2 and rec['flat'] is True
    assert LE.bracket(tree) == '{Column{TextView#0}{Button#1}{ImageView#2}}'


# ── 2. canonical form ──────────────────────────────────────

def test_orientation_labels_from_geometry():
    """Column for y-disjoint children (touching allowed), Row for x-disjoint, Box otherwise."""
    assert LE.orientation_of([[0, 0, 10, 10], [0, 10, 10, 20]]) == 'Column'     # touching
    assert LE.orientation_of([[0, 0, 10, 10], [10, 2, 20, 12]]) == 'Row'
    assert LE.orientation_of([[0, 0, 10, 10], [5, 5, 15, 15]]) == 'Box'
    assert LE.orientation_of([[0, 0, 10, 10], [0, 9.5, 10, 20]]) == 'Box'       # tolerance 0
    assert LE.orientation_of([[0, 0, 10, 10]]) == 'Column'
    assert LE.orientation_of([]) == 'Column'
    # Diagonal (disjoint on both axes) -> Column (y is checked first).
    assert LE.orientation_of([[0, 0, 10, 10], [20, 20, 30, 30]]) == 'Column'


def test_canonical_tree_is_independent_of_the_child_order():
    """Shuffled children give the same canonical tree; single-child chains collapse; empty containers vanish."""
    def make(order: List[int]) -> LE.TreeNode:
        """Builds the same tree with the root's children in the given order.

        Input:  order — permutation of 0..3
        Output: TreeNode
        """
        kids = [grp(grp(det('TextView', 0, [0, 0, 50, 10]))),            # chain -> leaf
                grp(det('Button', 1, [0, 20, 40, 30]), det('ImageView', 2, [41, 21, 60, 29])),
                grp(),                                                    # empty -> pruned
                det('ListView', 3, [0, 40, 60, 90], [grp(det('TextView', 4, [0, 40, 60, 60]),
                                                         det('TextView', 5, [0, 60, 60, 80]))])]
        return grp(*[kids[i] for i in order])

    ref = LE.bracket(LE.canonicalize(make([0, 1, 2, 3])))
    assert ref == ('{Column{TextView#0}{Row{Button#1}{ImageView#2}}'
                   '{ListView#3{Column{TextView#4}{TextView#5}}}}')
    rng = random.Random(0)
    for _ in range(10):
        order = [0, 1, 2, 3]
        rng.shuffle(order)
        assert LE.bracket(LE.canonicalize(make(order))) == ref


def test_reading_order_of_rows_ignores_one_pixel_top_differences():
    """Row children are read left to right even when a right item starts one pixel higher."""
    t = LE.canonicalize(grp(det('ImageView', 0, [10, 238, 375, 597]),
                            det('ImageView', 1, [419, 237, 782, 598]),
                            det('ImageView', 2, [831, 238, 1185, 597])))
    assert LE.bracket(t) == '{Row{ImageView#0}{ImageView#1}{ImageView#2}}'
    # A Column is read top to bottom even when a lower item starts further left.
    t = LE.canonicalize(grp(det('TextView', 0, [50, 0, 100, 10]), det('TextView', 1, [0, 20, 40, 30])))
    assert LE.bracket(t) == '{Column{TextView#0}{TextView#1}}'


def test_root_wrapping():
    """A single detection and an empty tree get a container root."""
    t = LE.canonicalize(grp(grp(det('Button', 7, [0, 0, 5, 5]))))
    assert LE.bracket(t) == '{Column{Button#7}}' and t.src == 'root'
    e = LE.canonicalize(None)
    assert LE.bracket(e) == '{Column}' and LE.tree_stats(e)['nodes'] == 1
    assert LE.bracket(LE.canonicalize(grp(grp()))) == '{Column}'


# ── 3. distances ───────────────────────────────────────────

def _three() -> LE.TreeNode:
    """Root container with three stacked leaves.

    Input:  none
    Output: canonical TreeNode
    """
    return LE.canonicalize(grp(det('TextView', 0, [0, 0, 10, 10]), det('Button', 1, [0, 20, 10, 30]),
                               det('ImageView', 2, [0, 40, 10, 50])))


def test_teds_identity_and_known_edits():
    """Identical trees: TED 0, TEDS 1; one relabel / one extra container / one missing leaf: exact values."""
    pytest.importorskip('apted')
    ref = _three()
    for v in LE.VARIANTS:
        r = LE.tree_edit(ref, _three(), v)
        assert (r['ted'], r['teds'], r['insert'], r['delete'], r['rename']) == (0, 1.0, 0, 0, 0)
    relabel = LE.canonicalize(grp(det('TextView', 0, [0, 0, 10, 10]), det('Switch', 1, [0, 20, 10, 30]),
                                  det('ImageView', 2, [0, 40, 10, 50])))
    r = LE.tree_edit(ref, relabel, 'g')
    assert (r['ted'], r['rename'], r['teds']) == (1, 1, 0.75)
    extra = LE.canonicalize(grp(grp(det('TextView', 0, [0, 0, 10, 10]), det('Button', 1, [0, 20, 10, 30])),
                                det('ImageView', 2, [0, 40, 10, 50])))
    r = LE.tree_edit(ref, extra, 'g')
    assert (r['ted'], r['delete'], r['insert'], r['n_gen'], r['n_ref']) == (1, 1, 0, 5, 4)
    assert r['teds'] == pytest.approx(0.8)
    missing = LE.canonicalize(grp(det('TextView', 0, [0, 0, 10, 10]), det('Button', 1, [0, 20, 10, 30])))
    r = LE.tree_edit(ref, missing, 'g')
    assert (r['ted'], r['insert'], r['teds']) == (1, 1, 0.75)


def test_teds_o_sees_orientation_but_teds_g_does_not():
    """A Row where the reference has a Column costs one rename in TEDS-o and nothing in TEDS-g."""
    pytest.importorskip('apted')
    col = LE.canonicalize(grp(det('TextView', 0, [0, 0, 10, 10]), det('TextView', 1, [0, 20, 10, 30])))
    row = LE.canonicalize(grp(det('TextView', 0, [0, 0, 10, 10]), det('TextView', 1, [20, 5, 30, 15])))
    assert LE.tree_edit(col, row, 'g')['ted'] == 0
    r = LE.tree_edit(col, row, 'o')
    assert (r['ted'], r['rename']) == (1, 1)


def test_apted_agrees_with_zss_on_random_trees():
    """APTED and Zhang-Shasha give the same unit-cost distance; the mapping cost equals the distance."""
    pytest.importorskip('apted')
    pytest.importorskip('zss')
    rng = random.Random(1)
    classes = ['TextView', 'Button', 'ImageView']

    def rand_tree(n: int) -> LE.TreeNode:
        """Random tree of n leaves with random grouping.

        Input:  n — number of leaves
        Output: canonical TreeNode
        """
        nodes = [det(rng.choice(classes), i, [rng.randint(0, 50), y := rng.randint(0, 100),
                                              60 + rng.randint(0, 50), y + rng.randint(1, 30)])
                 for i in range(n)]
        while len(nodes) > 1:
            k = rng.randint(2, min(4, len(nodes)))
            take = [nodes.pop(rng.randrange(len(nodes))) for _ in range(k)]
            nodes.append(grp(*take))
        return LE.canonicalize(nodes[0])

    for _ in range(25):
        a, b = rand_tree(rng.randint(1, 12)), rand_tree(rng.randint(1, 12))
        for v in LE.VARIANTS:
            assert LE.tree_edit(a, b, v)['ted'] == LE.zss_distance(a, b, v)


# ── 4. grouping and matching ───────────────────────────────

def test_grouping_scores_hand_made():
    """Over-grouping halves F1; a lost detection groups with nothing; no pair -> undefined."""
    pairs = [(i, i) for i in range(4)]
    ref = {0: 1, 1: 1, 2: 2, 3: 2}
    s = LE.grouping_scores(pairs, ref, {0: 5, 1: 5, 2: 5, 3: 5})
    assert (s['tp'], s['n_ref'], s['n_gen']) == (2, 2, 6)
    assert (s['precision'], s['recall'], s['f1']) == (pytest.approx(1 / 3), 1.0, 0.5)
    s = LE.grouping_scores(pairs, ref, {0: 5, 1: 5, 2: 5})
    assert (s['tp'], s['n_gen'], s['f1']) == (1, 3, pytest.approx(0.4))
    s = LE.grouping_scores(pairs, ref, {0: 1, 1: 1, 2: 2, 3: 2})
    assert s['f1'] == 1.0
    s = LE.grouping_scores([(0, 0)], {0: 1}, {0: 1})
    assert (s['precision'], s['recall'], s['f1']) == (None, None, None)


def test_detection_parents_and_duplicates():
    """Parents are pre-order numbers of the canonical tree; a duplicate keeps its first occurrence."""
    t = LE.canonicalize(grp(det('TextView', 0, [0, 0, 10, 10]),
                            grp(det('TextView', 1, [0, 20, 10, 30]), det('Button', 2, [20, 20, 30, 30])),
                            det('TextView', 1, [0, 40, 10, 50])))
    parents, occ = LE.detection_parents(t)
    assert parents == {0: 0, 1: 2, 2: 2}
    assert occ[1] == 2


def test_hungarian_matching_beats_greedy():
    """The assignment maximises the number of IoU >= 0.5 pairs (greedy by IoU would match one)."""
    pytest.importorskip('scipy')
    gt = [[0, 0, 100, 100], [50, 0, 150, 100]]
    pred = [[20, 0, 120, 100], [0, 0, 60, 100]]
    m = LE.match_boxes(gt, pred)
    assert [(i, j) for i, j, _ in m] == [(0, 1), (1, 0)]
    assert m[0][2] == pytest.approx(0.6) and m[1][2] == pytest.approx(70 / 130)
    assert LE.match_boxes(gt, [[0, 0, 40, 100]]) == []           # IoU 0.4 < 0.5
    assert LE.match_boxes([], pred) == [] and LE.match_boxes(gt, []) == []
    iou = LE.iou_matrix([[0, 0, 0, 0]], [[0, 0, 0, 0]])
    assert iou.shape == (1, 1) and iou[0, 0] == 0.0


# ── 5. generated trees ─────────────────────────────────────

BOXES = [
    {'id': 0, 'class_name': 'ListView', 'bbox': [0, 0, 1200, 800], 'confidence': 0.9},
    {'id': 1, 'class_name': 'TextView', 'bbox': [0, 0, 1200, 300], 'confidence': 0.9},
    {'id': 2, 'class_name': 'TextView', 'bbox': [0, 400, 1200, 700], 'confidence': 0.9},
    {'id': 3, 'class_name': 'ImageView', 'bbox': [0, 900, 1200, 1500], 'confidence': 0.9},
    {'id': 4, 'class_name': 'TextView', 'bbox': [100, 1000, 500, 1100], 'confidence': 0.9},
]


def test_v120_composites_keep_their_container_detection():
    """LazyColumn -> list-class node with its items; containment Box -> container-class node with its children."""
    tree, code, n_fb = LE.generate_v120(BOXES, (1200, 1920), 'xycut')
    t = LE.canonicalize(tree)
    assert LE.bracket(t) == '{Column{ListView#0{TextView#1}{TextView#2}}{ImageView#3{TextView#4}}}'
    assert n_fb == 0 and code.count(LE.PLACEHOLDER_MARK) == 0
    assert 'LazyColumn(' in code and 'Box(' in code
    flat, _, _ = LE.generate_v120(BOXES, (1200, 1920), 'flat')
    f = LE.canonicalize(flat)
    assert LE.tree_stats(f) == {'nodes': 6, 'depth': 2, 'containers': 1, 'detections': 5,
                                'composites': 0}


def test_box_repr_container_scores_the_widget_as_a_sibling_of_the_content():
    """Sensitivity variant: a containment Box becomes a container of its widget leaf and content; LazyColumns stay."""
    tree, code, _ = LE.generate_v120(BOXES, (1200, 1920), 'xycut', box_repr='container')
    t = LE.canonicalize(tree)
    assert LE.bracket(t) == '{Column{ListView#0{TextView#1}{TextView#2}}{Box{ImageView#3}{TextView#4}}}'
    assert LE.tree_stats(t)['composites'] == 1                 # the LazyColumn only
    primary, code_p, _ = LE.generate_v120(BOXES, (1200, 1920), 'xycut')
    assert code == code_p                                      # same Kotlin, other scoring
    assert LE.DEFAULT_BOX_REPR == 'composite'
    with pytest.raises(ValueError, match='box_repr'):
        LE.generate_v120(BOXES, (1200, 1920), 'xycut', box_repr='sibling')
    assert LE.parse_args(['--data-dir', 'd', '--out', 'o']).box_repr == 'composite'
    assert LE.parse_args(['--data-dir', 'd', '--out', 'o', '--box-repr', 'container']).box_repr \
        == 'container'


def test_condition_records_for_errors_and_empty_predictions(monkeypatch):
    """No predictions -> excluded; zero predictions -> scored empty root; oversized tree -> counted error."""
    pytest.importorskip('apted')
    ref = _three()
    gt = [{'id': i, 'class_name': n.cls, 'bbox': n.box, 'confidence': 1.0}
          for i, n in enumerate(ref.children)]
    p = LE.default_params()
    r = LE.evaluate_condition('C4', 's', ref, gt, (100, 100), None, p)
    assert r['status'] == 'excluded:no_predictions'
    r = LE.evaluate_condition('C4', 's', ref, gt, (100, 100), [], p)
    assert r['status'] == 'ok' and r['gen_nodes'] == 1 and r['n_matched'] == 0
    assert r['teds_g'] == pytest.approx(0.25) and r['grouping_f1'] is None
    monkeypatch.setattr(LE, 'MAX_TED_NODES', 3)
    r = LE.evaluate_condition('C1', 's', ref, gt, (100, 100), None, p)
    assert r['status'].startswith('error:tree_too_large')


def test_v120_fallback_nodes_are_counted():
    """Overlapping boxes without a cut give a fallback Column, counted once."""
    boxes = [{'id': 0, 'class_name': 'TextView', 'bbox': [0, 0, 600, 600]},
             {'id': 1, 'class_name': 'Button', 'bbox': [300, 300, 900, 900]}]
    _, _, n_fb = LE.generate_v120(boxes, (1200, 1920), 'xycut')
    assert n_fb == 1


@pytest.fixture(scope='module')
def v110_code(tmp_path_factory) -> LE.V110Code:
    """The v1.1.0 graph builder and generator, written from the repository's v1.1.0 tag.

    Input:  tmp_path_factory — pytest factory
    Output: V110Code (skips the test without git or the tag)
    """
    d = tmp_path_factory.mktemp('v110')
    for name in LE.V110_SHA256:
        try:
            src = subprocess.run(['git', '-C', str(REPO), 'show',
                                  f'v1.1.0:src/ui2compose/infer/{name}'],
                                 check=True, capture_output=True).stdout
        except (OSError, subprocess.CalledProcessError):
            pytest.skip('git or the v1.1.0 tag is not available')
        (d / name).write_bytes(src)
    return LE.V110Code(d)


def test_v110_code_is_checked_against_the_tag(tmp_path, v110_code):
    """The loader accepts the tag's files and refuses modified ones."""
    assert v110_code.is_v110_tag
    for name in LE.V110_SHA256:
        (tmp_path / name).write_bytes(Path(v110_code.dir, name).read_bytes() + b'\n# changed\n')
    with pytest.raises(ValueError):
        LE.V110Code(tmp_path)


def test_v110_duplicates_lost_containers_and_fallback(v110_code):
    """v1.1.0 lays out only leaves: a box inside two overlapping containers appears twice, the containers are lost."""
    boxes = [{'id': 0, 'class_name': 'ImageView', 'bbox': [0, 0, 600, 600]},
             {'id': 1, 'class_name': 'ImageView', 'bbox': [300, 0, 900, 600]},
             {'id': 2, 'class_name': 'TextView', 'bbox': [350, 100, 550, 300]}]
    tree, code, n_fb = LE.generate_v110(boxes, v110_code)
    t = LE.canonicalize(tree)
    assert LE.bracket(t) == '{Box{TextView#2}{TextView#2}}'
    assert n_fb == 1
    _, occ = LE.detection_parents(t)
    assert occ == {2: 2}
    # Classes without a v1.1.0 template become "Unknown" placeholders.
    _, code, _ = LE.generate_v110([{'id': 0, 'class_name': 'SeekBar', 'bbox': [0, 0, 10, 10]}],
                                  v110_code)
    assert code.count(LE.PLACEHOLDER_MARK) == 1


def test_v110_tree_equals_its_own_xy_cut(v110_code):
    """The tracing subclass returns exactly the tree v1.1.0's _xy_cut builds (code unchanged)."""
    graph = v110_code.graph.UIGraphBuilder().build_graph(
        [dict(b, center=[(b['bbox'][0] + b['bbox'][2]) / 2, (b['bbox'][1] + b['bbox'][3]) / 2])
         for b in BOXES])
    plain = v110_code.codegen.ComposeCodeGenerator()
    tracing = v110_code.generator_cls()
    assert tracing.generate(graph) == plain.generate(graph)
    expected = plain._xy_cut(plain._collect_leaves(graph['hierarchy']['roots']))

    def strip(n: Dict) -> Dict:
        """Removes the 'rule' tags.

        Input:  n — layout dict
        Output: copy without 'rule'
        """
        return {k: ([strip(c) for c in v] if k == 'children' else v) for k, v in n.items()
                if k != 'rule'}
    assert strip(tracing.last_layout_tree) == expected


# ── 6. bootstrap ───────────────────────────────────────────

def test_bootstrap_constant_values_and_groups():
    """Constant values give a degenerate interval; replicates only take group-resample values."""
    b = LE.Bootstrap(['a1', 'a2', 'b1'], {'a1': 'A', 'a2': 'A', 'b1': 'B'}, n=200, seed=0)
    iv = b.mean([0.5, 0.5, 0.5])
    assert iv['point'] == iv['lo'] == iv['hi'] == 0.5
    sums, cnts = b.unit_sums([0.0, 0.0, 1.0])
    reps = b.ratio_replicates(sums, cnts)
    assert set(np.round(reps, 6)) <= {0.0, round(1 / 3, 6), 1.0}
    assert b.mean([0.0, 0.0, 1.0])['point'] == pytest.approx(1 / 3)
    nan = b.mean([float('nan')] * 3)
    assert nan['point'] is None and nan['lo'] is None


def test_bootstrap_median_replicates_match_explicit_resamples():
    """The vectorised replicate median equals np.median of the explicitly resampled values."""
    rng = np.random.default_rng(3)
    stems = [f"s{i}" for i in range(40)]
    groups = {s: f"g{i % 13}" for i, s in enumerate(stems)}
    values = rng.normal(size=40)
    values[[3, 17]] = np.nan
    b = LE.Bootstrap(stems, groups, n=50, seed=1)
    reps = b.median_replicates(values)
    for k in range(50):
        mult = b.counts[k, b.unit_of].astype(int)
        explicit = np.repeat(values, mult)
        explicit = explicit[np.isfinite(explicit)]
        assert reps[k] == pytest.approx(np.median(explicit))


# ── 7. end to end ──────────────────────────────────────────

SIMPLE_XML = hierarchy(xml_node('android.widget.FrameLayout', (0, 0, 120, 192), [
    xml_node('android.widget.LinearLayout', (0, 0, 120, 40), [
        xml_node('android.widget.ImageView', (0, 0, 30, 40)),
        xml_node('android.widget.TextView', (30, 0, 120, 40)),
    ]),
    xml_node('android.widget.TextView', (0, 60, 120, 80)),
    xml_node('android.widget.Button', (0, 100, 120, 120)),
]))


def make_package(root: Path) -> Dict[str, str]:
    """Writes a four-screen synthetic val package (README_PACKAGE layout) and a prediction cache.

    s1 (app a): normal, predictions = jittered label boxes; s2 (app a): no
    predictions; s3 (app b): broken XML; s4 (app c): no raw XML mapping.

    Input:  root — directory
    Output: {'data': package dir, 'pred': prediction cache dir}
    """
    data, pred = root / 'pkg', root / 'pred'
    (data / 'dataset').mkdir(parents=True)
    (data / 'dataset' / 'dataset.yaml').write_text(
        'nc: 13\nnames:\n' + ''.join(f'  - {n}\n' for n in NAMES), encoding='utf-8')
    rows, manifest = [], []
    for stem, pkg, xml in (('a-screens_screenshot_1', 'a', SIMPLE_XML),
                           ('a-screens_screenshot_2', 'a', MAIN_XML),
                           ('b-screens_screenshot_1', 'b', '<hierarchy><node'),
                           ('c-screens_screenshot_1', 'c', SIMPLE_XML)):
        img = data / 'dataset' / 'images' / 'val' / f'{stem}.png'
        img.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (W, H), 'white').save(img)
        xml_rel = f'raw_xml/val/{pkg}-screens/hierarchy_{stem[-1]}.xml'
        (data / xml_rel).parent.mkdir(parents=True, exist_ok=True)
        good = SIMPLE_XML if pkg == 'b' else xml
        (data / xml_rel).write_text(good, encoding='utf-8')
        write_labels_like_parse_redraw(data / xml_rel, data / 'dataset' / 'labels' / 'val' / f'{stem}.txt')
        if pkg == 'b':
            (data / xml_rel).write_text(xml, encoding='utf-8')
        if pkg != 'c':
            rows.append({'split': 'val', 'stem': stem, 'xml_in_zip': xml_rel})
        manifest.append({'stem': stem, 'package': pkg, 'group_id': {'a': 0, 'b': 1, 'c': 2}[pkg]})
    with open(data / 'raw_xml' / 'mapping.csv', 'w', encoding='utf-8') as f:
        f.write('split,stem,xml_in_zip\n' + ''.join(f"{r['split']},{r['stem']},{r['xml_in_zip']}\n"
                                                    for r in rows))
    (data / 'splits' / 'full').mkdir(parents=True)
    with open(data / 'splits' / 'full' / 'manifest.csv', 'w', encoding='utf-8') as f:
        f.write('stem,package,group_id\n' + ''.join(f"{m['stem']},{m['package']},{m['group_id']}\n"
                                                   for m in manifest))
    (data / 'dataset' / 'unseen_val.txt').write_text(
        './images/val/a-screens_screenshot_1.png\n./images/val/b-screens_screenshot_1.png\n',
        encoding='utf-8')
    (data / 'dataset' / 'unseen_val_package_only.txt').write_text(
        './images/val/a-screens_screenshot_1.png\n', encoding='utf-8')
    gt = LE.parse_label_lines(LE.read_label_lines(
        data / 'dataset' / 'labels' / 'val' / 'a-screens_screenshot_1.txt'), W, H, NAMES)
    dets = [{'id': k, 'class_name': g['class_name'], 'class_id': g['class_id'],
             'confidence': 0.9 - 0.1 * k, 'bbox': [v + 0.3 for v in g['bbox']],
             'img_w': W, 'img_h': H} for k, g in enumerate(gt)]
    pred.mkdir()
    (pred / 'a-screens_screenshot_1.json').write_text(json.dumps(dets), encoding='utf-8')
    return {'data': str(data), 'pred': str(pred)}


def _strip_times(obj):
    """Removes timing and creation fields (for determinism checks).

    Input:  obj — JSON-like object
    Output: copy without 'time_s', 'timing', 'wall_seconds', 'created_utc'
    """
    if isinstance(obj, dict):
        return {k: _strip_times(v) for k, v in obj.items()
                if k not in ('time_s', 'timing', 'wall_seconds', 'created_utc')}
    if isinstance(obj, list):
        return [_strip_times(v) for v in obj]
    return obj


def test_end_to_end_on_a_synthetic_package(tmp_path):
    """Exclusions are counted, files are written, perfect boxes reproduce the reference, results are deterministic."""
    pytest.importorskip('apted')
    pytest.importorskip('zss')
    pytest.importorskip('scipy')
    pk = make_package(tmp_path)
    kw = dict(conditions=['C1', 'C2', 'C4', 'C5'], pred_cache=pk['pred'], n_boot=50,
              zss_check=4, progress_every=0)
    agg = LE.run_evaluation(pk['data'], 'val', tmp_path / 'out1', **kw)
    out = tmp_path / 'out1'
    assert {p.name for p in out.iterdir()} == {'aggregate.json', 'reference.jsonl', 'C1.jsonl',
                                               'C2.jsonl', 'C4.jsonl', 'C5.jsonl'}
    ref = agg['reference']['all']
    assert ref['status'] == {'excluded:no_mapping': 1, 'excluded:xml_parse_error': 1, 'ok': 2}
    assert ref['alignment']['label_match_rate'] == 1.0
    c1 = {json.loads(line)['stem']: json.loads(line) for line in (out / 'C1.jsonl').read_text().splitlines()}
    s1 = c1['a-screens_screenshot_1']
    assert (s1['teds_g'], s1['teds_o'], s1['grouping_f1']) == (1.0, 1.0, 1.0)
    c2 = {json.loads(line)['stem']: json.loads(line) for line in (out / 'C2.jsonl').read_text().splitlines()}
    assert c2['a-screens_screenshot_1']['teds_g'] == pytest.approx(1 - 1 / 6)
    assert c2['a-screens_screenshot_1']['grouping_f1'] == pytest.approx(0.5)
    c4 = agg['conditions']['C4']['all']
    assert c4['status'] == {'excluded:no_mapping': 1, 'excluded:no_predictions': 1,
                            'excluded:xml_parse_error': 1, 'ok': 1}
    assert c4['teds_g']['mean'] == 1.0 and c4['matching']['n_matched'] == 4
    assert agg['zss_cross_check']['C1'] == {'n_checked': 2, 'n_agree': 2, 'disagreements': []}
    assert agg['meta']['subsets'] == {'all': 4, 'unseen': 2, 'unseen_package_only': 1}
    # Resampling schemes: the package's full grouping (primary) and screens.
    assert agg['meta']['primary_grouping'] == 'full'
    assert list(agg['bootstrap']['all']) == ['full', 'screen']
    assert agg['bootstrap']['all']['full']['n_units'] == 3          # apps a, b, c
    assert agg['bootstrap']['all']['screen']['n_units'] == 4
    assert set(c4['teds_g']['ci_mean']) == {'full', 'screen'}
    expected = np.mean([1 / 6, c1['a-screens_screenshot_2']['teds_g']
                        - c2['a-screens_screenshot_2']['teds_g']])
    for scheme in ('full', 'screen'):
        diff = agg['paired_differences']['C1-C2']['all'][scheme]['teds_g']
        assert diff['point'] == pytest.approx(expected) and diff['n_screens_both'] == 2
    # Same results with two worker processes (only meta['workers'] / times differ).
    agg2 = LE.run_evaluation(pk['data'], 'val', tmp_path / 'out2', workers=2, **kw)
    assert agg2['meta']['workers'] == 2
    assert _strip_times({k: v for k, v in agg2.items() if k != 'meta'}) == \
        _strip_times({k: v for k, v in agg.items() if k != 'meta'})
    for name in ('reference.jsonl', 'C1.jsonl', 'C4.jsonl'):
        a = [_strip_times(json.loads(x)) for x in (out / name).read_text().splitlines()]
        b = [_strip_times(json.loads(x)) for x in (tmp_path / 'out2' / name).read_text().splitlines()]
        assert a == b


def test_test_split_needs_allow_test(tmp_path):
    """The test split is refused without allow_test; unknown conditions and missing inputs are refused."""
    pk = make_package(tmp_path)
    with pytest.raises(ValueError, match='allow_test'):
        LE.run_evaluation(pk['data'], 'test', tmp_path / 'o')
    with pytest.raises(ValueError, match='unknown conditions'):
        LE.run_evaluation(pk['data'], 'val', tmp_path / 'o', conditions=['C9'])
    with pytest.raises(ValueError, match='pred_cache'):
        LE.run_evaluation(pk['data'], 'val', tmp_path / 'o', conditions=['C4'])
    with pytest.raises(ValueError, match='v110_dir'):
        LE.run_evaluation(pk['data'], 'val', tmp_path / 'o', conditions=['C1p'])
    with pytest.raises(ValueError, match='box_repr'):
        LE.run_evaluation(pk['data'], 'val', tmp_path / 'o', conditions=['C1'], box_repr='x')
    assert LE.main(['--data-dir', pk['data'], '--split', 'test', '--out', str(tmp_path / 'o'),
                    '--conditions', 'C1']) == 2


def test_cli_runs_gt_conditions(tmp_path, capsys):
    """ui2compose-eval-layout writes the aggregate for the GT conditions without predictions."""
    pytest.importorskip('apted')
    pk = make_package(tmp_path)
    code = LE.main(['--data-dir', pk['data'], '--out', str(tmp_path / 'cli'), '--conditions',
                    'C1,C2', '--bootstrap', '20', '--zss-check', '0', '--no-trees'])
    assert code == 0
    agg = json.loads((tmp_path / 'cli' / 'aggregate.json').read_text())
    assert agg['meta']['params']['cut_tol_frac'] == LE.DEFAULT_CUT_TOL_FRAC
    assert agg['meta']['box_repr'] == 'composite'
    assert 'tree' not in json.loads((tmp_path / 'cli' / 'C1.jsonl').read_text().splitlines()[0])
    assert 'C1:' in capsys.readouterr().out


def test_console_script_is_registered():
    """pyproject.toml declares ui2compose-eval-layout -> layout_eval:main."""
    text = (REPO / 'pyproject.toml').read_text(encoding='utf-8')
    assert 'ui2compose-eval-layout = "ui2compose.evaluation.layout_eval:main"' in text
