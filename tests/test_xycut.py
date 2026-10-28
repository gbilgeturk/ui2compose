# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Tests of the layout stage: units, XY-cut, fallbacks, the layout tree and explain_path.

Pure tests: only the layout functions of generate_compose_code are imported
(no torch / ultralytics).
"""

import copy
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pytest

from ui2compose.infer.generate_compose_code import (
    DEFAULT_CUT_TOL_FRAC,
    _gaps_between,
    _split_on_gap,
    build_layout_tree,
    explain_path,
)

IMG = (1200, 1920)                      # ReDraw screenshot size
# Tolerance the tolerance-cut and trace tests exercise (0.0025, the value selected on the
# screenshot-level split). The shipped default, DEFAULT_CUT_TOL_FRAC, is 0.0 since the
# re-selection on the app-level split (results/eval_appsplit/selected_thresholds.json).
TOL_FRAC = 0.0025
TOL_Y = TOL_FRAC * IMG[1]   # 4.8 px
TOL_X = TOL_FRAC * IMG[0]   # 3.0 px
NODE_KEYS = {'type', 'bbox', 'members', 'rule', 'axis', 'gaps', 'tolerance_px', 'children'}
RULES = {'gap', 'gap_tolerance', 'fallback_y_sorted', 'least_overlap', 'containment',
         'list', 'flat', 'leaf'}


# ── helpers ────────────────────────────────────────────────

def node(i: int, bbox: List[float], cls: str = 'TextView', conf: Optional[float] = 0.9,
         children: Optional[List[Dict]] = None, **extra) -> Dict:
    """Builds a hierarchy node.

    Input:  i — id; bbox — [x1, y1, x2, y2]; cls — class; conf — confidence
            (None: no 'confidence' key); children — child nodes; extra — more keys
    Output: node dictionary
    """
    n = {'id': i, 'class': cls, 'bbox': list(bbox), 'children': list(children or [])}
    if conf is not None:
        n['confidence'] = conf
    n.update(extra)
    return n


def graph(roots: List[Dict], **extra) -> Dict:
    """Wraps root nodes into a minimal UI graph.

    Input:  roots — root nodes; extra — more top-level keys (e.g. image_size)
    Output: graph dictionary
    """
    g = {'nodes': [], 'edges': [], 'hierarchy': {'roots': roots}}
    g.update(extra)
    return g


def emitted_ids(tree: Dict) -> List:
    """Lists the ids the code generator writes: leaves and LazyColumn list containers.

    Input:  tree — layout node
    Output: list of ids (with repetitions, if any)
    """
    out = []
    if tree['type'] == 'leaf':
        out.append(tree['node']['id'])
    if tree['type'] == 'LazyColumn':
        out.append(tree['container']['id'])
    for c in tree['children']:
        out.extend(emitted_ids(c))
    return out


def signature(tree: Dict) -> Tuple:
    """Reduces a layout tree to its structure (types, rules, axes, members, split rules).

    Input:  tree — layout node
    Output: nested tuple that ignores pixel values
    """
    return (tree['type'], tree['rule'], tree['axis'], tuple(tree['members']),
            tree.get('split_rule'), tuple(signature(c) for c in tree['children']))


def walk(tree: Dict):
    """Yields every node of a layout tree (pre-order).

    Input:  tree — layout node
    Output: generator of layout nodes
    """
    yield tree
    for c in tree['children']:
        yield from walk(c)


def scale_graph(g: Dict, s: float) -> Dict:
    """Scales every bbox of a graph (hierarchy and nodes) by a factor.

    Input:  g — graph; s — scale factor
    Output: scaled deep copy
    """
    g = copy.deepcopy(g)

    def sc(n):
        """Scales one node and its subtree in place.

        Input:  n — node
        Output: none
        """
        n['bbox'] = [v * s for v in n['bbox']]
        for c in n.get('children', []):
            sc(c)
    for r in g['hierarchy']['roots']:
        sc(r)
    return g


def v110_xy_cut(boxes: List[Dict], prefer: str = 'y') -> Tuple:
    """Reference copy of the v1.1.0 XY-cut (zero tolerance, y-centre fallback) for regression tests.

    v1.1.0 left exact y-centre ties in the fallback in input order; the copy
    breaks them by x-centre, then id, as v1.2.0 documents.

    Input:  boxes — leaf dictionaries with 'id' and 'bbox'; prefer — first axis
    Output: nested tuple ('Column'|'Row', children) or ('leaf', id)
    """
    if len(boxes) == 1:
        return ('leaf', boxes[0]['id'])
    for axis in (['y', 'x'] if prefer == 'y' else ['x', 'y']):
        lo, hi = (1, 3) if axis == 'y' else (0, 2)
        ordered = sorted(boxes, key=lambda b: b['bbox'][lo])
        groups, cur, running = [], [ordered[0]], ordered[0]['bbox'][hi]
        for b in ordered[1:]:
            if b['bbox'][lo] > running:
                groups.append(cur)
                cur, running = [b], b['bbox'][hi]
            else:
                cur.append(b)
                running = max(running, b['bbox'][hi])
        groups.append(cur)
        if len(groups) >= 2:
            nxt = 'x' if axis == 'y' else 'y'
            return ('Column' if axis == 'y' else 'Row',
                    tuple(v110_xy_cut(g, nxt) for g in groups))
    ordered = sorted(boxes, key=lambda b: ((b['bbox'][1] + b['bbox'][3]) / 2,
                                           (b['bbox'][0] + b['bbox'][2]) / 2, b['id']))
    return ('Column', tuple(('leaf', b['id']) for b in ordered))


def as_v110_shape(tree: Dict) -> Tuple:
    """Reduces a v1.2.0 layout tree to the v1.1.0 reference shape.

    Input:  tree — layout node of a leaf-only graph
    Output: nested tuple comparable with v110_xy_cut
    """
    if tree['type'] == 'leaf':
        return ('leaf', tree['node']['id'])
    return (tree['type'], tuple(as_v110_shape(c) for c in tree['children']))


def screen_graph() -> Dict:
    """A realistic screen: title, tolerance-cut row pair, button row, image container, list, overlaps.

    Input:  none
    Output: v1.2.0-style graph with synthetic wrappers and image_size 1200x1920
    """
    roots = [
        node(-1, [0, 0, 1200, 1920], 'SyntheticColumn', None, synthetic=True, children=[
            node(0, [100, 40, 1100, 140], 'TextView', 0.95),
            # Two rows overlapping by 4 px (< TOL_Y = 9.6 px) -> tolerance cut
            node(1, [100, 200, 500, 304], 'TextView', 0.8),
            node(2, [700, 200, 1100, 304], 'ImageView', 0.7),
            node(3, [100, 300, 500, 400], 'TextView', 0.6),
            node(4, [700, 300, 1100, 400], 'TextView', 0.6),
            # Button row
            node(-2, [100, 500, 1100, 600], 'SyntheticRow', None, synthetic=True, children=[
                node(5, [100, 500, 500, 600], 'Button', 0.9),
                node(6, [700, 500, 1100, 600], 'Button', 0.9),
            ]),
            # Image container with a play icon and a caption
            node(7, [100, 700, 1100, 1100], 'ImageView', 0.5, children=[
                node(8, [500, 800, 700, 1000], 'ImageView', 0.45),
                node(9, [900, 1040, 1080, 1090], 'TextView', 0.4),
            ]),
            # List with three stacked items
            node(10, [0, 1200, 1200, 1700], 'ListView', 0.9, children=[
                node(11, [20, 1220, 1180, 1320], 'TextView', 0.8),
                node(12, [20, 1360, 1180, 1460], 'TextView', 0.8),
                node(13, [20, 1500, 1180, 1600], 'TextView', 0.8),
            ]),
            # Three mutually overlapping boxes (no cut on either axis)
            node(14, [0, 1750, 600, 1900], 'TextView', 0.3),
            node(15, [300, 1780, 900, 1910], 'TextView', 0.3),
            node(16, [550, 1760, 1200, 1890], 'ImageView', 0.3),
        ])]
    return graph(roots, image_size=list(IMG))


# ── _split_on_gap ──────────────────────────────────────────

def test_split_on_gap_clean_cut_and_gap_intervals():
    """Three vertically separated boxes give three groups and their separating gaps."""
    boxes = [{'id': 2, 'bbox': [0, 500, 10, 600]}, {'id': 0, 'bbox': [0, 0, 10, 100]},
             {'id': 1, 'bbox': [0, 200, 10, 300]}]
    groups = _split_on_gap(boxes, 'y', 0.0)
    assert [[b['id'] for b in g] for g in groups] == [[0], [1], [2]]
    assert _gaps_between(groups, 'y') == [[100, 200], [300, 500]]
    assert _split_on_gap(boxes, 'x', 0.0) is None          # all share the same x range


def test_split_on_gap_zero_tolerance_needs_a_strictly_positive_gap():
    """At tolerance 0 touching boxes are not cut (v1.1.0 rule); any positive tolerance cuts them."""
    boxes = [{'id': 0, 'bbox': [0, 0, 100, 10]}, {'id': 1, 'bbox': [100, 0, 200, 10]}]
    assert _split_on_gap(boxes, 'x', 0.0) is None
    groups = _split_on_gap(boxes, 'x', 0.5)
    assert _gaps_between(groups, 'x') == [[100, 100]]


def test_split_on_gap_tolerates_overlap_up_to_tol_px():
    """A cut is allowed where next.lo > running_max - tol_px."""
    boxes = [{'id': 0, 'bbox': [0, 0, 10, 105]}, {'id': 1, 'bbox': [0, 100, 10, 200]}]   # overlap 5 px
    assert _split_on_gap(boxes, 'y', 4.9) is None
    groups = _split_on_gap(boxes, 'y', 5.1)
    assert [[b['id'] for b in g] for g in groups] == [[0], [1]]
    assert _gaps_between(groups, 'y') == [[105, 100]]       # a > b: the overlap


def test_split_on_gap_single_box_has_no_cut():
    """Fewer than two boxes cannot be split."""
    assert _split_on_gap([{'id': 0, 'bbox': [0, 0, 1, 1]}], 'y', 10.0) is None


# ── XY-cut tree ────────────────────────────────────────────

def test_clean_cuts_build_nested_column_and_rows():
    """Separated rows give a gap Column whose row groups are gap Rows (axes alternate)."""
    roots = [node(0, [100, 40, 1100, 140]),
             node(1, [100, 200, 500, 300], 'Button'), node(2, [700, 200, 1100, 300], 'Button'),
             node(3, [100, 400, 1100, 500], 'EditText')]
    tree = build_layout_tree(graph(roots), image_size=IMG, cut_tol_frac=TOL_FRAC)
    assert (tree['type'], tree['rule'], tree['axis']) == ('Column', 'gap', 'y')
    assert tree['gaps'] == [[140, 200], [300, 400]]
    assert tree['tolerance_px'] == pytest.approx(TOL_Y)
    row = tree['children'][1]
    assert (row['type'], row['rule'], row['axis'], row['members']) == ('Row', 'gap', 'x', [1, 2])
    assert row['gaps'] == [[500, 700]] and row['tolerance_px'] == pytest.approx(TOL_X)
    assert [c['type'] for c in tree['children']] == ['leaf', 'Row', 'leaf']


def test_tolerance_cut_versus_zero_tolerance():
    """A 4 px overlap is cut with a tolerance of 0.0025 (gap_tolerance) but not with cut_tol_frac=0."""
    roots = [node(0, [0, 0, 100, 104]), node(1, [0, 100, 100, 200])]
    tree = build_layout_tree(graph(roots), image_size=IMG, cut_tol_frac=TOL_FRAC)
    assert (tree['type'], tree['rule'], tree['axis']) == ('Column', 'gap_tolerance', 'y')
    a, b = tree['gaps'][0]
    assert (a, b) == (104, 100) and a > b
    zero = build_layout_tree(graph(roots), image_size=IMG, cut_tol_frac=0.0)
    assert zero['rule'] == 'fallback_y_sorted' and zero['gaps'] == []
    # Touching boxes (zero-width gap): v1.1.0 did not cut them; with tolerance they are cut
    touching = [node(0, [0, 0, 100, 100]), node(1, [0, 100, 100, 200])]
    assert build_layout_tree(graph(touching), image_size=IMG, cut_tol_frac=0.0)['rule'] == 'fallback_y_sorted'
    t = build_layout_tree(graph(touching), image_size=IMG, cut_tol_frac=TOL_FRAC)
    assert t['rule'] == 'gap_tolerance' and t['gaps'] == [[100, 100]]
    assert explain_path(t, 1)[0].endswith("after gap y in [100,100] (touching)")
    # A real gap stays rule 'gap' with or without tolerance
    sep = [node(0, [0, 0, 100, 90]), node(1, [0, 100, 100, 200])]
    assert build_layout_tree(graph(sep), image_size=IMG, cut_tol_frac=TOL_FRAC)['rule'] == 'gap'
    assert build_layout_tree(graph(sep), image_size=IMG, cut_tol_frac=0.0)['rule'] == 'gap'


@pytest.mark.parametrize('grid', [False, True])
def test_zero_tolerance_reproduces_v110_on_leaf_only_graphs(grid):
    """With cut_tol_frac=0 and fallback y_sorted the tree equals the v1.1.0 XY-cut on random leaf boxes.

    grid=True puts the boxes on a 50 px grid, so touching boxes and ties occur.
    """
    rng = random.Random(f"v110-{grid}")
    for _ in range(300):
        boxes = []
        for i in range(rng.randint(1, 14)):
            if grid:
                x1, y1 = 50 * rng.randint(0, 20), 50 * rng.randint(0, 34)
                w, h = 50 * rng.randint(1, 6), 50 * rng.randint(1, 6)
            else:
                x1, y1 = rng.uniform(0, 1100), rng.uniform(0, 1800)
                w, h = rng.uniform(5, 300), rng.uniform(5, 300)
            boxes.append(node(i, [x1, y1, x1 + w, y1 + h]))
        tree = build_layout_tree(graph(copy.deepcopy(boxes)), image_size=IMG, cut_tol_frac=0.0)
        assert as_v110_shape(tree) == v110_xy_cut(boxes)


def test_fallback_y_sorted_orders_members_by_y_then_x_centre():
    """Mutually overlapping boxes give one Column ordered by y-centre, ties broken by x-centre."""
    roots = [node(0, [0, 50, 600, 250]), node(1, [300, 0, 900, 200]),     # cy 150, 100
             node(2, [500, 60, 1100, 240]), node(3, [100, 60, 700, 240])]  # cy 150, 150
    tree = build_layout_tree(graph(roots), image_size=IMG)
    assert (tree['type'], tree['rule'], tree['axis']) == ('Column', 'fallback_y_sorted', 'y')
    assert [c['node']['id'] for c in tree['children']] == [1, 0, 3, 2]
    assert tree['gaps'] == []


def test_least_overlap_split_and_recursion():
    """least_overlap chooses the split with minimum total pairwise overlap and recurses with the other axis first."""
    roots = [node(0, [0, 0, 100, 100]), node(1, [50, 50, 150, 150]), node(2, [120, 120, 220, 220])]
    tree = build_layout_tree(graph(roots), image_size=IMG, cut_tol_frac=0.0, fallback='least_overlap')
    # y: {0,1}|{2} costs ov(1,2)=30 < {0}|{1,2} costs ov(0,1)=50; x ties with y -> y (preferred) wins
    assert (tree['type'], tree['rule'], tree['axis']) == ('Column', 'least_overlap', 'y')
    assert tree['overlap_px'] == pytest.approx(30)
    assert tree['gaps'] == [[150, 120]]
    first, second = tree['children']
    assert (first['type'], first['rule'], first['axis'], first['members']) == ('Row', 'least_overlap', 'x', [0, 1])
    assert first['overlap_px'] == pytest.approx(50)
    assert second['type'] == 'leaf' and second['members'] == [2]
    # The default fallback keeps the v1.1.0 single y-sorted Column
    y_sorted = build_layout_tree(graph(roots), image_size=IMG, cut_tol_frac=0.0)
    assert y_sorted['rule'] == 'fallback_y_sorted'
    assert [c['node']['id'] for c in y_sorted['children']] == [0, 1, 2]


def test_least_overlap_equal_cost_goes_to_the_shorter_prefix():
    """Two splits with the same total overlap on the same axis: the shorter prefix wins (documented tie-break)."""
    # y overlaps: 0-1 = 10 px, 1-2 = 10 px, 0-2 = 0 -> {0}|{1,2} and {0,1}|{2} both cost 10;
    # on x all three intervals coincide (cost 200), so the split is along y.
    roots = [node(0, [0, 0, 100, 100]), node(1, [0, 90, 100, 200]), node(2, [0, 190, 100, 300])]
    tree = build_layout_tree(graph(roots), image_size=IMG, cut_tol_frac=0.0, fallback='least_overlap')
    assert (tree['rule'], tree['axis']) == ('least_overlap', 'y')
    assert tree['overlap_px'] == pytest.approx(10)
    assert [c['members'] for c in tree['children']] == [[0], [1, 2]]


@pytest.mark.parametrize('fallback', ['y_sorted', 'least_overlap'])
def test_identical_boxes_terminate_and_keep_every_id(fallback):
    """Degenerate input (all boxes identical) terminates and every id appears once."""
    roots = [node(i, [10, 10, 200, 200]) for i in range(6)]
    tree = build_layout_tree(graph(roots), image_size=IMG, fallback=fallback)
    assert sorted(emitted_ids(tree)) == list(range(6))


@pytest.mark.parametrize('fallback', ['y_sorted', 'least_overlap'])
@pytest.mark.parametrize('frac', [0.0, 0.005, 0.05])
def test_random_forests_emit_every_id_exactly_once(fallback, frac):
    """Random forests (with containers, lists, synthetic wrappers and duplicates) keep every id once."""
    rng = random.Random(f"{fallback}-{frac}")      # str seeds are deterministic
    classes = ['TextView', 'ImageView', 'Button', 'ListView', 'RecyclerView', 'CheckBox']
    for _ in range(60):
        next_id = [0]

        def make(depth):
            """Builds a random subtree.

            Input:  depth — remaining depth
            Output: node dictionary (may be synthetic)
            """
            x1, y1 = rng.uniform(0, 1000), rng.uniform(0, 1700)
            bbox = [x1, y1, x1 + rng.uniform(1, 400), y1 + rng.uniform(1, 400)]
            kids = [make(depth - 1) for _ in range(rng.randint(0, 3))] if depth > 0 else []
            if rng.random() < 0.15:
                return node(-100 - next_id[0], bbox, 'SyntheticRow', None, synthetic=True, children=kids)
            i = next_id[0]
            next_id[0] += 1
            return node(i, bbox, rng.choice(classes), rng.random(), children=kids)

        roots = [make(2) for _ in range(rng.randint(1, 6))]
        # v1.1.0-style duplicate: re-list an existing subtree under another root
        if next_id[0] and rng.random() < 0.5:
            roots.append(node(10_000, [0, 0, 1200, 1920], 'ImageView', 0.2,
                              children=[copy.deepcopy(roots[0])]))
        tree = build_layout_tree(graph(roots), image_size=IMG, cut_tol_frac=frac, fallback=fallback)
        ids = emitted_ids(tree)
        assert len(ids) == len(set(ids)), "an id was emitted twice"
        assert set(ids) == set(tree['members']) == set(tree_ids(roots))


def tree_ids(roots: List[Dict]) -> List:
    """Collects the non-synthetic ids of a hierarchy.

    Input:  roots — hierarchy nodes
    Output: list of ids (unique)
    """
    out = set()

    def visit(n):
        """Adds the id of n and its subtree.

        Input:  n — node
        Output: none
        """
        if n['id'] >= 0 and not n.get('synthetic'):
            out.add(n['id'])
        for c in n.get('children', []):
            visit(c)
    for r in roots:
        visit(r)
    return sorted(out)


@pytest.mark.parametrize('fallback', ['y_sorted', 'least_overlap'])
@pytest.mark.parametrize('s', [0.5, 1.5])
def test_scale_invariance(s, fallback):
    """Scaling boxes and image size by 0.5 / 1.5 leaves types, members, rules and axes unchanged."""
    g = screen_graph()
    base = build_layout_tree(g, fallback=fallback, cut_tol_frac=TOL_FRAC)
    scaled = scale_graph(g, s)
    scaled['image_size'] = [IMG[0] * s, IMG[1] * s]
    other = build_layout_tree(scaled, fallback=fallback, cut_tol_frac=TOL_FRAC)
    assert signature(other) == signature(base)
    assert other['meta']['tolerance_px']['y'] == pytest.approx(base['meta']['tolerance_px']['y'] * s)
    # The screen exercises every rule of the xycut layout
    rules = {n['rule'] for n in walk(base)}
    expected = {'gap', 'gap_tolerance', 'containment', 'list', 'leaf'}
    expected.add('fallback_y_sorted' if fallback == 'y_sorted' else 'least_overlap')
    assert expected <= rules


# ── units: synthetic nodes, duplicates, composites ─────────

def test_synthetic_nodes_are_transparent():
    """SyntheticRow/Column (negative id, 'Synthetic*' class or 'synthetic': True) do not change the tree."""
    plain = [node(0, [0, 0, 100, 100]), node(1, [200, 0, 300, 100]), node(2, [0, 200, 300, 300])]
    wrapped = [node(-5, [0, 0, 300, 300], 'SyntheticColumn', None, children=[
        node(-6, [0, 0, 300, 100], 'SyntheticRow', None, children=copy.deepcopy(plain[:2])),
        node(77, [0, 200, 300, 300], 'Group', None, synthetic=True, children=[copy.deepcopy(plain[2])]),
    ])]
    a = build_layout_tree(graph(plain), image_size=IMG)
    b = build_layout_tree(graph(wrapped), image_size=IMG)
    assert signature(a) == signature(b)
    assert 77 not in b['members'] and -5 not in b['members']


def test_duplicate_child_is_kept_under_its_first_parent_only():
    """A child listed under two parents (v1.1.0) appears once, under the first parent in DFS order."""
    shared = [node(10, [465, 562, 740, 845], 'ImageView', 0.51), node(11, [700, 862, 774, 936], 'TextView', 0.48)]
    roots = [node(-2, [70, 482, 954, 1107], 'SyntheticRow', None, children=[
        node(8, [70, 482, 954, 1087], 'ImageView', 0.38, children=copy.deepcopy(shared)),
        node(9, [322, 538, 868, 1107], 'ImageView', 0.31, children=copy.deepcopy(shared)),
    ])]
    tree = build_layout_tree(graph(roots), image_size=IMG)
    assert Counter(emitted_ids(tree)) == Counter({8: 1, 9: 1, 10: 1, 11: 1})
    box = next(n for n in walk(tree) if n['type'] == 'Box')
    assert box['container']['id'] == 8 and box['members'] == [8, 10, 11]
    leaf9 = next(n for n in walk(tree) if n['type'] == 'leaf' and n['node']['id'] == 9)
    assert leaf9['rule'] == 'leaf'


def test_children_listed_only_under_a_later_occurrence_are_not_lost():
    """Children that only a later occurrence of an id lists are attached to the kept occurrence."""
    roots = [node(1, [0, 0, 500, 500], 'ImageView', children=[node(2, [10, 10, 50, 50])]),
             node(-1, [0, 0, 500, 500], 'SyntheticRow', None, children=[
                 node(1, [0, 0, 500, 500], 'ImageView', children=[node(3, [100, 100, 150, 150])])])]
    tree = build_layout_tree(graph(roots), image_size=IMG)
    assert sorted(emitted_ids(tree)) == [1, 2, 3]
    assert tree['type'] == 'Box' and tree['members'] == [1, 2, 3]


def test_composite_takes_part_with_its_own_box():
    """At its level a composite uses its own bbox, even if a child sticks out of it."""
    roots = [node(0, [0, 0, 1200, 50]),
             node(1, [0, 100, 1200, 600], 'ImageView', children=[node(2, [400, 550, 600, 690])]),
             node(3, [0, 650, 1200, 700])]
    tree = build_layout_tree(graph(roots), image_size=IMG)
    assert tree['rule'] == 'gap' and [c['members'] for c in tree['children']] == [[0], [1, 2], [3]]
    assert tree['gaps'] == [[50, 100], [600, 650]]
    box = tree['children'][1]
    assert box['type'] == 'Box' and box['bbox'] == [0, 100, 1200, 690]   # union of members


def test_list_children_become_items():
    """A ListView with vertically stacked children gives one item per group; side-by-side children one item."""
    stacked = [node(0, [0, 0, 1200, 900], 'ListView', children=[
        node(1, [0, 0, 300, 100], 'ImageView'), node(2, [400, 0, 1200, 100]),
        node(3, [0, 200, 300, 300], 'ImageView'), node(4, [400, 200, 1200, 300])])]
    tree = build_layout_tree(graph(stacked), image_size=IMG)
    assert (tree['type'], tree['rule'], tree['split_rule']) == ('LazyColumn', 'list', 'gap')
    assert [c['type'] for c in tree['children']] == ['Row', 'Row']
    assert tree['gaps'] == [[100, 200]] and tree['container']['id'] == 0
    side = [node(0, [0, 0, 1200, 900], 'RecyclerView', children=[
        node(1, [0, 0, 300, 100]), node(2, [400, 0, 1200, 100])])]
    tree = build_layout_tree(graph(side), image_size=IMG)
    assert tree['split_rule'] == 'single_item' and [c['type'] for c in tree['children']] == ['Row']


def test_plain_detection_list_is_accepted():
    """The layout stage runs on a plain list of detector/GT boxes ('class_name', no hierarchy)."""
    dets = [{'id': 0, 'class_name': 'Button', 'confidence': 0.7, 'bbox': [0, 0, 100, 50]},
            {'id': 1, 'class_name': 'TextView', 'confidence': 0.6, 'bbox': [0, 100, 100, 150]}]
    tree = build_layout_tree(dets, image_size=IMG)
    assert tree['rule'] == 'gap' and [c['node']['class'] for c in tree['children']] == ['Button', 'TextView']
    assert build_layout_tree({'hierarchy': {'roots': dets}}, image_size=IMG)['members'] == [0, 1]


# ── flat mode ──────────────────────────────────────────────

def test_flat_mode_is_one_column_of_all_detections():
    """layout='flat': every detection is a leaf of one Column ordered by y-centre, then x-centre."""
    g = screen_graph()
    tree = build_layout_tree(g, layout='flat')
    assert (tree['type'], tree['rule'], tree['axis']) == ('Column', 'flat', 'y')
    assert all(c['type'] == 'leaf' and c['rule'] == 'leaf' for c in tree['children'])
    ids = [c['node']['id'] for c in tree['children']]
    assert sorted(ids) == list(range(17))
    keys = [((c['bbox'][1] + c['bbox'][3]) / 2, (c['bbox'][0] + c['bbox'][2]) / 2) for c in tree['children']]
    assert keys == sorted(keys)
    by_id = {c['node']['id']: c for c in tree['children']}
    assert by_id[10]['flattened_children'] == [11, 12, 13]
    assert by_id[7]['flattened_children'] == [8, 9] and by_id[0]['flattened_children'] == []
    one = build_layout_tree(graph([node(0, [0, 0, 10, 10])]), image_size=IMG, layout='flat')
    assert one['type'] == 'Column' and len(one['children']) == 1


def test_empty_graph_gives_empty_column():
    """No detections: an empty Column (rule 'flat') in both layout modes."""
    for layout in ('xycut', 'flat'):
        tree = build_layout_tree(graph([]), layout=layout)
        assert (tree['type'], tree['rule'], tree['members'], tree['children']) == ('Column', 'flat', [], [])


# ── trace fields ───────────────────────────────────────────

def test_trace_fields_are_complete_and_consistent():
    """Every node carries the documented fields; members/bbox are unions; leaves carry 'node'."""
    tree = build_layout_tree(screen_graph(), cut_tol_frac=TOL_FRAC)
    for n in walk(tree):
        assert NODE_KEYS <= set(n)
        assert n['rule'] in RULES
        assert n['members'] == sorted(n['members'])
        assert n['axis'] in ('x', 'y', None)
        if n['type'] == 'leaf':
            assert set(n['node']) == {'id', 'class', 'bbox', 'confidence'}
            assert n['members'] == [n['node']['id']] and n['children'] == []
            continue
        child_members = sorted(m for c in n['children'] for m in c['members'])
        if n['type'] == 'LazyColumn':
            child_members = sorted(child_members + [n['container']['id']])
        assert n['members'] == child_members
        boxes = [c['bbox'] for c in n['children']] + ([n['container']['bbox']] if 'container' in n else [])
        assert n['bbox'] == [min(b[0] for b in boxes), min(b[1] for b in boxes),
                             max(b[2] for b in boxes), max(b[3] for b in boxes)]
        if n['rule'] in ('gap', 'gap_tolerance'):
            assert len(n['gaps']) == len(n['children']) - 1
            assert n['tolerance_px'] == pytest.approx(TOL_Y if n['axis'] == 'y' else TOL_X)
            assert (n['rule'] == 'gap_tolerance') == any(a >= b for a, b in n['gaps'])
        if n['type'] == 'Box':
            assert n['children'][0]['type'] == 'leaf'
            assert n['children'][0]['node']['id'] == n['container']['id']
    meta = tree['meta']
    assert meta == {'layout': 'xycut', 'fallback': 'y_sorted', 'cut_tol_frac': 0.0025,
                    'image_size': [1200.0, 1920.0], 'image_size_source': 'graph',
                    'tolerance_px': {'x': pytest.approx(TOL_X), 'y': pytest.approx(TOL_Y)},
                    'n_detections': 17}


def test_default_tolerance_is_the_app_split_selection():
    """The shipped cut_tol_frac is 0.0 (selected on the app-level val split): touching rows need the fallback."""
    assert DEFAULT_CUT_TOL_FRAC == 0.0
    tree = build_layout_tree(screen_graph())
    assert tree['meta']['cut_tol_frac'] == 0.0 and tree['meta']['tolerance_px'] == {'x': 0.0, 'y': 0.0}
    assert 'gap_tolerance' not in {n['rule'] for n in walk(tree)}


def test_confidence_is_taken_from_graph_nodes_when_missing():
    """v1.1.0 hierarchy nodes have no confidence: it is looked up in graph['nodes'] (None if absent)."""
    g = graph([node(3, [0, 0, 10, 10], conf=None), node(4, [0, 50, 10, 60], conf=None)])
    g['nodes'] = [{'id': 3, 'class_name': 'TextView', 'confidence': 0.48, 'bbox': [0, 0, 10, 10]}]
    tree = build_layout_tree(g, image_size=IMG)
    confs = {c['node']['id']: c['node']['confidence'] for c in tree['children']}
    assert confs == {3: 0.48, 4: None}


# ── image size and parameters ──────────────────────────────

def test_image_size_resolution_order():
    """image_size argument > graph['image_size'] > box extent; the source is recorded."""
    g = graph([node(0, [0, 0, 600, 960])], image_size=[1200, 1920])
    assert build_layout_tree(g)['meta']['image_size_source'] == 'graph'
    t = build_layout_tree(g, image_size=(2400, 3840), cut_tol_frac=TOL_FRAC)
    assert t['meta']['image_size'] == [2400.0, 3840.0] and t['meta']['image_size_source'] == 'argument'
    assert t['meta']['tolerance_px'] == {'x': pytest.approx(6.0), 'y': pytest.approx(9.6)}
    del g['image_size']
    t = build_layout_tree(g)
    assert t['meta']['image_size'] == [600.0, 960.0] and t['meta']['image_size_source'] == 'inferred_from_boxes'
    g['image_size'] = [None, None]                         # unknown size in the graph -> inferred
    assert build_layout_tree(g)['meta']['image_size_source'] == 'inferred_from_boxes'


@pytest.mark.parametrize('kwargs', [
    {'layout': 'grid'}, {'fallback': 'random'}, {'cut_tol_frac': -0.01},
    {'cut_tol_frac': float('nan')}, {'image_size': (0, 100)}, {'image_size': (100,)},
])
def test_invalid_parameters_raise(kwargs):
    """Unknown modes, negative/NaN tolerance and non-positive image sizes are rejected."""
    with pytest.raises(ValueError):
        build_layout_tree(graph([node(0, [0, 0, 10, 10])]), **kwargs)


@pytest.mark.parametrize('bad', [{'id': 0, 'class': 'TextView'},
                                 {'id': 0, 'class': 'TextView', 'bbox': [10, 0, 5, 5]},
                                 {'id': 0, 'bbox': [0, 0, 5, 5]}])
def test_invalid_nodes_raise(bad):
    """Nodes without bbox / class, or with inverted boxes, raise a clear ValueError."""
    with pytest.raises(ValueError):
        build_layout_tree([bad], image_size=IMG)


# ── explain_path ───────────────────────────────────────────

def test_explain_path_names_rules_and_evidence():
    """explain_path lists one line per node from the root, with rule and geometric evidence."""
    tree = build_layout_tree(screen_graph(), cut_tol_frac=TOL_FRAC)
    lines = explain_path(tree, 9)
    assert lines[0] == ("Column (rule=gap_tolerance, axis=y, 6 cuts, 1 within tolerance 4.8 px) "
                        "-> group 5 of 7, between gaps y in [600,700] and [1100,1200]")
    assert lines[1] == ("Box (rule=containment, container ImageView#7 (conf 0.50) bbox "
                        "[100,700,1100,1100]) -> the contained children")
    assert lines[2] == ("Column (rule=gap, axis=y, 1 cut, tolerance 4.8 px) -> group 2 of 2, "
                        "after gap y in [1000,1040]")
    assert lines[3] == "leaf TextView#9 (conf 0.40, bbox [900,1040,1080,1090])"

    tol = explain_path(tree, 3)
    assert tol[0].endswith("group 3 of 7, between gaps y in [304,300] (overlap 4 px) and [400,500]")

    fb = explain_path(tree, 15)
    assert fb[1].startswith("Column (rule=fallback_y_sorted: no cut along y (tolerance 4.8 px) "
                            "or x (tolerance 3 px); members ordered by y-centre) -> position")

    item = explain_path(tree, 12)
    assert item[1].startswith("LazyColumn (rule=list, container ListView#10 (conf 0.90)")
    assert "items split by gap along y" in item[1] and item[1].endswith(
        "item 2 of 3, between gaps y in [1320,1360] and [1460,1500]")
    assert explain_path(tree, 10)[-1].endswith("-> this detection is the list container itself")
    assert explain_path(tree, 7)[1].endswith("-> the container widget itself")


@pytest.mark.parametrize('a, b, text', [
    (119.3864, 117.6465, '[119.39,117.65] (overlap 1.74 px)'),   # Shazam tab bar (README trace)
    (304, 300, '[304,300] (overlap 4 px)'),
    (100.0, 100.0, '[100,100] (touching)'),
    (612.0, 640.04, '[612,640]'),
])
def test_fmt_gap_examples(a, b, text):
    """A cut is printed as [end of the earlier group, start of the next one]; overlaps get two decimals."""
    from ui2compose.infer.generate_compose_code import _fmt_gap
    assert _fmt_gap(a, b) == text


def test_fmt_gap_printed_overlap_equals_the_difference_of_the_printed_endpoints():
    """For any overlapping cut, a - b computed from the printed numbers is exactly the printed overlap."""
    import re
    from decimal import Decimal

    from ui2compose.infer.generate_compose_code import _fmt_gap
    rng = random.Random(7)
    for _ in range(2000):
        b = rng.uniform(0, 2000)
        a = b + rng.uniform(0.001, 12)
        m = re.fullmatch(r'\[([\d.]+),([\d.]+)\] \(overlap ([\d.]+) px\)', _fmt_gap(a, b))
        assert m, _fmt_gap(a, b)
        assert Decimal(m.group(1)) - Decimal(m.group(2)) == Decimal(m.group(3))
        assert abs(float(m.group(3)) - (a - b)) <= 0.01 + 1e-9


def test_explain_path_least_overlap_and_flat():
    """least_overlap lines give the split and total overlap; flat lines give the ordering."""
    roots = [node(0, [0, 0, 100, 100]), node(1, [50, 50, 150, 150]), node(2, [120, 120, 220, 220])]
    tree = build_layout_tree(graph(roots), image_size=IMG, cut_tol_frac=0.0, fallback='least_overlap')
    lines = explain_path(tree, 2)
    assert lines[0] == ("Column (rule=least_overlap, axis=y: no cut along y (tolerance 0 px) or x "
                        "(tolerance 0 px); split between y 150 (end of group 1) and 120 (start of "
                        "group 2), total pairwise overlap 30 px) -> group 2 of 2")
    flat = build_layout_tree(graph(roots), image_size=IMG, layout='flat')
    assert explain_path(flat, 1) == [
        "Column (rule=flat: all 3 detections ordered by y-centre, then x-centre; containment "
        "ignored) -> position 2 of 3",
        "leaf TextView#1 (conf 0.90, bbox [50,50,150,150])"]


def test_explain_path_unknown_id_raises_clear_error():
    """An id that is not in the tree raises ValueError naming the id and the known ids."""
    tree = build_layout_tree(graph([node(0, [0, 0, 10, 10]), node(1, [0, 20, 10, 30])]), image_size=IMG)
    with pytest.raises(ValueError, match=r"detection id 99 is not in the layout tree \(known ids: 0, 1\)"):
        explain_path(tree, 99)
    with pytest.raises(ValueError):
        explain_path(tree, -1)                              # synthetic ids are not detections
