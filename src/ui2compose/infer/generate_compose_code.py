# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Stage 4 of the pipeline (layout and code generation): UI graph -> Jetpack Compose code.

What the stage does
-------------------
1. Units.  The containment forest in ``graph['hierarchy']['roots']`` is read.
   Synthetic grouping nodes (negative id, class name starting with
   ``Synthetic``, or ``'synthetic': True``) are transparent: they are replaced
   by their children, at every depth.  Every detection id is kept exactly once:
   the forest is walked depth-first in stored order, the first occurrence of an
   id is kept, and a later occurrence (a v1.1.0 graph can list a detection
   under two partly overlapping parents) is skipped; children listed only
   under a later occurrence are attached to the kept one.  After this step each
   top-level node is a *unit*; a unit with children is a *composite*.
2. Layout (``layout='xycut'``, default).  The units of one level are grouped by
   recursive XY-cut on their own bounding boxes (a composite takes part with
   its own box).  A cut along an axis is allowed where the next box (sorted by
   its start) starts after the running maximum end minus a tolerance of
   ``cut_tol_frac`` times the image extent along that axis; a cut along y gives
   a Column, along x a Row, and the axes alternate going down (y first at the
   root and inside every composite).  When neither axis can be cut, the
   ``fallback`` decides: ``'y_sorted'`` (v1.1.0 behaviour) emits a Column of the
   units ordered by y-centre; ``'least_overlap'`` splits at the position with
   the least total pairwise overlap and recurses.  Each composite is laid out
   independently from its children:

   * ListView / RecyclerView -> ``LazyColumn`` with a height bound derived from
     the detected box; every top-level group of the children's vertical
     layout becomes one ``item { }`` (rule ``'list'``);
   * any other class -> ``Box { <container's own widget>; <children layout> }``
     (rule ``'containment'``), e.g. an ImageView with a play icon inside.

   ``layout='flat'`` is a baseline: one Column holding every detection as a
   leaf, ordered by y-centre, then x-centre (rule ``'flat'``); containment is
   ignored.  In both layouts every detection id of the hierarchy appears in
   the code exactly once (no detection is dropped or duplicated).
3. Templates.  Each leaf is written with a fixed template for its detector
   class; all 13 classes of ``configs/dataset.yaml`` have one.  An unknown class
   name gets an explicit ``Text("Unknown: <class>")`` placeholder.  The texts,
   images and states of the templates are placeholders; only the component
   type and the layout structure are recovered from the screenshot.

The spatial edges of the graph (``graph['edges']``) are not read by this stage:
the layout is recomputed from the boxes by XY-cut.  The edges are descriptive
output of the graph stage only.

Every layout decision is recorded in a layout tree (``build_layout_tree``);
each node names its rule and the geometric evidence (cut axis, gap intervals,
tolerance, or the containing detection).  ``explain_path`` renders the path
from the root to one detection as readable lines.

Target toolchain of the emitted code: Kotlin 2.0.21 with the Compose BOM
2024.06.00 (material3 1.2.1).  Only stable APIs are used (no
``@ExperimentalMaterial3Api``) and only the imports the code uses are emitted.
"""

import argparse
import json
import math
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

#: Detector classes emitted as a LazyColumn (lists of items).
LIST_CLASSES = frozenset({'ListView', 'RecyclerView'})

#: Layout modes accepted by build_layout_tree / generate.
LAYOUT_MODES = ('xycut', 'flat')

#: Strategies used when XY-cut finds no cut on either axis.
FALLBACK_MODES = ('y_sorted', 'least_overlap')

#: XY-cut overlap tolerance as a fraction of the image extent along the cut
#: axis (height for a cut along y, width for a cut along x).  Selected on the
#: validation split of the app-level split (results/eval_appsplit/
#: selected_thresholds.json; grid 0 / 0.0025 / 0.005 / 0.01 / 0.02, a-priori
#: value 0.005); the earlier selection on the screenshot-level split
#: (results/eval/selected_thresholds.json) had chosen 0.0025.  0.0 is the
#: zero-tolerance rule of v1.1.0: a cut needs a gap (touching boxes may be cut).
DEFAULT_CUT_TOL_FRAC: float = 0.0

#: Default fallback when no cut exists ('y_sorted' is the v1.1.0 behaviour; kept
#: by the selection on the validation split against 'least_overlap',
#: results/eval_appsplit/selected_thresholds.json).
DEFAULT_FALLBACK: str = 'y_sorted'

#: Logical width (dp) of the target screen onto which the screenshot width is
#: mapped when pixel sizes are converted: dp = px * screen_width_dp / image
#: width (one uniform scale for both axes).  411 dp is a common phone width
#: (1080 px at 420 dpi); it is not the density of the screenshot.
DEFAULT_SCREEN_WIDTH_DP: float = 411.0

#: A ProgressBar box whose longer side is at most this many times its shorter
#: side counts as "roughly square" and becomes a CircularProgressIndicator;
#: otherwise a LinearProgressIndicator.  Provisional: not tuned, to be checked
#: on the validation split.
PROGRESS_SQUARE_MAX_ASPECT: float = 1.5

#: Smallest size (dp) written for a box-derived size, so that a degenerate
#: (zero-height or zero-width) box still yields a positive Compose size.
MIN_SIZE_DP: int = 1

# Import paths of the symbols the templates use (explicit imports only).
_FOUNDATION_LAYOUT = 'androidx.compose.foundation.layout.'
_MATERIAL3 = 'androidx.compose.material3.'
_RUNTIME = 'androidx.compose.runtime.'
_STATE_IMPORTS = (_RUNTIME + 'getValue', _RUNTIME + 'setValue', _RUNTIME + 'remember')

# Modifier extension functions -> import path ('weight' is a RowScope member
# and needs no import).
_MODIFIER_IMPORTS = {
    'padding': _FOUNDATION_LAYOUT + 'padding',
    'fillMaxWidth': _FOUNDATION_LAYOUT + 'fillMaxWidth',
    'size': _FOUNDATION_LAYOUT + 'size',
    'width': _FOUNDATION_LAYOUT + 'width',
    'height': _FOUNDATION_LAYOUT + 'height',
    'heightIn': _FOUNDATION_LAYOUT + 'heightIn',
    'weight': None,
}

# Rules whose Column children are stacked vertically and can therefore become
# the items of a LazyColumn.
_VERTICAL_SPLIT_RULES = ('gap', 'gap_tolerance', 'least_overlap', 'fallback_y_sorted')


# ---------------------------------------------------------------------------
# Units: flattening synthetic nodes and de-duplicating ids
# ---------------------------------------------------------------------------

def _is_synthetic(node: Dict) -> bool:
    """Tells whether a hierarchy node is a synthetic grouping node (transparent for layout).

    Input:  node — hierarchy node dictionary
    Output: True if the node has 'synthetic': True, a class name starting with
            'Synthetic', or a negative id; False otherwise
    """
    if node.get('synthetic') is True:
        return True
    cls = node.get('class', node.get('class_name'))
    if isinstance(cls, str) and cls.startswith('Synthetic'):
        return True
    nid = node.get('id')
    return isinstance(nid, (int, float)) and not isinstance(nid, bool) and nid < 0


def _roots_of(graph: Union[Dict, List[Dict]]) -> List[Dict]:
    """Returns the root node list of a UI graph, or the list itself for a plain box list.

    Input:  graph — UI graph dictionary with graph['hierarchy']['roots'], or a
            plain list of node/detection dictionaries
    Output: list of root node dictionaries
    """
    if isinstance(graph, list):
        return graph
    if isinstance(graph, dict):
        if 'hierarchy' in graph:
            return list(graph['hierarchy'].get('roots', []) or [])
        if 'roots' in graph:
            return list(graph['roots'] or [])
    raise ValueError("graph must be a dict with graph['hierarchy']['roots'] or a list of nodes")


def _make_unit(node: Dict, conf_lookup: Dict[Any, float]) -> Dict:
    """Converts one hierarchy node into an internal unit dictionary (without children yet).

    Input:  node — hierarchy or detection dictionary with 'id', 'bbox' and
            'class' (or 'class_name'); conf_lookup — id -> confidence taken from
            graph['nodes'], used when the node itself has no 'confidence'
    Output: unit dictionary {'id', 'class', 'bbox', 'confidence', 'children': []}
    """
    if 'id' not in node:
        raise ValueError(f"node without 'id': {node!r}")
    nid = node['id']
    cls = node.get('class', node.get('class_name'))
    if not isinstance(cls, str) or not cls:
        raise ValueError(f"node {nid!r} has no 'class' / 'class_name'")
    bbox = node.get('bbox')
    if bbox is None or len(bbox) != 4:
        raise ValueError(f"node {nid!r} needs a bbox [x1, y1, x2, y2], got {bbox!r}")
    bbox = [float(v) for v in bbox]
    if not all(math.isfinite(v) for v in bbox) or bbox[2] < bbox[0] or bbox[3] < bbox[1]:
        raise ValueError(f"node {nid!r} has an invalid bbox {bbox!r} (need x1<=x2, y1<=y2)")
    conf = node.get('confidence')
    if conf is None:
        conf = conf_lookup.get(nid)
    return {
        'id': nid,
        'class': cls,
        'bbox': bbox,
        'confidence': None if conf is None else float(conf),
        'children': [],
    }


def _prepare_units(graph: Union[Dict, List[Dict]]) -> Tuple[List[Dict], List[Dict]]:
    """Flattens synthetic nodes and de-duplicates ids, giving the top-level units and all detections.

    The forest is walked depth-first in stored order.  The first occurrence of
    an id is kept; a later occurrence is skipped and the children it lists that
    were not seen yet are attached to the kept occurrence.  Synthetic nodes are
    replaced by their children.  The same dictionary object reached twice (a
    shared or cyclic in-memory structure) is processed once.

    Input:  graph — UI graph dictionary or plain list of nodes
    Output: (units, detections) — units: top-level unit dictionaries (each with
            its unit children); detections: every unit once, in first-occurrence order
    """
    roots = _roots_of(graph)
    conf_lookup: Dict[Any, float] = {}
    if isinstance(graph, dict):
        for n in graph.get('nodes', []) or []:
            if isinstance(n, dict) and 'id' in n and n.get('confidence') is not None:
                conf_lookup[n['id']] = float(n['confidence'])

    by_id: Dict[Any, Dict] = {}
    detections: List[Dict] = []
    visited_objects = set()

    def visit(nodes: List[Dict], out: List[Dict]):
        """Appends the units of `nodes` to `out`, flattening synthetic nodes recursively.

        Input:  nodes — hierarchy nodes in stored order; out — children list to fill
        Output: none (side effect of filling `out`, by_id and detections)
        """
        for node in nodes:
            if id(node) in visited_objects:
                continue
            visited_objects.add(id(node))
            children = node.get('children', []) or []
            if _is_synthetic(node):
                visit(children, out)
                continue
            nid = node.get('id')
            if nid in by_id:
                # Later occurrence of an id: keep the first one, but do not lose
                # children that only this occurrence lists.
                visit(children, by_id[nid]['children'])
                continue
            unit = _make_unit(node, conf_lookup)
            by_id[unit['id']] = unit
            detections.append(unit)
            out.append(unit)
            visit(children, unit['children'])

    top: List[Dict] = []
    visit(roots, top)
    return top, detections


def _resolve_image_size(graph: Union[Dict, List[Dict]],
                        image_size: Optional[Sequence[float]],
                        detections: List[Dict]) -> Tuple[Tuple[float, float], str]:
    """Chooses the image size used for tolerances and px->dp conversion.

    Priority: the image_size argument, then graph['image_size'], then the
    extent of the boxes (max x2, max y2) when the size is unknown.

    Input:  graph — UI graph or box list; image_size — (width, height) in px or
            None; detections — all units (used only for the inferred extent)
    Output: ((width, height), source) with source in
            {'argument', 'graph', 'inferred_from_boxes', 'unknown'}
    """
    source = 'argument'
    size = image_size
    if size is None and isinstance(graph, dict):
        g_size = graph.get('image_size')
        if g_size is not None and len(g_size) == 2 and all(v is not None for v in g_size):
            size, source = g_size, 'graph'
    if size is None:
        if not detections:
            return (1.0, 1.0), 'unknown'
        size = (max(d['bbox'][2] for d in detections), max(d['bbox'][3] for d in detections))
        source = 'inferred_from_boxes'
    if len(size) != 2:
        raise ValueError(f"image_size must be (width, height), got {size!r}")
    w, h = float(size[0]), float(size[1])
    if not (math.isfinite(w) and math.isfinite(h) and w > 0 and h > 0):
        raise ValueError(f"image_size must be positive, got {size!r}")
    return (w, h), source


# ---------------------------------------------------------------------------
# XY-cut
# ---------------------------------------------------------------------------

def _axis_index(axis: str) -> Tuple[int, int]:
    """Returns the bbox indices (start, end) of an axis.

    Input:  axis — 'y' or 'x'
    Output: (1, 3) for 'y', (0, 2) for 'x'
    """
    if axis == 'y':
        return 1, 3
    if axis == 'x':
        return 0, 2
    raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


def _split_on_gap(boxes: List[Dict], axis: str, tol_px: float = 0.0) -> Optional[List[List[Dict]]]:
    """Splits boxes into >=2 groups along an axis where a gap (or an overlap of less than tol_px) separates them.

    The boxes are sorted by their start along the axis (ties: end, then id) and
    swept; a cut is placed before a box whose start is greater than the running
    maximum end of the current group minus tol_px.  tol_px = 0 is the v1.1.0
    rule (a strictly positive gap is required).
    axis='y' -> horizontal cuts (groups stacked vertically -> Column);
    axis='x' -> vertical cuts (groups side by side -> Row).

    Input:  boxes — dictionaries with a 'bbox' [x1, y1, x2, y2] (and optionally
            'id'); axis — 'y' or 'x'; tol_px — overlap tolerance in pixels (>= 0;
            a cut needs an overlap strictly less than tol_px)
    Output: list of >=2 groups (each a list of boxes, sorted by start) or None
            if no cut exists
    """
    if len(boxes) < 2:
        return None
    lo, hi = _axis_index(axis)
    ordered = sorted(boxes, key=lambda b: (b['bbox'][lo], b['bbox'][hi], _id_key(b)))
    groups: List[List[Dict]] = []
    cur: List[Dict] = [ordered[0]]
    running_max = ordered[0]['bbox'][hi]
    for b in ordered[1:]:
        if b['bbox'][lo] > running_max - tol_px:
            groups.append(cur)
            cur = [b]
            running_max = b['bbox'][hi]
        else:
            cur.append(b)
            running_max = max(running_max, b['bbox'][hi])
    groups.append(cur)
    return groups if len(groups) >= 2 else None


def _id_key(box: Dict) -> Tuple[int, Any]:
    """Returns a sort key for a box id that never compares None with a number.

    Input:  box — dictionary that may hold an 'id'
    Output: tuple usable as the last element of a sort key
    """
    nid = box.get('id')
    return (0, nid) if isinstance(nid, (int, float)) else (1, str(nid))


def _gaps_between(groups: List[List[Dict]], axis: str) -> List[List[float]]:
    """Returns the separating interval [a, b] between consecutive groups along an axis.

    a is the largest end of the earlier group and b the smallest start of the
    next group; a < b is a real gap, a >= b means the cut needed the tolerance
    (a > b: the groups overlap by a - b pixels).

    Input:  groups — groups as returned by _split_on_gap; axis — 'y' or 'x'
    Output: list of [a, b] pairs, one per cut
    """
    lo, hi = _axis_index(axis)
    gaps = []
    for g, nxt in zip(groups, groups[1:]):
        gaps.append([max(b['bbox'][hi] for b in g), min(b['bbox'][lo] for b in nxt)])
    return gaps


def _interval_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Returns the length of the intersection of two 1-D intervals.

    Input:  a, b — (start, end) intervals
    Output: overlap length (0.0 if disjoint or touching)
    """
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _least_overlap_split(boxes: List[Dict], axes: Sequence[str]) -> Tuple[str, List[List[Dict]], float]:
    """Finds the two-group split with the least total pairwise overlap (fallback 'least_overlap').

    For each axis the boxes are ordered by their centre along that axis (ties:
    start, end, id); every split of that order into a non-empty prefix and
    suffix is a candidate whose cost is the sum, over pairs (a in prefix, b in
    suffix), of the overlap length of their intervals on that axis.  The
    candidate with the minimum cost wins; ties go to the earlier axis in `axes`,
    then to the shorter prefix.

    Input:  boxes — >=2 dictionaries with 'bbox'; axes — axis order, e.g. ('y', 'x')
    Output: (axis, [prefix, suffix], total_overlap_px)
    """
    best: Optional[Tuple[Tuple[float, int, int], str, List[Dict], List[Dict]]] = None
    for rank, axis in enumerate(axes):
        lo, hi = _axis_index(axis)
        ordered = sorted(boxes, key=lambda b: ((b['bbox'][lo] + b['bbox'][hi]) / 2.0,
                                               b['bbox'][lo], b['bbox'][hi], _id_key(b)))
        iv = [(b['bbox'][lo], b['bbox'][hi]) for b in ordered]
        n = len(ordered)
        ov = [[_interval_overlap(iv[i], iv[j]) for j in range(n)] for i in range(n)]
        total = sum(ov[0][j] for j in range(1, n))          # prefix of length 1
        for k in range(1, n):
            if k > 1:
                m = k - 1                                    # element m moves to the prefix
                total += sum(ov[m][j] for j in range(k, n)) - sum(ov[i][m] for i in range(m))
            cand = (total, rank, k)
            if best is None or cand < best[0]:
                best = (cand, axis, ordered[:k], ordered[k:])
    assert best is not None
    (total, _, _), axis, first, second = best
    return axis, [first, second], max(0.0, total)


def _union_bbox(bboxes: List[List[float]]) -> List[float]:
    """Returns the union box of a list of boxes.

    Input:  bboxes — list of [x1, y1, x2, y2]
    Output: [min x1, min y1, max x2, max y2]
    """
    return [min(b[0] for b in bboxes), min(b[1] for b in bboxes),
            max(b[2] for b in bboxes), max(b[3] for b in bboxes)]


def _node_info(unit: Dict) -> Dict:
    """Returns the public description of a detection used in the layout tree.

    Input:  unit — internal unit dictionary
    Output: dictionary {'id', 'class', 'bbox', 'confidence'}
    """
    return {'id': unit['id'], 'class': unit['class'], 'bbox': list(unit['bbox']),
            'confidence': unit['confidence']}


def _leaf_node(unit: Dict) -> Dict:
    """Builds the layout-tree leaf of one detection.

    Input:  unit — internal unit dictionary
    Output: layout node of type 'leaf' (rule 'leaf') carrying 'node'
    """
    return {
        'type': 'leaf', 'bbox': list(unit['bbox']), 'members': [unit['id']],
        'rule': 'leaf', 'axis': None, 'gaps': [], 'tolerance_px': None,
        'children': [], 'node': _node_info(unit),
    }


def _container_node(ctype: str, rule: str, axis: Optional[str], gaps: List[List[float]],
                    tolerance_px: Optional[float], children: List[Dict],
                    container: Optional[Dict] = None) -> Dict:
    """Builds a non-leaf layout node; members and bbox are the union over the children (and container).

    Input:  ctype — 'Column' | 'Row' | 'Box' | 'LazyColumn'; rule — layout rule name;
            axis — arrangement axis or None; gaps — separating intervals;
            tolerance_px — tolerance of the cut axis or None; children — child
            layout nodes; container — unit of the containing detection (Box /
            LazyColumn) or None
    Output: layout node dictionary
    """
    members = [m for c in children for m in c['members']]
    boxes = [c['bbox'] for c in children]
    if container is not None:
        if container['id'] not in members:
            members.append(container['id'])
        boxes.append(container['bbox'])
    node = {
        'type': ctype,
        'bbox': _union_bbox(boxes) if boxes else [0.0, 0.0, 0.0, 0.0],
        'members': sorted(members, key=lambda v: _id_key({'id': v})),
        'rule': rule, 'axis': axis, 'gaps': gaps, 'tolerance_px': tolerance_px,
        'children': children,
    }
    if container is not None:
        node['container'] = _node_info(container)
    return node


class _LayoutBuilder:
    """Recursive XY-cut over units with a fixed tolerance per axis and a fixed fallback."""

    def __init__(self, tol_x: float, tol_y: float, fallback: str):
        """Stores the per-axis tolerances and the fallback strategy.

        Input:  tol_x, tol_y — tolerated overlap (px) for cuts along x / y;
                fallback — 'y_sorted' or 'least_overlap'
        Output: none
        """
        self.tol = {'x': tol_x, 'y': tol_y}
        self.fallback = fallback

    def layout(self, units: List[Dict], prefer: str = 'y') -> Dict:
        """Lays out the units of one level by recursive XY-cut.

        Input:  units — >=1 unit dictionaries of the same level; prefer — axis tried first
        Output: layout node (a unit's own subtree when there is a single unit)
        """
        if len(units) == 1:
            return self.unit_tree(units[0])
        order = ('y', 'x') if prefer == 'y' else ('x', 'y')
        for axis in order:
            groups = _split_on_gap(units, axis, self.tol[axis])
            if groups:
                gaps = _gaps_between(groups, axis)
                rule = 'gap_tolerance' if any(a >= b for a, b in gaps) else 'gap'
                nxt = 'x' if axis == 'y' else 'y'
                children = [self.layout(g, prefer=nxt) for g in groups]
                return _container_node('Column' if axis == 'y' else 'Row', rule, axis,
                                       gaps, self.tol[axis], children)
        if self.fallback == 'least_overlap':
            axis, groups, overlap = _least_overlap_split(units, order)
            nxt = 'x' if axis == 'y' else 'y'
            children = [self.layout(g, prefer=nxt) for g in groups]
            node = _container_node('Column' if axis == 'y' else 'Row', 'least_overlap', axis,
                                   _gaps_between(groups, axis), self.tol[axis], children)
            node['overlap_px'] = overlap
            return node
        # 'y_sorted' (v1.1.0): honest vertical ordering of the overlapping units.
        ordered = sorted(units, key=_centre_key)
        return _container_node('Column', 'fallback_y_sorted', 'y', [], self.tol['y'],
                               [self.unit_tree(u) for u in ordered])

    def unit_tree(self, unit: Dict) -> Dict:
        """Builds the subtree of one unit: a leaf, a containment Box or a list LazyColumn.

        Input:  unit — unit dictionary (its children are laid out from scratch, y first)
        Output: layout node
        """
        if not unit['children']:
            return _leaf_node(unit)
        inner = self.layout(unit['children'], prefer='y')
        if unit['class'] in LIST_CLASSES:
            if inner['type'] == 'Column' and inner['rule'] in _VERTICAL_SPLIT_RULES:
                items, split_rule = inner['children'], inner['rule']
                gaps, tol = inner['gaps'], inner['tolerance_px']
            else:
                items, split_rule, gaps, tol = [inner], 'single_item', [], None
            node = _container_node('LazyColumn', 'list', 'y', gaps, tol, items, container=unit)
            node['split_rule'] = split_rule
            if 'overlap_px' in inner and split_rule == 'least_overlap':
                node['overlap_px'] = inner['overlap_px']
            return node
        return _container_node('Box', 'containment', None, [], None,
                               [_leaf_node(unit), inner], container=unit)


def _centre_key(unit: Dict) -> Tuple[float, float, Tuple[int, Any]]:
    """Returns the (y-centre, x-centre, id) ordering key of a unit.

    Input:  unit — dictionary with 'bbox'
    Output: sort key tuple
    """
    b = unit['bbox']
    return ((b[1] + b[3]) / 2.0, (b[0] + b[2]) / 2.0, _id_key(unit))


def build_layout_tree(graph: Union[Dict, List[Dict]],
                      image_size: Optional[Sequence[float]] = None,
                      layout: str = 'xycut',
                      cut_tol_frac: float = DEFAULT_CUT_TOL_FRAC,
                      fallback: str = DEFAULT_FALLBACK) -> Dict:
    """Builds the layout tree (with its per-node trace) that the code generator emits.

    Node fields: 'type' ('Column'|'Row'|'Box'|'LazyColumn'|'leaf'); 'bbox'
    (union of the members' boxes); 'members' (sorted detection ids in the
    subtree); 'rule' ('gap' | 'gap_tolerance' | 'fallback_y_sorted' |
    'least_overlap' | 'containment' | 'list' | 'flat' | 'leaf'); 'axis'
    (arrangement axis: 'y' for Column/LazyColumn, 'x' for Row, None for Box
    and leaves); 'gaps' ([a, b] separating intervals in px of the cuts; a >= b
    marks a cut accepted only through the tolerance, a > b being an overlap);
    'tolerance_px' (tolerance of the cut axis, or None); 'children'.  Leaves
    carry 'node' {id, class, bbox, confidence}; Box and LazyColumn carry
    'container' (the containing detection); a LazyColumn carries 'split_rule'
    (the rule that split its children into items, or 'single_item');
    'least_overlap' nodes carry 'overlap_px'; flat-mode leaves carry
    'flattened_children' (ids of the detection's direct children).  The
    root carries 'meta' with the parameters, the image size and its source.
    A rule 'gap' node only uses cuts with a strictly positive gap (the v1.1.0
    rule); a zero-width (touching) or overlapping cut makes it 'gap_tolerance'.
    An empty graph gives an empty Column with rule 'flat'.

    A plain list is laid out as given: this function does NOT compute
    containment, so boxes nest only through 'children' keys already present
    in the list.  To lay out plain boxes (e.g. ground-truth boxes) with the
    same algorithm the pipeline applies to detections, build the graph first
    (UIGraphBuilder().build_graph) or use pipeline.layout_from_boxes.

    Input:  graph — UI graph (graph['hierarchy']['roots'], optional graph['nodes']
            for confidences and graph['image_size']) or a plain list of
            node/detection dictionaries ('id', 'bbox', 'class' or 'class_name',
            optional 'children', 'confidence');
            image_size — (width, height) in px; default graph['image_size'], else
            the extent of the boxes; layout — 'xycut' or 'flat';
            cut_tol_frac — XY-cut tolerance as a fraction of the image extent
            along the cut axis (>= 0; default 0.0, selected on the validation split);
            fallback — 'y_sorted' or 'least_overlap' (used when no cut exists)
    Output: root layout node dictionary
    """
    if layout not in LAYOUT_MODES:
        raise ValueError(f"layout must be one of {LAYOUT_MODES}, got {layout!r}")
    if fallback not in FALLBACK_MODES:
        raise ValueError(f"fallback must be one of {FALLBACK_MODES}, got {fallback!r}")
    cut_tol_frac = float(cut_tol_frac)
    if not math.isfinite(cut_tol_frac) or cut_tol_frac < 0:
        raise ValueError(f"cut_tol_frac must be a finite number >= 0, got {cut_tol_frac!r}")

    units, detections = _prepare_units(graph)
    (img_w, img_h), size_source = _resolve_image_size(graph, image_size, detections)
    tol_x, tol_y = cut_tol_frac * img_w, cut_tol_frac * img_h

    if not detections:
        tree = _container_node('Column', 'flat', 'y', [], None, [])
    elif layout == 'flat':
        leaves = []
        for unit in sorted(detections, key=_centre_key):
            leaf = _leaf_node(unit)
            leaf['flattened_children'] = sorted((c['id'] for c in unit['children']),
                                                key=lambda v: _id_key({'id': v}))
            leaves.append(leaf)
        tree = _container_node('Column', 'flat', 'y', [], None, leaves)
    else:
        tree = _LayoutBuilder(tol_x, tol_y, fallback).layout(units, prefer='y')

    tree['meta'] = {
        'layout': layout,
        'fallback': fallback,
        'cut_tol_frac': cut_tol_frac,
        'image_size': [img_w, img_h],
        'image_size_source': size_source,
        'tolerance_px': {'x': tol_x, 'y': tol_y},
        'n_detections': len(detections),
    }
    return tree


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float]) -> str:
    """Formats a pixel value compactly (integer if within 0.05 of one, else one decimal).

    Input:  v — number or None
    Output: string such as '612', '9.6' or 'n/a'
    """
    if v is None:
        return 'n/a'
    r = round(v)
    return str(int(r)) if abs(v - r) < 0.05 else f"{v:.1f}"


def _fmt_box(b: Sequence[float]) -> str:
    """Formats a bbox as '[x1,y1,x2,y2]'.

    Input:  b — [x1, y1, x2, y2]
    Output: formatted string
    """
    return '[' + ','.join(_fmt(v) for v in b) + ']'


def _fmt2(v: float) -> str:
    """Formats a pixel value with at most two decimals (trailing zeros removed).

    Input:  v — number
    Output: string such as '119.39', '117.6' or '304'
    """
    s = f"{v:.2f}".rstrip('0').rstrip('.')
    return '0' if s in ('', '-0') else s


def _fmt_gap(a: float, b: float) -> str:
    """Formats one separating interval [a, b] of a cut.

    a is where the earlier group ends and b where the next group starts along
    the cut axis: a < b is a gap, a == b touching boxes, a > b an overlap of
    a - b px (accepted only through the tolerance).  An overlap is printed
    with two decimals and its size is computed from the printed endpoints, so
    the printed numbers add up (e.g. '[119.39,117.65] (overlap 1.74 px)').

    Input:  a — end of the earlier group; b — start of the next group (px)
    Output: e.g. '[612,640]', '[700,695] (overlap 5 px)' or '[640,640] (touching)'
    """
    if a > b:
        ra, rb = round(a, 2), round(b, 2)
        return f"[{_fmt2(ra)},{_fmt2(rb)}] (overlap {_fmt2(ra - rb)} px)"
    s = f"[{_fmt(a)},{_fmt(b)}]"
    if a == b:
        s += " (touching)"
    return s


def _fmt_cut_position(axis: str, gaps: List[List[float]], idx: int, n: int, noun: str) -> str:
    """Describes which gaps bound the idx-th group of a cut node.

    Input:  axis — cut axis; gaps — the node's separating intervals; idx —
            0-based group index; n — number of groups; noun — 'group' or 'item'
    Output: e.g. 'group 2 of 3, between gaps y in [158,482] and [1107,1165]'
    """
    head = f"{noun} {idx + 1} of {n}"
    before = gaps[idx - 1] if 0 < idx <= len(gaps) else None
    after = gaps[idx] if idx < len(gaps) else None
    if before and after:
        return f"{head}, between gaps {axis} in {_fmt_gap(*before)} and {_fmt_gap(*after)}"
    if after:
        return f"{head}, before gap {axis} in {_fmt_gap(*after)}"
    if before:
        return f"{head}, after gap {axis} in {_fmt_gap(*before)}"
    return head


def _fmt_cut_summary(gaps: List[List[float]], tolerance_px: Optional[float]) -> str:
    """Summarises the cuts of a node: their number and how many needed the tolerance.

    Input:  gaps — separating intervals; tolerance_px — tolerance of the cut axis
    Output: e.g. '6 cuts, 2 within tolerance 9.6 px' or '1 cut, tolerance 9.6 px'
    """
    n_cuts = len(gaps)
    n_tol = sum(1 for a, b in gaps if a >= b)
    s = f"{n_cuts} cut{'s' if n_cuts != 1 else ''}"
    if n_tol:
        return f"{s}, {n_tol} within tolerance {_fmt(tolerance_px)} px"
    return f"{s}, tolerance {_fmt(tolerance_px)} px"


def _fmt_conf(conf: Optional[float]) -> str:
    """Formats a detection confidence with two decimals.

    Input:  conf — confidence or None
    Output: e.g. '0.48', or 'n/a' when unknown
    """
    return 'n/a' if conf is None else f"{conf:.2f}"


def _fmt_det(info: Dict) -> str:
    """Formats a detection as 'Class#id (conf 0.48)'.

    Input:  info — dictionary {'id', 'class', 'confidence', ...}
    Output: formatted string
    """
    return f"{info['class']}#{info['id']} (conf {_fmt_conf(info.get('confidence'))})"


def _describe(node: Dict, idx: Optional[int], meta: Dict) -> str:
    """Describes one non-leaf layout node and the position of the followed child.

    Input:  node — layout node; idx — index of the child on the path (None if
            the path ends at this node); meta — root 'meta' dictionary
    Output: one readable line
    """
    t, rule, n = node['type'], node['rule'], len(node['children'])
    tol = meta.get('tolerance_px', {})
    no_cut = (f"no cut along y (tolerance {_fmt(tol.get('y'))} px) "
              f"or x (tolerance {_fmt(tol.get('x'))} px)")
    if rule in ('gap', 'gap_tolerance'):
        head = (f"{t} (rule={rule}, axis={node['axis']}, "
                f"{_fmt_cut_summary(node['gaps'], node['tolerance_px'])})")
        if idx is None:
            return head
        return f"{head} -> {_fmt_cut_position(node['axis'], node['gaps'], idx, n, 'group')}"
    if rule == 'fallback_y_sorted':
        pos = '' if idx is None else f" -> position {idx + 1} of {n}"
        return f"{t} (rule=fallback_y_sorted: {no_cut}; members ordered by y-centre){pos}"
    if rule == 'least_overlap':
        (a, b), = node['gaps']
        pos = '' if idx is None else f" -> group {idx + 1} of {n}"
        return (f"{t} (rule=least_overlap, axis={node['axis']}: {no_cut}; split between "
                f"{node['axis']} {_fmt(a)} (end of group 1) and {_fmt(b)} (start of group 2), "
                f"total pairwise overlap {_fmt(node.get('overlap_px'))} px){pos}")
    if rule == 'containment':
        c = node['container']
        what = '' if idx is None else (' -> the container widget itself' if idx == 0
                                       else ' -> the contained children')
        return f"Box (rule=containment, container {_fmt_det(c)} bbox {_fmt_box(c['bbox'])}){what}"
    if rule == 'list':
        c = node['container']
        split = node.get('split_rule')
        if split in ('gap', 'gap_tolerance'):
            how = f"items split by {split} along y, {_fmt_cut_summary(node['gaps'], node['tolerance_px'])}"
        elif split == 'least_overlap':
            how = (f"items split by least_overlap along y ({no_cut}; total pairwise overlap "
                   f"{_fmt(node.get('overlap_px'))} px)")
        elif split == 'fallback_y_sorted':
            how = f"items split by fallback_y_sorted ({no_cut}; ordered by y-centre)"
        else:
            how = "a single item (the children are not stacked vertically)"
        if idx is None:
            tail = " -> this detection is the list container itself"
        elif split in ('gap', 'gap_tolerance'):
            tail = f" -> {_fmt_cut_position('y', node['gaps'], idx, n, 'item')}"
        else:
            tail = f" -> item {idx + 1} of {n}"
        return f"LazyColumn (rule=list, container {_fmt_det(c)} bbox {_fmt_box(c['bbox'])}; {how}){tail}"
    if rule == 'flat':
        pos = '' if idx is None else f" -> position {idx + 1} of {n}"
        return (f"Column (rule=flat: all {n} detections ordered by y-centre, then x-centre; "
                f"containment ignored){pos}")
    pos = '' if idx is None else f" -> child {idx + 1} of {n}"
    return f"{t} (rule={rule}){pos}"


def explain_path(tree: Dict, det_id: Any) -> List[str]:
    """Explains, line by line, why a detection ended up where it is in the layout.

    Each line names one node on the path from the root to the detection: its
    container type, its rule and the geometric evidence (cut axis with the gap
    intervals in px and the tolerance, the fallback used, or the containing
    detection), and which child the path follows.  The last line is the leaf,
    e.g. "leaf TextView#11 (conf 0.48, bbox [700,862,774,936])", or the
    LazyColumn when the detection is a list container.

    Input:  tree — root layout node from build_layout_tree; det_id — detection id
    Output: list of readable lines from the root to the detection
    """
    if det_id not in tree.get('members', []):
        known = ', '.join(str(m) for m in tree.get('members', []))
        raise ValueError(f"detection id {det_id!r} is not in the layout tree (known ids: {known})")
    meta = tree.get('meta', {})
    lines: List[str] = []
    node = tree
    while True:
        if node['type'] == 'leaf':
            info = node['node']
            lines.append(f"leaf {info['class']}#{info['id']} (conf {_fmt_conf(info.get('confidence'))}, "
                         f"bbox {_fmt_box(info['bbox'])})")
            return lines
        idx = next((i for i, c in enumerate(node['children']) if det_id in c['members']), None)
        lines.append(_describe(node, idx, meta))
        if idx is None:          # the detection is this LazyColumn's own list container
            return lines
        node = node['children'][idx]


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def _kotlin_string(text: str) -> str:
    """Escapes text for use inside a Kotlin string literal.

    Input:  text — raw text
    Output: escaped text (backslash, double quote, dollar sign and newlines)
    """
    return (text.replace('\\', '\\\\').replace('"', '\\"').replace('$', '\\$')
            .replace('\n', '\\n').replace('\r', '\\r'))


class ComposeCodeGenerator:
    """Writes a layout tree (see build_layout_tree) as a single @Composable GeneratedScreen()."""

    #: Detector class -> composable the template emits for a leaf of that class.
    COMPONENT_MAP = {
        'Button': 'Button',
        'TextView': 'Text',
        'EditText': 'TextField',
        'ImageView': 'Image',
        'CheckBox': 'Checkbox',
        'Switch': 'Switch',
        'RadioButton': 'RadioButton',
        'SeekBar': 'Slider',
        'ProgressBar': 'LinearProgressIndicator / CircularProgressIndicator',
        'Spinner': 'OutlinedButton + DropdownMenu',
        'WebView': 'AndroidView(WebView)',
        'ListView': 'LazyColumn',
        'RecyclerView': 'LazyColumn',
    }

    def __init__(self, indent_size: int = 4, trace_comments: bool = False):
        """Initializes the code generator.

        Input:  indent_size — number of spaces per indentation level;
                trace_comments — if True, every emitted container and leaf is
                preceded by a '//' comment naming its rule or detection
                (e.g. '// leaf TextView#11 (conf 0.48)'), linking the code to the
                layout tree and to explain_path
        Output: none (side effect of preparing the generator state)
        """
        self.indent = " " * indent_size
        self.trace_comments = trace_comments
        self.code_lines: List[str] = []
        self.last_layout_tree: Optional[Dict] = None
        self._imports: set = set()
        self._state_counter = 0
        self._size_from_boxes = False
        self._px_to_dp = 1.0

    def generate(self, graph: Union[Dict, List[Dict]], use_xy_cut: bool = True,
                 image_size: Optional[Sequence[float]] = None, layout: Optional[str] = None,
                 size_from_boxes: bool = False, screen_width_dp: float = DEFAULT_SCREEN_WIDTH_DP,
                 cut_tol_frac: float = DEFAULT_CUT_TOL_FRAC, fallback: str = DEFAULT_FALLBACK) -> str:
        """Generates the complete Jetpack Compose screen from the UI graph.

        The graph's spatial edges are not used.  The output is deterministic
        for a given input and the input graph is not modified.

        Input:  graph — UI graph from build_ui_graph.py (or a plain node list);
                use_xy_cut — kept for v1.1.0 compatibility and used only when
                layout is None: True -> 'xycut', False -> 'flat'.  (In v1.1.0,
                False selected a legacy recursive emitter that no entry point
                of the package called; that emitter has been removed.);
                image_size — (width, height) in px, default graph['image_size'],
                else the extent of the boxes; layout — 'xycut' | 'flat' | None;
                size_from_boxes — False keeps the v1.1.0 fixed sizes; True sizes
                leaves and containment Boxes, and sets LazyColumn widths, from
                their boxes (dp = px * screen_width_dp / image width).  In both
                modes the LazyColumn maximum height and the WebView height
                are taken from the box (they have no intrinsic height);
                screen_width_dp — target screen width in dp (> 0);
                cut_tol_frac, fallback — XY-cut parameters (see build_layout_tree)
        Output: full Kotlin source (single string); the layout tree is kept in
                self.last_layout_tree
        """
        if layout is None:
            layout = 'xycut' if use_xy_cut else 'flat'
        screen_width_dp = float(screen_width_dp)
        if not (math.isfinite(screen_width_dp) and screen_width_dp > 0):
            raise ValueError(f"screen_width_dp must be > 0, got {screen_width_dp!r}")
        tree = build_layout_tree(graph, image_size=image_size, layout=layout,
                                 cut_tol_frac=cut_tol_frac, fallback=fallback)
        self.last_layout_tree = tree
        meta = tree['meta']
        img_w, img_h = meta['image_size']

        self._size_from_boxes = bool(size_from_boxes)
        self._px_to_dp = screen_width_dp / img_w
        self._state_counter = 0  # ensures unique local var names for stateful widgets
        self._imports = {_RUNTIME + 'Composable'}
        self.code_lines = []

        self._emit(tree, level=1, in_row=False)
        body = self.code_lines

        params = f"layout={layout}"
        if layout == 'xycut':
            params += f", cut_tol_frac={meta['cut_tol_frac']:g}, fallback={fallback}"
        params += (f", size_from_boxes={'true' if self._size_from_boxes else 'false'}"
                   f", image_size={_fmt(img_w)}x{_fmt(img_h)} ({meta['image_size_source']})"
                   f", screen_width_dp={_fmt(screen_width_dp)}")
        lines = [f"// Generated by ui2compose ({params})", ""]
        lines += [f"import {imp}" for imp in sorted(self._imports)]
        lines += ["", "@Composable", "fun GeneratedScreen() {"]
        lines += body
        lines.append("}")
        self.code_lines = lines
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tree emission
    # ------------------------------------------------------------------
    def _emit(self, node: Dict, level: int, in_row: bool):
        """Emits one layout node (and its subtree) as Kotlin.

        Input:  node — layout node; level — indentation level; in_row — True if
                the node is a direct child of a Row
        Output: none (side effect of appending Kotlin lines to code_lines)
        """
        t = node['type']
        if t == 'leaf':
            self._trace(level, f"leaf {_fmt_det(node['node'])}")
            self._emit_leaf(node['node'], level, in_row, node.get('flattened_children', []))
        elif t in ('Column', 'Row'):
            self._trace(level, f"{t} (rule={node['rule']})")
            self._emit_linear(node, level, in_row)
        elif t == 'Box':
            self._trace(level, f"Box (rule=containment, container {_fmt_det(node['container'])})")
            self._emit_box(node, level, in_row)
        elif t == 'LazyColumn':
            self._trace(level, f"LazyColumn (rule=list, list container {_fmt_det(node['container'])})")
            self._emit_lazy_column(node, level, in_row)
        else:
            raise ValueError(f"unknown layout node type {t!r}")

    def _trace(self, level: int, text: str):
        """Adds a trace comment line when trace_comments is enabled.

        Input:  level — indentation level; text — comment text
        Output: none (side effect of appending a comment line)
        """
        if self.trace_comments:
            self._add_line(f"{self.indent * level}// {text}")

    def _emit_linear(self, node: Dict, level: int, in_row: bool):
        """Emits a Column or Row container (v1.1.0 modifiers and arrangement) with its children.

        Input:  node — 'Column' or 'Row' layout node; level — indentation level;
                in_row — True if the container is a direct child of a Row
                (weight(1f) replaces fillMaxWidth())
        Output: none (side effect of appending Kotlin lines)
        """
        indent = self.indent * level
        ctype = node['type']
        self._use(_FOUNDATION_LAYOUT + ctype, _FOUNDATION_LAYOUT + 'Arrangement',
                  'androidx.compose.ui.Alignment')
        self._register_chain(['weight(1f)' if in_row else 'fillMaxWidth()', 'padding(4.dp)'])
        self._add_line(f"{indent}{ctype}(")
        self._add_line(f"{indent}    modifier = Modifier")
        # A child of a Row shares horizontal space via weight; otherwise it
        # spans the available width.
        if in_row:
            self._add_line(f"{indent}        .weight(1f)")
        else:
            self._add_line(f"{indent}        .fillMaxWidth()")
        self._add_line(f"{indent}        .padding(4.dp),")
        if ctype == 'Row':
            self._add_line(f"{indent}    horizontalArrangement = Arrangement.spacedBy(8.dp),")
            self._add_line(f"{indent}    verticalAlignment = Alignment.CenterVertically")
        else:
            self._add_line(f"{indent}    verticalArrangement = Arrangement.spacedBy(8.dp),")
            self._add_line(f"{indent}    horizontalAlignment = Alignment.CenterHorizontally")
        self._add_line(f"{indent}) {{")
        for child in node['children']:
            self._emit(child, level + 1, in_row=(ctype == 'Row'))
        self._add_line(f"{indent}}}")
        self._add_blank_line()

    def _emit_box(self, node: Dict, level: int, in_row: bool = False):
        """Emits a containment Box: the container's own widget, then its children's layout on top.

        Without size_from_boxes the Box has no fixed size, and its children's
        layout (a Column, Row or LazyColumn) uses fillMaxWidth().  Inside a Row
        such a Box would take the whole remaining width and leave its later
        siblings 0 dp wide, so a Box that is a direct child of a Row gets
        weight(1f), like a Column or Row in a Row.  With size_from_boxes the
        Box has the fixed size of its container's box.

        Input:  node — 'Box' layout node (children[0] is the container's leaf,
                children[1] the children's layout); level — indentation level;
                in_row — True if the Box is a direct child of a Row
        Output: none (side effect of appending Kotlin lines)
        """
        indent = self.indent * level
        self._use(_FOUNDATION_LAYOUT + 'Box')
        if self._size_from_boxes:
            mod = self._chain([self._size_call(node['container']['bbox'])])
        elif in_row:
            mod = self._chain(['weight(1f)', 'padding(4.dp)'])
        else:
            mod = self._chain(['padding(4.dp)'])
        self._add_line(f"{indent}Box(")
        self._add_line(f"{indent}    modifier = {mod}")
        self._add_line(f"{indent}) {{")
        for child in node['children']:
            self._emit(child, level + 1, in_row=False)
        self._add_line(f"{indent}}}")
        self._add_blank_line()

    def _emit_lazy_column(self, node: Dict, level: int, in_row: bool):
        """Emits a list container as a LazyColumn with a bounded height and one item { } per child.

        Input:  node — 'LazyColumn' layout node; level — indentation level;
                in_row — True if the list is a direct child of a Row
        Output: none (side effect of appending Kotlin lines)
        """
        indent = self.indent * level
        self._lazy_column_open(node['container']['bbox'], level, in_row)
        for child in node['children']:
            self._add_line(f"{indent}    item {{")
            self._emit(child, level + 2, in_row=False)
            self._add_line(f"{indent}    }}")
        self._add_line(f"{indent}}}")
        self._add_blank_line()

    def _lazy_column_open(self, bbox: List[float], level: int, in_row: bool):
        """Writes the opening of a LazyColumn whose maximum height is always derived from its box.

        A LazyColumn must have a bounded height (it may sit inside another
        vertically scrolling list), so heightIn(max = ...) is emitted in every
        sizing mode.  Width: fillMaxWidth() (weight(1f) inside a Row), or the
        box width when size_from_boxes is True.

        Input:  bbox — box of the list detection; level — indentation level;
                in_row — True if the list is a direct child of a Row
        Output: none (side effect of appending Kotlin lines)
        """
        indent = self.indent * level
        self._use('androidx.compose.foundation.lazy.LazyColumn')
        if self._size_from_boxes:
            width_call = f"width({self._dp(bbox[2] - bbox[0])}.dp)"
        else:
            width_call = 'weight(1f)' if in_row else 'fillMaxWidth()'
        height_call = f"heightIn(max = {self._dp(bbox[3] - bbox[1])}.dp)"
        self._register_chain([width_call, height_call])
        self._add_line(f"{indent}LazyColumn(")
        self._add_line(f"{indent}    modifier = Modifier")
        self._add_line(f"{indent}        .{width_call}")
        self._add_line(f"{indent}        .{height_call}")
        self._add_line(f"{indent}) {{")

    # ------------------------------------------------------------------
    # Leaf templates
    # ------------------------------------------------------------------
    def _emit_leaf(self, info: Dict, level: int, in_row: bool,
                   flattened_children: Optional[List[Any]] = None):
        """Emits the template of one detection (placeholder content, real component type).

        Input:  info — leaf 'node' dictionary {id, class, bbox, confidence};
                level — indentation level; in_row — True if the leaf is a direct
                child of a Row (full-width templates then use weight(1f));
                flattened_children — ids of the detection's direct children,
                which the flat layout emits as separate leaves
        Output: none (side effect of appending Kotlin lines)
        """
        ind = self.indent * level
        cls = info['class']
        bbox = info['bbox']
        L = self._add_line

        if cls == 'Button':
            self._use(_MATERIAL3 + 'Button', _MATERIAL3 + 'Text')
            L(f'{ind}Button(')
            L(f'{ind}    onClick = {{ /* TODO */ }},')
            L(f'{ind}    modifier = {self._leaf_mod(bbox, ["padding(8.dp)"])}')
            L(f'{ind}) {{')
            L(f'{ind}    Text("Button")')
            L(f'{ind}}}')

        elif cls == 'TextView':
            self._use(_MATERIAL3 + 'Text')
            L(f'{ind}Text(')
            L(f'{ind}    text = "Text Content",')
            L(f'{ind}    modifier = {self._leaf_mod(bbox, ["padding(4.dp)"])}')
            L(f'{ind})')

        elif cls == 'EditText':
            var = self._state_var('textValue', 'mutableStateOf')
            self._use(_MATERIAL3 + 'TextField', _MATERIAL3 + 'Text')
            L(f'{ind}var {var} by remember {{ mutableStateOf("") }}')
            L(f'{ind}TextField(')
            L(f'{ind}    value = {var},')
            L(f'{ind}    onValueChange = {{ {var} = it }},')
            L(f'{ind}    label = {{ Text("Input") }},')
            L(f'{ind}    modifier = {self._leaf_mod(bbox, ["fillMaxWidth()", "padding(8.dp)"], in_row)}')
            L(f'{ind})')

        elif cls == 'ImageView':
            self._use('androidx.compose.foundation.Image', 'androidx.compose.ui.graphics.Color',
                      'androidx.compose.ui.graphics.painter.ColorPainter')
            L(f'{ind}Image(')
            L(f'{ind}    painter = ColorPainter(Color.LightGray),')
            L(f'{ind}    contentDescription = "Image placeholder",')
            L(f'{ind}    modifier = {self._leaf_mod(bbox, ["size(100.dp)"])}')
            L(f'{ind})')

        elif cls in ('CheckBox', 'Switch'):
            composable = 'Checkbox' if cls == 'CheckBox' else 'Switch'
            var = self._state_var('checked' if cls == 'CheckBox' else 'switched', 'mutableStateOf')
            self._use(_MATERIAL3 + composable)
            L(f'{ind}var {var} by remember {{ mutableStateOf(false) }}')
            L(f'{ind}{composable}(')
            L(f'{ind}    checked = {var},')
            if self._size_from_boxes:
                L(f'{ind}    onCheckedChange = {{ {var} = it }},')
                L(f'{ind}    modifier = {self._leaf_mod(bbox, [])}')
            else:   # v1.1.0 template: no modifier
                L(f'{ind}    onCheckedChange = {{ {var} = it }}')
            L(f'{ind})')

        elif cls == 'RadioButton':
            var = self._state_var('selected', 'mutableStateOf')
            self._use(_MATERIAL3 + 'RadioButton')
            L(f'{ind}var {var} by remember {{ mutableStateOf(false) }}')
            L(f'{ind}RadioButton(')
            L(f'{ind}    selected = {var},')
            if self._size_from_boxes:
                L(f'{ind}    onClick = {{ {var} = !{var} }},')
                L(f'{ind}    modifier = {self._leaf_mod(bbox, [])}')
            else:
                L(f'{ind}    onClick = {{ {var} = !{var} }}')
            L(f'{ind})')

        elif cls == 'SeekBar':
            var = self._state_var('sliderValue', 'mutableFloatStateOf')
            self._use(_MATERIAL3 + 'Slider')
            L(f'{ind}var {var} by remember {{ mutableFloatStateOf(0.5f) }}')
            L(f'{ind}Slider(')
            L(f'{ind}    value = {var},')
            L(f'{ind}    onValueChange = {{ {var} = it }},')
            L(f'{ind}    modifier = {self._leaf_mod(bbox, ["fillMaxWidth()", "padding(8.dp)"], in_row)}')
            L(f'{ind})')

        elif cls == 'ProgressBar':
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            square = min(w, h) > 0 and max(w, h) / min(w, h) <= PROGRESS_SQUARE_MAX_ASPECT
            if square:
                self._use(_MATERIAL3 + 'CircularProgressIndicator')
                L(f'{ind}CircularProgressIndicator(')
                L(f'{ind}    modifier = {self._leaf_mod(bbox, ["padding(8.dp)"])}')
            else:
                self._use(_MATERIAL3 + 'LinearProgressIndicator')
                L(f'{ind}LinearProgressIndicator(')
                L(f'{ind}    modifier = {self._leaf_mod(bbox, ["fillMaxWidth()", "padding(8.dp)"], in_row)}')
            L(f'{ind})')

        elif cls == 'Spinner':
            var = self._state_var('expanded', 'mutableStateOf')
            self._use(_FOUNDATION_LAYOUT + 'Box', _MATERIAL3 + 'OutlinedButton', _MATERIAL3 + 'Text',
                      _MATERIAL3 + 'DropdownMenu', _MATERIAL3 + 'DropdownMenuItem')
            L(f'{ind}var {var} by remember {{ mutableStateOf(false) }}')
            L(f'{ind}Box(')
            L(f'{ind}    modifier = {self._leaf_mod(bbox, ["padding(8.dp)"])}')
            L(f'{ind}) {{')
            L(f'{ind}    OutlinedButton(onClick = {{ {var} = true }}) {{')
            L(f'{ind}        Text("Select")')
            L(f'{ind}    }}')
            L(f'{ind}    DropdownMenu(')
            L(f'{ind}        expanded = {var},')
            L(f'{ind}        onDismissRequest = {{ {var} = false }}')
            L(f'{ind}    ) {{')
            L(f'{ind}        DropdownMenuItem(')
            L(f'{ind}            text = {{ Text("Option 1") }},')
            L(f'{ind}            onClick = {{ {var} = false }}')
            L(f'{ind}        )')
            L(f'{ind}    }}')
            L(f'{ind}}}')

        elif cls == 'WebView':
            # A WebView has no intrinsic Compose height before it loads content,
            # so its height is always taken from the detected box.
            mod = self._leaf_mod(bbox, ["fillMaxWidth()", f"height({self._dp(bbox[3] - bbox[1])}.dp)",
                                        "padding(8.dp)"], in_row)
            self._use('androidx.compose.ui.platform.LocalInspectionMode',
                      'androidx.compose.ui.viewinterop.AndroidView',
                      _FOUNDATION_LAYOUT + 'Box', _MATERIAL3 + 'Text')
            L(f'{ind}if (LocalInspectionMode.current) {{')
            L(f'{ind}    Box(')
            L(f'{ind}        modifier = {mod}')
            L(f'{ind}    ) {{')
            L(f'{ind}        Text("WebView")')
            L(f'{ind}    }}')
            L(f'{ind}}} else {{')
            L(f'{ind}    AndroidView(')
            L(f'{ind}        factory = {{ context -> android.webkit.WebView(context) }},')
            L(f'{ind}        modifier = {mod}')
            L(f'{ind}    )')
            L(f'{ind}}}')

        elif cls in LIST_CLASSES:
            self._use(_MATERIAL3 + 'Text')
            if flattened_children:
                note = (f"{cls}: {len(flattened_children)} child detection(s) "
                        f"emitted separately (flat layout)")
            else:
                note = f"{cls}: no items detected"
            self._lazy_column_open(bbox, level, in_row)
            L(f'{ind}    item {{')
            L(f'{ind}        Text("{_kotlin_string(note)}")')
            L(f'{ind}    }}')
            L(f'{ind}}}')

        else:
            # Explicit placeholder for class names outside the detector's classes
            self._use(_FOUNDATION_LAYOUT + 'Box', _MATERIAL3 + 'Text')
            L(f'{ind}Box(')
            L(f'{ind}    modifier = {self._leaf_mod(bbox, ["padding(8.dp)"])}')
            L(f'{ind}) {{')
            L(f'{ind}    Text("Unknown: {_kotlin_string(cls)}")')
            L(f'{ind}}}')

        self._add_blank_line()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _dp(self, px: float) -> int:
        """Converts a pixel length into whole dp (rounded half up, at least MIN_SIZE_DP).

        Input:  px — length in screenshot pixels
        Output: length in dp (px * screen_width_dp / image width)
        """
        return max(MIN_SIZE_DP, int(math.floor(px * self._px_to_dp + 0.5)))

    def _size_call(self, bbox: List[float]) -> str:
        """Returns the 'size(width = .., height = ..)' modifier call for a box.

        Input:  bbox — [x1, y1, x2, y2] in px
        Output: modifier call string (without the leading 'Modifier.')
        """
        return f"size(width = {self._dp(bbox[2] - bbox[0])}.dp, height = {self._dp(bbox[3] - bbox[1])}.dp)"

    def _leaf_mod(self, bbox: List[float], default_calls: List[str], in_row: bool = False) -> str:
        """Returns the modifier expression of a leaf template.

        With size_from_boxes the box size is used; otherwise the template's
        fixed v1.1.0 calls, where a leading fillMaxWidth() becomes weight(1f)
        for a direct child of a Row (fillMaxWidth would starve later siblings).

        Input:  bbox — leaf box; default_calls — fixed modifier calls; in_row —
                True if the leaf is a direct child of a Row
        Output: modifier expression such as 'Modifier.padding(8.dp)'
        """
        if self._size_from_boxes:
            return self._chain([self._size_call(bbox)])
        calls = list(default_calls)
        if in_row and calls and calls[0] == 'fillMaxWidth()':
            calls[0] = 'weight(1f)'
        return self._chain(calls)

    def _chain(self, calls: List[str]) -> str:
        """Builds 'Modifier.a().b()' and registers the imports of the calls.

        Input:  calls — modifier calls such as 'padding(8.dp)'
        Output: modifier expression ('Modifier' alone for an empty list)
        """
        self._register_chain(calls)
        return 'Modifier' + ''.join('.' + c for c in calls)

    def _register_chain(self, calls: List[str]):
        """Registers the imports needed by a chain of modifier calls.

        Input:  calls — modifier calls such as 'heightIn(max = 120.dp)'
        Output: none (side effect of adding import paths)
        """
        self._use('androidx.compose.ui.Modifier')
        for c in calls:
            name = c.split('(', 1)[0]
            if name not in _MODIFIER_IMPORTS:
                raise ValueError(f"modifier call {c!r} has no registered import")
            if _MODIFIER_IMPORTS[name]:
                self._use(_MODIFIER_IMPORTS[name])
            if '.dp' in c:
                self._use('androidx.compose.ui.unit.dp')

    def _state_var(self, prefix: str, factory: str) -> str:
        """Returns a fresh state-variable name and registers the state imports.

        Input:  prefix — name prefix (e.g. 'checked'); factory — state factory
                function ('mutableStateOf' or 'mutableFloatStateOf')
        Output: unique variable name such as 'checked3'
        """
        name = f"{prefix}{self._state_counter}"
        self._state_counter += 1
        self._use(*_STATE_IMPORTS, _RUNTIME + factory)
        return name

    def _use(self, *imports: str):
        """Registers fully qualified imports used by the emitted code.

        Input:  imports — import paths
        Output: none (side effect of adding to the import set)
        """
        self._imports.update(imports)

    def _add_line(self, line: str):
        """Adds a single line to the code list.

        Input:  line — Kotlin line to add
        Output: none (side effect of appending to the code_lines list)
        """
        self.code_lines.append(line)

    def _add_blank_line(self):
        """Adds a blank line to the code list.

        Input:  none
        Output: none (side effect of appending a blank line to the code_lines list)
        """
        self.code_lines.append("")

    def save_to_file(self, code: str, output_path: str):
        """Saves the generated Compose code to a file as UTF-8.

        Input:  code — Kotlin code (string); output_path — target .kt file path
        Output: none (side effect of writing to disk and printing a console notice)
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"✓ Compose code saved: {output_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point: generates Kotlin from a saved UI graph and optionally explains detections.

    The code is written to --output, or printed to stdout when neither
    --output nor --explain is given.  --explain prints explain_path for each
    id; --layout-json writes the layout tree.

    Input:  argv — command-line arguments (default: sys.argv[1:])
    Output: exit status (0 on success, 2 for an unknown --explain id)
    """
    parser = argparse.ArgumentParser(
        description="Generate Jetpack Compose code from a ui2compose UI graph (3_ui_graph.json).")
    parser.add_argument("graph", help="UI graph JSON produced by the graph stage")
    parser.add_argument("-o", "--output", help="Kotlin output file (default: stdout)")
    parser.add_argument("--image-size", nargs=2, type=float, metavar=("W", "H"),
                        help="screenshot size in px (default: graph['image_size'] or box extent)")
    parser.add_argument("--layout", choices=LAYOUT_MODES, default="xycut")
    parser.add_argument("--fallback", choices=FALLBACK_MODES, default=DEFAULT_FALLBACK)
    parser.add_argument("--cut-tol-frac", type=float, default=DEFAULT_CUT_TOL_FRAC)
    parser.add_argument("--size-from-boxes", action="store_true")
    parser.add_argument("--screen-width-dp", type=float, default=DEFAULT_SCREEN_WIDTH_DP)
    parser.add_argument("--trace-comments", action="store_true",
                        help="precede every emitted node with a '//' trace comment")
    parser.add_argument("--explain", type=int, nargs="+", metavar="ID",
                        help="print the layout path of these detection ids")
    parser.add_argument("--layout-json", help="write the layout tree (with trace) to this JSON file")
    args = parser.parse_args(argv)

    with open(args.graph, 'r', encoding='utf-8') as f:
        graph = json.load(f)

    generator = ComposeCodeGenerator(trace_comments=args.trace_comments)
    code = generator.generate(graph, image_size=args.image_size, layout=args.layout,
                              size_from_boxes=args.size_from_boxes,
                              screen_width_dp=args.screen_width_dp,
                              cut_tol_frac=args.cut_tol_frac, fallback=args.fallback)
    tree = generator.last_layout_tree

    if args.output:
        generator.save_to_file(code, args.output)
    elif not args.explain:
        print(code)
    if args.layout_json:
        with open(args.layout_json, 'w', encoding='utf-8') as f:
            json.dump(tree, f, indent=2)
    if args.explain:
        for det_id in args.explain:
            try:
                lines = explain_path(tree, det_id)
            except ValueError as err:
                print(f"error: {err}", file=sys.stderr)
                return 2
            print(f"Detection {det_id}:")
            for depth, line in enumerate(lines):
                print("  " * (depth + 1) + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
