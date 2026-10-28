# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""UI graph construction (stage 3 of the pipeline).

Turns a flat list of component boxes (detections, or plain ground-truth boxes)
into a UI graph with three parts. Only the first one is consumed by code
generation:

1. Containment hierarchy (``graph['hierarchy']``) — a forest in which every
   input id appears exactly once. A box B is *contained* in a box A when at
   least ``containment_ratio`` of B's area lies inside A and A ranks above B
   (larger area; for equal areas higher confidence, then lower id). Each box
   is attached to exactly one parent: the smallest-area box that contains it.
   Code generation (``generate_compose_code.py``) consumes only this
   containment structure.
2. Synthetic layout groups — the top-level nodes of the forest are wrapped in
   ``SyntheticRow`` / ``SyntheticColumn`` nodes (negative ids,
   ``'synthetic': True``) by a row-clustering heuristic. They are an
   inspection/visualisation export (e.g. the layout overlay of the Streamlit
   demo) and are NOT used by code generation, which treats them as transparent
   and computes its own layout from the boxes.
3. Edges (``graph['edges']``) — one ``parent_child`` edge per hierarchy link,
   plus spatial edges (``above`` / ``below`` / ``left_of`` / ``right_of``)
   between boxes that share the same containment parent. The edges are an
   inspection/visualisation export (the demo's graph and adjacency views) and
   are NOT used by code generation.

The alignment tolerance of the spatial edges is relative to the image width
(``align_tol_frac``), so the same screen at a different resolution yields the
same graph.
"""

import json
import warnings
from dataclasses import dataclass
from enum import Enum
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

# Fallback image size (width, height) in pixels: the resolution of the ReDraw
# screenshots. It is used only when the size can be determined neither from the
# `image_size` argument nor from the detections (see resolve_image_size).
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (1200, 1920)

# Minimum fraction of a box's area that must lie inside another box for the box
# to count as contained in it. Default selected on the validation split
# (results/eval_appsplit/selected_thresholds.json; grid 1.0 / 0.95 / 0.9, a-priori value
# 0.95). 1.0 reproduces the strict geometric containment test of v1.1.0.
DEFAULT_CONTAINMENT_RATIO: float = 0.9

# Spatial-edge alignment tolerance as a fraction of the image width. 20/1200 is
# identical to the fixed 20 px used by v1.1.0 on the 1200 px wide ReDraw
# screenshots. Not tuned (edges are an inspection export only).
DEFAULT_ALIGN_TOL_FRAC: float = 20 / 1200

# Row clustering of the synthetic layout groups (inspection export only; values
# taken over unchanged from v1.1.0, not tuned): two consecutive top-level boxes
# are in the same row when their y-centres differ by less than
# DEFAULT_LAYOUT_ROW_TOL x the mean height of the top-level boxes ...
DEFAULT_LAYOUT_ROW_TOL: float = 0.6
# ... and two consecutive rows with the same number (>= 2) of boxes are merged
# into columns when their sorted x-centres pairwise differ by at most
# DEFAULT_LAYOUT_COL_TOL x that mean height.
DEFAULT_LAYOUT_COL_TOL: float = 1.5

# Keyword arguments of the v1.1.0 constructor that are still accepted (with a
# DeprecationWarning) and mapped to `align_tol_px`.
_DEPRECATED_TOLERANCE_KWARGS = ("vertical_threshold", "horizontal_threshold")


class RelationType(Enum):
    """Relation types of the graph edges."""
    PARENT_CHILD = "parent_child"  # Containment (one edge per hierarchy link)
    ABOVE = "above"  # source is above target (centres aligned in x)
    BELOW = "below"  # source is below target (centres aligned in x)
    LEFT_OF = "left_of"  # source is left of target (centres aligned in y)
    RIGHT_OF = "right_of"  # source is right of target (centres aligned in y)


@dataclass
class Edge:
    """Graph edge between two components.

    `coverage` is set for PARENT_CHILD edges only: the fraction of the child's
    area that lies inside the parent.
    """
    source_id: int
    target_id: int
    relation_type: RelationType
    coverage: Optional[float] = None


# ── Box geometry ───────────────────────────────────────────


def box_area(bbox: Sequence[float]) -> float:
    """Computes the area of a box; degenerate (inverted) sides count as zero.

    Input:  bbox — box in (x1, y1, x2, y2) pixel format
    Output: area in square pixels (>= 0)
    """
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    """Computes the area of the intersection of two boxes.

    Input:  a, b — boxes in (x1, y1, x2, y2) pixel format
    Output: intersection area in square pixels (0.0 if the boxes do not overlap)
    """
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h


def coverage(inner: Sequence[float], outer: Sequence[float]) -> float:
    """Computes the fraction of the inner box's area that lies inside the outer box.

    A zero-area inner box has no meaningful area fraction; it counts as fully
    covered (1.0) if it lies inside the closed outer box, otherwise as 0.0.

    Input:  inner, outer — boxes in (x1, y1, x2, y2) pixel format
    Output: coverage in [0, 1]
    """
    area = box_area(inner)
    if area <= 0:
        inside = (outer[0] <= inner[0] and outer[1] <= inner[1] and
                  inner[2] <= outer[2] and inner[3] <= outer[3])
        return 1.0 if inside else 0.0
    return intersection_area(inner, outer) / area


def _rank_key(node: Dict) -> Tuple[float, float, int]:
    """Returns the containment rank of a node: only a higher-ranked node can be its parent.

    The rank orders by area, then confidence, then lower id. It is a strict
    total order, so containment can never form a cycle and of two identical
    boxes exactly one becomes the parent of the other.

    Input:  node — normalised node with 'bbox', 'confidence', 'id'
    Output: tuple (area, confidence, -id); larger means more parent-like
    """
    return (box_area(node['bbox']), node['confidence'], -node['id'])


def _parent_preference_key(node: Dict) -> Tuple[float, float, int]:
    """Returns the sort key that picks the parent among the containing nodes.

    The smallest-area container wins; ties go to the higher confidence, then
    to the lower id.

    Input:  node — normalised node with 'bbox', 'confidence', 'id'
    Output: tuple (area, -confidence, id); smallest is preferred
    """
    return (box_area(node['bbox']), -node['confidence'], node['id'])


def _containers_of(child: Dict, nodes: List[Dict], containment_ratio: float) -> List[Dict]:
    """Lists the nodes that contain the given node, best parent candidate first.

    Input:  child — node whose containers are searched; nodes — all nodes;
            containment_ratio — minimum coverage of the child by a container
    Output: containing nodes sorted by parent preference (smallest area first)
    """
    child_rank = _rank_key(child)
    found = [n for n in nodes
             if n['id'] != child['id']
             and _rank_key(n) > child_rank
             and coverage(child['bbox'], n['bbox']) >= containment_ratio]
    found.sort(key=_parent_preference_key)
    return found


# ── Image size ─────────────────────────────────────────────


def resolve_image_size(detections: List[Dict],
                       image_size: Optional[Sequence[float]] = None
                       ) -> Tuple[Tuple[float, float], str]:
    """Determines the image size (width, height) the boxes refer to.

    Order of precedence: (1) the `image_size` argument; (2) the 'img_w'/'img_h'
    fields of the detections (written by ComponentDetector.detect); (3) an
    estimate from 'bbox' vs 'bbox_norm' (box width / normalised width, median
    over the detections, rounded to whole pixels); (4) DEFAULT_IMAGE_SIZE, the
    ReDraw resolution 1200 x 1920 — only correct for ReDraw-sized screens.

    Input:  detections — component dictionaries; image_size — optional (w, h)
    Output: ((width, height), source) with source one of 'argument',
            'detections', 'bbox_norm', 'default'
    """
    if image_size is not None:
        w, h = image_size
        if w <= 0 or h <= 0:
            raise ValueError(f"image_size must be positive, got {image_size!r}")
        return (w, h), 'argument'

    sizes = {(d['img_w'], d['img_h']) for d in detections
             if d.get('img_w') is not None and d.get('img_h') is not None}
    if len(sizes) > 1:
        raise ValueError(f"Detections refer to different image sizes: {sorted(sizes)}")
    if sizes:
        return sizes.pop(), 'detections'

    widths, heights = [], []
    for d in detections:
        norm = d.get('bbox_norm')
        if not norm or len(norm) != 4:
            continue
        x1, y1, x2, y2 = d['bbox']
        if norm[2] > 0 and x2 > x1:
            widths.append((x2 - x1) / norm[2])
        if norm[3] > 0 and y2 > y1:
            heights.append((y2 - y1) / norm[3])
    if widths and heights:
        return (int(round(median(widths))), int(round(median(heights)))), 'bbox_norm'

    return DEFAULT_IMAGE_SIZE, 'default'


# ── Hierarchy helpers ──────────────────────────────────────


def containment_roots(hierarchy: Dict) -> List[Dict]:
    """Returns the top-level detected nodes of the hierarchy, with the synthetic layout groups removed.

    Synthetic nodes ('synthetic': True, or negative id for graphs written by
    v1.1.0) are replaced by their children, recursively; the result is the
    pure containment forest in depth-first order of the hierarchy.

    Input:  hierarchy — hierarchy dictionary containing the 'roots' list
    Output: list of hierarchy nodes of detected components (not copied)
    """
    out: List[Dict] = []

    def walk(node: Dict):
        """Appends a detected node, or descends through a synthetic one.

        Input:  node — hierarchy node
        Output: none (side effect of appending to the out list)
        """
        if node.get('synthetic', node['id'] < 0):
            for child in node.get('children', []):
                walk(child)
        else:
            out.append(node)

    for root in hierarchy['roots']:
        walk(root)
    return out


def explain_containment(graph: Dict, det_id: int, max_near_misses: int = 3) -> List[str]:
    """Explains, in plain text, why a node has the containment parent it has in the graph.

    The containment rule is re-evaluated on graph['nodes'] with the
    containment_ratio recorded in graph['params'], so the explanation follows
    the same code path as build_graph. Near misses are higher-ranked boxes that
    overlap the node but cover less than containment_ratio of it (they would
    have been containers at a lower ratio).

    Input:  graph — output of UIGraphBuilder.build_graph; det_id — node id;
            max_near_misses — how many near misses to list at most
    Output: list of explanation lines
    """
    ratio = graph['params']['containment_ratio']
    nodes = _normalise_detections(graph['nodes'])
    by_id = {n['id']: n for n in nodes}
    if det_id not in by_id:
        return [f"id {det_id}: not a node of the graph (removed before graph "
                f"construction, or never detected)"]

    node = by_id[det_id]
    lines = [f"id {det_id} {node['class_name']} conf={node['confidence']:.2f} "
             f"bbox={_fmt_box(node['bbox'])} area={box_area(node['bbox']):.0f}; "
             f"containment rule: coverage >= {ratio:g} and higher rank "
             f"(area, then confidence, then lower id)"]

    containers = _containers_of(node, nodes, ratio)
    if containers:
        cand = ", ".join(
            f"{c['id']} {c['class_name']} (coverage {coverage(node['bbox'], c['bbox']):.3f}, "
            f"area {box_area(c['bbox']):.0f})" for c in containers)
        lines.append(f"  contained in: {cand}")
        parent = containers[0]
        reason = ("only container" if len(containers) == 1 else
                  "smallest-area container (ties: higher confidence, then lower id)")
        lines.append(f"  parent: {parent['id']} {parent['class_name']} — {reason}")
    else:
        lines.append("  contained in: none -> top-level node of the containment forest")

    rank = _rank_key(node)
    misses = [(coverage(node['bbox'], n['bbox']), n) for n in nodes
              if n['id'] != det_id and _rank_key(n) > rank]
    misses = [(c, n) for c, n in misses if 0.0 < c < ratio]
    misses.sort(key=lambda cn: (-cn[0], cn[1]['id']))
    for c, n in misses[:max_near_misses]:
        lines.append(f"  near miss: {n['id']} {n['class_name']} covers {c:.3f} < {ratio:g}")

    children = [n for n in nodes
                if n['id'] != det_id and _containers_of(n, nodes, ratio)[:1] == [node]]
    if children:
        lines.append("  children: " + ", ".join(f"{c['id']} {c['class_name']}" for c in children))
    return lines


def _fmt_box(bbox: Sequence[float]) -> str:
    """Formats a box with whole-pixel coordinates for explanation lines.

    Input:  bbox — box in (x1, y1, x2, y2) pixel format
    Output: string like '[12, 30, 200, 90]'
    """
    return "[" + ", ".join(f"{v:.0f}" for v in bbox) + "]"


def _normalise_detections(detections: List[Dict]) -> List[Dict]:
    """Copies the input boxes and fills in the optional fields, so plain box lists are accepted.

    Required per box: 'bbox' (x1, y1, x2, y2 pixels) and 'class_name'.
    Optional: 'id' (default: position in the list) and 'confidence' (default
    1.0, e.g. for ground-truth boxes; ties then fall through to the id). Ids
    must be unique and non-negative (negative ids are reserved for synthetic
    layout nodes).

    Input:  detections — list of component dictionaries
    Output: list of shallow copies with 'id', 'class_name', 'bbox', 'confidence' set
    """
    nodes = []
    for idx, det in enumerate(detections):
        if 'bbox' not in det or 'class_name' not in det:
            raise ValueError(f"Box #{idx} needs 'bbox' and 'class_name': {det!r}")
        node = dict(det)
        node.setdefault('id', idx)
        node.setdefault('confidence', 1.0)
        if node['confidence'] is None:
            node['confidence'] = 1.0
        nodes.append(node)

    ids = [n['id'] for n in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Box ids must be unique, got {ids}")
    if any(i < 0 for i in ids):
        raise ValueError(f"Box ids must be non-negative (negative ids are reserved "
                         f"for synthetic layout nodes), got {ids}")
    return nodes


class UIGraphBuilder:
    """Builds the UI graph (containment hierarchy, layout groups, edges) from component boxes."""

    def __init__(self, *,
                 containment_ratio: float = DEFAULT_CONTAINMENT_RATIO,
                 align_tol_frac: float = DEFAULT_ALIGN_TOL_FRAC,
                 align_tol_px: Optional[float] = None,
                 layout_row_tol: float = DEFAULT_LAYOUT_ROW_TOL,
                 layout_col_tol: float = DEFAULT_LAYOUT_COL_TOL,
                 **deprecated: float):
        """Initializes the graph builder with its containment and alignment thresholds.

        Input:  containment_ratio — minimum fraction of a box's area inside another box for
                    containment, in (0, 1] (default 0.9, selected on the validation split,
                    results/eval_appsplit/selected_thresholds.json; 1.0 = strict v1.1.0 test);
                align_tol_frac — spatial-edge alignment tolerance as a fraction of the image
                    width (default 20/1200 = v1.1.0's 20 px at 1200 px width; inspection only);
                align_tol_px — optional absolute tolerance in pixels that overrides align_tol_frac;
                layout_row_tol, layout_col_tol — row clustering / column merging tolerances of the
                    synthetic layout groups, as multiples of the mean top-level box height
                    (v1.1.0 values 0.6 / 1.5; inspection only);
                deprecated — v1.1.0 keywords vertical_threshold / horizontal_threshold (pixels),
                    accepted with a DeprecationWarning and mapped to align_tol_px
        Output: none (side effect of assigning the thresholds to object fields)
        """
        unknown = set(deprecated) - set(_DEPRECATED_TOLERANCE_KWARGS)
        if unknown:
            raise TypeError(f"UIGraphBuilder got unexpected keyword arguments: {sorted(unknown)}")
        if deprecated:
            values = {deprecated[k] for k in _DEPRECATED_TOLERANCE_KWARGS if k in deprecated}
            if len(values) > 1:
                raise ValueError("vertical_threshold and horizontal_threshold differ; v1.2.0 "
                                 "uses one isotropic tolerance — pass align_tol_px instead")
            if align_tol_px is not None:
                raise ValueError("Pass either align_tol_px or the deprecated "
                                 "vertical_threshold/horizontal_threshold, not both")
            warnings.warn("vertical_threshold/horizontal_threshold are deprecated; use "
                          "align_tol_frac (relative to image width) or align_tol_px",
                          DeprecationWarning, stacklevel=2)
            align_tol_px = values.pop()

        if not 0.0 < containment_ratio <= 1.0:
            raise ValueError(f"containment_ratio must be in (0, 1], got {containment_ratio}")
        if align_tol_frac < 0:
            raise ValueError(f"align_tol_frac must be >= 0, got {align_tol_frac}")
        if align_tol_px is not None and align_tol_px < 0:
            raise ValueError(f"align_tol_px must be >= 0, got {align_tol_px}")

        self.containment_ratio = containment_ratio
        self.align_tol_frac = align_tol_frac
        self.align_tol_px = align_tol_px
        self.layout_row_tol = layout_row_tol
        self.layout_col_tol = layout_col_tol

    def alignment_tolerance_px(self, image_width: float) -> float:
        """Returns the spatial-edge alignment tolerance in pixels for a given image width.

        Input:  image_width — width of the image the boxes refer to (pixels)
        Output: align_tol_px if set, otherwise align_tol_frac * image_width
        """
        if self.align_tol_px is not None:
            return self.align_tol_px
        return self.align_tol_frac * image_width

    def build_graph(self, detections: List[Dict],
                    image_size: Optional[Sequence[float]] = None) -> Dict:
        """Builds the full UI graph: containment forest, synthetic layout groups and edges.

        Only graph['hierarchy'] is consumed by code generation, and of it only the
        containment structure: the synthetic SyntheticRow/SyntheticColumn groups
        and all graph['edges'] are an inspection/visualisation export.

        Input:  detections — component dictionaries (detector output, or plain boxes with
                    'bbox' and 'class_name'; see _normalise_detections); the input is not modified;
                image_size — optional (width, height) in pixels; if None it is taken from the
                    detections (see resolve_image_size)
        Output: dictionary {'nodes': copies of the input boxes (missing 'id'/'confidence'
                filled in), 'edges': parent_child + spatial edges, 'hierarchy': {'roots': [...]},
                'image_size': [w, h], 'params': thresholds actually used}
        """
        nodes = _normalise_detections(detections)
        (img_w, img_h), size_source = resolve_image_size(nodes, image_size)
        tolerance_px = self.alignment_tolerance_px(img_w)

        # 1. Containment: exactly one parent (or none) per node
        parent_of, containment_edges = self._find_containment_relations(nodes)

        # 2. Containment forest
        hierarchy = self._build_hierarchy_tree(nodes, parent_of)

        # 3. Synthetic Row/Column groups around the top-level nodes (inspection only)
        hierarchy = self._infer_layout_groups(hierarchy)

        # 4. Spatial relations between nodes with the same containment parent (inspection only)
        spatial_edges = self._find_spatial_relations(nodes, parent_of, tolerance_px)

        return {
            'nodes': nodes,
            'edges': [self._edge_to_dict(e) for e in containment_edges + spatial_edges],
            'hierarchy': hierarchy,
            'image_size': [img_w, img_h],
            'params': {
                'containment_ratio': self.containment_ratio,
                'align_tol_frac': self.align_tol_frac,
                'align_tol_px': self.align_tol_px,
                'tolerance_px': tolerance_px,
                'layout_row_tol': self.layout_row_tol,
                'layout_col_tol': self.layout_col_tol,
                'image_size_source': size_source,
            },
        }

    # ── Containment ────────────────────────────────────────────

    def _find_containment_relations(self, nodes: List[Dict]
                                    ) -> Tuple[Dict[int, Optional[int]], List[Edge]]:
        """Assigns every node its single containment parent (the smallest-area container) or none.

        Input:  nodes — normalised component nodes
        Output: (parent_of mapping {id: parent id or None}, list of PARENT_CHILD Edges in node order)
        """
        parent_of: Dict[int, Optional[int]] = {}
        edges: List[Edge] = []
        for node in nodes:
            containers = _containers_of(node, nodes, self.containment_ratio)
            if containers:
                parent = containers[0]
                parent_of[node['id']] = parent['id']
                edges.append(Edge(source_id=parent['id'],
                                  target_id=node['id'],
                                  relation_type=RelationType.PARENT_CHILD,
                                  coverage=coverage(node['bbox'], parent['bbox'])))
            else:
                parent_of[node['id']] = None
        return parent_of, edges

    def _build_hierarchy_tree(self, nodes: List[Dict],
                              parent_of: Dict[int, Optional[int]]) -> Dict:
        """Builds the containment forest; children and roots keep the input order.

        Input:  nodes — normalised component nodes; parent_of — {id: parent id or None}
        Output: hierarchy dictionary {'roots': [tree nodes]}; each tree node has
                'id', 'class', 'confidence', 'synthetic' (False), 'bbox', 'children'
        """
        by_id = {n['id']: n for n in nodes}
        children_of: Dict[int, List[int]] = {n['id']: [] for n in nodes}
        root_ids: List[int] = []
        for node in nodes:
            parent = parent_of[node['id']]
            if parent is None:
                root_ids.append(node['id'])
            else:
                children_of[parent].append(node['id'])

        def build_subtree(node_id: int) -> Dict:
            """Builds the subtree rooted at the given node.

            Input:  node_id — id of the subtree root
            Output: tree node dictionary
            """
            node = by_id[node_id]
            return {
                'id': node_id,
                'class': node['class_name'],
                'confidence': node['confidence'],
                'synthetic': False,
                'bbox': list(node['bbox']),
                'children': [build_subtree(cid) for cid in children_of[node_id]],
            }

        return {'roots': [build_subtree(rid) for rid in root_ids]}

    # ── Synthetic layout groups (inspection/visualisation export) ──

    def _infer_layout_groups(self, hierarchy: Dict) -> Dict:
        """Clusters the top-level nodes into rows (and aligned columns) and wraps them in synthetic containers.

        The result is an inspection/visualisation export; code generation treats the
        synthetic nodes as transparent. Every detected node keeps exactly one place.

        Input:  hierarchy — hierarchy dictionary containing the 'roots' list
        Output: updated hierarchy dictionary with roots wrapped in synthetic containers
        """
        roots = hierarchy['roots']
        if len(roots) < 2:
            return hierarchy

        # Step 1: Sort roots by y-center
        for r in roots:
            bbox = r['bbox']
            r['_cy'] = (bbox[1] + bbox[3]) / 2
            r['_cx'] = (bbox[0] + bbox[2]) / 2
            r['_h'] = bbox[3] - bbox[1]
        roots.sort(key=lambda r: r['_cy'])

        # Step 2: Cluster into horizontal rows (similar y-center)
        avg_height = sum(r['_h'] for r in roots) / len(roots)
        y_threshold = avg_height * self.layout_row_tol
        rows = []
        current_row = [roots[0]]

        for i in range(1, len(roots)):
            if abs(roots[i]['_cy'] - current_row[-1]['_cy']) < y_threshold:
                current_row.append(roots[i])
            else:
                rows.append(current_row)
                current_row = [roots[i]]
        rows.append(current_row)

        # Step 2.5: Merge consecutive rows with matching x-alignment
        rows = self._merge_vertically_aligned_rows(rows, avg_height)

        # Step 3: Build final structure from rows
        new_roots = []
        synthetic_id = -1

        for row in rows:
            if len(row) == 1:
                node = row[0]
                self._clean_temp_fields(node)
                new_roots.append(node)
            else:
                row.sort(key=lambda r: r['_cx'])

                # Check if this row contains merged vertical groups
                has_vertical = any(isinstance(r.get('_vgroup'), list) for r in row)

                if has_vertical:
                    # Build Row with Column sub-groups
                    row_children = []
                    for item in row:
                        vgroup = item.pop('_vgroup', None)
                        if vgroup and len(vgroup) > 1:
                            # Vertical sub-group → SyntheticColumn
                            vgroup.sort(key=lambda r: r['_cy'])
                            for v in vgroup:
                                self._clean_temp_fields(v)
                            col_node = self._make_synthetic_node(
                                synthetic_id, 'SyntheticColumn', vgroup
                            )
                            synthetic_id -= 1
                            row_children.append(col_node)
                        else:
                            self._clean_temp_fields(item)
                            row_children.append(item)

                    row_node = self._make_synthetic_node(
                        synthetic_id, 'SyntheticRow', row_children
                    )
                    synthetic_id -= 1
                    new_roots.append(row_node)
                else:
                    # Simple horizontal row
                    for item in row:
                        self._clean_temp_fields(item)
                    row_node = self._make_synthetic_node(
                        synthetic_id, 'SyntheticRow', row
                    )
                    synthetic_id -= 1
                    new_roots.append(row_node)

        # Step 4: Wrap everything in a root SyntheticColumn
        if len(new_roots) > 1:
            root_column = self._make_synthetic_node(
                synthetic_id, 'SyntheticColumn', new_roots
            )
            hierarchy['roots'] = [root_column]
        else:
            hierarchy['roots'] = new_roots

        return hierarchy

    def _merge_vertically_aligned_rows(self, rows: List[List[Dict]],
                                       avg_height: float) -> List[List[Dict]]:
        """Merges consecutive rows whose items align vertically (matching x-centers) into vertical groups (_vgroup).

        Example: if ["Reward Points", "Travel Trips", "Bucket List"] (row A) and
               ["360", "238", "473"] (row B) are aligned, they merge into a single row:
               [("Reward Points","360"), ("Travel Trips","238"), ("Bucket List","473")]

        Input:  rows — rows of root nodes (list of lists); avg_height — mean top-level box height,
                scaled by layout_col_tol to give the x-alignment tolerance
        Output: merged row list (representative nodes carry a _vgroup list with a copy of
                themselves and their partner; the representative itself is not emitted)
        """
        if len(rows) < 2:
            return rows

        x_threshold = avg_height * self.layout_col_tol  # x-alignment tolerance
        merged = []
        i = 0

        while i < len(rows):
            if i + 1 < len(rows):
                row_a = rows[i]
                row_b = rows[i + 1]

                # Only try merge if both rows have same number of elements (2+)
                if len(row_a) >= 2 and len(row_a) == len(row_b):
                    # Sort both by x-center
                    row_a_sorted = sorted(row_a, key=lambda n: n['_cx'])
                    row_b_sorted = sorted(row_b, key=lambda n: n['_cx'])

                    # Check if x-centers align pairwise
                    aligned = True
                    for a, b in zip(row_a_sorted, row_b_sorted):
                        if abs(a['_cx'] - b['_cx']) > x_threshold:
                            aligned = False
                            break

                    if aligned:
                        # Merge: create virtual groups with _vgroup marker
                        merged_row = []
                        for a, b in zip(row_a_sorted, row_b_sorted):
                            # Use 'a' as the representative, attach vertical group
                            a['_vgroup'] = [a.copy(), b]
                            merged_row.append(a)
                        merged.append(merged_row)
                        i += 2  # skip both rows
                        continue

            merged.append(rows[i])
            i += 1

        return merged

    def _make_synthetic_node(self, node_id: int, class_name: str,
                             children: List[Dict]) -> Dict:
        """Creates a synthetic container node (SyntheticRow/SyntheticColumn) with a bbox enclosing the children's boxes.

        Input:  node_id — negative synthetic id; class_name — 'SyntheticRow' or 'SyntheticColumn';
                children — child nodes to place in the container
        Output: container node dictionary with 'synthetic': True, 'confidence': None
                (not a detection) and id, class, class_name, bbox, children fields
        """
        all_bboxes = []

        def collect(n):
            """Collects the bboxes of the node and its whole subtree into the all_bboxes list.

            Input:  n — node dictionary that may have bbox and children fields
            Output: none (side effect of appending to the all_bboxes list)
            """
            if 'bbox' in n:
                all_bboxes.append(n['bbox'])
            for c in n.get('children', []):
                collect(c)
        for c in children:
            collect(c)

        if all_bboxes:
            bbox = [
                min(b[0] for b in all_bboxes),
                min(b[1] for b in all_bboxes),
                max(b[2] for b in all_bboxes),
                max(b[3] for b in all_bboxes),
            ]
        else:
            bbox = [0, 0, 0, 0]

        return {
            'id': node_id,
            'class': class_name,
            'class_name': class_name,
            'confidence': None,
            'synthetic': True,
            'bbox': bbox,
            'children': children,
        }

    def _clean_temp_fields(self, node: Dict):
        """Removes the temporary fields (_cy, _cx, _h) used during layout inference from the node.

        Input:  node — node dictionary to clean
        Output: none (side effect of updating the node in place)
        """
        node.pop('_cy', None)
        node.pop('_cx', None)
        node.pop('_h', None)

    # ── Spatial relations (inspection/visualisation export) ────

    def _find_spatial_relations(self, nodes: List[Dict],
                                parent_of: Dict[int, Optional[int]],
                                tolerance_px: float) -> List[Edge]:
        """Finds spatial (above/below/left/right) edges between nodes with the same containment parent.

        Top-level nodes (no parent) form one sibling group; the synthetic layout groups
        play no role here. Pairs are visited in input order (i < j).

        Input:  nodes — normalised component nodes; parent_of — {id: parent id or None};
                tolerance_px — alignment tolerance in pixels
        Output: list of Edges of type ABOVE/BELOW/LEFT_OF/RIGHT_OF
        """
        edges = []
        siblings_by_parent: Dict[Optional[int], List[Dict]] = {}
        for node in nodes:
            siblings_by_parent.setdefault(parent_of[node['id']], []).append(node)

        for siblings in siblings_by_parent.values():
            for i, node_i in enumerate(siblings):
                for node_j in siblings[i + 1:]:
                    relation = self._determine_spatial_relation(node_i, node_j, tolerance_px)
                    if relation:
                        edges.append(Edge(source_id=node_i['id'],
                                          target_id=node_j['id'],
                                          relation_type=relation))
        return edges

    def _determine_spatial_relation(self, node1: Dict, node2: Dict,
                                    tolerance_px: float) -> Optional[RelationType]:
        """Determines the spatial relation between two boxes from their centres (same rule as v1.1.0).

        If the x-centres differ by less than tolerance_px the boxes are vertically
        aligned (ABOVE if node1's centre is higher, otherwise BELOW); else, if the
        y-centres differ by less than tolerance_px, they are horizontally aligned
        (LEFT_OF / RIGHT_OF); otherwise there is no edge.

        Input:  node1, node2 — nodes with 'bbox'; tolerance_px — alignment tolerance (pixels)
        Output: RelationType (ABOVE/BELOW/LEFT_OF/RIGHT_OF) or None if there is no alignment
        """
        b1, b2 = node1['bbox'], node2['bbox']
        dx = (b2[0] + b2[2]) / 2 - (b1[0] + b1[2]) / 2
        dy = (b2[1] + b2[3]) / 2 - (b1[1] + b1[3]) / 2

        # Vertical relationships
        if abs(dx) < tolerance_px:
            return RelationType.ABOVE if dy > 0 else RelationType.BELOW

        # Horizontal relationships
        if abs(dy) < tolerance_px:
            return RelationType.LEFT_OF if dx > 0 else RelationType.RIGHT_OF

        return None

    def _edge_to_dict(self, edge: Edge) -> Dict:
        """Converts an Edge object into a JSON-serializable dictionary.

        Input:  edge — Edge carrying source, target, relation type (and coverage for containment)
        Output: dictionary with source, target, relation keys (+ coverage for parent_child)
        """
        out = {
            'source': edge.source_id,
            'target': edge.target_id,
            'relation': edge.relation_type.value,
        }
        if edge.coverage is not None:
            out['coverage'] = edge.coverage
        return out

    def visualize_graph(self, graph: Dict, output_path: str = "graph.json"):
        """Saves the graph as a JSON file for visualization.

        Input:  graph — dictionary output by build_graph; output_path — target JSON file path
        Output: none (side effect of writing JSON to disk and printing a console notice)
        """
        with open(output_path, 'w') as f:
            json.dump(graph, f, indent=2)
        print(f"✓ Graph saved: {output_path}")


def main():
    """Tries the graph builder end to end on saved detection output and prints the hierarchy.

    Input:  none (reads from the output/detections.json file)
    Output: none (side effect of saving output/ui_graph.json and printing a summary to the console)
    """
    # Load detections from previous step
    with open("output/detections.json", 'r') as f:
        detections = json.load(f)

    # Build graph
    builder = UIGraphBuilder()
    graph = builder.build_graph(detections)

    print(f"\n🌳 Graph Statistics:")
    print(f"  Nodes: {len(graph['nodes'])}")
    print(f"  Edges: {len(graph['edges'])}")
    print(f"  Root components: {len(graph['hierarchy']['roots'])}")
    print(f"  Image size: {graph['image_size']} ({graph['params']['image_size_source']})")

    # Save graph
    builder.visualize_graph(graph, "output/ui_graph.json")

    # Print hierarchy
    print(f"\n📊 Hierarchy:")

    def print_tree(node, indent=0):
        """Prints the hierarchy tree to the console in indented form.

        Input:  node — tree node to print; indent — indentation level
        Output: none (side effect of printing to the console)
        """
        print("  " * indent + f"└─ {node['class']} (id={node['id']})")
        for child in node.get('children', []):
            print_tree(child, indent + 1)

    for root in graph['hierarchy']['roots'][:3]:  # Show first 3 roots
        print_tree(root)


if __name__ == "__main__":
    main()
