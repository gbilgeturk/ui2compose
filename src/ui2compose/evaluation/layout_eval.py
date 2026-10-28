# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Layout evaluation: generated layout trees vs the ReDraw view hierarchies (TEDS, grouping F1).

Typical use (validation split of the regenerated ReDraw package; the
predictions come from the cache written by ``ui2compose.evaluation.pred_cache``):
    ui2compose-eval-layout --data-dir <package> --split val \
        --pred-cache <outside the repo>/pred_cache/val \
        --v110-code <outside the repo>/v110_code --out <outside the repo>/layout_val

The method is fixed here and documented once; results/eval/README.md repeats
it for readers of the results.

1. Reference tree (one per screen, from raw_xml/<split>/<pkg>-screens/hierarchy_<n>.xml)
   * The uiautomator dump is parsed with ElementTree. The labelling of
     parse_redraw.py is replayed node by node (trace_labels): same traversal
     order, container-name filter, bounds parsing, area filter, ALIASES /
     short_class_name, 13-class whitelist and same-class IoU de-duplication,
     with the parse_redraw command-line defaults the dataset was written with
     (PARSE_DROP_CONTAINER, PARSE_MIN_AREA, PARSE_DEDUP_IOU). Every
     replayed label is formatted exactly like a label line
     (format_label_line), so each line of the screen's YOLO label file is
     linked to the XML node that produced it (align_labels): first as a
     whole sequence, then line by line (exact string), then near-exactly
     (same class, every coordinate within NEAR_EXACT_TOL_PX pixels after
     to_xywhn). Unmatched label lines and unused replayed nodes are counted
     and reported; a screen with an unmatched label line is excluded
     ('alignment_incomplete'), so the reference detection set always equals
     the screen's ground-truth label boxes.
   * Detection nodes = the label boxes (class from the label, box
     denormalised from the label line with the image size and snapped to
     whole pixels within SNAP_PX, which undoes the 6-decimal rounding of the
     label lines; exactly the boxes the GT conditions receive). Container nodes = all other XML nodes
     with at least one detection node below them (their label is the short
     XML class, kept only for inspection). A label node with detection nodes
     below it (e.g. a ListView with its items, a Spinner with its TextView)
     stays an internal node labelled with its class.
2. Generated tree (one per screen and condition) = the layout tree the Kotlin
   is written from: ComposeCodeGenerator.last_layout_tree (v1.2.0) or the
   XY-cut tree of the v1.1.0 generator. Column / Row (all rules, including
   the fallbacks and the flat baseline) -> container nodes; leaves ->
   detection nodes. Composites keep their container detection: a LazyColumn
   becomes a detection node labelled with the list's class whose children
   are its items; a containment Box becomes a detection node labelled with
   the container's class whose children are the Box's children without the
   container's own widget leaf, i.e. the layout of the contained detections
   (the Column / Row / leaf the code emits inside the Box). This is the
   mapping the evaluation was specified with ('composite', the primary
   metric). The Kotlin itself draws Box { widget; content } with the widget
   and its content as siblings, which is how a dump describes a background
   image under its content (a FrameLayout with both as children), while a
   13-class container that is a ViewGroup in the dump (a Spinner with its
   TextView, a ListView with its items) is a parent there. A post-hoc
   sensitivity variant, added after the review of the revision and run once
   per split ('container', --box-repr container), scores a containment Box
   as a container node whose children are the container's widget leaf and
   the Box's other children; LazyColumns are unchanged. Only conditions with
   containment Boxes (C1, C4) change under it. The v1.1.0
   generator lays out only the leaves of its containment forest, so it has
   no composites; a detection it emits twice (a v1.1.0 hierarchy lists a
   box under every container whose box contains it) appears twice, and a
   detection it never emits is missing (both are counted as 'duplicates' /
   'lost').
3. Canonicalisation (canonicalize, identical for both trees): containers
   without a detection below them are pruned; a container with exactly one
   child is replaced by that child (recursively, so single-child chains
   collapse; detection nodes are never removed); children are sorted in
   reading order: top of the union box of their detections, then left (then
   bottom, right, label), except among siblings that are pairwise disjoint
   in x but not in y (a 'Row' by the rule of item 4), which are sorted left
   first (then top, right, bottom, label). With a strict top-then-left
   order, the items of a row whose tops differ by one pixel (common for
   predicted boxes) would be scrambled and every such row would cost
   spurious delete/insert pairs. The root is the top-most remaining container (for
   the generated tree the root Column of the code whenever it has two or
   more children); if the collapsed tree is a single detection (or empty),
   it is wrapped in one root container, in both trees alike.
4. Labels. TEDS-g: detection nodes by class, every container 'Group'.
   TEDS-o: detection nodes by class; a container is labelled by the
   orientation of its children, inferred from geometry with one rule for
   both trees (orientation_of): 'Column' if the children's y-intervals
   (of their union boxes) are pairwise non-overlapping, else 'Row' if their
   x-intervals are, else 'Box'; the tolerance is 0 (touching intervals do
   not overlap); a container with fewer than two children is a 'Column'.
5. Distance. Ordered tree edit distance with unit costs (insert, delete,
   rename = 1, rename of equal labels = 0) computed with APTED (package
   'apted'); TEDS = 1 - TED / max(|T_ref|, |T_gen|). The optimal edit
   mapping APTED returns is split into inserts, deletes and renames
   (structure-only correction proxy: the edits a developer would make to the
   generated tree to obtain the reference tree, ignoring texts, sizes and
   styles); their sum equals TED (checked). A sample of screens
   (--zss-check) is recomputed with the Zhang-Shasha implementation of the
   package 'zss' as a cross-check.
6. Grouping F1. Ground-truth boxes and the boxes given to the layout stage
   are matched one to one: identity for the GT conditions; for predicted
   boxes the Hungarian algorithm (scipy.optimize.linear_sum_assignment) on
   cost 1 - IoU with every pair of IoU < MATCH_IOU (0.5) priced out, which
   maximises the number of pairs with IoU >= 0.5 and then their total IoU;
   the class need not agree (class mismatches are counted separately). For
   every unordered pair of matched boxes, the pair is 'grouped' in a tree
   when the two detection nodes have the same direct parent in the
   canonical tree (for a detection emitted twice, its first occurrence in
   reading order). With TP = pairs grouped in both trees, precision = TP /
   grouped-in-generated, recall = TP / grouped-in-reference and F1 = 2 TP /
   (grouped-in-generated + grouped-in-reference); F1 is undefined (and not
   averaged) when no pair is grouped in either tree. A matched detection
   that the layout never emits has no parent and groups with nothing.
   Reported per screen (mean over screens with a defined F1) and
   micro-averaged (sums of the counts over screens). The means of two
   conditions are therefore over different screens (with predicted boxes,
   over what the detector found); paired differences use the screens where
   both define F1, and layout_report gives the numbers of screens and the
   means on the screens every condition defines.
7. Conditions (CONDITIONS): C1 GT boxes + v1.2.0 containment and XY-cut
   (no de-duplication: GT boxes have no detector duplicates); C2 GT boxes +
   flat Column; C3 raw predictions + the v1.1.0 graph builder and code
   generator as submitted (loaded from a copy of the v1.1.0 tag, checked
   against V110_SHA256, no de-duplication); C4 predictions + v1.2.0
   de-duplication, containment and XY-cut; C5 predictions + v1.2.0
   de-duplication, then every detection in one flat Column; C1p (C1') GT
   boxes + the v1.1.0 layout. Predictions are the cached outputs of
   ComponentDetector.detect() at the pipeline operating point (conf 0.3,
   NMS IoU 0.7, the checkpoint's imgsz 640, class-aware NMS, max_det 300).
   The thresholds of de-duplication, containment, the cut tolerance and the
   fallback are parameters of the run (default: the v1.2.0 pipeline defaults,
   selected on the validation split by layout_sweep, see
   results/eval_appsplit/selected_thresholds.json).
8. Exclusions (never silent; counted per reason in the aggregate):
   'no_mapping' (no raw XML for the image), 'no_label_file', 'empty_labels',
   'label_parse_error', 'image_unreadable', 'xml_parse_error',
   'alignment_incomplete' (reference level, all conditions), and per
   condition 'no_predictions' (screen missing from the cache),
   'error:generation:<message>' (the graph / code generation raised) and
   'error:tree_too_large' (generated tree above MAX_TED_NODES nodes; only
   possible for v1.1.0, see there). A screen with zero predicted boxes is NOT excluded:
   its generated tree is an empty root, scored as such.
9. Aggregation per condition and subset (all screens of the split, the
   unseen-app subset dataset/unseen_<split>.txt and the package-only unseen
   subset dataset/unseen_<split>_package_only.txt): mean / median /
   quartiles of TEDS-g, TEDS-o and grouping F1, micro grouping P / R / F1,
   median (IQR) of the edit operations, node counts and depths, fallback
   nodes, placeholders ('Text("Unknown: ...")' in the code), duplicates,
   lost detections and de-duplication removals. 95 % percentile bootstrap
   intervals (1,000 replicates, seed 0) resample app groups: every replicate
   draws as many groups as the subset has, with replacement, and takes all
   screens of every drawn group (units as ap._resampling_units). Intervals
   are given per resampling scheme and always named: 'full' (group_id of
   splits/full/manifest.csv, the primary scheme), 'package_md5'
   (splits/package_md5/manifest.csv) and 'screen' (each screen its own
   unit). The full grouping chains near-duplicate screens of different apps
   transitively into one group of 878 of the 1,869 val screens, so its
   intervals are wide and asymmetric (that group is drawn 0, 1, 2, ...
   times); the package_md5 grouping keeps apps apart without that chaining.
   The draws of a scheme are the same for every condition, so the
   differences C1-C2, C4-C5, C4-C3, C1-C4 (and C1-C1p) are paired
   (per-screen differences on the screens valid in both conditions).

Split guard: --split test requires --allow-test, because test numbers are
to be computed once, after every choice has been fixed on the validation split.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as _dt
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ui2compose import __version__
from ui2compose.evaluation import ap as apm
from ui2compose.infer.build_ui_graph import DEFAULT_CONTAINMENT_RATIO, UIGraphBuilder
from ui2compose.infer.generate_compose_code import (DEFAULT_CUT_TOL_FRAC, DEFAULT_FALLBACK,
                                                    FALLBACK_MODES, ComposeCodeGenerator)
from ui2compose.infer.postprocess import (DEFAULT_CROSS_CLASS_IOU, DEFAULT_SAME_CLASS_IOMIN,
                                          DEFAULT_SAME_CLASS_IOU, DETECTOR_CLASSES,
                                          deduplicate_detections)
from ui2compose.preprocess import parse_redraw as pr

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

#: parse_redraw.py command-line defaults (--drop-container, --min-area,
#: --dedup-iou) with which the dataset's label files were written; the label
#: replay uses the same values (a test checks them against parse_redraw).
PARSE_DROP_CONTAINER: bool = True
PARSE_MIN_AREA: float = 0.00005
PARSE_DEDUP_IOU: float = 0.85

#: Largest coordinate difference (pixels, after to_xywhn) of a near-exact
#: label/XML match, used only for label lines whose replayed string differs.
#: Chosen a priori (one pixel), not tuned.
NEAR_EXACT_TOL_PX: float = 1.0

#: Ground-truth coordinates within this distance (px) of a whole pixel are
#: snapped to it. Label lines carry 6 decimals, so integer XML bounds come
#: back within ~0.001 px (0.5e-6 x 1920); without snapping, touching boxes
#: would overlap by such amounts and the tolerance-0 orientation rule would
#: call a stack of touching boxes a 'Box'. Boxes clamped by to_xywhn (partly
#: off-screen) are not near-integer and stay as they are.
SNAP_PX: float = 0.01

#: IoU a predicted box needs with a ground-truth box to be matched to it.
MATCH_IOU: float = 0.5

#: A reference tree with at most this many levels (root = level 1) is "flat":
#: every detection hangs directly below the root container.
FLAT_MAX_DEPTH: int = 2

#: TEDS-g label of every container node.
GROUP_LABEL: str = 'Group'

#: Largest generated canonical tree (nodes) whose edit distance is computed.
#: v1.2.0 trees have at most 2 x max_det = 600 nodes; only a v1.1.0 tree that
#: repeats boxes listed under several containers can exceed it, and APTED on
#: such a tree would take minutes. A larger tree is recorded as
#: 'error:tree_too_large' (counted, excluded). Chosen a priori, not tuned.
MAX_TED_NODES: int = 1500

#: Scoring of a v1.2.0 containment Box (item 2 of the module docstring):
#: 'composite' (primary, as specified: a node of the container's class whose
#: children are the Box's content) or 'container' (post-hoc sensitivity: a
#: container node holding the container's widget leaf and the content).
BOX_REPRS: Tuple[str, ...] = ('composite', 'container')
DEFAULT_BOX_REPR: str = 'composite'

#: Layout rules counted as fallback nodes (no XY-cut existed).
FALLBACK_RULES: Tuple[str, ...] = ('fallback_y_sorted', 'least_overlap')

#: Text the generators emit for a component without a template.
PLACEHOLDER_MARK: str = 'Text("Unknown: '

#: Edit-mapping operations reported per variant.
EDIT_OPS: Tuple[str, ...] = ('insert', 'delete', 'rename')

#: Label variants of the tree edit distance.
VARIANTS: Tuple[str, ...] = ('g', 'o')

#: Conditions: box source ('gt' | 'pred'), code ('v120' | 'v110'), layout
#: mode of the v1.2.0 generator, de-duplication on/off, description.
CONDITIONS: Dict[str, Dict[str, Any]] = {
    'C1': {'boxes': 'gt', 'code': 'v120', 'layout': 'xycut', 'dedup': False,
           'description': 'ground-truth boxes + v1.2.0 containment and XY-cut'},
    'C2': {'boxes': 'gt', 'code': 'v120', 'layout': 'flat', 'dedup': False,
           'description': 'ground-truth boxes + flat Column (baseline)'},
    'C3': {'boxes': 'pred', 'code': 'v110', 'layout': None, 'dedup': False,
           'description': 'raw predictions + v1.1.0 graph and code generation as submitted'},
    'C4': {'boxes': 'pred', 'code': 'v120', 'layout': 'xycut', 'dedup': True,
           'description': 'predictions + v1.2.0 de-duplication, containment and XY-cut'},
    'C5': {'boxes': 'pred', 'code': 'v120', 'layout': 'flat', 'dedup': True,
           'description': 'predictions + v1.2.0 de-duplication, then one flat Column (baseline)'},
    'C1p': {'boxes': 'gt', 'code': 'v110', 'layout': None, 'dedup': False,
            'description': "ground-truth boxes + v1.1.0 layout (C1')"},
}

#: Conditions run when --conditions is not given.
DEFAULT_CONDITIONS: Tuple[str, ...] = ('C1', 'C2', 'C3', 'C4', 'C5')

#: Paired differences (first minus second) reported when both are run.
DEFAULT_DIFFS: Tuple[Tuple[str, str], ...] = (('C1', 'C2'), ('C4', 'C5'), ('C4', 'C3'),
                                              ('C1', 'C4'), ('C1', 'C1p'))

#: Subsets of the aggregate besides 'all': name -> list file in dataset/.
SUBSET_FILES: Dict[str, str] = {'unseen': 'unseen_{split}.txt',
                                'unseen_package_only': 'unseen_{split}_package_only.txt'}

#: Number of screens (first in stem order) recomputed with zss.
DEFAULT_ZSS_CHECK: int = 20

#: v1.1.0 files C3 / C1p need, and their SHA-256 at the v1.1.0 tag (commit
#: 865bc2e5b110fd3d3f60a8fc62bdeca772f4d616, src/ui2compose/infer/).
V110_SHA256: Dict[str, str] = {
    'build_ui_graph.py': '91bb1af29e319e3ec6ed9bed83bbff45fdad7db68885b36f7206b8eac12fa77d',
    'generate_compose_code.py': 'e505586dc4cab069598e0e9a1d41d6eb8cba25a151f073638d172f06848cab90',
}

#: Per-screen metrics that get bootstrap intervals.
CI_METRICS: Tuple[str, ...] = ('teds_g', 'teds_o', 'grouping_f1')

#: Default resampling schemes of the bootstrap: name -> manifest relative to
#: the package (column group_id). 'full' (app_split's full grouping: package
#: OR MD5 OR dHash <= 4, transitively closed) is the primary scheme;
#: 'package_md5' (app_split --near-dup-dist -1) is reported as well, because
#: the full grouping chains about half of val and test into one group (878 of
#: the 1,869 val screens), which then decides most of the interval width.
DEFAULT_GROUPINGS: Dict[str, str] = {'full': 'splits/full/manifest.csv',
                                     'package_md5': 'splits/package_md5/manifest.csv'}

#: Name of the scheme in which every screen is its own unit.
SCREEN_SCHEME: str = 'screen'


# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------

class TreeNode:
    """Node of a reference or generated layout tree.

    A detection node (cls is a class name) stands for one box — a ground-truth
    label (det_id = line index in the label file) or a detection (det_id = the
    detector's id) — and has children only if it is a composite. A container
    node (cls None) groups its children. 'src' records where the node comes
    from (XML class, layout rule), 'extent' is the union box of the
    detections in its subtree and 'orient' the geometric orientation of its
    children (both set by canonicalize).
    """

    __slots__ = ('cls', 'det_id', 'box', 'src', 'children', 'extent', 'orient')

    def __init__(self, cls: Optional[str], det_id: Any, box: Optional[Sequence[float]],
                 src: str, children: Optional[List['TreeNode']] = None):
        """Creates a node.

        Input:  cls — detector class (None for a container); det_id — box id
                (None for a container); box — [x1, y1, x2, y2] px (None for a
                container); src — provenance string; children — child nodes
        Output: none
        """
        self.cls = cls
        self.det_id = det_id
        self.box = None if box is None else [float(v) for v in box]
        self.src = src
        self.children: List[TreeNode] = list(children or [])
        self.extent: Optional[List[float]] = None
        self.orient: Optional[str] = None

    @property
    def is_container(self) -> bool:
        """Tells whether the node is a container (no box of its own).

        Input:  none
        Output: True for a container node
        """
        return self.cls is None

    def __repr__(self) -> str:
        """Short description for debugging.

        Input:  none
        Output: e.g. "TreeNode(TextView#3, 0 children)"
        """
        name = 'container:' + self.src if self.cls is None else f"{self.cls}#{self.det_id}"
        return f"TreeNode({name}, {len(self.children)} children)"


def container(src: str, children: Optional[List[TreeNode]] = None) -> TreeNode:
    """Creates a container node.

    Input:  src — provenance string (e.g. 'xml:LinearLayout', 'Column:gap'); children
    Output: TreeNode with cls None
    """
    return TreeNode(None, None, None, src, children)


def detection(cls: str, det_id: Any, box: Sequence[float], src: str = 'leaf',
              children: Optional[List[TreeNode]] = None) -> TreeNode:
    """Creates a detection node.

    Input:  cls — class name; det_id — box id; box — [x1, y1, x2, y2] px;
            src — provenance string; children — child nodes (composites only)
    Output: TreeNode
    """
    return TreeNode(cls, det_id, box, src, children)


def union_box(boxes: Iterable[Optional[Sequence[float]]]) -> Optional[List[float]]:
    """Union box of the given boxes (None entries are skipped).

    Input:  boxes — iterable of [x1, y1, x2, y2] or None
    Output: [min x1, min y1, max x2, max y2], or None if there is no box
    """
    bs = [b for b in boxes if b is not None]
    if not bs:
        return None
    return [min(b[0] for b in bs), min(b[1] for b in bs),
            max(b[2] for b in bs), max(b[3] for b in bs)]


def intervals_disjoint(extents: Sequence[Sequence[float]], axis: str) -> bool:
    """Tells whether the boxes' intervals along an axis are pairwise non-overlapping (tolerance 0).

    Two intervals that only touch (end == start) do not overlap.

    Input:  extents — boxes [x1, y1, x2, y2]; axis — 'y' or 'x'
    Output: True if no two intervals overlap by a positive length
    """
    lo, hi = (1, 3) if axis == 'y' else (0, 2)
    ivs = sorted((e[lo], e[hi]) for e in extents)
    if not ivs:
        return True
    run = ivs[0][1]
    for a, b in ivs[1:]:
        if a < run:
            return False
        run = max(run, b)
    return True


def orientation_of(extents: Sequence[Sequence[float]]) -> str:
    """Orientation label of a container from the geometry of its children (shared rule of TEDS-o).

    Input:  extents — union boxes of the children
    Output: 'Column' if the y-intervals are pairwise disjoint (also for fewer
            than two children), else 'Row' if the x-intervals are, else 'Box'
    """
    if intervals_disjoint(extents, 'y'):
        return 'Column'
    if intervals_disjoint(extents, 'x'):
        return 'Row'
    return 'Box'


def label_g(node: TreeNode) -> str:
    """TEDS-g label: the class of a detection node, 'Group' for a container.

    Input:  node — TreeNode
    Output: label string
    """
    return GROUP_LABEL if node.cls is None else node.cls


def label_o(node: TreeNode) -> str:
    """TEDS-o label: the class of a detection node, the geometric orientation of a container.

    Input:  node — canonical TreeNode (orient set by canonicalize)
    Output: label string
    """
    if node.cls is not None:
        return node.cls
    if node.orient is None:
        node.orient = orientation_of([c.extent for c in node.children if c.extent is not None])
    return node.orient


LABELERS = {'g': label_g, 'o': label_o}


def _reading_key(node: TreeNode, orient: str = 'Column') -> Tuple:
    """Reading-order sort key of a canonical node among its siblings.

    Top, left, bottom, right of its extent, then label; among the children
    of a 'Row' (x-disjoint but not y-disjoint siblings, see orientation_of)
    left, top, right, bottom, then label, so that a row is read left to right
    even when the tops of its items differ by a pixel.

    Input:  node — node with extent set; orient — orientation of the parent
    Output: sort key tuple
    """
    e = node.extent or [math.inf, math.inf, math.inf, math.inf]
    if orient == 'Row':
        return (e[0], e[1], e[2], e[3], label_g(node))
    return (e[1], e[0], e[3], e[2], label_g(node))


def canonicalize(root: Optional[TreeNode]) -> TreeNode:
    """Returns the canonical copy of a tree: pruned, single-child containers collapsed, children in reading order.

    Steps (bottom-up): a container without a detection below it is removed;
    a container with exactly one remaining child is replaced by that child;
    detection nodes are always kept; each node's extent is the union box of
    the detections in its subtree, the orientation of its children is
    computed with orientation_of (stored as 'orient' on containers) and its
    children are sorted by _reading_key with that orientation. If the result
    is not a container (a single detection, or nothing), it is wrapped in a
    root container (src 'root'), so every canonical tree has a container root.

    Input:  root — TreeNode or None (not modified)
    Output: new canonical TreeNode
    """
    def canon(node: TreeNode) -> Optional[TreeNode]:
        """Canonical copy of one subtree, or None if it holds no detection.

        Input:  node — TreeNode
        Output: TreeNode or None
        """
        kids = [k for k in (canon(c) for c in node.children) if k is not None]
        if node.cls is None:
            if not kids:
                return None
            if len(kids) == 1:
                return kids[0]
            out = container(node.src, kids)
        else:
            out = detection(node.cls, node.det_id, node.box, node.src, kids)
        out.extent = union_box([out.box] + [k.extent for k in kids])
        orient = orientation_of([k.extent for k in kids])
        out.children.sort(key=lambda k: _reading_key(k, orient))
        if out.cls is None:
            out.orient = orient
        return out

    top = canon(root) if root is not None else None
    if top is None:
        top = container('root', [])
        top.orient = orientation_of([])
    elif top.cls is not None:
        wrapped = container('root', [top])
        wrapped.extent = top.extent
        wrapped.orient = orientation_of([top.extent])
        top = wrapped
    return top


def iter_preorder(root: TreeNode) -> Iterator[Tuple[TreeNode, Optional[TreeNode], int]]:
    """Walks a tree in pre-order (children in stored order).

    Input:  root — TreeNode
    Output: iterator of (node, parent or None, level) with level 1 for the root
    """
    stack: List[Tuple[TreeNode, Optional[TreeNode], int]] = [(root, None, 1)]
    while stack:
        node, parent, level = stack.pop()
        yield node, parent, level
        for c in reversed(node.children):
            stack.append((c, node, level + 1))


def tree_stats(root: TreeNode) -> Dict[str, int]:
    """Size statistics of a tree.

    Input:  root — TreeNode
    Output: {'nodes', 'depth' (levels, root = 1), 'containers' (including the
             root), 'detections' (detection nodes), 'composites' (detection
             nodes with children)}
    """
    nodes = depth = containers = dets = composites = 0
    for node, _, level in iter_preorder(root):
        nodes += 1
        depth = max(depth, level)
        if node.cls is None:
            containers += 1
        else:
            dets += 1
            composites += int(bool(node.children))
    return {'nodes': nodes, 'depth': depth, 'containers': containers, 'detections': dets,
            'composites': composites}


def bracket(root: TreeNode, with_ids: bool = True) -> str:
    """Bracket notation of a canonical tree with TEDS-o labels (which also determine the TEDS-g labels).

    Detection nodes are written 'Class#id' (with_ids) and containers by
    their orientation, e.g. '{Column{TextView#0}{Row{Button#1}{ImageView#2}}}'.

    Input:  root — canonical TreeNode; with_ids — append '#id' to detection labels
    Output: string
    """
    def rec(n: TreeNode) -> str:
        """Bracket string of one subtree.

        Input:  n — TreeNode
        Output: string
        """
        lab = label_o(n)
        if n.cls is not None and with_ids:
            lab = f"{lab}#{n.det_id}"
        return '{' + lab + ''.join(rec(c) for c in n.children) + '}'
    return rec(root)


def render(root: TreeNode, max_lines: Optional[int] = None) -> List[str]:
    """Indented text rendering of a canonical tree (TEDS-o labels, ids and rounded extents).

    Input:  root — canonical TreeNode; max_lines — truncate after this many lines
    Output: list of lines
    """
    lines: List[str] = []
    for node, _, level in iter_preorder(root):
        e = node.extent
        ext = '' if e is None else ' [' + ','.join(f"{v:.0f}" for v in e) + ']'
        if node.cls is None:
            name = f"{label_o(node)} ({node.src})"
        else:
            name = f"{node.cls}#{node.det_id}" + (' (composite)' if node.children else '')
        lines.append('  ' * (level - 1) + name + ext)
        if max_lines is not None and len(lines) >= max_lines:
            lines.append('  ...')
            break
    return lines


def detection_parents(root: TreeNode) -> Tuple[Dict[Any, int], Counter]:
    """Direct parent of every detection id in a canonical tree (first occurrence in reading order).

    Nodes are numbered in pre-order (root = 0); the parent of a detection is
    the number of its parent node.

    Input:  root — canonical TreeNode
    Output: ({det_id: parent number}, Counter of occurrences per det_id)
    """
    number: Dict[int, int] = {}
    parents: Dict[Any, int] = {}
    occurrences: Counter = Counter()
    for k, (node, parent, _) in enumerate(iter_preorder(root)):
        number[id(node)] = k
        if node.cls is not None:
            occurrences[node.det_id] += 1
            if node.det_id not in parents and parent is not None:
                parents[node.det_id] = number[id(parent)]
    return parents, occurrences


# ---------------------------------------------------------------------------
# Tree edit distance
# ---------------------------------------------------------------------------

class _LabelNode:
    """Minimal labelled tree node for apted / zss ('name' and 'children')."""

    __slots__ = ('name', 'children')

    def __init__(self, name: str, children: List['_LabelNode']):
        """Creates a labelled node.

        Input:  name — label; children — child nodes
        Output: none
        """
        self.name = name
        self.children = children


def label_tree(root: TreeNode, variant: str) -> _LabelNode:
    """Converts a canonical tree into a labelled tree of one TEDS variant.

    Input:  root — canonical TreeNode; variant — 'g' or 'o'
    Output: _LabelNode tree
    """
    lab = LABELERS[variant]

    def rec(n: TreeNode) -> _LabelNode:
        """Labelled copy of one subtree.

        Input:  n — TreeNode
        Output: _LabelNode
        """
        return _LabelNode(lab(n), [rec(c) for c in n.children])
    return rec(root)


def _size(node: _LabelNode) -> int:
    """Number of nodes of a labelled tree.

    Input:  node — _LabelNode
    Output: node count
    """
    n, stack = 0, [node]
    while stack:
        x = stack.pop()
        n += 1
        stack.extend(x.children)
    return n


def tree_edit(ref: TreeNode, gen: TreeNode, variant: str) -> Dict[str, Any]:
    """Unit-cost tree edit distance (APTED), TEDS and the operations of the optimal edit mapping.

    The mapping edits the generated tree into the reference tree: 'insert' =
    reference node without a counterpart (missing in the generated tree),
    'delete' = generated node without a counterpart, 'rename' = mapped pair
    with different labels.

    Input:  ref, gen — canonical TreeNodes; variant — 'g' or 'o'
    Output: {'ted', 'teds', 'n_ref', 'n_gen', 'insert', 'delete', 'rename'};
            raises RuntimeError if the mapping cost differs from the distance
    """
    from apted import APTED, Config

    a, b = label_tree(gen, variant), label_tree(ref, variant)
    algo = APTED(a, b, Config())
    ted = int(algo.compute_edit_distance())
    mapping = algo.compute_edit_mapping()
    ops = {'insert': 0, 'delete': 0, 'rename': 0}
    for x, y in mapping:
        if x is None:
            ops['insert'] += 1
        elif y is None:
            ops['delete'] += 1
        elif x.name != y.name:
            ops['rename'] += 1
    if sum(ops.values()) != ted:
        raise RuntimeError(f"edit mapping cost {sum(ops.values())} != distance {ted}")
    n_gen, n_ref = _size(a), _size(b)
    return {'ted': ted, 'teds': 1.0 - ted / max(n_ref, n_gen), 'n_ref': n_ref, 'n_gen': n_gen,
            **ops}


def zss_distance(ref: TreeNode, gen: TreeNode, variant: str) -> int:
    """Unit-cost tree edit distance with the Zhang-Shasha implementation of 'zss' (cross-check).

    Input:  ref, gen — canonical TreeNodes; variant — 'g' or 'o'
    Output: distance (int)
    """
    import zss

    return int(zss.simple_distance(label_tree(gen, variant), label_tree(ref, variant),
                                   get_children=lambda n: n.children,
                                   get_label=lambda n: n.name,
                                   label_dist=lambda u, v: int(u != v)))


# ---------------------------------------------------------------------------
# Box matching and grouping
# ---------------------------------------------------------------------------

def iou_matrix(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> np.ndarray:
    """Pairwise IoU of two box lists (float64; 0 where the union is empty).

    Input:  a — N boxes; b — M boxes, [x1, y1, x2, y2]
    Output: (N, M) array
    """
    A = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    B = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    iw = np.clip(np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0]),
                 0, None)
    ih = np.clip(np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1]),
                 0, None)
    inter = iw * ih
    area_a = np.clip(A[:, 2] - A[:, 0], 0, None) * np.clip(A[:, 3] - A[:, 1], 0, None)
    area_b = np.clip(B[:, 2] - B[:, 0], 0, None) * np.clip(B[:, 3] - B[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)
    return out


def match_boxes(gt: Sequence[Sequence[float]], pred: Sequence[Sequence[float]],
                iou_thr: float = MATCH_IOU) -> List[Tuple[int, int, float]]:
    """One-to-one matching of ground-truth and predicted boxes (Hungarian on IoU, IoU >= iou_thr).

    Pairs below iou_thr get a prohibitive cost, so the assignment maximises
    the number of pairs with IoU >= iou_thr and, among those, the total IoU;
    pairs below the threshold are then dropped. Classes are not compared.

    Input:  gt — ground-truth boxes; pred — predicted boxes ([x1, y1, x2, y2]);
            iou_thr — minimum IoU of a match
    Output: list of (gt index, pred index, IoU) sorted by gt index
    """
    if not len(gt) or not len(pred):
        return []
    from scipy.optimize import linear_sum_assignment

    iou = iou_matrix(gt, pred)
    big = 1.0 + float(min(iou.shape))
    cost = np.where(iou >= iou_thr, 1.0 - iou, big)
    rows, cols = linear_sum_assignment(cost)
    out = [(int(i), int(j), float(iou[i, j])) for i, j in zip(rows, cols) if iou[i, j] >= iou_thr]
    return sorted(out)


def _pairs(n: int) -> int:
    """Number of unordered pairs among n items.

    Input:  n — count
    Output: n * (n - 1) / 2
    """
    return n * (n - 1) // 2


def grouping_scores(pairs: Sequence[Tuple[Any, Any]], ref_parent: Mapping[Any, int],
                    gen_parent: Mapping[Any, int]) -> Dict[str, Any]:
    """Grouping precision / recall / F1 over the pairs of matched boxes.

    Two matched boxes are grouped in a tree when their detection nodes have
    the same direct parent. A generated id without a parent (not emitted by
    the layout) groups with nothing.

    Input:  pairs — matched (reference id, generated id) pairs;
            ref_parent / gen_parent — {id: parent number} (detection_parents)
    Output: {'tp', 'n_ref' (pairs grouped in the reference), 'n_gen' (pairs
             grouped in the generated tree), 'precision', 'recall', 'f1'}
             (None where the denominator is 0)
    """
    ref_groups: Counter = Counter()
    gen_groups: Counter = Counter()
    cells: Counter = Counter()
    for r, g in pairs:
        rp = ref_parent[r]
        ref_groups[rp] += 1
        if g in gen_parent:
            gp = gen_parent[g]
            gen_groups[gp] += 1
            cells[(rp, gp)] += 1
    tp = sum(_pairs(v) for v in cells.values())
    n_ref = sum(_pairs(v) for v in ref_groups.values())
    n_gen = sum(_pairs(v) for v in gen_groups.values())
    return {'tp': tp, 'n_ref': n_ref, 'n_gen': n_gen,
            'precision': tp / n_gen if n_gen else None,
            'recall': tp / n_ref if n_ref else None,
            'f1': 2.0 * tp / (n_gen + n_ref) if (n_gen + n_ref) else None}


# ---------------------------------------------------------------------------
# Reference trees
# ---------------------------------------------------------------------------

def format_label_line(cid: int, box: Sequence[float]) -> str:
    """Formats one label exactly as parse_redraw.py writes it.

    Input:  cid — class id; box — (xc, yc, w, h) normalised
    Output: label line without newline
    """
    xc, yc, w, h = box
    return f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def read_label_lines(path: str | Path) -> List[str]:
    """Reads the non-empty lines of a YOLO label file (stripped).

    Input:  path — label .txt
    Output: list of lines
    """
    return [line.strip() for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def snap(v: float, snap_px: float = SNAP_PX) -> float:
    """Snaps a coordinate to the nearest whole pixel if it lies within snap_px of it.

    Input:  v — coordinate (px); snap_px — snapping distance (0 disables)
    Output: float(round(v)) or v
    """
    r = round(v)
    return float(r) if abs(v - r) <= snap_px else v


def parse_label_lines(lines: Sequence[str], img_w: int, img_h: int, names: Sequence[str],
                      snap_px: float = SNAP_PX) -> List[Dict[str, Any]]:
    """Converts label lines into pixel boxes (the ground-truth boxes of the GT conditions).

    The coordinates are denormalised with the image size and snapped to
    whole pixels within snap_px (undoes the 6-decimal rounding of the label
    lines, which moves integer XML bounds by up to ~0.001 px).

    Input:  lines — 'cid xc yc w h' lines; img_w, img_h — image size; names —
            class names; snap_px — see snap
    Output: list of {'id' (line index), 'class_id', 'class_name', 'bbox'
            [x1, y1, x2, y2] float px, 'confidence' 1.0}; raises ValueError
            for a malformed line or an unknown class id
    """
    out = []
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"label line {i} has {len(parts)} fields: {line!r}")
        cid = int(parts[0])
        if not 0 <= cid < len(names):
            raise ValueError(f"label line {i}: class id {cid} outside 0..{len(names) - 1}")
        xc, yc, w, h = (float(v) for v in parts[1:])
        box = [(xc - w / 2) * img_w, (yc - h / 2) * img_h, (xc + w / 2) * img_w, (yc + h / 2) * img_h]
        out.append({'id': i, 'class_id': cid, 'class_name': names[cid],
                    'bbox': [snap(v, snap_px) for v in box], 'confidence': 1.0})
    return out


def _node_box(n: ET.Element) -> Optional[Tuple[int, int, int, int]]:
    """Box of an XML node exactly as parse_redraw.extract_labels reads it.

    Input:  n — XML element
    Output: (x1, y1, x2, y2) pixels or None
    """
    b = None
    if 'bounds' in n.attrib:
        b = pr.parse_bounds(n.attrib.get('bounds'))
    if not b:
        try:
            x = int(n.attrib.get('x', '0'))
            y = int(n.attrib.get('y', '0'))
            w = int(n.attrib.get('width', '0'))
            h = int(n.attrib.get('height', '0'))
            b = (x, y, x + w, y + h) if (w > 0 and h > 0) else None
        except Exception:
            b = None
    return b


def _node_class(n: ET.Element) -> str:
    """Full class attribute of an XML node as parse_redraw reads it.

    Input:  n — XML element
    Output: class string ('' if absent)
    """
    return n.attrib.get('class') or n.attrib.get('className') or ''


def trace_labels(root: ET.Element, img_w: int, img_h: int, name2id: Mapping[str, int],
                 drop_container: bool = PARSE_DROP_CONTAINER, min_area: float = PARSE_MIN_AREA,
                 dedup_iou: float = PARSE_DEDUP_IOU
                 ) -> List[Tuple[ET.Element, int, Tuple[float, float, float, float]]]:
    """Replays parse_redraw.extract_labels on a parsed hierarchy and keeps the XML node of every label.

    Same traversal (parse_redraw.walk_nodes), container filter, bounds
    (with the x/y/width/height fallback), area filter, short_class_name with
    ALIASES, whitelist and same-class IoU de-duplication, in the same order.

    Input:  root — XML root element; img_w, img_h — screenshot size;
            name2id — class name -> id; drop_container, min_area, dedup_iou —
            parse_redraw settings (default: its command-line defaults)
    Output: list of (element, class id, (xc, yc, w, h)) in label order
    """
    labels: List[Tuple[ET.Element, int, Tuple[float, float, float, float]]] = []
    for n in pr.walk_nodes(root):
        cls_full = _node_class(n)
        if drop_container and 'container' in cls_full.lower():
            continue
        b = _node_box(n)
        if not b:
            continue
        box = pr.to_xywhn(b, img_w, img_h)
        if box[2] * box[3] < min_area:
            continue
        cname = pr.short_class_name(cls_full)
        if cname not in name2id:
            continue
        cid = name2id[cname]
        if any(c2 == cid and pr.iou_xywh(box, b2) >= dedup_iou for _, c2, b2 in labels):
            continue
        labels.append((n, cid, box))
    return labels


def align_labels(label_lines: Sequence[str],
                 traced: Sequence[Tuple[ET.Element, int, Tuple[float, float, float, float]]],
                 root: ET.Element, img_w: int, img_h: int, name2id: Mapping[str, int],
                 tol_px: float = NEAR_EXACT_TOL_PX
                 ) -> Tuple[List[Optional[ET.Element]], Dict[str, Any]]:
    """Links every label line of a screen to the XML node that produced it.

    1. 'sequence': the replayed labels, formatted, equal the label lines in order.
    2. Otherwise each line takes the first unused replayed label with the same
       string ('exact'); 3. a remaining line takes the closest unused XML node
       of the same class whose to_xywhn box differs by at most tol_px pixels
       in every coordinate ('near'; ties: traversal order).

    Input:  label_lines — lines of the label file; traced — trace_labels output;
            root — XML root (for the near-exact search); img_w, img_h — image
            size; name2id — class name -> id; tol_px — near-exact tolerance
    Output: (element per label line or None, report {'method', 'n_labels',
             'n_exact', 'n_near', 'n_unmatched', 'unmatched_lines' (indices),
             'n_traced', 'n_traced_unused'})
    """
    lines = list(label_lines)
    strings = [format_label_line(cid, box) for _, cid, box in traced]
    report: Dict[str, Any] = {'n_labels': len(lines), 'n_traced': len(traced)}
    if strings == lines:
        report.update(method='sequence', n_exact=len(lines), n_near=0, n_unmatched=0,
                      unmatched_lines=[], n_traced_unused=0)
        return [e for e, _, _ in traced], report

    pool: Dict[str, deque] = defaultdict(deque)
    for (e, _, _), s in zip(traced, strings):
        pool[s].append(e)
    elems: List[Optional[ET.Element]] = [None] * len(lines)
    used = set()
    for i, line in enumerate(lines):
        if pool.get(line):
            e = pool[line].popleft()
            elems[i] = e
            used.add(id(e))
    n_exact = sum(e is not None for e in elems)

    candidates = []
    for order, n in enumerate(pr.walk_nodes(root)):
        b = _node_box(n)
        cname = pr.short_class_name(_node_class(n))
        if b and cname in name2id:
            candidates.append((order, n, name2id[cname], pr.to_xywhn(b, img_w, img_h)))
    n_near = 0
    for i, line in enumerate(lines):
        if elems[i] is not None:
            continue
        parts = line.split()
        try:
            cid = int(parts[0])
            xc, yc, w, h = (float(v) for v in parts[1:5])
        except (ValueError, IndexError):
            continue
        best = None
        for order, n, c2, (x2, y2, w2, h2) in candidates:
            if c2 != cid or id(n) in used:
                continue
            dev = max(abs(xc - x2) * img_w, abs(yc - y2) * img_h,
                      abs(w - w2) * img_w, abs(h - h2) * img_h)
            if dev <= tol_px and (best is None or (dev, order) < best[0]):
                best = ((dev, order), n)
        if best is not None:
            elems[i] = best[1]
            used.add(id(best[1]))
            n_near += 1
    unmatched = [i for i, e in enumerate(elems) if e is None]
    report.update(method='per_line', n_exact=n_exact, n_near=n_near, n_unmatched=len(unmatched),
                  unmatched_lines=unmatched,
                  n_traced_unused=sum(1 for e, _, _ in traced if id(e) not in used))
    return elems, report


def build_reference_tree(root: ET.Element, label_of: Mapping[int, int],
                         gt_boxes: Sequence[Mapping[str, Any]]) -> Optional[TreeNode]:
    """Builds the (uncanonicalised) reference tree from the XML hierarchy and the aligned labels.

    Input:  root — XML root element; label_of — {id(element): label index};
            gt_boxes — parse_label_lines output (class and pixel box per label)
    Output: TreeNode tree (None if no label lies below root): label nodes are
            detection nodes (src 'xml:<short class>'), every other node with
            a label below it is a container
    """
    def rec(elem: ET.Element) -> Optional[TreeNode]:
        """Tree of one XML subtree.

        Input:  elem — XML element
        Output: TreeNode or None
        """
        kids = [k for k in (rec(c) for c in elem) if k is not None]
        src = 'xml:' + (pr.short_class_name(_node_class(elem)) or elem.tag)
        li = label_of.get(id(elem))
        if li is not None:
            b = gt_boxes[li]
            return detection(b['class_name'], b['id'], b['bbox'], src, kids)
        if kids:
            return container(src, kids)
        return None
    return rec(root)


def xml_shape(root: ET.Element) -> Dict[str, int]:
    """Size of the raw XML hierarchy.

    Input:  root — XML root element
    Output: {'xml_nodes' (elements, including the root), 'xml_depth' (levels)}
    """
    n, depth, stack = 0, 0, [(root, 1)]
    while stack:
        e, d = stack.pop()
        n += 1
        depth = max(depth, d)
        stack.extend((c, d + 1) for c in e)
    return {'xml_nodes': n, 'xml_depth': depth}


def build_reference(stem: str, xml_path: Optional[str | Path], label_path: str | Path,
                    image_path: str | Path, names: Sequence[str]
                    ) -> Tuple[Dict[str, Any], Optional[TreeNode], List[Dict[str, Any]]]:
    """Builds the reference tree of one screen.

    Input:  stem — image stem; xml_path — raw hierarchy XML (None: no mapping);
            label_path — YOLO label file; image_path — screenshot (for its
            size); names — class names in id order
    Output: (record, canonical reference tree or None, ground-truth boxes);
            record holds 'stem', 'status' ('ok' or 'excluded:<reason>'),
            'image_size', 'alignment', XML and tree statistics, 'flat',
            'time_s' and (status ok) 'tree' in bracket notation
    """
    t0 = time.perf_counter()
    rec: Dict[str, Any] = {'stem': stem}

    def done(status: str) -> Tuple[Dict[str, Any], None, List]:
        """Finishes an excluded record.

        Input:  status — exclusion status
        Output: (record, None, [])
        """
        rec['status'] = status
        rec['time_s'] = time.perf_counter() - t0
        return rec, None, []

    if xml_path is None:
        return done('excluded:no_mapping')
    if not Path(label_path).is_file():
        return done('excluded:no_label_file')
    lines = read_label_lines(label_path)
    if not lines:
        return done('excluded:empty_labels')
    try:
        with Image.open(image_path) as im:
            img_w, img_h = im.size
    except Exception:
        return done('excluded:image_unreadable')
    rec['image_size'] = [img_w, img_h]
    try:
        gt = parse_label_lines(lines, img_w, img_h, names)
    except ValueError:
        return done('excluded:label_parse_error')
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError):
        return done('excluded:xml_parse_error')
    name2id = {n: i for i, n in enumerate(names)}
    traced = trace_labels(root, img_w, img_h, name2id)
    elems, report = align_labels(lines, traced, root, img_w, img_h, name2id)
    rec['alignment'] = report
    rec.update(xml_shape(root))
    if report['n_unmatched']:
        return done('excluded:alignment_incomplete')
    raw = build_reference_tree(root, {id(e): i for i, e in enumerate(elems)}, gt)
    tree = canonicalize(raw)
    stats = tree_stats(tree)
    rec.update({f"ref_{k}": v for k, v in stats.items()})
    rec['flat'] = stats['depth'] <= FLAT_MAX_DEPTH
    rec['n_gt'] = len(gt)
    rec['status'] = 'ok'
    rec['tree'] = bracket(tree)
    rec['time_s'] = time.perf_counter() - t0
    return rec, tree, gt


# ---------------------------------------------------------------------------
# Generated trees
# ---------------------------------------------------------------------------

def tree_from_layout(node: Mapping[str, Any], box_repr: str = DEFAULT_BOX_REPR
                     ) -> Tuple[TreeNode, int]:
    """Converts a v1.2.0 layout tree (build_layout_tree / last_layout_tree) into a TreeNode tree.

    Column / Row -> container (src '<type>:<rule>'); leaf -> detection node;
    Box (rule containment) -> with box_repr 'composite', a detection node of
    the container's class whose children are the Box's children except the
    container's own widget leaf; with 'container', a container node (src
    'Box:containment') whose children are the widget leaf and the other
    children; LazyColumn -> detection node of the list's class whose children
    are its items (both variants).

    Input:  node — layout node dictionary; box_repr — one of BOX_REPRS
    Output: (TreeNode, number of fallback nodes: rule in FALLBACK_RULES, or a
             LazyColumn whose items were split by such a rule)
    """
    if box_repr not in BOX_REPRS:
        raise ValueError(f"box_repr must be one of {BOX_REPRS}, got {box_repr!r}")
    t = node['type']
    if t == 'leaf':
        info = node['node']
        return detection(info['class'], info['id'], info['bbox'], 'leaf'), 0
    n_fb = int(node.get('rule') in FALLBACK_RULES)
    if t in ('Column', 'Row'):
        kids = [tree_from_layout(c, box_repr) for c in node['children']]
        return (container(f"{t}:{node['rule']}", [k for k, _ in kids]),
                n_fb + sum(f for _, f in kids))
    if t == 'Box':
        c = node['container']
        widget, inner = node['children'][0], node['children'][1:]
        if widget.get('type') != 'leaf' or widget['node']['id'] != c['id']:
            raise ValueError(f"Box {c['id']}: first child is not the container's own widget")
        kids = [tree_from_layout(x, box_repr) for x in inner]
        n_kids = n_fb + sum(f for _, f in kids)
        if box_repr == 'container':
            leaf = detection(c['class'], c['id'], c['bbox'], 'leaf')
            return container('Box:containment', [leaf] + [k for k, _ in kids]), n_kids
        return detection(c['class'], c['id'], c['bbox'], 'Box', [k for k, _ in kids]), n_kids
    if t == 'LazyColumn':
        c = node['container']
        kids = [tree_from_layout(x, box_repr) for x in node['children']]
        n_fb += int(node.get('split_rule') in FALLBACK_RULES)
        return (detection(c['class'], c['id'], c['bbox'], f"LazyColumn:{node.get('split_rule')}",
                          [k for k, _ in kids]),
                n_fb + sum(f for _, f in kids))
    raise ValueError(f"unknown layout node type {t!r}")


def tree_from_v110(node: Optional[Mapping[str, Any]]) -> Tuple[Optional[TreeNode], int]:
    """Converts the XY-cut tree of the v1.1.0 generator into a TreeNode tree.

    Input:  node — v1.1.0 layout dictionary ('leaf' with 'node' {id, class,
            bbox}, or 'Column' / 'Row' with 'children', optional 'rule' tag
            set by the tracing generator), or None (nothing laid out)
    Output: (TreeNode or None, number of fallback nodes)
    """
    if node is None:
        return None, 0
    if node['type'] == 'leaf':
        info = node['node']
        return detection(info['class'], info['id'], info['bbox'], 'leaf'), 0
    rule = node.get('rule', 'gap')
    kids = [tree_from_v110(c) for c in node['children']]
    return (container(f"{node['type']}:{rule}", [k for k, _ in kids if k is not None]),
            int(rule in FALLBACK_RULES) + sum(f for _, f in kids))


class V110Code:
    """The v1.1.0 graph builder and code generator, loaded from files of the v1.1.0 tag."""

    def __init__(self, code_dir: str | Path, check_sha256: bool = True):
        """Loads build_ui_graph.py and generate_compose_code.py of v1.1.0 under private module names.

        Input:  code_dir — directory holding the two files (e.g. written with
                'git show v1.1.0:src/ui2compose/infer/<file>'); check_sha256 —
                require the SHA-256 of the v1.1.0 tag (V110_SHA256)
        Output: none (attributes graph, codegen, generator_cls, sha256,
                dir); raises FileNotFoundError / ValueError
        """
        d = Path(code_dir)
        self.dir = str(d)
        self.sha256: Dict[str, str] = {}
        for name in V110_SHA256:
            p = d / name
            if not p.is_file():
                raise FileNotFoundError(f"v1.1.0 code file missing: {p}")
            self.sha256[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        self.is_v110_tag = self.sha256 == V110_SHA256
        if check_sha256 and not self.is_v110_tag:
            raise ValueError(f"{d}: files differ from the v1.1.0 tag (sha256 {self.sha256})")
        self.graph = _load_module('_ui2compose_v110_build_ui_graph', d / 'build_ui_graph.py')
        self.codegen = _load_module('_ui2compose_v110_generate_compose_code',
                                    d / 'generate_compose_code.py')
        self.generator_cls = _tracing_v110_generator(self.codegen.ComposeCodeGenerator)


def _load_module(name: str, path: Path):
    """Imports a Python file as a module under the given name.

    Input:  name — module name; path — .py file
    Output: module object
    """
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tracing_v110_generator(base: type) -> type:
    """Subclass of the v1.1.0 ComposeCodeGenerator that keeps its XY-cut tree and marks fallbacks.

    The code and the tree are unchanged: _xy_cut only records, for every call,
    whether neither axis had a cut (the v1.1.0 'honest vertical ordering'
    fallback, tagged rule 'fallback_y_sorted') and keeps the outermost tree
    in last_layout_tree.

    Input:  base — v1.1.0 ComposeCodeGenerator class
    Output: subclass
    """
    class TracingV110Generator(base):
        """v1.1.0 generator that records its layout tree."""

        def generate(self, graph: Dict, use_xy_cut: bool = True) -> str:
            """Runs the v1.1.0 generate() and keeps the XY-cut tree.

            Input:  graph — v1.1.0 UI graph; use_xy_cut — as in v1.1.0
            Output: Kotlin code; self.last_layout_tree is the tree or None
            """
            self.last_layout_tree = None
            self._xy_depth = 0
            return super().generate(graph, use_xy_cut)

        def _xy_cut(self, boxes: List[Dict], prefer: str = 'y') -> Dict:
            """The v1.1.0 _xy_cut, plus a fallback tag and the outermost tree.

            Input:  boxes, prefer — as in v1.1.0
            Output: the v1.1.0 layout tree dictionary
            """
            no_cut = (len(boxes) > 1 and self._split_on_gap(boxes, 'y') is None
                      and self._split_on_gap(boxes, 'x') is None)
            self._xy_depth += 1
            try:
                tree = super()._xy_cut(boxes, prefer)
            finally:
                self._xy_depth -= 1
            if no_cut:
                tree['rule'] = 'fallback_y_sorted'
            if self._xy_depth == 0:
                self.last_layout_tree = tree
            return tree

    return TracingV110Generator


def generate_v120(boxes: Sequence[Mapping[str, Any]], image_size: Sequence[float], layout: str,
                  containment_ratio: float = DEFAULT_CONTAINMENT_RATIO,
                  cut_tol_frac: float = DEFAULT_CUT_TOL_FRAC,
                  fallback: str = DEFAULT_FALLBACK,
                  box_repr: str = DEFAULT_BOX_REPR) -> Tuple[TreeNode, str, int]:
    """Runs the v1.2.0 graph and code generation as run_pipeline does and returns the layout tree.

    Input:  boxes — detections or ground-truth boxes ('id', 'class_name',
            'bbox', optional 'confidence'); image_size — (w, h) px; layout —
            'xycut' or 'flat'; containment_ratio, cut_tol_frac, fallback —
            v1.2.0 thresholds; box_repr — scoring of containment Boxes
            (BOX_REPRS)
    Output: (TreeNode tree, Kotlin code, number of fallback nodes)
    """
    graph = UIGraphBuilder(containment_ratio=containment_ratio).build_graph(
        [dict(b) for b in boxes], image_size=image_size)
    gen = ComposeCodeGenerator()
    code = gen.generate(graph, image_size=image_size, layout=layout,
                        cut_tol_frac=cut_tol_frac, fallback=fallback)
    tree, n_fb = tree_from_layout(gen.last_layout_tree, box_repr)
    return tree, code, n_fb


def generate_v110(boxes: Sequence[Mapping[str, Any]], v110: V110Code
                  ) -> Tuple[Optional[TreeNode], str, int]:
    """Runs the v1.1.0 graph builder and code generator (as v1.1.0's run_pipeline) and returns the layout tree.

    The boxes are deep-copied; 'center' is added where missing (the v1.1.0
    spatial relations read it; detector output already has it).

    Input:  boxes — detections or ground-truth boxes; v110 — V110Code
    Output: (TreeNode tree or None, Kotlin code, number of fallback nodes)
    """
    dets = []
    for b in boxes:
        d = copy.deepcopy(dict(b))
        if 'center' not in d:
            x1, y1, x2, y2 = d['bbox']
            d['center'] = [(x1 + x2) / 2, (y1 + y2) / 2]
        dets.append(d)
    graph = v110.graph.UIGraphBuilder().build_graph(dets)
    gen = v110.generator_cls()
    code = gen.generate(graph)
    tree, n_fb = tree_from_v110(gen.last_layout_tree)
    return tree, code, n_fb


# ---------------------------------------------------------------------------
# One screen
# ---------------------------------------------------------------------------

def default_params() -> Dict[str, Any]:
    """Default thresholds of a run (the v1.2.0 defaults).

    Input:  none
    Output: {'dedup': {same_class_iou, same_class_iomin, cross_class_iou},
             'containment_ratio', 'cut_tol_frac', 'fallback', 'match_iou'}
    """
    return {'dedup': {'same_class_iou': DEFAULT_SAME_CLASS_IOU,
                      'same_class_iomin': DEFAULT_SAME_CLASS_IOMIN,
                      'cross_class_iou': DEFAULT_CROSS_CLASS_IOU},
            'containment_ratio': DEFAULT_CONTAINMENT_RATIO,
            'cut_tol_frac': DEFAULT_CUT_TOL_FRAC,
            'fallback': DEFAULT_FALLBACK,
            'match_iou': MATCH_IOU}


def evaluate_condition(cond: str, stem: str, ref_tree: TreeNode, gt: Sequence[Mapping[str, Any]],
                       image_size: Sequence[float], dets: Optional[Sequence[Mapping[str, Any]]],
                       params: Mapping[str, Any], v110: Optional[V110Code] = None,
                       zss_check: bool = False, save_tree: bool = True,
                       box_repr: str = DEFAULT_BOX_REPR) -> Dict[str, Any]:
    """Evaluates one condition on one screen against its reference tree.

    Input:  cond — key of CONDITIONS; stem — image stem; ref_tree — canonical
            reference tree; gt — ground-truth boxes (parse_label_lines);
            image_size — (w, h) px; dets — cached raw detections (None if not
            cached); params — default_params()-shaped thresholds; v110 —
            V110Code (C3 / C1p); zss_check — also compute the zss distances;
            save_tree — include the generated tree in bracket notation;
            box_repr — scoring of v1.2.0 containment Boxes (BOX_REPRS)
    Output: per-screen record ('status' 'ok', 'excluded:no_predictions',
            'error:generation:<message>' or 'error:tree_too_large:<n> nodes';
            for 'ok' the counts, TEDS and edit operations per variant,
            grouping scores and timings)
    """
    spec = CONDITIONS[cond]
    rec: Dict[str, Any] = {'stem': stem, 'condition': cond}
    t0 = time.perf_counter()
    if spec['boxes'] == 'gt':
        layout_in = [dict(b) for b in gt]
        n_input, n_removed = len(gt), 0
    else:
        if dets is None:
            rec['status'] = 'excluded:no_predictions'
            return rec
        n_input = len(dets)
        if spec['dedup']:
            kept, removed = deduplicate_detections(list(dets), **params['dedup'])
            layout_in, n_removed = list(kept), len(removed)
        else:
            layout_in, n_removed = list(dets), 0
    try:
        if spec['code'] == 'v120':
            raw, code, n_fb = generate_v120(layout_in, image_size, spec['layout'],
                                            containment_ratio=params['containment_ratio'],
                                            cut_tol_frac=params['cut_tol_frac'],
                                            fallback=params['fallback'], box_repr=box_repr)
        else:
            if v110 is None:
                raise ValueError(f"condition {cond} needs the v1.1.0 code (--v110-code)")
            raw, code, n_fb = generate_v110(layout_in, v110)
    except Exception as e:
        rec['status'] = f"error:generation:{type(e).__name__}: {e}"
        return rec
    t1 = time.perf_counter()

    gen = canonicalize(raw)
    gstats = tree_stats(gen)
    if gstats['nodes'] > MAX_TED_NODES:
        rec['status'] = f"error:tree_too_large:{gstats['nodes']} nodes > {MAX_TED_NODES}"
        return rec
    gen_parent, occurrences = detection_parents(gen)
    ref_parent, _ = detection_parents(ref_tree)
    if spec['boxes'] == 'gt':
        pairs = [(b['id'], b['id']) for b in gt]
        mismatch, ious = 0, []
    else:
        m = match_boxes([b['bbox'] for b in gt], [d['bbox'] for d in layout_in],
                        params.get('match_iou', MATCH_IOU))
        pairs = [(gt[i]['id'], layout_in[j]['id']) for i, j, _ in m]
        mismatch = sum(gt[i]['class_name'] != layout_in[j]['class_name'] for i, j, _ in m)
        ious = [v for _, _, v in m]
    in_ids = [d['id'] for d in layout_in]
    rec.update({
        'status': 'ok',
        'n_gt': len(gt), 'n_input': n_input, 'n_layout_input': len(layout_in),
        'n_dedup_removed': n_removed, 'n_matched': len(pairs), 'n_class_mismatch': mismatch,
        'matched_iou_mean': float(np.mean(ious)) if ious else None,
        'gen_nodes': gstats['nodes'], 'gen_depth': gstats['depth'],
        'gen_containers': gstats['containers'], 'gen_composites': gstats['composites'],
        'n_fallback': n_fb, 'n_placeholder': code.count(PLACEHOLDER_MARK),
        'n_duplicates': sum(v - 1 for v in occurrences.values() if v > 1),
        'n_lost': sum(1 for i in in_ids if i not in occurrences),
    })
    for v in VARIANTS:
        r = tree_edit(ref_tree, gen, v)
        rec[f"ted_{v}"] = r['ted']
        rec[f"teds_{v}"] = r['teds']
        rec[f"ops_{v}"] = {k: r[k] for k in EDIT_OPS}
        rec['ref_nodes'] = r['n_ref']
    grouping = grouping_scores(pairs, ref_parent, gen_parent)
    rec['grouping'] = grouping
    rec['grouping_f1'] = grouping['f1']
    t2 = time.perf_counter()
    rec['time_s'] = {'generate': t1 - t0, 'metrics': t2 - t1}
    if zss_check:
        rec['zss'] = {v: zss_distance(ref_tree, gen, v) for v in VARIANTS}
        rec['zss_agrees'] = all(rec['zss'][v] == rec[f"ted_{v}"] for v in VARIANTS)
        rec['time_s']['zss'] = time.perf_counter() - t2
    if save_tree:
        rec['tree'] = bracket(gen)
    return rec


_WORKER: Dict[str, Any] = {}


def _worker_v110(code_dir: Optional[str]) -> Optional[V110Code]:
    """Loads (once per process) the v1.1.0 code of a directory.

    Input:  code_dir — directory or None
    Output: V110Code or None
    """
    if code_dir is None:
        return None
    if _WORKER.get('v110_dir') != code_dir:
        _WORKER['v110'] = V110Code(code_dir)
        _WORKER['v110_dir'] = code_dir
    return _WORKER['v110']


def load_predictions_file(path: Optional[str | Path]) -> Optional[List[Dict[str, Any]]]:
    """Reads one screen's cached detections (format of 1_detections_raw.json, see pred_cache.py).

    Input:  path — cache file or None
    Output: list of detection dicts, or None if the path is None or missing
    """
    if path is None or not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text(encoding='utf-8'))


def evaluate_screen(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Builds the reference of one screen and evaluates every requested condition on it.

    Input:  task — {'stem', 'xml', 'label', 'image', 'pred' (cache file or
            None), 'names', 'conditions', 'params', 'v110_dir', 'zss',
            'save_trees', optional 'box_repr'}
    Output: {'reference': reference record, 'conditions': {cond: record}};
            a condition of an excluded reference gets the reference's status
    """
    ref_rec, ref_tree, gt = build_reference(task['stem'], task['xml'], task['label'],
                                            task['image'], task['names'])
    if not task.get('save_trees', True):
        ref_rec.pop('tree', None)
    out: Dict[str, Any] = {'reference': ref_rec, 'conditions': {}}
    dets = load_predictions_file(task.get('pred'))
    if dets is not None and ref_rec.get('image_size'):
        sizes = {(d.get('img_w'), d.get('img_h')) for d in dets if d.get('img_w') is not None}
        ref_rec['pred_size_mismatch'] = bool(sizes) and sizes != {tuple(ref_rec['image_size'])}
    v110 = _worker_v110(task.get('v110_dir'))
    for cond in task['conditions']:
        if ref_tree is None:
            out['conditions'][cond] = {'stem': task['stem'], 'condition': cond,
                                       'status': ref_rec['status']}
            continue
        out['conditions'][cond] = evaluate_condition(
            cond, task['stem'], ref_tree, gt, ref_rec['image_size'], dets, task['params'],
            v110=v110, zss_check=task.get('zss', False), save_tree=task.get('save_trees', True),
            box_repr=task.get('box_repr', DEFAULT_BOX_REPR))
    return out


# ---------------------------------------------------------------------------
# Inputs of a run
# ---------------------------------------------------------------------------

def load_names(data_dir: str | Path) -> List[str]:
    """Class names of the package's dataset.yaml (must equal the detector classes).

    Input:  data_dir — package root
    Output: names in id order; raises ValueError if they differ from DETECTOR_CLASSES
    """
    names, _ = pr.load_names(Path(data_dir) / 'dataset' / 'dataset.yaml')
    if tuple(names) != DETECTOR_CLASSES:
        raise ValueError(f"dataset.yaml classes {names} differ from the detector classes")
    return list(names)


def load_mapping(data_dir: str | Path, split: str) -> Dict[str, Dict[str, str]]:
    """Rows of raw_xml/mapping.csv for one split, by image stem.

    Input:  data_dir — package root; split — 'val' or 'test'
    Output: {stem: row}
    """
    with open(Path(data_dir) / 'raw_xml' / 'mapping.csv', newline='', encoding='utf-8') as f:
        return {r['stem']: r for r in csv.DictReader(f) if r['split'] == split}


def read_stem_list(path: str | Path) -> List[str]:
    """Image stems of an Ultralytics image-list file (lines './images/<split>/<file>').

    Input:  path — list file
    Output: stems in file order
    """
    out = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            out.append(Path(line).stem)
    return out


def load_subsets(data_dir: str | Path, split: str) -> Dict[str, List[str]]:
    """Stems of the unseen-app subsets of a split (SUBSET_FILES); missing files are skipped.

    Input:  data_dir — package root; split — split name
    Output: {subset name: stems}
    """
    out = {}
    for name, pattern in SUBSET_FILES.items():
        p = Path(data_dir) / 'dataset' / pattern.format(split=split)
        if p.is_file():
            out[name] = read_stem_list(p)
    return out


def default_grouping_files(data_dir: str | Path) -> Dict[str, Path]:
    """Manifests of the default resampling schemes that exist in a package (DEFAULT_GROUPINGS order).

    Input:  data_dir — package root
    Output: {scheme name: manifest.csv path}
    """
    out = {}
    for name, rel in DEFAULT_GROUPINGS.items():
        p = Path(data_dir) / rel
        if p.is_file():
            out[name] = p
    return out


def load_groupings(files: Mapping[str, str | Path], stems: Sequence[str]
                   ) -> Dict[str, Dict[str, str]]:
    """Reads the group of every screen per resampling scheme, plus the 'screen' scheme.

    Input:  files — {scheme name: manifest.csv with columns 'stem' and
            'group_id'} (order kept; the first is the primary scheme);
            stems — screens of the run
    Output: {scheme name: {stem: group}} with SCREEN_SCHEME last (each screen
            its own unit)
    """
    from ui2compose.evaluation.evaluate_detector import load_groups as _load

    out: Dict[str, Dict[str, str]] = {name: _load(path) for name, path in files.items()}
    out[SCREEN_SCHEME] = {s: s for s in stems}
    return out


def select_stems(data_dir: str | Path, split: str, images: Optional[str] = None,
                 limit: Optional[int] = None) -> List[str]:
    """Stems to evaluate: every image of the split, or the images of a list / directory.

    Input:  data_dir — package root; split — split name; images — image-list
            file or directory (default dataset/images/<split>); limit — keep
            the first N stems (sorted)
    Output: sorted unique stems
    """
    from ui2compose.evaluation.evaluate_detector import read_image_list

    spec = images or str(Path(data_dir) / 'dataset' / 'images' / split)
    stems = sorted({p.stem for p in read_image_list(spec)})
    return stems[:limit] if limit else stems


def prediction_meta(pred_cache: Optional[str | Path]) -> Optional[Dict[str, Any]]:
    """Settings recorded by pred_cache.py for a cache directory (<dir>/../meta_<dir name>.json).

    Input:  pred_cache — cache directory or None
    Output: {'file', 'weights', 'weights_md5', 'settings', 'ultralytics_version'}
            or None when no metadata file exists
    """
    if pred_cache is None:
        return None
    p = Path(pred_cache)
    meta = p.parent / f"meta_{p.name}.json"
    if not meta.is_file():
        return None
    m = json.loads(meta.read_text(encoding='utf-8'))
    return {'file': str(meta), **{k: m.get(k) for k in ('weights', 'weights_md5', 'settings',
                                                         'ultralytics_version')}}


def make_tasks(data_dir: str | Path, split: str, stems: Sequence[str], conditions: Sequence[str],
               params: Mapping[str, Any], pred_cache: Optional[str | Path],
               v110_dir: Optional[str | Path], zss_check: int = DEFAULT_ZSS_CHECK,
               save_trees: bool = True) -> List[Dict[str, Any]]:
    """Builds the per-screen tasks of a run.

    Input:  data_dir — package root; split — split name; stems — screens;
            conditions — condition keys; params — thresholds; pred_cache —
            directory of <stem>.json detections (None: no predicted
            conditions possible); v110_dir — v1.1.0 code directory;
            zss_check — number of first screens to cross-check with zss;
            save_trees — keep the trees in the records
    Output: list of task dictionaries for evaluate_screen
    """
    d = Path(data_dir)
    names = load_names(d)
    mapping = load_mapping(d, split)
    tasks = []
    for k, stem in enumerate(stems):
        row = mapping.get(stem)
        tasks.append({
            'stem': stem,
            'xml': str(d / row['xml_in_zip']) if row else None,
            'label': str(d / 'dataset' / 'labels' / split / f"{stem}.txt"),
            'image': str(d / 'dataset' / 'images' / split / f"{stem}.png"),
            'pred': str(Path(pred_cache) / f"{stem}.json") if pred_cache else None,
            'names': names, 'conditions': list(conditions), 'params': dict(params),
            'v110_dir': str(v110_dir) if v110_dir else None,
            'zss': k < zss_check, 'save_trees': save_trees,
        })
    return tasks


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _num(v: Any) -> float:
    """Converts a metric value to float (None -> NaN).

    Input:  v — number or None
    Output: float
    """
    return math.nan if v is None else float(v)


def describe(values: Sequence[float]) -> Dict[str, Any]:
    """Mean, median, quartiles, min and max of the finite values.

    Input:  values — numbers (NaN / None ignored)
    Output: {'n', 'mean', 'median', 'q1', 'q3', 'min', 'max'} (only 'n' for no value)
    """
    v = np.asarray([_num(x) for x in values], dtype=np.float64)
    v = v[np.isfinite(v)]
    if not v.size:
        return {'n': 0}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {'n': int(v.size), 'mean': float(v.mean()), 'median': float(med), 'q1': float(q1),
            'q3': float(q3), 'min': float(v.min()), 'max': float(v.max())}


class Bootstrap:
    """Group-level (app) percentile bootstrap over a fixed list of screens, with shared draws.

    Each replicate draws as many app groups as the subset has, with
    replacement (NumPy default_rng(seed), as ap.bootstrap_ci), and takes every
    screen of every drawn group (a screen of a group drawn twice counts
    twice). The draws are made once, so every metric and every condition of
    a subset is resampled identically (paired differences).
    """

    def __init__(self, stems: Sequence[str], groups: Mapping[str, Any], n: int,
                 seed: int = apm.DEFAULT_BOOTSTRAP_SEED, alpha: float = apm.DEFAULT_ALPHA):
        """Draws the resampled app groups once.

        Screens without a group in `groups` form a group of their own (counted
        in n_missing_groups).

        Input:  stems — screens of the subset; groups — {stem: group id};
                n — replicates (0: no intervals); seed — RNG seed; alpha — two-sided level
        Output: none
        """
        self.stems = list(stems)
        self.index = {s: i for i, s in enumerate(self.stems)}
        self.n_missing_groups = sum(1 for s in self.stems if s not in groups)
        g = {s: groups.get(s, f"__screen__{s}") for s in self.stems}
        self.units, _ = apm._resampling_units(self.stems, g)
        self.seed, self.alpha = int(seed), float(alpha)
        n_units = len(self.units)
        self.n = int(n) if n_units else 0
        rng = np.random.default_rng(self.seed)
        draws = rng.integers(0, n_units, size=(self.n, n_units)) if self.n else np.zeros((0, 0), int)
        # counts[b, u] = how often group u is drawn in replicate b
        self.counts = np.zeros((self.n, n_units), dtype=np.float64)
        if self.n:
            np.add.at(self.counts, (np.repeat(np.arange(self.n), n_units), draws.ravel()), 1.0)
        self.unit_of = np.empty(len(self.stems), dtype=int)
        for u, idx in enumerate(self.units):
            self.unit_of[idx] = u

    def unit_sums(self, values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Per-group sums and counts of the finite values.

        Input:  values — per-screen values aligned with self.stems (NaN = missing)
        Output: (sums, counts) arrays per group
        """
        v = np.asarray([_num(x) for x in values], dtype=np.float64)
        ok = np.isfinite(v)
        n_units = len(self.units)
        sums = np.bincount(self.unit_of[ok], weights=v[ok], minlength=n_units)
        cnts = np.bincount(self.unit_of[ok], minlength=n_units).astype(np.float64)
        return sums, cnts

    def ratio_replicates(self, num_sums: np.ndarray, den_sums: np.ndarray) -> np.ndarray:
        """Replicate values of sum(num) / sum(den) over the drawn groups (NaN where sum(den) = 0).

        Input:  num_sums, den_sums — per-group sums
        Output: array of n replicate values
        """
        num = self.counts @ num_sums
        den = self.counts @ den_sums
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    def interval(self, reps: np.ndarray, point: Optional[float]) -> Dict[str, Any]:
        """Percentile interval of replicate values plus the fraction of replicates <= 0.

        Input:  reps — replicate values (NaN = undefined); point — estimate on the subset
        Output: {'point', 'lo', 'hi', 'n_valid', 'p_le_0'}
        """
        out = apm._interval(np.asarray(reps, dtype=np.float64), point, self.alpha)
        valid = reps[np.isfinite(reps)] if len(reps) else reps
        out['p_le_0'] = float(np.mean(valid <= 0)) if len(valid) else None
        return out

    def mean(self, values: Sequence[float]) -> Dict[str, Any]:
        """Point estimate and interval of the mean over the finite values.

        Input:  values — per-screen values aligned with self.stems
        Output: {'point', 'lo', 'hi', 'n_valid', 'p_le_0'} (p_le_0: fraction of
                replicates <= 0, meaningful for differences)
        """
        sums, cnts = self.unit_sums(values)
        point = float(sums.sum() / cnts.sum()) if cnts.sum() else None
        return self.interval(self.ratio_replicates(sums, cnts), point)

    def ratio(self, num: Sequence[float], den: Sequence[float], scale: float = 1.0
              ) -> Dict[str, Any]:
        """Point estimate and interval of scale * sum(num) / sum(den) (e.g. micro F1).

        Screens where either value is missing are left out of both sums.

        Input:  num, den — per-screen values aligned with self.stems (NaN = missing);
                scale — factor (2 for F1 = 2 TP / (n_gen + n_ref))
        Output: interval dictionary as in mean()
        """
        a = np.asarray([_num(x) for x in num], dtype=np.float64)
        b = np.asarray([_num(x) for x in den], dtype=np.float64)
        ok = np.isfinite(a) & np.isfinite(b)
        sa, _ = self.unit_sums(np.where(ok, scale * a, np.nan))
        sb, _ = self.unit_sums(np.where(ok, b, np.nan))
        point = float(sa.sum() / sb.sum()) if sb.sum() > 0 else None
        return self.interval(self.ratio_replicates(sa, sb), point)

    def median_replicates(self, values: Sequence[float]) -> np.ndarray:
        """Replicate medians: the median of each resampled multiset of the finite values.

        A screen of a group drawn k times counts k times; even sizes average
        the two middle values, as np.median.

        Input:  values — per-screen values aligned with self.stems
        Output: array of n replicate medians (NaN for a replicate without a value)
        """
        v = np.asarray([_num(x) for x in values], dtype=np.float64)
        ok = np.flatnonzero(np.isfinite(v))
        reps = np.full(self.n, np.nan)
        if not self.n or not ok.size:
            return reps
        order = ok[np.argsort(v[ok], kind='stable')]
        x = v[order]
        c = np.cumsum(self.counts[:, self.unit_of[order]], axis=1)   # (n, n_valid)
        total = c[:, -1]
        has = total > 0
        i1 = np.argmax(c > np.floor((total - 1) / 2)[:, None], axis=1)
        i2 = np.argmax(c > np.floor(total / 2)[:, None], axis=1)
        reps[has] = ((x[i1] + x[i2]) / 2)[has]
        return reps

    def median(self, values: Sequence[float]) -> Dict[str, Any]:
        """Point estimate and interval of the median over the finite values.

        Input:  values — per-screen values aligned with self.stems
        Output: {'point', 'lo', 'hi', 'n_valid', 'p_le_0'}
        """
        v = np.asarray([_num(x) for x in values], dtype=np.float64)
        ok = np.isfinite(v)
        point = float(np.median(v[ok])) if ok.any() else None
        if not self.n or not ok.any():
            return {'point': point, 'lo': None, 'hi': None, 'n_valid': 0, 'p_le_0': None}
        return self.interval(self.median_replicates(v), point)


def summarize_reference(records: Mapping[str, Mapping[str, Any]], stems: Sequence[str]
                        ) -> Dict[str, Any]:
    """Summary of the reference trees of a subset: exclusions, label alignment, flatness, size.

    Input:  records — {stem: reference record}; stems — subset
    Output: summary dictionary
    """
    recs = [records[s] for s in stems if s in records]
    ok = [r for r in recs if r.get('status') == 'ok']
    status = Counter(r.get('status') for r in recs)
    aligned = [r for r in recs if 'alignment' in r]
    labels = sum(r['alignment']['n_labels'] for r in aligned)
    exact = sum(r['alignment']['n_exact'] for r in aligned)
    near = sum(r['alignment']['n_near'] for r in aligned)
    unmatched = sum(r['alignment']['n_unmatched'] for r in aligned)
    depth_hist = Counter(r['ref_depth'] for r in ok)
    return {
        'n_screens': len(stems), 'n_records': len(recs), 'n_ok': len(ok),
        'status': dict(sorted(status.items())),
        'alignment': {
            'n_screens_aligned_attempted': len(aligned),
            'n_screens_sequence_identical': sum(r['alignment']['method'] == 'sequence'
                                                for r in aligned),
            'n_screens_fully_matched': sum(r['alignment']['n_unmatched'] == 0 for r in aligned),
            'n_labels': labels, 'n_exact': exact, 'n_near': near, 'n_unmatched': unmatched,
            'label_match_rate': (exact + near) / labels if labels else None,
            'n_traced': sum(r['alignment']['n_traced'] for r in aligned),
            'n_traced_unused': sum(r['alignment']['n_traced_unused'] for r in aligned),
        },
        'flat': {'n_flat': sum(bool(r['flat']) for r in ok),
                 'rate': (sum(bool(r['flat']) for r in ok) / len(ok)) if ok else None,
                 'max_depth': FLAT_MAX_DEPTH,
                 'depth_histogram': {str(k): depth_hist[k] for k in sorted(depth_hist)}},
        'n_gt': describe([r['n_gt'] for r in ok]),
        'ref_nodes': describe([r['ref_nodes'] for r in ok]),
        'ref_depth': describe([r['ref_depth'] for r in ok]),
        'ref_containers_below_root': describe([r['ref_containers'] - 1 for r in ok]),
        'ref_composites': describe([r['ref_composites'] for r in ok]),
        'screens_with_composite': sum(r['ref_composites'] > 0 for r in ok),
        'xml_nodes': describe([r['xml_nodes'] for r in recs if 'xml_nodes' in r]),
        'xml_depth': describe([r['xml_depth'] for r in recs if 'xml_depth' in r]),
        'pred_size_mismatch': sum(bool(r.get('pred_size_mismatch')) for r in recs),
    }


def _count_summary(values: Sequence[float]) -> Dict[str, Any]:
    """Mean, total and number of screens with a positive count.

    Input:  values — per-screen counts
    Output: {'mean', 'total', 'screens_with'}
    """
    v = [float(x) for x in values if x is not None]
    return {'mean': float(np.mean(v)) if v else None, 'total': float(sum(v)),
            'screens_with': sum(1 for x in v if x > 0)}


def summarize_condition(records: Mapping[str, Mapping[str, Any]],
                        boots: Mapping[str, Bootstrap]) -> Dict[str, Any]:
    """Summary of one condition on the screens of a subset, with intervals per resampling scheme.

    Input:  records — {stem: condition record}; boots — {scheme name:
            Bootstrap}, all over the same stems (the subset)
    Output: summary dictionary (see the module docstring, item 9); intervals
            are {scheme: interval} under 'ci_mean' / 'ci_median' / 'ci_f1'
    """
    stems = next(iter(boots.values())).stems
    recs = [records.get(s, {'status': 'missing'}) for s in stems]
    ok = [r for r in recs if r.get('status') == 'ok']
    out: Dict[str, Any] = {'n_screens': len(recs), 'n_evaluated': len(ok),
                           'status': dict(sorted(Counter(r.get('status') for r in recs).items()))}

    def col(key: str) -> List[float]:
        """Per-screen values aligned with the subset's stems (NaN when not evaluated).

        Input:  key — record key
        Output: list of floats
        """
        return [_num(r.get(key)) if r.get('status') == 'ok' else math.nan for r in recs]

    for m in CI_METRICS:
        vals = col(m)
        out[m] = {**describe(vals),
                  'ci_mean': {k: b.mean(vals) for k, b in boots.items()},
                  'ci_median': {k: b.median(vals) for k, b in boots.items()}}
    out['grouping_f1']['n_undefined'] = sum(1 for r in ok if r.get('grouping_f1') is None)
    g = [r['grouping'] if r.get('status') == 'ok' else None for r in recs]
    tp = [math.nan if x is None else x['tp'] for x in g]
    ng = [math.nan if x is None else x['n_gen'] for x in g]
    nr = [math.nan if x is None else x['n_ref'] for x in g]
    den = [a + b for a, b in zip(ng, nr)]
    stp, sng, snr = (float(np.nansum(x)) for x in (tp, ng, nr))
    out['grouping_micro'] = {'tp': stp, 'n_gen': sng, 'n_ref': snr,
                             'precision': stp / sng if sng else None,
                             'recall': stp / snr if snr else None,
                             'f1': 2 * stp / (sng + snr) if (sng + snr) else None,
                             'ci_f1': {k: b.ratio(tp, den, scale=2.0) for k, b in boots.items()}}
    for v in VARIANTS:
        out[f"edit_ops_{v}"] = {k: describe([r[f"ops_{v}"][k] for r in ok]) for k in EDIT_OPS}
        out[f"edit_ops_{v}"]['total'] = describe([r[f"ted_{v}"] for r in ok])
    out['nodes'] = {'reference': describe([r['ref_nodes'] for r in ok]),
                    'generated': describe([r['gen_nodes'] for r in ok])}
    out['depth'] = {'generated': describe([r['gen_depth'] for r in ok])}
    out['generated_containers'] = describe([r['gen_containers'] for r in ok])
    out['generated_composites'] = describe([r['gen_composites'] for r in ok])
    for k in ('n_fallback', 'n_placeholder', 'n_duplicates', 'n_lost', 'n_dedup_removed'):
        out[k[2:]] = _count_summary([r[k] for r in ok])
    n_gt = sum(r['n_gt'] for r in ok)
    n_in = sum(r['n_layout_input'] for r in ok)
    n_m = sum(r['n_matched'] for r in ok)
    out['matching'] = {'n_gt': n_gt, 'n_input': sum(r['n_input'] for r in ok),
                       'n_layout_input': n_in, 'n_matched': n_m,
                       'n_class_mismatch': sum(r['n_class_mismatch'] for r in ok),
                       'gt_recall': n_m / n_gt if n_gt else None,
                       'layout_input_precision': n_m / n_in if n_in else None}
    return out


def paired_difference(rec_a: Mapping[str, Mapping[str, Any]], rec_b: Mapping[str, Mapping[str, Any]],
                      boot: Bootstrap) -> Dict[str, Any]:
    """Paired differences (a minus b) of the CI metrics and of micro grouping F1, for one resampling scheme.

    Per-screen differences are taken on the screens with a value in both
    conditions; the micro-F1 difference uses the screens evaluated in both.

    Input:  rec_a, rec_b — {stem: record} of two conditions; boot — Bootstrap of the subset
    Output: {metric: interval of the mean per-screen difference (plus
             'n_screens_both'), 'grouping_micro_f1': interval of the
             difference of the micro F1s}
    """
    out: Dict[str, Any] = {}

    def val(rec: Mapping[str, Any], key: str) -> float:
        """Value of a record (NaN unless status ok).

        Input:  rec — record; key — metric key
        Output: float
        """
        return _num(rec.get(key)) if rec.get('status') == 'ok' else math.nan

    for m in CI_METRICS:
        d = [val(rec_a.get(s, {}), m) - val(rec_b.get(s, {}), m) for s in boot.stems]
        out[m] = boot.mean(d)
        out[m]['n_screens_both'] = int(np.sum(np.isfinite(np.asarray(d, dtype=float))))
    both = [s for s in boot.stems
            if rec_a.get(s, {}).get('status') == 'ok' and rec_b.get(s, {}).get('status') == 'ok']
    reps = np.zeros(boot.n)
    points: List[float] = []
    for sign, rec in ((1.0, rec_a), (-1.0, rec_b)):
        tp = np.full(len(boot.stems), np.nan)
        den = np.full(len(boot.stems), np.nan)
        for s in both:
            g = rec[s]['grouping']
            tp[boot.index[s]] = 2.0 * g['tp']
            den[boot.index[s]] = g['n_gen'] + g['n_ref']
        st, _ = boot.unit_sums(tp)
        sd, _ = boot.unit_sums(den)
        reps = reps + sign * boot.ratio_replicates(st, sd)
        points.append(st.sum() / sd.sum() if sd.sum() > 0 else math.nan)
    point = points[0] - points[1] if both else math.nan
    iv = boot.interval(reps, float(point) if math.isfinite(point) else None)
    iv['n_screens_both'] = len(both)
    out['grouping_micro_f1'] = iv
    return out


def summarize_timing(cond_records: Mapping[str, Mapping[str, Mapping[str, Any]]],
                     ref_records: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Single-process timing per screen and per 1,000 screens (reference build, generation, metrics).

    Input:  cond_records — {cond: {stem: record}}; ref_records — {stem: reference record}
    Output: {'reference': {...}, cond: {'generate', 'metrics', 'total',
             'seconds_per_1000_screens'}}
    """
    ref_t = [r['time_s'] for r in ref_records.values() if 'time_s' in r]
    out: Dict[str, Any] = {'reference': {**describe(ref_t),
                                         'seconds_per_1000_screens':
                                             1000 * float(np.mean(ref_t)) if ref_t else None}}
    for cond, recs in cond_records.items():
        ok = [r for r in recs.values() if r.get('status') == 'ok']
        gen = [r['time_s']['generate'] for r in ok]
        met = [r['time_s']['metrics'] for r in ok]
        tot = [a + b for a, b in zip(gen, met)]
        out[cond] = {'generate': describe(gen), 'metrics': describe(met), 'total': describe(tot),
                     'seconds_per_1000_screens': 1000 * float(np.mean(tot)) if tot else None}
    return out


def zss_summary(cond_records: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> Dict[str, Any]:
    """Agreement of APTED and zss on the cross-checked screens.

    Input:  cond_records — {cond: {stem: record}}
    Output: {cond: {'n_checked', 'n_agree', 'disagreements' (stems)}}
    """
    out = {}
    for cond, recs in cond_records.items():
        checked = [r for r in recs.values() if 'zss_agrees' in r]
        out[cond] = {'n_checked': len(checked), 'n_agree': sum(r['zss_agrees'] for r in checked),
                     'disagreements': sorted(r['stem'] for r in checked if not r['zss_agrees'])}
    return out


def aggregate(ref_records: Mapping[str, Mapping[str, Any]],
              cond_records: Mapping[str, Mapping[str, Mapping[str, Any]]],
              subsets: Mapping[str, Sequence[str]], groupings: Mapping[str, Mapping[str, Any]],
              n_boot: int = apm.DEFAULT_BOOTSTRAP_N, seed: int = apm.DEFAULT_BOOTSTRAP_SEED,
              diffs: Sequence[Tuple[str, str]] = DEFAULT_DIFFS) -> Dict[str, Any]:
    """Aggregates the per-screen records of a run per subset, with intervals per resampling scheme.

    Input:  ref_records — {stem: reference record}; cond_records — {cond:
            {stem: record}}; subsets — {name: stems} (e.g. 'all', 'unseen',
            'unseen_package_only'); groupings — {scheme name: {stem: group}}
            (the first is the primary scheme; 'screen' = every screen its own
            unit); n_boot — bootstrap replicates; seed — RNG seed (the same
            for every scheme and subset); diffs — (a, b) pairs
    Output: {'reference': {subset: ...}, 'conditions': {cond: {subset: ...}},
             'paired_differences': {'a-b': {subset: {scheme: ...}}},
             'bootstrap': {subset: {scheme: {...}}}, 'zss_cross_check', 'timing'}
    """
    out: Dict[str, Any] = {'reference': {}, 'conditions': {c: {} for c in cond_records},
                           'paired_differences': {}, 'bootstrap': {}}
    for name, stems in subsets.items():
        stems = sorted(set(stems))
        boots = {scheme: Bootstrap(stems, groups, n_boot, seed) for scheme, groups in groupings.items()}
        out['bootstrap'][name] = {
            scheme: {'n': b.n, 'seed': b.seed, 'alpha': b.alpha, 'n_units': len(b.units),
                     'largest_unit_screens': max((len(u) for u in b.units), default=0),
                     'n_missing_groups': b.n_missing_groups}
            for scheme, b in boots.items()}
        out['reference'][name] = summarize_reference(ref_records, stems)
        for cond, recs in cond_records.items():
            out['conditions'][cond][name] = summarize_condition(recs, boots)
        for a, b in diffs:
            if a in cond_records and b in cond_records:
                out['paired_differences'].setdefault(f"{a}-{b}", {})[name] = {
                    scheme: paired_difference(cond_records[a], cond_records[b], boot)
                    for scheme, boot in boots.items()}
    out['zss_cross_check'] = zss_summary(cond_records)
    out['timing'] = summarize_timing(cond_records, ref_records)
    return out


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _versions() -> Dict[str, Optional[str]]:
    """Versions of the packages the evaluation depends on.

    Input:  none
    Output: {package: version or None}
    """
    out: Dict[str, Optional[str]] = {'python': platform.python_version(), 'ui2compose': __version__}
    for dist in ('apted', 'zss', 'scipy', 'numpy', 'Pillow'):
        try:
            out[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[dist] = None
    return out


def run_evaluation(data_dir: str | Path, split: str, out_dir: str | Path,
                   conditions: Sequence[str] = DEFAULT_CONDITIONS,
                   params: Optional[Mapping[str, Any]] = None,
                   pred_cache: Optional[str | Path] = None, v110_dir: Optional[str | Path] = None,
                   images: Optional[str] = None, limit: Optional[int] = None, workers: int = 1,
                   n_boot: int = apm.DEFAULT_BOOTSTRAP_N, seed: int = apm.DEFAULT_BOOTSTRAP_SEED,
                   zss_check: int = DEFAULT_ZSS_CHECK, save_trees: bool = True,
                   diffs: Sequence[Tuple[str, str]] = DEFAULT_DIFFS, allow_test: bool = False,
                   groupings: Optional[Mapping[str, str | Path]] = None,
                   progress_every: int = 200, box_repr: str = DEFAULT_BOX_REPR) -> Dict[str, Any]:
    """Runs the layout evaluation of one split and writes the per-screen JSONL files and the aggregate.

    Input:  data_dir — package root (README_PACKAGE.txt layout); split —
            'val' or 'test'; out_dir — output directory; conditions —
            condition keys; params — thresholds (default_params() shape;
            missing keys take the defaults); pred_cache — directory of
            <stem>.json detections (needed by C3 / C4 / C5); v110_dir —
            directory with the v1.1.0 build_ui_graph.py and
            generate_compose_code.py (needed by C3 / C1p); images — image
            list or directory (default: the whole split); limit — first N
            stems; workers — processes; n_boot, seed — bootstrap; zss_check —
            screens cross-checked with zss; save_trees — keep trees in the
            JSONL; diffs — paired differences; allow_test — required for
            split 'test'; groupings — {scheme name: manifest.csv} of the
            bootstrap (default: default_grouping_files(); the first is the
            primary scheme; the 'screen' scheme is always added);
            progress_every — progress line interval (0: none); box_repr —
            scoring of v1.2.0 containment Boxes ('composite' = the primary
            metric; 'container' = the post-hoc sensitivity variant)
    Output: the aggregate dictionary (also written to <out_dir>/aggregate.json);
            files <out_dir>/reference.jsonl and <out_dir>/<cond>.jsonl
    """
    if split == 'test' and not allow_test:
        raise ValueError("split 'test' needs allow_test=True (--allow-test): test numbers are "
                         "computed once, after every choice is fixed on the validation split")
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise ValueError(f"unknown conditions {unknown} (known: {sorted(CONDITIONS)})")
    p = default_params()
    for k, v in (params or {}).items():
        if k == 'dedup':
            p['dedup'].update(v)
        elif k in p:
            p[k] = v
        else:
            raise ValueError(f"unknown parameter {k!r}")
    if p['fallback'] not in FALLBACK_MODES:
        raise ValueError(f"fallback must be one of {FALLBACK_MODES}")
    if box_repr not in BOX_REPRS:
        raise ValueError(f"box_repr must be one of {BOX_REPRS}")
    needs_pred = [c for c in conditions if CONDITIONS[c]['boxes'] == 'pred']
    if needs_pred and pred_cache is None:
        raise ValueError(f"conditions {needs_pred} need pred_cache (--pred-cache)")
    v110_info = None
    if any(CONDITIONS[c]['code'] == 'v110' for c in conditions):
        if v110_dir is None:
            raise ValueError("conditions C3 / C1p need v110_dir (--v110-code)")
        v110 = V110Code(v110_dir)
        v110_info = {'dir': v110.dir, 'sha256': v110.sha256, 'is_v110_tag': v110.is_v110_tag}

    t_start = time.perf_counter()
    stems = select_stems(data_dir, split, images, limit)
    tasks = make_tasks(data_dir, split, stems, conditions, p, pred_cache,
                       v110_dir if v110_info else None, zss_check, save_trees)
    for t in tasks:
        t['box_repr'] = box_repr
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref_records: Dict[str, Dict[str, Any]] = {}
    cond_records: Dict[str, Dict[str, Dict[str, Any]]] = {c: {} for c in conditions}
    files = {c: open(out / f"{c}.jsonl", 'w', encoding='utf-8') for c in conditions}
    ref_file = open(out / 'reference.jsonl', 'w', encoding='utf-8')
    try:
        if workers > 1:
            import multiprocessing as mp

            ctx = mp.get_context('spawn')
            with ctx.Pool(workers) as pool:
                results = pool.imap(evaluate_screen, tasks, chunksize=4)
                _collect(results, len(tasks), ref_records, cond_records, ref_file, files,
                         progress_every)
        else:
            _collect((evaluate_screen(t) for t in tasks), len(tasks), ref_records, cond_records,
                     ref_file, files, progress_every)
    finally:
        ref_file.close()
        for f in files.values():
            f.close()
    wall = time.perf_counter() - t_start

    subsets: Dict[str, List[str]] = {'all': stems}
    selected = set(stems)
    for name, sub in load_subsets(data_dir, split).items():
        subsets[name] = [s for s in sub if s in selected]
    group_files = dict(groupings) if groupings is not None else default_grouping_files(data_dir)
    agg = aggregate(ref_records, cond_records, subsets, load_groupings(group_files, stems),
                    n_boot, seed, diffs)
    agg['meta'] = {
        'tool': 'ui2compose-eval-layout', 'ui2compose_version': __version__,
        'created_utc': _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        'data_dir': str(data_dir), 'split': split, 'images': images or f"dataset/images/{split}",
        'limit': limit, 'n_screens': len(stems), 'conditions': {c: CONDITIONS[c] for c in conditions},
        'params': p, 'box_repr': box_repr,
        'pred_cache': str(pred_cache) if pred_cache else None,
        'prediction_meta': prediction_meta(pred_cache), 'v110_code': v110_info,
        'subsets': {k: len(v) for k, v in subsets.items()},
        'groupings': {**{k: str(v) for k, v in group_files.items()},
                      SCREEN_SCHEME: 'every screen its own unit'},
        'primary_grouping': next(iter(group_files), SCREEN_SCHEME),
        'parse_settings': {'drop_container': PARSE_DROP_CONTAINER, 'min_area': PARSE_MIN_AREA,
                           'dedup_iou': PARSE_DEDUP_IOU, 'near_exact_tol_px': NEAR_EXACT_TOL_PX,
                           'snap_px': SNAP_PX},
        'flat_max_depth': FLAT_MAX_DEPTH, 'match_iou': p['match_iou'],
        'workers': workers, 'wall_seconds': wall, 'versions': _versions(),
        'files': {'reference': 'reference.jsonl', **{c: f"{c}.jsonl" for c in conditions}},
    }
    (out / 'aggregate.json').write_text(json.dumps(agg, indent=2), encoding='utf-8')
    return agg


def _collect(results: Iterable[Dict[str, Any]], n: int, ref_records: Dict, cond_records: Dict,
             ref_file, files: Mapping[str, Any], progress_every: int) -> None:
    """Stores and writes the per-screen results as they arrive.

    Input:  results — iterable of evaluate_screen outputs; n — number of
            screens; ref_records, cond_records — dictionaries to fill;
            ref_file, files — open JSONL files; progress_every — progress interval
    Output: none
    """
    t0 = time.perf_counter()
    for k, res in enumerate(results, 1):
        ref = res['reference']
        ref_records[ref['stem']] = ref
        ref_file.write(json.dumps(ref) + '\n')
        for cond, rec in res['conditions'].items():
            cond_records[cond][rec['stem']] = rec
            files[cond].write(json.dumps(rec) + '\n')
        if progress_every and k % progress_every == 0:
            print(f"  {k}/{n} screens ({time.perf_counter() - t0:.0f} s)", flush=True)


def read_run(out_dir: str | Path) -> Tuple[Dict[str, Dict[str, Any]],
                                           Dict[str, Dict[str, Dict[str, Any]]]]:
    """Reads the per-screen records of a finished run, e.g. to aggregate it again without recomputing.

    Input:  out_dir — output directory of run_evaluation
    Output: (reference records {stem: record}, {cond: {stem: record}} for
            every <cond>.jsonl present)
    """
    out = Path(out_dir)

    def read(p: Path) -> Dict[str, Dict[str, Any]]:
        """Records of one JSONL file by stem.

        Input:  p — JSONL file
        Output: {stem: record}
        """
        recs = (json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line)
        return {r['stem']: r for r in recs}

    conds = {c: read(out / f"{c}.jsonl") for c in CONDITIONS if (out / f"{c}.jsonl").is_file()}
    return read(out / 'reference.jsonl'), conds


def _threshold(text: str) -> Optional[float]:
    """Parses a de-duplication threshold ('none' disables the rule).

    Input:  text — command-line value
    Output: float or None
    """
    return None if text.lower() == 'none' else float(text)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Defines and reads the command-line arguments.

    Input:  argv — argument list (default sys.argv[1:])
    Output: argparse namespace
    """
    d = default_params()
    ap = argparse.ArgumentParser(description="Layout evaluation against the ReDraw view "
                                             "hierarchies (TEDS, grouping F1)")
    ap.add_argument('--data-dir', required=True, help="regenerated ReDraw val/test package")
    ap.add_argument('--split', default='val', choices=('val', 'test'))
    ap.add_argument('--allow-test', action='store_true',
                    help="required for --split test (test numbers are computed once)")
    ap.add_argument('--images', default=None,
                    help="image list (.txt, Ultralytics format) or directory "
                         "(default: dataset/images/<split>)")
    ap.add_argument('--limit', type=int, default=None, help="first N screens (sorted stems)")
    ap.add_argument('--conditions', default=','.join(DEFAULT_CONDITIONS),
                    help=f"comma-separated subset of {','.join(CONDITIONS)}")
    ap.add_argument('--pred-cache', default=None,
                    help="directory of <stem>.json raw detections (pred_cache.py)")
    ap.add_argument('--v110-code', default=None,
                    help="directory with build_ui_graph.py and generate_compose_code.py of v1.1.0")
    ap.add_argument('--same-class-iou', type=_threshold, default=d['dedup']['same_class_iou'])
    ap.add_argument('--same-class-iomin', type=_threshold, default=d['dedup']['same_class_iomin'])
    ap.add_argument('--cross-class-iou', type=_threshold, default=d['dedup']['cross_class_iou'])
    ap.add_argument('--containment-ratio', type=float, default=d['containment_ratio'])
    ap.add_argument('--cut-tol-frac', type=float, default=d['cut_tol_frac'])
    ap.add_argument('--fallback', choices=FALLBACK_MODES, default=d['fallback'])
    ap.add_argument('--out', required=True, help="output directory")
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--bootstrap', type=int, default=apm.DEFAULT_BOOTSTRAP_N,
                    help="bootstrap replicates (0 disables the intervals)")
    ap.add_argument('--seed', type=int, default=apm.DEFAULT_BOOTSTRAP_SEED)
    ap.add_argument('--zss-check', type=int, default=DEFAULT_ZSS_CHECK,
                    help="number of screens cross-checked with zss")
    ap.add_argument('--groups', action='append', default=None, metavar='NAME=CSV',
                    help="bootstrap resampling scheme: manifest.csv with stem,group_id "
                         "(repeatable; the first is primary; default: "
                         + ', '.join(f"{k}=<data-dir>/{v}" for k, v in DEFAULT_GROUPINGS.items())
                         + "; the 'screen' scheme is always added)")
    ap.add_argument('--no-trees', action='store_true', help="do not store trees in the JSONL files")
    ap.add_argument('--box-repr', choices=BOX_REPRS, default=DEFAULT_BOX_REPR,
                    help="scoring of containment Boxes: 'composite' (primary metric) or "
                         "'container' (post-hoc sensitivity: widget leaf and content as siblings)")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Runs the layout evaluation from the command line.

    Input:  argv — argument list (default sys.argv[1:])
    Output: exit code 0 (2 for invalid arguments); files in --out
    """
    args = parse_args(argv)
    conditions = [c.strip() for c in args.conditions.split(',') if c.strip()]
    params = {'dedup': {'same_class_iou': args.same_class_iou,
                        'same_class_iomin': args.same_class_iomin,
                        'cross_class_iou': args.cross_class_iou},
              'containment_ratio': args.containment_ratio, 'cut_tol_frac': args.cut_tol_frac,
              'fallback': args.fallback}
    groupings = None
    if args.groups:
        groupings = {}
        for spec in args.groups:
            name, sep, path = spec.partition('=')
            if not sep or not name or not path or name in groupings or name == SCREEN_SCHEME:
                print(f"[ERR] --groups expects a new NAME=CSV, got {spec!r}")
                return 2
            groupings[name] = path
    try:
        agg = run_evaluation(args.data_dir, args.split, args.out, conditions, params,
                             pred_cache=args.pred_cache, v110_dir=args.v110_code,
                             images=args.images, limit=args.limit, workers=args.workers,
                             n_boot=args.bootstrap, seed=args.seed, zss_check=args.zss_check,
                             save_trees=not args.no_trees, allow_test=args.allow_test,
                             groupings=groupings, box_repr=args.box_repr)
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERR] {e}")
        return 2
    ref = agg['reference']['all']
    print(f"{agg['meta']['n_screens']} screens, reference ok {ref['n_ok']}, "
          f"label match rate {ref['alignment']['label_match_rate']}, "
          f"flat {ref['flat']['n_flat']}/{ref['n_ok']}")
    for cond in conditions:
        s = agg['conditions'][cond]['all']
        print(f"  {cond}: n={s['n_evaluated']} TEDS-g {s['teds_g'].get('mean')} "
              f"TEDS-o {s['teds_o'].get('mean')} grouping F1 {s['grouping_f1'].get('mean')} "
              f"(micro {s['grouping_micro']['f1']})")
    print(f"written: {Path(args.out) / 'aggregate.json'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
