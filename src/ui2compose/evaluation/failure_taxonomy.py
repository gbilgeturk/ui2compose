# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Failure taxonomy of the layout evaluation: every edit of a C4 screen attributed to one cause.

Typical use (the rules below were fixed on the validation split before the
test split was evaluated; the sample is drawn from the whole split):
    python -m ui2compose.evaluation.failure_taxonomy run --data-dir <package> --split test \
        --allow-test --pred-cache <outside>/pred_cache/test --run <outside>/layout_test \
        --out-dir <outside>/failure_taxonomy_test
    python -m ui2compose.evaluation.failure_taxonomy cases --data-dir <package> --split test \
        --allow-test --pred-cache <outside>/pred_cache/test \
        --taxonomy <outside>/failure_taxonomy_test/failure_taxonomy.json \
        --out-dir <outside>/failure_taxonomy_test/cases --results-dir results/eval

Method (one rule set, applied identically to every screen):

1. Sample. SAMPLE_SIZE (50) screens drawn without replacement from all stems
   of the split (sorted) with numpy.random.default_rng(SAMPLE_SEED = 42)
   .choice. Each is evaluated under condition C4 of layout_eval (cached
   predictions + v1.2.0 de-duplication, containment and XY-cut, the shipped
   thresholds) with the same reference tree, canonical form and matching as
   ui2compose-eval-layout. With --run, every sampled screen's TED and both
   canonical trees are checked against that run's C4.jsonl / reference.jsonl.
2. Edits. The TEDS-o variant is used (container labels = geometric
   orientation, so a Column-vs-Row error is an edit). An optimal unit-cost
   edit mapping (APTED) edits the generated tree into the reference tree:
   'delete' = generated node without counterpart, 'insert' = reference node
   without counterpart, 'rename' = mapped pair with different labels
   (renames of equal labels cost 0 and are not edits); the number of edits
   equals layout_eval's TED-o (checked). Unit costs often admit several
   optimal mappings (e.g. three Buttons detected where the reference has
   three ImageViews of which one is undetected); the tie is broken towards
   pairing detections that the box matching paired: every edit costs
   TIE_SCALE = 2^16 and pairing two detections not matched to each other
   costs 1 more, which never outweighs one edit. The edit kinds can
   therefore differ from the counts of layout_eval.tree_edit (APTED's own
   tie-break); their total cannot.
3. Evidence per screen: the one-to-one IoU >= 0.5 Hungarian matching of the
   ground-truth boxes with the de-duplicated detections (as for grouping F1,
   class not required) and with the raw detections (before de-duplication);
   the XML element of every reference node; the dumped window (union of the
   top-level XML nodes' bounds); the grey-level standard deviation of the
   screenshot inside every ground-truth box.
4. Categories (CATEGORIES); every edit gets exactly one, by the first rule
   that applies, in the order written here. Notation: m(.) = the box
   matching of item 3; a 'composite' is a detection node with children (a
   containment Box or a LazyColumn).
   * FP type (an unmatched detection g): reference_artifact if less than
     WINDOW_MIN_INSIDE (0.5) of its area lies inside the dumped window (the
     dump describes a dialog or popup only, or leaves out the status bar,
     while the screenshot shows more), or if the ground-truth box it
     overlaps most (IoU > 0) is an unmatched box flagged as beyond the
     screen (see FN type); else detector_fp.
   * FN type (an unmatched ground-truth box r): dedup_removed if r is
     matched among the raw detections (de-duplication removed the box that
     matched it); reference_artifact if r's XML bounds or its label box
     extend beyond the screenshot (parse_redraw clamps centre and size
     separately, which shifts such labels) or the screenshot is uniform
     inside r (grey-level standard deviation < PIXEL_STD_MAX = 2: a dump
     taken while the screen still showed a spinner, a rotated or black
     screenshot); reference_overlap if r is hidden or stacked: fully
     covered by the bounds of a later-drawn sibling (of r or of an
     ancestor) whose subtree holds a label, inside a DrawerLayout child
     followed by a later child that holds a label (open drawer over dimmed
     content), or with at least SHARED_MIN (0.5) of its area under other
     labels that are neither its XML ancestors nor descendants; a covering
     node or label counts only if its area is at most COVER_AREA_RATIO_MAX
     (4) times r's (a background, banner or card around r is a surface r is
     drawn on; on the val sample such texts and icons were visible); else
     detector_fn.
   * Stacked / flat reference container: a reference container whose
     children overlap in both axes (TEDS-o label 'Box') is 'stacked' if two
     children's boxes intersect with positive area (FrameLayout-style
     overlays) and 'flat 2D' otherwise (RelativeLayout / ConstraintLayout
     style, e.g. an image left of two stacked texts without a nested group,
     which XY-cut nests like a Compose Row { Image; Column { .. } }).
   * Tolerance split: a generated 'gap_tolerance' Column / Row (or
     LazyColumn item split) separated two of its children through the
     tolerance if they overlap along its cut axis by a positive length.
   * Structural cause S(g, r) of a matched pair g = m(r) that the mapping
     does not pair (the box is in the wrong place): the FP type of g's
     nearest composite ancestor if that ancestor is an unmatched detection
     (e.g. a hallucinated RecyclerView swallowing the screen); containment
     if g's nearest composite ancestor (as its matched ground-truth id)
     differs from r's nearest composite ancestor in the reference; the
     majority FN type of the unmatched detections under r's reference
     parent if fewer than two of that parent's children hold a matched
     detection (the reference group cannot be rebuilt because its other
     boxes were not detected); xycut_fallback if g's direct parent is a
     fallback node (no cut existed: 'fallback_y_sorted' / 'least_overlap',
     also as a LazyColumn's item split); tolerance_cut if g's direct parent
     separated g from a sibling through the tolerance; reference_overlap /
     reference_flat2d if r's reference parent is stacked / flat 2D; else
     grouping.
   * insert of a reference detection r: FN type if unmatched, else
     S(m(r), r). delete of a generated detection g: FP type if unmatched,
     else S(g, m(g)).
   * rename of two detections g -> r: wrong_class if m(g) = r (the class
     differs); FP type if g is unmatched; FN type if r is unmatched; else
     S(g, m(g)).
   * rename of a generated container into a reference detection r (a
     composite such as a ListView with items): FN type if r is unmatched,
     else containment. rename of a generated composite g into a reference
     container q: FP type if g is unmatched; reference_overlap if q is
     stacked and holds m(g) as a direct child (the reference draws the
     contents over the widget as siblings in a FrameLayout where ui2compose
     nests them in a Box: the same picture); else containment.
   * rename of two containers (the orientation differs): xycut_fallback if
     the generated one is a fallback node; tolerance_cut if it is a
     'gap_tolerance' node two of whose children overlap along its cut axis
     (so its geometric orientation cannot be the emitted Column / Row);
     reference_overlap / reference_flat2d if the reference one is stacked /
     flat 2D; else grouping.
   * delete of a generated container c (members = matched detections in its
     subtree): xycut_fallback if c is a fallback node; the majority FP type
     of its unmatched detections if fewer than two members are matched;
     tolerance_cut if c's children overlap along its axis (as for renames);
     containment if a member's composite ancestor differs between the
     trees; reference_overlap / reference_flat2d if the reference lowest
     common ancestor of the members is stacked / flat 2D; else grouping.
   * insert of a reference container q: the majority FN type of its
     unmatched detections if fewer than two of its detections are matched;
     xycut_fallback if the generated lowest common ancestor L of the matched
     detections is a fallback node; tolerance_cut if L separated two of its
     children that hold q's detections through the tolerance; containment
     (as above); reference_overlap / reference_flat2d if q is stacked /
     flat 2D; reference_artifact if q's XML bounds do not contain its
     detections (more than ARTIFACT_PX outside: a zero-size or misplaced,
     i.e. invisible, container); else grouping.
   'grouping' is the residual: both trees hold the boxes, but XY-cut groups
   them differently from the developer's view hierarchy (e.g. nested
   vertical LinearLayouts, which one XY-cut level flattens, or cards cut
   by rows) and no rule above explains it locally. Majority ties are broken
   in CATEGORIES order. The rules are heuristics on the geometry of the
   dump and the screenshot: reference_overlap is an upper bound of
   occlusion, and reference_artifact relies on uniform pixels and window
   bounds; they were checked by eye on the val sample only.
5. Per screen: counts per category, the dominant category (most edits;
   ties in CATEGORIES order; 'none' for a perfect screen) and detector-level
   counts (FP, FN, class mismatches, de-duplication removals that matched a
   label). Aggregate: edits per category (count, share), screens per
   dominant category, screens with at least one edit of a category, the
   cross-table edit kind x category, and the families (FAMILIES).
6. Illustrative cases (select_cases): one screen per family (detector,
   layout rules, reference). Only screens with at least one edit of their
   dominant category that leaves a detection to trace are eligible (an
   undetected box, i.e. detector_fn or an FN-type artifact, has no
   --explain trace). Among them, the family's category that dominates most
   screens; among the screens it dominates, the one with the highest share
   of edits in it (ties: more edits of it, then stem). The pipeline is run
   on that screenshot with '--explain <id>' for the detection of the first
   such edit (ui2compose CLI, released weights, shipped defaults); its raw
   detections are compared with the cached ones and the trace is stored,
   with the line that shows the rule (key_trace_line). The eligibility
   condition was added after the first test run, whose reference-family
   case (an undetected ImageView over uniform pixels) had nothing to trace;
   no number depends on the case selection.

The test split needs --allow-test (the categories and the case rule are
fixed before; nothing is tuned on test).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ui2compose import __version__
from ui2compose.evaluation import layout_eval as LE
from ui2compose.infer.build_ui_graph import UIGraphBuilder
from ui2compose.infer.generate_compose_code import ComposeCodeGenerator
from ui2compose.infer.postprocess import deduplicate_detections
from ui2compose.preprocess import parse_redraw as pr

# ---------------------------------------------------------------------------
# Parameters (fixed on the validation split)
# ---------------------------------------------------------------------------

#: Screens in the sample and the seed of numpy.random.default_rng.
SAMPLE_SIZE: int = 50
SAMPLE_SEED: int = 42

#: Condition analysed and the TEDS label variant whose edit mapping is used.
CONDITION: str = 'C4'
VARIANT: str = 'o'

#: Least fraction of a detection's area inside the dumped window for it to
#: count as a detector false positive (else the dump does not cover it).
WINDOW_MIN_INSIDE: float = 0.5

#: Grey-level standard deviation (0-255) below which the screenshot counts
#: as uniform inside a ground-truth box (nothing visible there). On a val
#: sample, 16 % of the undetected boxes and < 1 % of the detected ones fall
#: below it.
PIXEL_STD_MAX: float = 2.0

#: Least fraction of a ground-truth box's area under other labels (not its
#: XML ancestors or descendants) for the box to count as stacked.
SHARED_MIN: float = 0.5

#: Largest area ratio (covering node / covered box) of a node that can hide
#: or stack a box. Larger nodes are surfaces drawn around content
#: (backgrounds, banners, cards): on the val sample, texts and icons drawn
#: on them are visible, so they do not count.
COVER_AREA_RATIO_MAX: float = 4.0

#: Pixels by which a reference container's XML bounds may miss its
#: detections before the container counts as an artifact (rounding).
ARTIFACT_PX: float = 1.0

#: Categories, in tie-break order, with a one-line description.
CATEGORIES: Dict[str, str] = {
    'detector_fp': 'detection without a ground-truth box (IoU < 0.5) inside the dumped window',
    'detector_fn': 'ground-truth box without a detection, not explained by the rules below',
    'wrong_class': 'box found, class differs from the label',
    'dedup_removed': 'box found by the detector but removed by de-duplication',
    'containment': 'composite nesting (Box / LazyColumn) differs from the reference',
    'xycut_fallback': 'no XY-cut existed; the fallback ordering grouped the boxes',
    'tolerance_cut': 'cut accepted only through the tolerance; its children overlap',
    'reference_overlap': 'reference node hidden or stacked (covered by a later sibling, open '
                         'drawer, pixels shared with other labels, FrameLayout-style overlay)',
    'reference_flat2d': 'reference container lays out its children in 2D without nesting '
                        '(RelativeLayout style); XY-cut nests them',
    'reference_artifact': 'the dump does not describe the pixels (detection outside the dumped '
                          'window, uniform pixels, box beyond the screen, invisible container)',
    'grouping': 'residual: both trees hold the boxes, XY-cut groups them differently from '
                'the view hierarchy',
}

#: Families of categories (summary and illustrative cases).
FAMILIES: Dict[str, Tuple[str, ...]] = {
    'detector': ('detector_fp', 'detector_fn', 'wrong_class'),
    'layout_rules': ('dedup_removed', 'containment', 'xycut_fallback', 'tolerance_cut', 'grouping'),
    'reference': ('reference_overlap', 'reference_flat2d', 'reference_artifact'),
}

EDIT_KINDS: Tuple[str, ...] = ('insert', 'delete', 'rename')

#: Substrings of the --explain trace line that shows the rule behind a
#: category (key_trace_line; the first matching line is quoted).
KEY_TRACE: Dict[str, Tuple[str, ...]] = {
    'detector_fp': ('[detection]',), 'wrong_class': ('[detection]',),
    'dedup_removed': ('removed by de-duplication',),
    'containment': ('rule=containment', 'contained in', 'LazyColumn (rule=list'),
    'xycut_fallback': ('fallback_y_sorted', 'least_overlap'),
    'tolerance_cut': ('within tolerance',),
    'reference_overlap': ('rule=containment', 'rule=gap', '[detection]'),
    'reference_flat2d': ('Row (rule=', 'Column (rule='),
    'reference_artifact': ('[detection]',),
    'grouping': ('Row (rule=', 'Column (rule='),
}

#: (category, reason, detection id to trace or None)
Verdict = Tuple[str, str, Optional[int]]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _contains(outer: Sequence[float], inner: Sequence[float], tol: float = 0.0) -> bool:
    """Tells whether box outer contains box inner (within tol px).

    Input:  outer, inner — [x1, y1, x2, y2]; tol — slack in px
    Output: bool
    """
    return (outer[0] <= inner[0] + tol and outer[1] <= inner[1] + tol
            and outer[2] >= inner[2] - tol and outer[3] >= inner[3] - tol)


def _area(b: Sequence[float]) -> float:
    """Area of a box (0 for degenerate boxes).

    Input:  b — [x1, y1, x2, y2]
    Output: float
    """
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _clip(a: Sequence[float], b: Sequence[float]) -> List[float]:
    """Intersection box of two boxes (may be degenerate).

    Input:  a, b — [x1, y1, x2, y2]
    Output: [x1, y1, x2, y2]
    """
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


def union_area(boxes: Sequence[Sequence[float]]) -> float:
    """Exact area of the union of boxes (coordinate compression).

    Input:  boxes — [x1, y1, x2, y2] (degenerate boxes contribute nothing)
    Output: float
    """
    bs = [b for b in boxes if _area(b) > 0]
    if not bs:
        return 0.0
    xs = sorted({v for b in bs for v in (b[0], b[2])})
    ys = sorted({v for b in bs for v in (b[1], b[3])})
    xi = {v: k for k, v in enumerate(xs)}
    yi = {v: k for k, v in enumerate(ys)}
    cover = np.zeros((len(ys) - 1, len(xs) - 1), dtype=bool)
    for b in bs:
        cover[yi[b[1]]:yi[b[3]], xi[b[0]]:xi[b[2]]] = True
    w = np.diff(np.asarray(xs, dtype=np.float64))
    h = np.diff(np.asarray(ys, dtype=np.float64))
    return float((cover * np.outer(h, w)).sum())


def stacked_or_flat(node: Optional[LE.TreeNode]) -> Optional[str]:
    """Kind of a reference container whose children overlap in both axes.

    Input:  node — reference node or None
    Output: 'reference_overlap' if two children's boxes intersect with
            positive area, 'reference_flat2d' if the TEDS-o label is 'Box'
            without such an intersection, None otherwise (or for a detection)
    """
    if node is None or node.cls is not None or LE.label_o(node) != 'Box':
        return None
    ext = [c.extent for c in node.children if c.extent is not None]
    for i in range(len(ext)):
        for j in range(i + 1, len(ext)):
            if _area(_clip(ext[i], ext[j])) > 0:
                return 'reference_overlap'
    return 'reference_flat2d'


# ---------------------------------------------------------------------------
# Reference evidence
# ---------------------------------------------------------------------------

class XmlEvidence:
    """XML and pixel facts about one reference: elements, dumped window, hidden / invisible labels."""

    def __init__(self, root: ET.Element, label_elems: Sequence[ET.Element],
                 gt_boxes: Sequence[Mapping[str, Any]], image_size: Sequence[int],
                 gray: Optional[np.ndarray] = None):
        """Indexes the hierarchy and tests every label.

        Input:  root — XML root; label_elems — element of every label line
                (layout_eval.align_labels); gt_boxes — parse_label_lines
                output (same order); image_size — (w, h) px; gray — screenshot
                as a 2-D grey-level array (None: no pixel test)
        Output: none (attributes: elems, index, parent, window, hidden
                {label id: reason}, artifact {label id: reason})
        """
        self.elems: List[ET.Element] = list(root.iter())
        self.index = {id(e): k for k, e in enumerate(self.elems)}
        self.parent: Dict[int, ET.Element] = {id(c): p for p in self.elems for c in p}
        self.label_index = {id(e): i for i, e in enumerate(label_elems)}
        has_label: Dict[int, bool] = {}
        for e in reversed(self.elems):        # children come after their parents in iter()
            has_label[id(e)] = id(e) in self.label_index or any(has_label[id(c)] for c in e)
        self.has_label = has_label
        tops = [b for b in (LE._node_box(n) for n in root) if b]
        self.window = [float(v) for v in LE.union_box(tops)] if tops else None
        w, h = image_size
        ancestors = {i: self._ancestors(e) for i, e in enumerate(label_elems)}
        self.hidden: Dict[int, str] = {}
        self.artifact: Dict[int, str] = {}
        for i, e in enumerate(label_elems):
            b = LE._node_box(e)
            lb = gt_boxes[i]['bbox']
            if b is not None and (b[0] < 0 or b[1] < 0 or b[2] > w or b[3] > h):
                self.artifact[i] = f"XML bounds {list(b)} extend beyond the {w}x{h} screenshot"
            elif lb[0] < 0 or lb[1] < 0 or lb[2] > w or lb[3] > h:
                self.artifact[i] = (f"label box {[round(v) for v in lb]} extends beyond the "
                                    f"{w}x{h} screenshot")
            elif gray is not None:
                std = pixel_std(gray, gt_boxes[i]['bbox'])
                if std is None or std < PIXEL_STD_MAX:
                    self.artifact[i] = ("screenshot uniform inside the box (grey-level std "
                                        f"{'n/a' if std is None else f'{std:.2f}'})")
            reason = self._covered(e, b) if b is not None else None
            if reason is None:
                bb = gt_boxes[i]['bbox']
                a = _area(bb)
                others = [gt_boxes[j]['bbox'] for j, f in enumerate(label_elems)
                          if j != i and id(f) not in ancestors[i] and id(e) not in ancestors[j]
                          and _area(gt_boxes[j]['bbox']) <= COVER_AREA_RATIO_MAX * a]
                shared = union_area([_clip(bb, o) for o in others]) / a if a > 0 else 0.0
                if shared >= SHARED_MIN:
                    reason = f"{shared:.0%} of its area under other labels (stacked)"
            if reason:
                self.hidden[i] = reason

    def _ancestors(self, e: ET.Element) -> set:
        """Ids of the XML ancestors of an element.

        Input:  e — element
        Output: set of id(element)
        """
        out, node = set(), e
        while id(node) in self.parent:
            node = self.parent[id(node)]
            out.add(id(node))
        return out

    def _covered(self, e: ET.Element, b: Sequence[int]) -> Optional[str]:
        """Whether a label's node is covered by later-drawn content.

        Input:  e — label element; b — its bounds
        Output: 'covered by later sibling <class> [bounds]', 'under an open
                drawer ...' or None
        """
        node = e
        while id(node) in self.parent:
            par = self.parent[id(node)]
            kids = list(par)
            later = kids[kids.index(node) + 1:]
            for s in later:
                sb = LE._node_box(s)
                if (sb and self.has_label[id(s)] and _contains(sb, b)
                        and _area(sb) <= COVER_AREA_RATIO_MAX * _area(b)):
                    name = pr.short_class_name(LE._node_class(s)) or s.tag
                    return f"covered by later sibling {name} {list(sb)}"
            if 'drawerlayout' in LE._node_class(par).lower() and any(self.has_label[id(s)] for s in later):
                return 'under an open drawer (a later DrawerLayout child holds labels)'
            node = par
        return None

    def element_of(self, src: str) -> Optional[ET.Element]:
        """XML element of a reference node built by indexed_reference_tree.

        Input:  src — node src 'xml:<class>@<index>'
        Output: element or None
        """
        if '@' not in src:
            return None
        k = int(src.rsplit('@', 1)[1])
        return self.elems[k] if 0 <= k < len(self.elems) else None


def pixel_std(gray: np.ndarray, box: Sequence[float]) -> Optional[float]:
    """Grey-level standard deviation of the screenshot inside a box.

    Input:  gray — 2-D array; box — [x1, y1, x2, y2] px (rounded, clipped to the image)
    Output: float, or None when less than one pixel remains
    """
    h, w = gray.shape
    x1, y1 = max(0, int(round(box[0]))), max(0, int(round(box[1])))
    x2, y2 = min(w, int(round(box[2]))), min(h, int(round(box[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return float(gray[y1:y2, x1:x2].std())


def indexed_reference_tree(root: ET.Element, label_of: Mapping[int, int],
                           gt_boxes: Sequence[Mapping[str, Any]], ev: XmlEvidence) -> Optional[LE.TreeNode]:
    """layout_eval.build_reference_tree with the element index in every node's src ('xml:<class>@<k>').

    The tree (and its canonical form) is identical to layout_eval's; only
    src differs, which no label, order or distance uses (analyse_screen
    checks the bracket notation).

    Input:  root — XML root; label_of — {id(element): label index}; gt_boxes —
            parse_label_lines output; ev — XmlEvidence of the same root
    Output: TreeNode or None
    """
    def rec(elem: ET.Element) -> Optional[LE.TreeNode]:
        """Tree of one XML subtree.

        Input:  elem — XML element
        Output: TreeNode or None
        """
        kids = [k for k in (rec(c) for c in elem) if k is not None]
        src = f"xml:{pr.short_class_name(LE._node_class(elem)) or elem.tag}@{ev.index[id(elem)]}"
        li = label_of.get(id(elem))
        if li is not None:
            b = gt_boxes[li]
            return LE.detection(b['class_name'], b['id'], b['bbox'], src, kids)
        if kids:
            return LE.container(src, kids)
        return None
    return rec(root)


# ---------------------------------------------------------------------------
# Edit mapping with node identity
# ---------------------------------------------------------------------------

class _MapNode:
    """Labelled node for APTED that remembers its TreeNode."""

    __slots__ = ('name', 'children', 'node')

    def __init__(self, name: str, children: List['_MapNode'], node: LE.TreeNode):
        """Creates a node.

        Input:  name — label; children — child nodes; node — the TreeNode
        Output: none
        """
        self.name = name
        self.children = children
        self.node = node


#: Cost scale of the tie-breaking edit distance: every unit edit costs
#: TIE_SCALE, pairing two detections that the IoU matching did not pair costs
#: 1 more. Trees have at most layout_eval.MAX_TED_NODES < TIE_SCALE nodes, so
#: the penalties of a mapping never add up to one unit edit.
TIE_SCALE: int = 1 << 16


def _tie_config(matched: Optional[set]):
    """APTED cost model: unit costs, optionally with the matched-pair tie-break.

    Input:  matched — set of (generated det id, reference det id) pairs, or
            None for plain unit costs
    Output: apted Config instance
    """
    from apted import Config

    class TieBreakConfig(Config):
        """Unit costs scaled by TIE_SCALE, + 1 for pairing two unmatched-to-each-other detections."""

        def delete(self, node) -> int:
            """Cost of deleting a node.

            Input:  node — _MapNode
            Output: TIE_SCALE
            """
            return TIE_SCALE

        def insert(self, node) -> int:
            """Cost of inserting a node.

            Input:  node — _MapNode
            Output: TIE_SCALE
            """
            return TIE_SCALE

        def rename(self, a, b) -> int:
            """Cost of mapping generated node a onto reference node b.

            Input:  a, b — _MapNode
            Output: TIE_SCALE * (labels differ) + 1 if both are detections
                    not paired by the box matching
            """
            cost = TIE_SCALE * int(a.name != b.name)
            if (matched is not None and a.node.cls is not None and b.node.cls is not None
                    and (a.node.det_id, b.node.det_id) not in matched):
                cost += 1
            return cost

    return TieBreakConfig()


def edit_mapping(ref: LE.TreeNode, gen: LE.TreeNode, variant: str = VARIANT,
                 matched: Optional[set] = None
                 ) -> Tuple[int, List[Tuple[Optional[LE.TreeNode], Optional[LE.TreeNode]]]]:
    """Optimal unit-cost edit mapping (APTED) from the generated to the reference tree, with node identity.

    With matched, the mapping is still optimal for unit costs, and among the
    optimal mappings it pairs as few detections as possible that the box
    matching did not pair (TIE_SCALE); without it, it is APTED's own choice.

    Input:  ref, gen — canonical TreeNodes; variant — 'g' or 'o'; matched —
            set of (generated det id, reference det id) pairs or None
    Output: (unit-cost distance, list of (generated node or None, reference
            node or None) for every edit: (g, None) delete, (None, r) insert,
            (g, r) rename with different labels); raises RuntimeError if the
            edits do not add up to the distance
    """
    from apted import APTED

    lab = LE.LABELERS[variant]

    def conv(n: LE.TreeNode) -> _MapNode:
        """Labelled copy of a subtree.

        Input:  n — TreeNode
        Output: _MapNode
        """
        return _MapNode(lab(n), [conv(c) for c in n.children], n)

    algo = APTED(conv(gen), conv(ref), _tie_config(matched))
    ted = int(algo.compute_edit_distance()) // TIE_SCALE
    edits = [(a.node if a else None, b.node if b else None)
             for a, b in algo.compute_edit_mapping()
             if a is None or b is None or a.name != b.name]
    if len(edits) != ted:
        raise RuntimeError(f"{len(edits)} edits != distance {ted}")
    return ted, edits


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _parents(root: LE.TreeNode) -> Dict[int, LE.TreeNode]:
    """Parent of every node of a tree.

    Input:  root — TreeNode
    Output: {id(node): parent node}
    """
    return {id(n): p for n, p, _ in LE.iter_preorder(root) if p is not None}


def _detections(node: LE.TreeNode) -> List[LE.TreeNode]:
    """Detection nodes of a subtree (pre-order, the node itself included).

    Input:  node — TreeNode
    Output: list of nodes with cls set
    """
    return [n for n, _, _ in LE.iter_preorder(node) if n.cls is not None]


def _rule(node: Optional[LE.TreeNode]) -> Tuple[Optional[str], Optional[str]]:
    """Emitted type and layout rule of a generated node, from its src.

    Input:  node — generated TreeNode or None
    Output: (type, rule): ('Column', 'gap'), ('Row', 'gap_tolerance'),
            ('LazyColumn', '<item split rule>'), ('Box', 'containment'),
            ('root', None) or (None, None)
    """
    if node is None:
        return None, None
    if node.cls is not None:
        if node.src.startswith('LazyColumn'):
            return 'LazyColumn', node.src.split(':', 1)[1] if ':' in node.src else None
        if node.src == 'Box':
            return 'Box', 'containment'
        return None, None
    if ':' in node.src:
        t, r = node.src.split(':', 1)
        return t, r
    return node.src, None


def is_fallback(node: Optional[LE.TreeNode]) -> bool:
    """Tells whether a generated node was laid out by a fallback (no cut existed).

    Input:  node — generated TreeNode or None
    Output: bool (container rule or LazyColumn item split in layout_eval.FALLBACK_RULES)
    """
    return _rule(node)[1] in LE.FALLBACK_RULES


def tolerance_split(node: Optional[LE.TreeNode], parts: Sequence[LE.TreeNode]) -> bool:
    """Tells whether a tolerance cut separated the given children of a node.

    True if node is a 'gap_tolerance' Column / Row (or LazyColumn item split)
    and two of the given children overlap along its cut axis by a positive
    length (they could only be separated through the tolerance).

    Input:  node — generated TreeNode or None; parts — some of its children
    Output: bool
    """
    t, rule = _rule(node)
    if rule != 'gap_tolerance' or node is None:
        return False
    lo, hi = (1, 3) if t in ('Column', 'LazyColumn') else (0, 2)
    ext = [c.extent for c in parts if c.extent is not None]
    return any(min(a[hi], b[hi]) - max(a[lo], b[lo]) > 0
               for i, a in enumerate(ext) for b in ext[i + 1:])


def _child_towards(node: LE.TreeNode, target: LE.TreeNode, parents: Mapping[int, LE.TreeNode]
                   ) -> Optional[LE.TreeNode]:
    """Child of node on the path to target (target itself if it is a child).

    Input:  node — ancestor; target — descendant; parents — _parents of the tree
    Output: child node or None if target is not below node
    """
    x = target
    while id(x) in parents:
        if parents[id(x)] is node:
            return x
        x = parents[id(x)]
    return None


def _first_det(node: Optional[LE.TreeNode]) -> Optional[int]:
    """Id of the first detection of a subtree (pre-order).

    Input:  node — TreeNode or None
    Output: detection id or None
    """
    dets = _detections(node) if node is not None else []
    return dets[0].det_id if dets else None


class ScreenClassifier:
    """Attributes every edit of one screen's C4 mapping to a category (module docstring, item 4)."""

    def __init__(self, ref: LE.TreeNode, gen: LE.TreeNode, gt: Sequence[Mapping[str, Any]],
                 kept: Sequence[Mapping[str, Any]], raw: Sequence[Mapping[str, Any]],
                 removed: Sequence[Mapping[str, Any]], ev: Optional[XmlEvidence],
                 match_iou: float = LE.MATCH_IOU):
        """Matches the boxes and indexes both trees.

        Input:  ref, gen — canonical trees (reference built by
                indexed_reference_tree); gt — ground-truth boxes; kept —
                de-duplicated detections (the layout input); raw — raw
                detections; removed — de-duplication removals; ev —
                XmlEvidence (None: no XML / pixel facts); match_iou — matching IoU
        Output: none
        """
        self.ref, self.gen, self.ev = ref, gen, ev
        self.gt = list(gt)
        m = LE.match_boxes([b['bbox'] for b in gt], [d['bbox'] for d in kept], match_iou)
        self.g2r = {kept[j]['id']: gt[i]['id'] for i, j, _ in m}
        self.r2g = {r: g for g, r in self.g2r.items()}
        mr = LE.match_boxes([b['bbox'] for b in gt], [d['bbox'] for d in raw], match_iou)
        removed_ids = {d['id'] for d in removed}
        raw_match = {gt[i]['id']: raw[j]['id'] for i, j, _ in mr}
        self.dedup_lost = {r: d for r, d in raw_match.items()
                           if r not in self.r2g and d in removed_ids}
        class_of_gt = {b['id']: b['class_name'] for b in gt}
        class_of_det = {d['id']: d['class_name'] for d in kept}
        self.gpar = _parents(gen)
        self.rpar = _parents(ref)
        self.gnode = {n.det_id: n for n in _detections(gen)}
        self.rnode = {n.det_id: n for n in _detections(ref)}
        self.n_class_mismatch = sum(class_of_det[g] != class_of_gt[r] for g, r in self.g2r.items())
        self.n_fp = len(kept) - len(self.g2r)
        self.n_fn = len(gt) - len(self.g2r)

    # -- per box -------------------------------------------------------------

    def fp_type(self, g: LE.TreeNode) -> Verdict:
        """Category of an unmatched detection.

        Input:  g — generated detection node
        Output: (category, reason, detection id)
        """
        if self.ev is not None and self.ev.window is not None and g.box is not None:
            a = _area(g.box)
            inside = _area(_clip(g.box, self.ev.window)) / a if a > 0 else 0.0
            if inside < WINDOW_MIN_INSIDE:
                return ('reference_artifact',
                        f"{g.cls}#{g.det_id}: {inside:.0%} inside the dumped window "
                        f"{[round(v) for v in self.ev.window]}", g.det_id)
        if self.ev is not None and self.gt and g.box is not None:
            ious = LE.iou_matrix([g.box], [b['bbox'] for b in self.gt])[0]
            k = int(np.argmax(ious))
            best = self.gt[k]['id']
            if ious[k] > 0 and best not in self.r2g and best in self.ev.artifact:
                return ('reference_artifact', f"{g.cls}#{g.det_id} overlaps GT #{best} most: "
                                              f"{self.ev.artifact[best]}", g.det_id)
        return 'detector_fp', f"{g.cls}#{g.det_id} matches no ground-truth box", g.det_id

    def fn_type(self, r: LE.TreeNode) -> Verdict:
        """Category of an unmatched ground-truth box.

        Input:  r — reference detection node
        Output: (category, reason, detection id or None)
        """
        i = r.det_id
        if i in self.dedup_lost:
            return ('dedup_removed',
                    f"GT {r.cls}#{i} matched raw detection {self.dedup_lost[i]}, removed by "
                    f"de-duplication", self.dedup_lost[i])
        if self.ev is not None and i in self.ev.artifact:
            return 'reference_artifact', f"GT {r.cls}#{i}: {self.ev.artifact[i]}", None
        if self.ev is not None and i in self.ev.hidden:
            return 'reference_overlap', f"GT {r.cls}#{i}: {self.ev.hidden[i]}", None
        return 'detector_fn', f"GT {r.cls}#{i} not detected", None

    def _gen_composite_key(self, g: LE.TreeNode) -> Optional[Tuple[str, Any]]:
        """Nearest composite ancestor of a generated node: ('gt', id) if matched, ('fp', det id) if not.

        Input:  g — generated node
        Output: key or None
        """
        p = self.gpar.get(id(g))
        while p is not None:
            if p.cls is not None:
                return ('gt', self.g2r[p.det_id]) if p.det_id in self.g2r else ('fp', p.det_id)
            p = self.gpar.get(id(p))
        return None

    def _ref_composite_key(self, r: LE.TreeNode) -> Optional[Tuple[str, Any]]:
        """Nearest composite ancestor of a reference node, as ('gt', id).

        Input:  r — reference node
        Output: key or None
        """
        p = self.rpar.get(id(r))
        while p is not None:
            if p.cls is not None:
                return ('gt', p.det_id)
            p = self.rpar.get(id(p))
        return None

    def structural(self, g: LE.TreeNode, r: LE.TreeNode) -> Verdict:
        """Cause S(g, r) of a matched pair that the mapping does not pair.

        Input:  g — generated detection; r — the reference detection it is matched to
        Output: (category, reason, detection id)
        """
        gk, rk = self._gen_composite_key(g), self._ref_composite_key(r)
        if gk is not None and gk[0] == 'fp':
            cat, why, _ = self.fp_type(self.gnode[gk[1]])
            return cat, f"{g.cls}#{g.det_id} nested in unmatched detection: {why}", gk[1]
        if gk != rk:
            return ('containment', f"{g.cls}#{g.det_id}: composite ancestor {gk} vs reference {rk}",
                    g.det_id)
        q = self.rpar.get(id(r))
        if q is not None:
            found = [c for c in q.children if any(d.det_id in self.r2g for d in _detections(c))]
            if len(found) < 2:
                missing = [d for d in _detections(q) if d.det_id not in self.r2g and d is not q]
                if missing:
                    cat, why, _ = self._majority([self.fn_type(d) for d in missing])
                    return cat, f"GT #{r.det_id}: its reference group is incomplete ({why})", None
        p = self.gpar.get(id(g))
        if is_fallback(p):
            return 'xycut_fallback', f"{g.cls}#{g.det_id} under {p.src} (no cut)", g.det_id
        if p is not None and any(tolerance_split(p, [g, c]) for c in p.children if c is not g):
            return ('tolerance_cut', f"{g.cls}#{g.det_id} cut from an overlapping sibling under {p.src}",
                    g.det_id)
        kind = stacked_or_flat(self.rpar.get(id(r)))
        if kind:
            return kind, f"GT {r.cls}#{r.det_id}: reference parent {kind[10:]}", g.det_id
        return 'grouping', f"{g.cls}#{g.det_id} grouped differently from GT #{r.det_id}", g.det_id

    @staticmethod
    def _majority(items: Sequence[Verdict]) -> Verdict:
        """Most frequent category of a list (ties in CATEGORIES order) with its first reason and id.

        Input:  items — verdicts (non-empty)
        Output: verdict
        """
        c = Counter(k for k, _, _ in items)
        order = list(CATEGORIES)
        best = sorted(c, key=lambda k: (-c[k], order.index(k)))[0]
        reason, eid = next((r, e) for k, r, e in items if k == best)
        return best, f"{c[best]} of {len(items)} unmatched: {reason}", eid

    @staticmethod
    def _lca(nodes: Sequence[LE.TreeNode], parents: Mapping[int, LE.TreeNode]) -> Optional[LE.TreeNode]:
        """Lowest common ancestor of nodes of one tree (a node counts as its own ancestor).

        Input:  nodes — non-empty list; parents — _parents of their tree
        Output: node or None
        """
        def chain(n: LE.TreeNode) -> List[LE.TreeNode]:
            """Path from a node to the root.

            Input:  n — node
            Output: list of nodes
            """
            out = [n]
            while id(out[-1]) in parents:
                out.append(parents[id(out[-1])])
            return out
        sets = [{id(x) for x in chain(n)} for n in nodes]
        return next((x for x in chain(nodes[0]) if all(id(x) in s for s in sets)), None)

    def _containment_differs(self, pairs: Sequence[Tuple[LE.TreeNode, LE.TreeNode]]
                             ) -> Optional[LE.TreeNode]:
        """First matched pair whose composite ancestors differ between the trees.

        Input:  pairs — (generated, reference) detection pairs
        Output: the generated node of that pair, or None
        """
        return next((g for g, r in pairs if self._gen_composite_key(g) != self._ref_composite_key(r)),
                    None)

    # -- per edit ------------------------------------------------------------

    def classify(self, g: Optional[LE.TreeNode], r: Optional[LE.TreeNode]) -> Tuple[str, str, str, Optional[int]]:
        """Category of one edit of the mapping.

        Input:  g — generated node or None; r — reference node or None
        Output: (edit kind 'insert' | 'delete' | 'rename', category, reason,
                detection id to trace or None)
        """
        if g is None:
            v = self._insert_det(r) if r.cls is not None else self._insert_cont(r)
            return ('insert',) + v
        if r is None:
            v = self._delete_det(g) if g.cls is not None else self._delete_cont(g)
            return ('delete',) + v
        return ('rename',) + self._rename(g, r)

    def _insert_det(self, r: LE.TreeNode) -> Verdict:
        """Insert of a reference detection.

        Input:  r — reference detection
        Output: verdict
        """
        if r.det_id not in self.r2g:
            return self.fn_type(r)
        return self.structural(self.gnode[self.r2g[r.det_id]], r)

    def _delete_det(self, g: LE.TreeNode) -> Verdict:
        """Delete of a generated detection.

        Input:  g — generated detection
        Output: verdict
        """
        if g.det_id not in self.g2r:
            return self.fp_type(g)
        return self.structural(g, self.rnode[self.g2r[g.det_id]])

    def _delete_cont(self, c: LE.TreeNode) -> Verdict:
        """Delete of a generated container.

        Input:  c — generated container
        Output: verdict
        """
        dets = _detections(c)
        pairs = [(d, self.rnode[self.g2r[d.det_id]]) for d in dets if d.det_id in self.g2r]
        if is_fallback(c):
            return ('xycut_fallback', f"container {c.src} (no cut) holding {len(dets)} detections",
                    _first_det(c))
        if len(pairs) < 2:
            return self._majority([self.fp_type(d) for d in dets if d.det_id not in self.g2r])
        if tolerance_split(c, c.children):
            return 'tolerance_cut', f"container {c.src} with overlapping children", _first_det(c)
        g = self._containment_differs(pairs)
        if g is not None:
            return 'containment', f"container {c.src}: members nested differently", g.det_id
        kind = stacked_or_flat(self._lca([r for _, r in pairs], self.rpar))
        if kind:
            return kind, f"container {c.src}: reference members {kind[10:]}", _first_det(c)
        return 'grouping', f"container {c.src} not in the reference", _first_det(c)

    def _insert_cont(self, q: LE.TreeNode) -> Verdict:
        """Insert of a reference container.

        Input:  q — reference container
        Output: verdict
        """
        dets = _detections(q)
        pairs = [(self.gnode[self.r2g[d.det_id]], d) for d in dets if d.det_id in self.r2g]
        name = q.src.rsplit('@', 1)[0]
        if len(pairs) < 2:
            return self._majority([self.fn_type(d) for d in dets if d.det_id not in self.r2g])
        lca = self._lca([g for g, _ in pairs], self.gpar)
        first = pairs[0][0].det_id
        if is_fallback(lca):
            return 'xycut_fallback', f"reference {name}: its boxes meet under {lca.src} (no cut)", first
        parts: List[LE.TreeNode] = []
        for g, _ in pairs:
            c = _child_towards(lca, g, self.gpar)
            if c is not None and all(c is not x for x in parts):
                parts.append(c)
        if tolerance_split(lca, parts):
            return ('tolerance_cut', f"reference {name}: its boxes were cut apart through the "
                                     f"tolerance under {lca.src}", first)
        g = self._containment_differs(pairs)
        if g is not None:
            return 'containment', f"reference {name}: members nested differently", g.det_id
        kind = stacked_or_flat(q)
        if kind:
            return kind, f"reference {name}: children {kind[10:]}", first
        if self.ev is not None:
            e = self.ev.element_of(q.src)
            b = LE._node_box(e) if e is not None else None
            if b is None or not _contains(b, q.extent, ARTIFACT_PX):
                return ('reference_artifact',
                        f"reference {name}: XML bounds {list(b) if b else None} do not contain "
                        f"its boxes {[round(v) for v in q.extent]}", first)
        return 'grouping', f"reference {name}: grouping not produced by XY-cut", first

    def _rename(self, g: LE.TreeNode, r: LE.TreeNode) -> Verdict:
        """Rename of a mapped pair with different labels.

        Input:  g — generated node; r — reference node
        Output: verdict
        """
        if g.cls is not None and r.cls is not None:
            if self.g2r.get(g.det_id) == r.det_id:
                return 'wrong_class', f"{g.cls}#{g.det_id} is GT {r.cls}#{r.det_id}", g.det_id
            if g.det_id not in self.g2r:
                return self.fp_type(g)
            if r.det_id not in self.r2g:
                return self.fn_type(r)
            return self.structural(g, self.rnode[self.g2r[g.det_id]])
        if g.cls is None and r.cls is not None:
            if r.det_id not in self.r2g:
                return self.fn_type(r)
            return ('containment', f"generated {g.src} where the reference has composite "
                                   f"{r.cls}#{r.det_id}", self.r2g[r.det_id])
        if g.cls is not None and r.cls is None:
            if g.det_id not in self.g2r:
                return self.fp_type(g)
            name = r.src.rsplit('@', 1)[0]
            own = self.rnode[self.g2r[g.det_id]]
            if stacked_or_flat(r) == 'reference_overlap' and self.rpar.get(id(own)) is r:
                return ('reference_overlap', f"composite {g.cls}#{g.det_id} (Box: widget with "
                                             f"contents drawn on it) where the reference stacks the "
                                             f"widget and the contents as siblings in {name}", g.det_id)
            return ('containment', f"composite {g.cls}#{g.det_id} where the reference has "
                                   f"container {name}", g.det_id)
        lab = f"{LE.label_o(g)}->{LE.label_o(r)}"
        if is_fallback(g):
            return 'xycut_fallback', f"{g.src} ({lab})", _first_det(g)
        if tolerance_split(g, g.children):
            return 'tolerance_cut', f"{g.src} children overlap ({lab})", _first_det(g)
        kind = stacked_or_flat(r)
        if kind:
            return kind, f"reference children {kind[10:]} ({lab})", _first_det(g)
        return 'grouping', f"orientation {lab}", _first_det(g)


def _describe(n: Optional[LE.TreeNode]) -> Optional[str]:
    """Short text of a node for the edit list.

    Input:  n — TreeNode or None
    Output: e.g. 'TextView#3 [0,10,40,30]', 'Column(Column:gap) [..]' or None
    """
    if n is None:
        return None
    e = n.extent or n.box
    box = '' if e is None else ' [' + ','.join(f"{v:.0f}" for v in e) + ']'
    if n.cls is None:
        return f"{LE.label_o(n)}({n.src.rsplit('@', 1)[0]}){box}"
    return f"{n.cls}#{n.det_id}" + (' (composite)' if n.children else '') + box


# ---------------------------------------------------------------------------
# One screen
# ---------------------------------------------------------------------------

def analyse_screen(stem: str, xml_path: Path, label_path: Path, image_path: Path,
                   dets: Optional[Sequence[Mapping[str, Any]]], names: Sequence[str],
                   params: Mapping[str, Any]) -> Dict[str, Any]:
    """Builds the C4 trees of one screen, maps them and classifies every edit.

    Input:  stem — image stem; xml_path, label_path, image_path — package
            files; dets — cached raw detections (None: not cached, status
            'excluded:no_predictions' as in layout_eval); names — class names;
            params — layout_eval.default_params()-shaped thresholds
    Output: record {'stem', 'status', counts, 'ted', 'teds', 'edits' [{kind,
            category, reason, gen, ref, explain_id}], 'categories',
            'dominant', trees}; an excluded reference keeps layout_eval's status
    """
    rec_ref, ref_official, gt = LE.build_reference(stem, xml_path, label_path, image_path, names)
    out: Dict[str, Any] = {'stem': stem, 'status': rec_ref['status']}
    if ref_official is None:
        return out
    if dets is None:
        out['status'] = 'excluded:no_predictions'
        return out
    img_w, img_h = rec_ref['image_size']
    root = ET.parse(xml_path).getroot()
    name2id = {n: i for i, n in enumerate(names)}
    traced = LE.trace_labels(root, img_w, img_h, name2id)
    elems, _ = LE.align_labels(LE.read_label_lines(label_path), traced, root, img_w, img_h, name2id)
    with Image.open(image_path) as im:
        gray = np.asarray(im.convert('L'), dtype=np.float32)
    ev = XmlEvidence(root, elems, gt, (img_w, img_h), gray)
    ref = LE.canonicalize(indexed_reference_tree(root, {id(e): i for i, e in enumerate(elems)}, gt, ev))
    if LE.bracket(ref) != LE.bracket(ref_official):
        raise RuntimeError(f"{stem}: indexed reference tree differs from layout_eval's")

    raw = [dict(d) for d in dets]
    kept, removed = deduplicate_detections([dict(d) for d in dets], **params['dedup'])
    graph = UIGraphBuilder(containment_ratio=params['containment_ratio']).build_graph(
        [dict(b) for b in kept], image_size=(img_w, img_h))
    codegen = ComposeCodeGenerator()
    codegen.generate(graph, image_size=(img_w, img_h), layout='xycut',
                     cut_tol_frac=params['cut_tol_frac'], fallback=params['fallback'])
    gen_raw, n_fb = LE.tree_from_layout(codegen.last_layout_tree)
    gen = LE.canonicalize(gen_raw)

    clf = ScreenClassifier(ref, gen, gt, kept, raw, removed, ev, params.get('match_iou', LE.MATCH_IOU))
    ted, edits = edit_mapping(ref, gen, VARIANT, set(clf.g2r.items()))
    official = LE.tree_edit(ref, gen, VARIANT)
    if official['ted'] != ted:
        raise RuntimeError(f"{stem}: mapping distance {ted} != tree_edit {official['ted']}")
    rows = []
    for g, r in edits:
        kind, cat, reason, eid = clf.classify(g, r)
        rows.append({'kind': kind, 'category': cat, 'reason': reason, 'gen': _describe(g),
                     'ref': _describe(r), 'explain_id': eid})
    cats = Counter(x['category'] for x in rows)
    order = list(CATEGORIES)
    dominant = sorted(cats, key=lambda k: (-cats[k], order.index(k)))[0] if cats else 'none'
    out.update({
        'status': 'ok', 'image_size': [img_w, img_h], 'n_gt': len(gt), 'n_raw': len(raw),
        'n_kept': len(kept), 'n_dedup_removed': len(removed),
        'n_dedup_removed_matching_gt': len(clf.dedup_lost), 'n_matched': len(clf.g2r),
        'n_fp': clf.n_fp, 'n_fn': clf.n_fn, 'n_class_mismatch': clf.n_class_mismatch,
        'n_fallback': n_fb, 'ref_nodes': official['n_ref'], 'gen_nodes': official['n_gen'],
        'ted': ted, 'teds': official['teds'],
        'ops': {k: official[k] for k in EDIT_KINDS},
        'categories': {k: cats[k] for k in order if cats[k]},
        'dominant': dominant,
        'dominant_share': cats[dominant] / len(rows) if rows else None,
        'window': ev.window, 'n_hidden_labels': len(ev.hidden),
        'n_artifact_labels': len(ev.artifact),
        'edits': rows,
        'ref_tree': LE.bracket(ref), 'gen_tree': LE.bracket(gen),
        'ref_render': LE.render(ref, 80), 'gen_render': LE.render(gen, 80),
    })
    return out


# ---------------------------------------------------------------------------
# Sample, aggregate, cases
# ---------------------------------------------------------------------------

def sample_stems(stems: Sequence[str], n: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> List[str]:
    """Random sample of screens without replacement (numpy default_rng(seed).choice over the sorted stems).

    Input:  stems — screens of the split; n — sample size; seed — RNG seed
    Output: sampled stems in draw order
    """
    s = sorted(stems)
    idx = np.random.default_rng(seed).choice(len(s), size=min(n, len(s)), replace=False)
    return [s[int(i)] for i in idx]


def summarize(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Counts of the taxonomy over the analysed screens.

    Input:  records — analyse_screen outputs
    Output: {'n_screens', 'n_ok', 'status', 'n_edits', 'edits_per_screen',
             'edits_per_category' {cat: {'n', 'share'}}, 'dominant' {cat:
             screens}, 'screens_with' {cat: screens}, 'kind_by_category',
             'families', 'dominant_by_family', 'detector_totals'}
    """
    ok = [r for r in records if r.get('status') == 'ok']
    edits = [e for r in ok for e in r['edits']]
    n = len(edits)
    cat = Counter(e['category'] for e in edits)
    dom = Counter(r['dominant'] for r in ok)
    order = list(CATEGORIES)
    fam = {f: sum(cat[c] for c in cs) for f, cs in FAMILIES.items()}
    return {
        'n_screens': len(records), 'n_ok': len(ok),
        'status': dict(sorted(Counter(r.get('status') for r in records).items())),
        'n_edits': n,
        'edits_per_screen': LE.describe([len(r['edits']) for r in ok]),
        'edits_per_category': {c: {'n': cat[c], 'share': cat[c] / n if n else None} for c in order},
        'dominant': {c: dom[c] for c in order + ['none']},
        'screens_with': {c: sum(1 for r in ok if r['categories'].get(c)) for c in order},
        'kind_by_category': {k: {c: sum(1 for e in edits if e['kind'] == k and e['category'] == c)
                                 for c in order} for k in EDIT_KINDS},
        'families': {f: {'n': v, 'share': v / n if n else None, 'categories': list(FAMILIES[f])}
                     for f, v in fam.items()},
        'dominant_by_family': {f: sum(dom[c] for c in cs) for f, cs in FAMILIES.items()},
        'detector_totals': {k: sum(r[k] for r in ok) for k in
                            ('n_gt', 'n_raw', 'n_kept', 'n_matched', 'n_fp', 'n_fn',
                             'n_class_mismatch', 'n_dedup_removed', 'n_dedup_removed_matching_gt',
                             'n_fallback')},
    }


def select_cases(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Illustrative cases: one screen per family (module docstring, item 6).

    Input:  records — analyse_screen outputs
    Output: list of {'family', 'category', 'stem', 'share', 'n_edits',
            'n_category', 'teds', 'explain_id', 'edit'} (a family without a
            screen whose dominant category has a traceable edit is skipped)
    """
    def traceable(r: Mapping[str, Any]) -> bool:
        """Tells whether a screen has an edit of its dominant category with a detection to trace.

        Input:  r — record
        Output: bool
        """
        return any(e['category'] == r['dominant'] and e['explain_id'] is not None for e in r['edits'])

    ok = [r for r in records if r.get('status') == 'ok' and r['edits'] and traceable(r)]
    order = list(CATEGORIES)
    out = []
    for fam, cats in FAMILIES.items():
        dom = Counter(r['dominant'] for r in ok if r['dominant'] in cats)
        if not dom:
            continue
        cat = sorted(dom, key=lambda c: (-dom[c], order.index(c)))[0]
        cands = [r for r in ok if r['dominant'] == cat]
        best = sorted(cands, key=lambda r: (-r['dominant_share'], -r['categories'][cat],
                                            r['stem']))[0]
        edit = next(e for e in best['edits'] if e['category'] == cat and e['explain_id'] is not None)
        out.append({'family': fam, 'category': cat, 'stem': best['stem'],
                    'share': best['dominant_share'], 'n_edits': len(best['edits']),
                    'n_category': best['categories'][cat], 'teds': best['teds'],
                    'explain_id': edit['explain_id'], 'edit': edit})
    return out


def run_taxonomy(data_dir: str | Path, split: str, pred_cache: str | Path, out_dir: str | Path,
                 n: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED, run_dir: Optional[str | Path] = None,
                 params: Optional[Mapping[str, Any]] = None, allow_test: bool = False) -> Dict[str, Any]:
    """Analyses a random sample of C4 screens and writes failure_taxonomy.json.

    Input:  data_dir — package root; split — 'val' or 'test'; pred_cache —
            cached raw detections of the split; out_dir — output directory;
            n, seed — sample; run_dir — finished ui2compose-eval-layout run of
            the same split to cross-check TED and trees against (optional;
            a mismatch raises RuntimeError); params — thresholds (default:
            layout_eval.default_params()); allow_test — required for 'test'
    Output: result dictionary (also <out_dir>/failure_taxonomy.json)
    """
    if split == 'test' and not allow_test:
        raise ValueError("split 'test' needs allow_test=True (--allow-test)")
    d = Path(data_dir)
    p = LE.default_params()
    for k, v in (params or {}).items():
        p[k] = v
    names = LE.load_names(d)
    mapping = LE.load_mapping(d, split)
    stems = LE.select_stems(d, split)
    sample = sample_stems(stems, n, seed)
    records = []
    for stem in sample:
        row = mapping.get(stem)
        dets = LE.load_predictions_file(Path(pred_cache) / f"{stem}.json")
        if row is None:
            records.append({'stem': stem, 'status': 'excluded:no_mapping'})
            continue
        records.append(analyse_screen(stem, d / row['xml_in_zip'],
                                      d / 'dataset' / 'labels' / split / f"{stem}.txt",
                                      d / 'dataset' / 'images' / split / f"{stem}.png",
                                      dets, names, p))
    check = None
    if run_dir is not None:
        ref_recs, conds = LE.read_run(run_dir)
        c4 = conds.get(CONDITION, {})
        bad = []
        for r in records:
            if r.get('status') != 'ok':
                continue
            o = c4.get(r['stem'], {})
            same = (o.get(f"ted_{VARIANT}") == r['ted'] and o.get('tree') in (None, r['gen_tree'])
                    and ref_recs.get(r['stem'], {}).get('tree') in (None, r['ref_tree']))
            if not same:
                bad.append(r['stem'])
        check = {'run_dir': str(run_dir), 'n_checked': sum(r.get('status') == 'ok' for r in records),
                 'n_mismatch': len(bad), 'mismatches': bad}
        if bad:
            raise RuntimeError(f"taxonomy differs from the run {run_dir} on {bad}")
    result = {
        'meta': {'tool': 'ui2compose.evaluation.failure_taxonomy', 'ui2compose_version': __version__,
                 'created_utc': _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                 'data_dir': str(data_dir), 'split': split, 'pred_cache': str(pred_cache),
                 'prediction_meta': LE.prediction_meta(pred_cache),
                 'condition': CONDITION, 'variant': VARIANT, 'params': p,
                 'sample': {'n': len(sample), 'seed': seed, 'method': 'numpy default_rng(seed).choice '
                            'over the sorted stems of the split, without replacement',
                            'n_split': len(stems), 'stems': sample},
                 'thresholds': {'window_min_inside': WINDOW_MIN_INSIDE, 'pixel_std_max': PIXEL_STD_MAX,
                                'shared_min': SHARED_MIN, 'cover_area_ratio_max': COVER_AREA_RATIO_MAX,
                                'artifact_px': ARTIFACT_PX},
                 'categories': CATEGORIES, 'families': {k: list(v) for k, v in FAMILIES.items()},
                 'run_check': check},
        'summary': summarize(records),
        'cases': select_cases(records),
        'screens': sorted(records, key=lambda r: r['stem']),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'failure_taxonomy.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


#: Categories whose key trace line is the line of the generated container
#: named in the edit (its type and rule); the others start with KEY_TRACE.
LAYOUT_KEYED: Tuple[str, ...] = ('containment', 'xycut_fallback', 'tolerance_cut', 'grouping',
                                 'reference_flat2d', 'reference_overlap')


def key_trace_line(lines: Optional[Sequence[str]], category: str,
                   edit: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """The trace line that shows the rule behind an edit.

    For layout categories (LAYOUT_KEYED) the line of the generated container
    named in the edit ('Column:fallback_y_sorted' in the edited node or the
    reason -> the line starting 'Column (rule=fallback_y_sorted'; a composite
    -> its 'Box (rule=containment' line) is searched first; then the
    category's KEY_TRACE keys, then the '[detection]' line. For the other
    categories the KEY_TRACE keys come first.

    Input:  lines — --explain output lines or None; category — taxonomy
            category; edit — the edit dictionary (optional)
    Output: first matching line (stripped) or None
    """
    if not lines:
        return None
    named: List[str] = []
    for text in ((edit or {}).get('gen') or '', (edit or {}).get('reason') or ''):
        for typ, rule in re.findall(r'\b(Column|Row|LazyColumn):([a-z_]+)', text):
            named.append('LazyColumn (rule=list' if typ == 'LazyColumn' else f"{typ} (rule={rule}")
        if '(composite)' in text:
            named.append('Box (rule=containment')
    keys = list(KEY_TRACE.get(category, ()))
    keys = (named + keys if category in LAYOUT_KEYED else keys + named) + ['[detection]']
    for key in keys:
        hit = next((line.strip() for line in lines if key in line), None)
        if hit:
            return hit
    return None


def run_cases(taxonomy: str | Path, data_dir: str | Path, split: str, pred_cache: str | Path,
              out_dir: str | Path, model: str | Path, allow_test: bool = False,
              pipeline_config: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    """Runs the pipeline with --explain on the illustrative cases and stores the traces.

    The pipeline (python -m ui2compose.infer.pipeline <image> --model <model>
    --explain <id> --output <dir> --no-viz) runs with its shipped defaults, or
    with --config <pipeline_config> when given; its raw detections are
    compared with the cached ones.

    Input:  taxonomy — failure_taxonomy.json; data_dir — package root; split —
            split of the taxonomy; pred_cache — cached detections; out_dir —
            pipeline output root; model — weights; allow_test — required for test;
            pipeline_config — optional pipeline config (thresholds of the taxonomy)
    Output: the cases with 'explain_lines', 'pipeline_dir',
            'detections_identical_to_cache' (also written back into the JSON)
    """
    if split == 'test' and not allow_test:
        raise ValueError("split 'test' needs allow_test=True (--allow-test)")
    tx = json.loads(Path(taxonomy).read_text(encoding='utf-8'))
    cases = tx['cases']
    for case in cases:
        stem = case['stem']
        img = Path(data_dir) / 'dataset' / 'images' / split / f"{stem}.png"
        odir = Path(out_dir) / stem
        cmd = [sys.executable, '-m', 'ui2compose.infer.pipeline', str(img), '--model', str(model),
               '--output', str(odir), '--no-viz']
        if pipeline_config is not None:
            cmd += ['--config', str(pipeline_config)]
        if case.get('explain_id') is not None:
            cmd += ['--explain', str(case['explain_id'])]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        case['pipeline_command'] = 'ui2compose ' + ' '.join(cmd[3:])
        case['pipeline_returncode'] = proc.returncode
        case['pipeline_dir'] = str(odir)
        ex = odir / f"2_explain_{case.get('explain_id')}.txt"
        case['explain_lines'] = (ex.read_text(encoding='utf-8').splitlines()
                                 if case.get('explain_id') is not None and ex.is_file() else None)
        case['key_line'] = key_trace_line(case['explain_lines'], case['category'], case.get('edit'))
        raw_out = odir / '1_detections_raw.json'
        cached = Path(pred_cache) / f"{stem}.json"
        if raw_out.is_file() and cached.is_file():
            a = json.loads(raw_out.read_text(encoding='utf-8'))
            b = json.loads(cached.read_text(encoding='utf-8'))
            keys = ('id', 'class_name', 'confidence', 'bbox')
            case['detections_identical_to_cache'] = (
                [{k: x.get(k) for k in keys} for x in a] == [{k: x.get(k) for k in keys} for x in b])
        else:
            case['detections_identical_to_cache'] = None
    tx['cases'] = cases
    Path(taxonomy).write_text(json.dumps(tx, indent=2), encoding='utf-8')
    return cases


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _pct(v: Optional[float]) -> str:
    """Formats a share as a percentage.

    Input:  v — share or None
    Output: e.g. '12.3 %'
    """
    return '–' if v is None else f"{100 * v:.1f} %"


def compact(tx: Mapping[str, Any]) -> Dict[str, Any]:
    """Repository version of the taxonomy: summary, cases and per-screen counts (no trees, no edit lists).

    Input:  tx — run_taxonomy result (optionally with run_cases traces)
    Output: dictionary
    """
    keep = ('stem', 'status', 'n_gt', 'n_raw', 'n_kept', 'n_matched', 'n_fp', 'n_fn',
            'n_class_mismatch', 'n_dedup_removed', 'n_dedup_removed_matching_gt', 'n_fallback',
            'ref_nodes', 'gen_nodes', 'ted', 'teds', 'ops', 'categories', 'dominant', 'dominant_share')
    return {'meta': tx['meta'], 'summary': tx['summary'], 'cases': tx['cases'],
            'screens': [{k: r.get(k) for k in keep if k in r} for r in tx['screens']]}


def markdown(tx: Mapping[str, Any]) -> str:
    """Readable report of a taxonomy run.

    Input:  tx — run_taxonomy result (optionally with run_cases traces)
    Output: Markdown text
    """
    m, s = tx['meta'], tx['summary']
    L: List[str] = []
    L.append(f"# Failure taxonomy of C4 ({m['split']} split, {m['sample']['n']} random screens, "
             f"seed {m['sample']['seed']})")
    L.append('')
    L.append("Generated by `python -m ui2compose.evaluation.failure_taxonomy` (method: module "
             "docstring). Every edit of the optimal TEDS-o edit mapping (generated tree -> "
             "reference tree) gets exactly one category by fixed rules; the rules were fixed on "
             "the validation split. Structure only: texts, sizes and styles are not compared.")
    L.append('')
    L.append(f"Screens: {s['n_ok']} analysed of {s['n_screens']} sampled "
             f"(status {s['status']}). Edits: {s['n_edits']} "
             f"(per screen median {s['edits_per_screen'].get('median', 0):.0f}, "
             f"IQR {s['edits_per_screen'].get('q1', 0):.0f}–{s['edits_per_screen'].get('q3', 0):.0f}, "
             f"max {s['edits_per_screen'].get('max', 0):.0f}).")
    if m.get('run_check'):
        rc = m['run_check']
        L.append(f"Cross-check against the evaluation run: {rc['n_checked'] - rc['n_mismatch']} of "
                 f"{rc['n_checked']} screens have the same TED-o and the same trees.")
    L.append('')
    L.append('| category | meaning | edits | share | screens where dominant | screens with ≥ 1 |')
    L.append('|---|---|---:|---:|---:|---:|')
    for c, desc in CATEGORIES.items():
        e = s['edits_per_category'][c]
        L.append(f"| {c} | {desc} | {e['n']} | {_pct(e['share'])} | {s['dominant'][c]} | "
                 f"{s['screens_with'][c]} |")
    L.append(f"| none | perfect screen (no edit) | – | – | {s['dominant']['none']} | – |")
    L.append('')
    L.append('Families: ' + '; '.join(f"{f} {v['n']} edits ({_pct(v['share'])}; dominant on "
                                       f"{s['dominant_by_family'][f]} screens; "
                                       f"{', '.join(v['categories'])})"
                                       for f, v in s['families'].items()) + '.')
    L.append('')
    L.append('Edit kind x category:')
    L.append('')
    L.append('| kind | ' + ' | '.join(CATEGORIES) + ' |')
    L.append('|---|' + '---:|' * len(CATEGORIES))
    for k in EDIT_KINDS:
        L.append(f"| {k} | " + ' | '.join(str(s['kind_by_category'][k][c]) for c in CATEGORIES) + ' |')
    L.append('')
    dt = s['detector_totals']
    L.append(f"Detector level on these screens: {dt['n_gt']} GT boxes, {dt['n_raw']} raw detections, "
             f"{dt['n_kept']} after de-duplication, {dt['n_matched']} matched (IoU >= 0.5), "
             f"{dt['n_fp']} unmatched detections, {dt['n_fn']} unmatched GT boxes, "
             f"{dt['n_class_mismatch']} class mismatches, {dt['n_dedup_removed']} de-duplication "
             f"removals of which {dt['n_dedup_removed_matching_gt']} matched a GT box, "
             f"{dt['n_fallback']} fallback nodes.")
    L.append('')
    L.append('## Illustrative cases')
    L.append('')
    L.append("One screen per family (module docstring item 6): among the screens whose dominant "
             "category has an edit with a detection to trace, the family's category that dominates "
             "most screens, and among those screens the highest share of edits in it. The "
             "traceability condition was added after the first test run (its reference-family "
             "case, an undetected ImageView over uniform pixels, had nothing to trace); no number "
             "depends on the case selection.")
    L.append('')
    for k, c in enumerate(tx['cases'], 1):
        L.append(f"{k}. **{c['stem']}** ({c['family']}: {c['category']}, {c['n_category']} of "
                 f"{c['n_edits']} edits, TEDS-o {c['teds']:.3f}). Edit: {c['edit']['kind']} "
                 f"{c['edit']['gen'] or c['edit']['ref']} — {c['edit']['reason']}.")
        if c.get('key_line'):
            L.append(f"   Key trace line: `{c['key_line']}`")
        if c.get('explain_lines'):
            L.append(f"   Trace (`{c.get('pipeline_command', 'ui2compose ... --explain')}`; raw "
                     f"detections identical to the cache: {c.get('detections_identical_to_cache')}):")
            L.append('')
            L.append('   ```')
            L.extend('   ' + line for line in c['explain_lines'])
            L.append('   ```')
        elif c.get('explain_id') is None:
            L.append('   No detection to trace (the edit concerns undetected boxes).')
        L.append('')
    L.append('## Per screen')
    L.append('')
    L.append('| stem | GT | det (kept) | matched | TEDS-o | edits | dominant | categories |')
    L.append('|---|---:|---:|---:|---:|---:|---|---|')
    for r in tx['screens']:
        if r.get('status') != 'ok':
            L.append(f"| {r['stem']} | | | | | | {r.get('status')} | |")
            continue
        cats = ', '.join(f"{k} {v}" for k, v in r['categories'].items())
        L.append(f"| {r['stem']} | {r['n_gt']} | {r['n_kept']} | {r['n_matched']} | {r['teds']:.3f} | "
                 f"{len(r['edits']) if 'edits' in r else r['ted']} | {r['dominant']} | {cats} |")
    L.append('')
    return '\n'.join(L)


def write_results(tx: Mapping[str, Any], results_dir: str | Path, name: str = 'failure_taxonomy'
                  ) -> Tuple[Path, Path]:
    """Writes the compact JSON and the Markdown report.

    Input:  tx — taxonomy result; results_dir — directory; name — file stem
    Output: (json path, md path)
    """
    d = Path(results_dir)
    d.mkdir(parents=True, exist_ok=True)
    pj, pm = d / f"{name}.json", d / f"{name}.md"
    pj.write_text(json.dumps(compact(tx), indent=2), encoding='utf-8')
    pm.write_text(markdown(tx), encoding='utf-8')
    return pj, pm


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Defines and reads the command-line arguments.

    Input:  argv — argument list (default sys.argv[1:])
    Output: argparse namespace
    """
    ap = argparse.ArgumentParser(description="Failure taxonomy of the C4 layout edits")
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help="analyse a random sample of screens")
    r.add_argument('--data-dir', required=True)
    r.add_argument('--split', default='val', choices=('val', 'test'))
    r.add_argument('--allow-test', action='store_true')
    r.add_argument('--pred-cache', required=True)
    r.add_argument('--out-dir', required=True, help="failure_taxonomy.json (full, outside the repo)")
    r.add_argument('--run', default=None, help="ui2compose-eval-layout output of the split (cross-check; "
                   "its thresholds are used)")
    r.add_argument('--n', type=int, default=SAMPLE_SIZE)
    r.add_argument('--seed', type=int, default=SAMPLE_SEED)
    r.add_argument('--results-dir', default=None, help="also write the compact JSON + Markdown here")
    r.add_argument('--name', default='failure_taxonomy')
    c = sub.add_parser('cases', help="pipeline --explain on the illustrative cases")
    c.add_argument('--taxonomy', required=True)
    c.add_argument('--data-dir', required=True)
    c.add_argument('--split', default='val', choices=('val', 'test'))
    c.add_argument('--allow-test', action='store_true')
    c.add_argument('--pred-cache', required=True)
    c.add_argument('--model', default='runs/oversample_5k/weights/best.pt')
    c.add_argument('--pipeline-config', default=None,
                   help="pipeline config with the thresholds of the taxonomy (default: shipped defaults)")
    c.add_argument('--out-dir', required=True)
    c.add_argument('--results-dir', default=None)
    c.add_argument('--name', default='failure_taxonomy')
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Runs a subcommand.

    Input:  argv — argument list
    Output: exit code (2 for refused arguments)
    """
    a = parse_args(argv)
    try:
        if a.cmd == 'run':
            params = None
            if a.run:
                agg = Path(a.run) / 'aggregate.json'
                if agg.is_file():
                    params = json.loads(agg.read_text(encoding='utf-8'))['meta']['params']
            tx = run_taxonomy(a.data_dir, a.split, a.pred_cache, a.out_dir, a.n, a.seed, a.run,
                              params=params, allow_test=a.allow_test)
        else:
            run_cases(a.taxonomy, a.data_dir, a.split, a.pred_cache, a.out_dir, a.model,
                      allow_test=a.allow_test, pipeline_config=a.pipeline_config)
            tx = json.loads(Path(a.taxonomy).read_text(encoding='utf-8'))
    except ValueError as e:
        print(f"[ERR] {e}")
        return 2
    s = tx['summary']
    print(f"{s['n_ok']} screens, {s['n_edits']} edits")
    for cat, v in s['edits_per_category'].items():
        print(f"  {cat:20s} {v['n']:5d}  dominant on {s['dominant'][cat]}")
    if a.results_dir:
        pj, pm = write_results(tx, a.results_dir, a.name)
        print(f"written: {pj}, {pm}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
