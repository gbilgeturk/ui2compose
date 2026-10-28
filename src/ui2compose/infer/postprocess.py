# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Detection post-processing (stage 2 of the pipeline): per-class confidence filtering and de-duplication.

The detector's non-maximum suppression (NMS) is class-aware and uses a single
IoU threshold, so two kinds of redundant boxes can survive it and reach the UI
graph and the generated code:

  * the same element detected twice with the same class, either as two
    largely coinciding boxes or as a tight box plus a loose box that encloses
    it (e.g. a wide TextView around a tight TextView over one text line);
  * the same region detected as two different classes (competing class
    hypotheses, e.g. a TextView box that almost coincides with a RecyclerView).

deduplicate_detections() removes such boxes with three explicit rules (see its
docstring) and records, for every removed box, which rule removed it, the
overlap value that triggered the rule and the id of the detection that caused
the removal, so that a removal can be traced in the output.

All thresholds are named parameters. Their default values were selected on
the validation split of the app-level split (results/eval_appsplit/
selected_thresholds.json; grid sweep of ui2compose.evaluation.layout_sweep,
objective: mean TEDS-o of the generated layout against the ReDraw view
hierarchies; the earlier sweep on the screenshot-level split, results/eval/,
had selected the same values). The a-priori values the sweep started from were
0.5 / 0.9 / 0.6. The test split and the bundled examples (examples/*.png) were
not used to choose them.

This module is pure Python (no NumPy, OpenCV, torch or ultralytics).
"""

import copy
import difflib
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

#: Class names of the released detector in class-id order (configs/dataset.yaml;
#: a test checks that both agree). Keys of class_conf are checked against them.
DETECTOR_CLASSES: Tuple[str, ...] = (
    'Button', 'CheckBox', 'EditText', 'ImageView', 'ListView', 'ProgressBar',
    'RadioButton', 'RecyclerView', 'SeekBar', 'Spinner', 'Switch', 'TextView', 'WebView',
)

# Classes that cannot legitimately contain another instance of the same class.
# For these, a same-class box that (almost) encloses or is enclosed by another
# one is treated as a duplicate (rule 'same_class_nested'). ImageView,
# ListView, RecyclerView and WebView are deliberately excluded: an avatar
# ImageView inside a banner ImageView or a list nested in a list item is a
# legitimate layout.
NON_NESTING_CLASSES: FrozenSet[str] = frozenset({
    'TextView', 'Button', 'EditText', 'CheckBox', 'RadioButton',
    'Switch', 'SeekBar', 'ProgressBar', 'Spinner',
})

# Rule names written to the 'dedup_rule' field of removed detections.
RULE_SAME_CLASS_IOU = 'same_class_iou'
RULE_SAME_CLASS_NESTED = 'same_class_nested'
RULE_CROSS_CLASS_IOU = 'cross_class_iou'

# Defaults selected on the validation split (results/eval_appsplit/selected_thresholds.json:
# setting 'looser' of the sweep, chosen among off / 0.6-0.95-0.7 / 0.5-0.9-0.6 /
# 0.4-0.85-0.5; never tuned on the bundled test examples).
# Same-class IoU above which the lower-confidence box is a duplicate.
# ultralytics' own NMS default is 0.7, so this rule catches same-class pairs
# that NMS let through (a-priori value 0.5, the usual matching criterion).
DEFAULT_SAME_CLASS_IOU = 0.4
# Intersection over the smaller box's area above which two same-class boxes of
# a NON_NESTING_CLASSES class are duplicates (one box almost inside the other;
# a-priori value 0.9).
DEFAULT_SAME_CLASS_IOMIN = 0.85
# Cross-class IoU above which the lower-confidence box is a competing class
# hypothesis for the same region. Higher than the same-class threshold
# because different classes legitimately overlap more often (e.g. a label
# inside a button; a-priori value 0.6).
DEFAULT_CROSS_CLASS_IOU = 0.5

_REQUIRED_KEYS = ('id', 'class_name', 'confidence', 'bbox')


def box_area(box: Sequence[float]) -> float:
    """Computes the area of an axis-aligned box; inverted or degenerate boxes have area 0.

    Input:  box — [x1, y1, x2, y2] in pixels
    Output: area as float (>= 0)
    """
    x1, y1, x2, y2 = box
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    """Computes the intersection area of two axis-aligned boxes.

    Input:  a, b — boxes [x1, y1, x2, y2] in pixels
    Output: intersection area as float (0 if the boxes do not overlap or only touch)
    """
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    return float(max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1))


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Computes the intersection over union (IoU) of two boxes.

    Input:  a, b — boxes [x1, y1, x2, y2] in pixels
    Output: IoU in [0, 1]; 0.0 when the union is empty (e.g. two zero-area boxes)
    """
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def intersection_over_min(a: Sequence[float], b: Sequence[float]) -> float:
    """Computes the intersection area divided by the area of the smaller box.

    A value near 1 means that one box lies (almost) entirely inside the other.

    Input:  a, b — boxes [x1, y1, x2, y2] in pixels
    Output: value in [0, 1]; 0.0 when either box has zero area
    """
    min_area = min(box_area(a), box_area(b))
    if min_area <= 0.0:
        return 0.0
    return float(intersection_area(a, b) / min_area)


def _check_threshold(name: str, value: Optional[float]) -> None:
    """Validates a de-duplication threshold.

    Input:  name — parameter name (for the error message); value — threshold or None
    Output: none (raises ValueError unless value is None or 0 < value <= 1)
    """
    if value is None:
        return
    if not (0.0 < float(value) <= 1.0):
        raise ValueError(f"{name} must be in (0, 1] or None, got {value!r}")


def _check_detections(detections: List[Dict]) -> None:
    """Validates the keys, box format and id uniqueness of a detection list.

    Input:  detections — list of detection dictionaries
    Output: none (raises ValueError on a missing key, a malformed bbox or a duplicate id)
    """
    seen = set()
    for index, d in enumerate(detections):
        missing = [k for k in _REQUIRED_KEYS if k not in d]
        if missing:
            raise ValueError(f"detection at index {index} lacks keys {missing}")
        if len(d['bbox']) != 4:
            raise ValueError(f"detection {d['id']!r}: bbox must be [x1, y1, x2, y2]")
        if d['id'] in seen:
            raise ValueError(f"duplicate detection id {d['id']!r}")
        seen.add(d['id'])


def _match_rule(candidate: Dict, kept: Dict,
                same_class_iou: Optional[float],
                same_class_iomin: Optional[float],
                cross_class_iou: Optional[float]) -> Optional[Tuple[str, float]]:
    """Checks whether an already-kept detection makes a candidate redundant.

    Rules are checked in the order same_class_iou, same_class_nested,
    cross_class_iou; the first one that holds is returned. A rule whose
    threshold is None is disabled.

    Input:  candidate — detection under test; kept — an already-kept detection;
            same_class_iou, same_class_iomin, cross_class_iou — thresholds (or None)
    Output: (rule name, overlap value) of the first rule that holds, or None
    """
    a, b = kept['bbox'], candidate['bbox']
    if candidate['class_name'] == kept['class_name']:
        if same_class_iou is not None:
            overlap = iou(a, b)
            if overlap >= same_class_iou:
                return RULE_SAME_CLASS_IOU, overlap
        if same_class_iomin is not None and candidate['class_name'] in NON_NESTING_CLASSES:
            overlap = intersection_over_min(a, b)
            if overlap >= same_class_iomin:
                return RULE_SAME_CLASS_NESTED, overlap
    elif cross_class_iou is not None:
        overlap = iou(a, b)
        if overlap >= cross_class_iou:
            return RULE_CROSS_CLASS_IOU, overlap
    return None


def deduplicate_detections(
        detections: List[Dict],
        same_class_iou: Optional[float] = DEFAULT_SAME_CLASS_IOU,
        same_class_iomin: Optional[float] = DEFAULT_SAME_CLASS_IOMIN,
        cross_class_iou: Optional[float] = DEFAULT_CROSS_CLASS_IOU,
) -> Tuple[List[Dict], List[Dict]]:
    """Removes duplicate and competing detections with three explicit, traceable rules.

    Detections are visited greedily in descending confidence (ties: lower 'id'
    first). A detection is kept unless an ALREADY-KEPT detection K makes it
    redundant by one of the rules below (removed detections never remove
    others). K is compared in the order the kept detections were accepted
    (highest confidence first) and, for each K, the rules in this order:
      (a) 'same_class_iou'    — K has the same class and IoU >= same_class_iou;
      (b) 'same_class_nested' — K has the same class, that class is in
          NON_NESTING_CLASSES, and intersection / min(area) >= same_class_iomin;
      (c) 'cross_class_iou'   — K has a different class and IoU >= cross_class_iou.
    The first (K, rule) pair that holds is recorded. Thresholds are inclusive;
    passing None disables a rule (useful for ablations). Boxes with zero area
    have overlap 0 with everything, so they are never removed and never remove.

    The default thresholds were selected on the validation split
    (results/eval_appsplit/selected_thresholds.json), not on the test split or
    the bundled examples.

    Input:  detections — list of detection dicts with at least 'id' (unique),
                'class_name', 'confidence' and 'bbox' [x1, y1, x2, y2] in pixels;
                the input list and dicts are not modified
            same_class_iou — IoU threshold of rule (a) in (0, 1], or None (default 0.4)
            same_class_iomin — intersection-over-min-area threshold of rule (b)
                in (0, 1], or None (default 0.85)
            cross_class_iou — IoU threshold of rule (c) in (0, 1], or None (default 0.5)
    Output: (kept, removed) —
                kept: the input dicts that survive (same objects, not copies),
                    in input order, with their original ids;
                removed: deep copies of the removed dicts, in input order, each
                    annotated with 'removed_by' (id of K), 'dedup_rule' (rule
                    name) and 'overlap' (the IoU or intersection-over-min value
                    that triggered the rule, as float).
            Raises ValueError on an invalid threshold, a detection without the
            required keys or with a malformed bbox, or duplicate ids.
    """
    _check_threshold('same_class_iou', same_class_iou)
    _check_threshold('same_class_iomin', same_class_iomin)
    _check_threshold('cross_class_iou', cross_class_iou)
    _check_detections(detections)

    # Greedy visiting order: descending confidence, ties broken by lower id.
    order = sorted(range(len(detections)),
                   key=lambda i: (-float(detections[i]['confidence']), detections[i]['id']))

    kept_in_order: List[Dict] = []
    removal: Dict[int, Tuple[object, str, float]] = {}  # input index -> (removed_by, rule, overlap)
    for i in order:
        candidate = detections[i]
        for k in kept_in_order:
            match = _match_rule(candidate, k, same_class_iou, same_class_iomin, cross_class_iou)
            if match is not None:
                removal[i] = (k['id'], match[0], float(match[1]))
                break
        else:
            kept_in_order.append(candidate)

    kept: List[Dict] = []
    removed: List[Dict] = []
    for i, d in enumerate(detections):
        if i in removal:
            removed_by, rule, overlap = removal[i]
            item = copy.deepcopy(d)
            item['removed_by'] = removed_by
            item['dedup_rule'] = rule
            item['overlap'] = overlap
            removed.append(item)
        else:
            kept.append(d)
    return kept, removed


def validate_class_conf(class_conf: Optional[Dict[str, float]],
                        known_classes: Optional[Iterable[str]] = DETECTOR_CLASSES) -> None:
    """Checks a per-class confidence mapping, so that a misspelt class name cannot be silently ignored.

    Input:  class_conf — mapping class name -> minimum confidence, or None
            known_classes — the detector's class names (default DETECTOR_CLASSES);
                None skips the name check (e.g. for a detector with other classes)
    Output: none; raises ValueError if class_conf is not a mapping, names a class
            outside known_classes (the message suggests the closest known name),
            or has a threshold outside [0, 1]
    """
    if class_conf is None:
        return
    if not isinstance(class_conf, dict):
        raise ValueError(f"class_conf must be a mapping class name -> confidence, got {class_conf!r}")
    known = None if known_classes is None else [str(c) for c in known_classes]
    for name, value in class_conf.items():
        if known is not None and name not in known:
            close = difflib.get_close_matches(str(name), known, n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            raise ValueError(f"class_conf names unknown class {name!r} (detector classes: "
                             f"{', '.join(known)}){hint}")
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"class_conf[{name!r}] must be a number in [0, 1], got {value!r}")
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"class_conf[{name!r}] must be in [0, 1], got {value!r}")


def apply_class_conf(detections: List[Dict],
                     class_conf: Optional[Dict[str, float]] = None,
                     known_classes: Optional[Iterable[str]] = DETECTOR_CLASSES) -> List[Dict]:
    """Applies per-class minimum confidence thresholds.

    A detection is kept only if confidence >= class_conf.get(class_name, 0),
    i.e. classes without an entry are not filtered. This filter is meant to run
    BEFORE deduplicate_detections(), so that a detection rejected by its class
    threshold cannot remove another detection. No per-class thresholds are set
    by default; any values used must be chosen on the validation split.

    Input:  detections — list of detection dicts with 'class_name' and 'confidence'
            class_conf — mapping class name -> minimum confidence in [0, 1], or None
                (None or an empty mapping keeps every detection)
            known_classes — class names class_conf may name (default
                DETECTOR_CLASSES; None = no name check), see validate_class_conf
    Output: new list with the kept input dicts (same objects), in input order.
            Raises ValueError for an unknown class name or a threshold outside [0, 1].
    """
    validate_class_conf(class_conf, known_classes)
    if not class_conf:
        return list(detections)
    return [d for d in detections
            if float(d['confidence']) >= float(class_conf.get(d['class_name'], 0.0))]
