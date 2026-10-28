# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Tests for detection post-processing and the detection stage.

Sections:
  1. Box overlap helpers (iou, intersection_over_min).
  2. deduplicate_detections: one test per rule, the nesting exception for
     ImageView, ordering/determinism, input validation, edge cases.
  3. apply_class_conf.
  4. ComponentDetector.detect / draw_detections with a fake model (needs cv2,
     no torch/ultralytics).
  5. Regression against the v1.1.0 reference detections with the released
     weights (skipped when ultralytics or the weights are not available).

The module itself imports only the pure post-processing code; anything that
needs OpenCV or ultralytics is imported inside the test via pytest.importorskip.
"""

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from _reference_data import require_reference_data
from ui2compose.infer.postprocess import (
    NON_NESTING_CLASSES,
    RULE_CROSS_CLASS_IOU,
    RULE_SAME_CLASS_IOU,
    RULE_SAME_CLASS_NESTED,
    apply_class_conf,
    deduplicate_detections,
    intersection_over_min,
    iou,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def det(det_id: int, class_name: str, confidence: float, bbox: List[float]) -> Dict:
    """Builds a minimal detection dictionary in the detector's output format.

    Input:  det_id — detection id; class_name — class label; confidence — score;
            bbox — [x1, y1, x2, y2] in pixels
    Output: detection dictionary with the keys used by the post-processing code
    """
    return {'id': det_id, 'class_name': class_name, 'confidence': confidence,
            'bbox': [float(v) for v in bbox]}


def ids(detections: List[Dict]) -> List[int]:
    """Returns the ids of a detection list in order.

    Input:  detections — list of detection dictionaries
    Output: list of 'id' values
    """
    return [d['id'] for d in detections]


# ---------------------------------------------------------------------------
# 1. Overlap helpers
# ---------------------------------------------------------------------------

def test_iou_identical_disjoint_and_partial():
    a = [0, 0, 10, 10]
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, [20, 20, 30, 30]) == 0.0
    # Overlap 5x10 = 50, union 100 + 100 - 50 = 150
    assert iou(a, [5, 0, 15, 10]) == pytest.approx(50 / 150)
    # Touching edges have zero intersection
    assert iou(a, [10, 0, 20, 10]) == 0.0


def test_intersection_over_min_nested_box_is_one():
    outer = [0, 0, 100, 100]
    inner = [10, 10, 20, 20]
    assert intersection_over_min(outer, inner) == pytest.approx(1.0)
    assert intersection_over_min(inner, outer) == pytest.approx(1.0)
    assert iou(outer, inner) == pytest.approx(100 / 10000)


def test_overlap_helpers_zero_area_boxes_do_not_crash():
    point = [5, 5, 5, 5]
    line = [0, 5, 10, 5]
    box = [0, 0, 10, 10]
    assert iou(point, point) == 0.0
    assert iou(point, box) == 0.0
    assert iou(line, box) == 0.0
    assert intersection_over_min(point, box) == 0.0
    assert intersection_over_min(point, point) == 0.0
    # Inverted coordinates are treated as an empty box, not a negative area
    assert iou([10, 10, 0, 0], box) == 0.0
    assert intersection_over_min([10, 10, 0, 0], box) == 0.0


# ---------------------------------------------------------------------------
# 2. deduplicate_detections
# ---------------------------------------------------------------------------

def test_non_nesting_classes_match_specification():
    assert NON_NESTING_CLASSES == frozenset({
        'TextView', 'Button', 'EditText', 'CheckBox', 'RadioButton',
        'Switch', 'SeekBar', 'ProgressBar', 'Spinner'})
    for container_like in ('ImageView', 'ListView', 'RecyclerView', 'WebView'):
        assert container_like not in NON_NESTING_CLASSES


def test_rule_same_class_iou_removes_lower_confidence_duplicate():
    # Two ImageView boxes with IoU = 81/119 ~ 0.68 (>= 0.5); ImageView may nest,
    # so only the IoU rule can fire here.
    high = det(0, 'ImageView', 0.38, [0, 0, 10, 10])
    low = det(1, 'ImageView', 0.31, [1, 1, 11, 11])
    kept, removed = deduplicate_detections([high, low])
    assert ids(kept) == [0]
    assert len(removed) == 1
    r = removed[0]
    assert r['id'] == 1
    assert r['removed_by'] == 0
    assert r['dedup_rule'] == RULE_SAME_CLASS_IOU == 'same_class_iou'
    assert r['overlap'] == pytest.approx(81 / 119)


def test_rule_same_class_iou_threshold_is_inclusive():
    # IoU exactly 0.5: [0,0,10,10] vs [0,0,10,5] -> 50 / 100
    a = det(0, 'ImageView', 0.9, [0, 0, 10, 10])
    b = det(1, 'ImageView', 0.8, [0, 0, 10, 5])
    kept, removed = deduplicate_detections([a, b], same_class_iou=0.5,
                                           same_class_iomin=None)
    assert ids(kept) == [0]
    assert removed[0]['dedup_rule'] == RULE_SAME_CLASS_IOU
    # Just above the measured IoU the pair is kept.
    kept, removed = deduplicate_detections([a, b], same_class_iou=0.51,
                                           same_class_iomin=None)
    assert ids(kept) == [0, 1]
    assert removed == []


def test_rule_same_class_nested_removes_wide_textview_around_tight_one():
    # Shazam-like case: a wide low-confidence TextView encloses a tight
    # higher-confidence TextView over the same text line. IoU is low (~0.36),
    # but intersection / min(area) ~ 1.
    tight = det(13, 'TextView', 0.55, [390, 1254, 807, 1306])
    wide = det(14, 'TextView', 0.41, [33, 1255, 1171, 1308])
    assert iou(tight['bbox'], wide['bbox']) < 0.5
    kept, removed = deduplicate_detections([tight, wide])
    assert ids(kept) == [13]
    assert removed[0]['id'] == 14
    assert removed[0]['removed_by'] == 13
    assert removed[0]['dedup_rule'] == RULE_SAME_CLASS_NESTED == 'same_class_nested'
    assert removed[0]['overlap'] == pytest.approx(
        intersection_over_min(tight['bbox'], wide['bbox']))
    assert removed[0]['overlap'] >= 0.9


def test_rule_same_class_nested_applies_to_every_non_nesting_class():
    for cls in sorted(NON_NESTING_CLASSES):
        outer = det(0, cls, 0.9, [0, 0, 100, 100])
        inner = det(1, cls, 0.5, [10, 10, 30, 30])
        kept, removed = deduplicate_detections([outer, inner])
        assert ids(kept) == [0], cls
        assert removed[0]['dedup_rule'] == RULE_SAME_CLASS_NESTED, cls


def test_nested_imageview_avatar_in_banner_is_kept():
    # An avatar ImageView inside a banner ImageView is a legitimate nesting:
    # ImageView is not in NON_NESTING_CLASSES and the IoU is small.
    banner = det(0, 'ImageView', 0.9, [0, 0, 1200, 400])
    avatar = det(1, 'ImageView', 0.6, [50, 200, 250, 380])
    kept, removed = deduplicate_detections([banner, avatar])
    assert ids(kept) == [0, 1]
    assert removed == []


def test_nested_containers_of_same_class_are_kept():
    for cls in ('ImageView', 'ListView', 'RecyclerView', 'WebView'):
        outer = det(0, cls, 0.9, [0, 0, 1000, 1000])
        inner = det(1, cls, 0.8, [100, 100, 400, 400])
        kept, removed = deduplicate_detections([outer, inner])
        assert ids(kept) == [0, 1], cls
        assert removed == [], cls


def test_rule_cross_class_iou_removes_competing_class_hypothesis():
    # Shazam-like: a TextView box almost coinciding with a RecyclerView box.
    rv = det(17, 'RecyclerView', 0.74, [12, 1470, 1189, 1920])
    tv = det(16, 'TextView', 0.41, [20, 1450, 1182, 1758])
    overlap = iou(rv['bbox'], tv['bbox'])
    assert overlap >= 0.6
    kept, removed = deduplicate_detections([tv, rv])
    assert ids(kept) == [17]
    assert removed[0]['id'] == 16
    assert removed[0]['removed_by'] == 17
    assert removed[0]['dedup_rule'] == RULE_CROSS_CLASS_IOU == 'cross_class_iou'
    assert removed[0]['overlap'] == pytest.approx(overlap)


def test_rule_cross_class_iou_threshold_is_inclusive():
    # IoU exactly 0.6 between different classes: [0,0,10,10] vs [0,0,10,6] -> 60 / 100
    img = det(0, 'ImageView', 0.9, [0, 0, 10, 10])
    txt = det(1, 'TextView', 0.8, [0, 0, 10, 6])
    assert iou(img['bbox'], txt['bbox']) == 0.6
    kept, removed = deduplicate_detections([img, txt], cross_class_iou=0.6)
    assert ids(kept) == [0]
    assert removed[0]['dedup_rule'] == RULE_CROSS_CLASS_IOU and removed[0]['overlap'] == 0.6
    # Just above the measured IoU the pair is kept.
    kept, removed = deduplicate_detections([img, txt], cross_class_iou=0.61)
    assert ids(kept) == [0, 1] and removed == []


def test_cross_class_nesting_is_not_removed():
    # A Button inside a RecyclerView (small IoU) and a TextView fully inside a
    # Button (IoU < 0.6) are both legitimate and must survive.
    rv = det(0, 'RecyclerView', 0.74, [0, 0, 1000, 1000])
    button = det(1, 'Button', 0.46, [0, 800, 200, 900])
    btn2 = det(2, 'Button', 0.9, [0, 0, 300, 100])
    label = det(3, 'TextView', 0.8, [100, 30, 200, 70])
    kept, removed = deduplicate_detections([rv, button, btn2, label])
    assert ids(kept) == [0, 1, 2, 3]
    assert removed == []


def test_lower_confidence_detection_cannot_remove_higher_one():
    low = det(0, 'Button', 0.2, [0, 0, 10, 10])
    high = det(1, 'Button', 0.9, [0, 0, 10, 10])
    kept, removed = deduplicate_detections([low, high])
    assert ids(kept) == [1]
    assert removed[0]['id'] == 0
    assert removed[0]['removed_by'] == 1


def test_removed_detection_cannot_remove_others():
    # A (0.9) removes B (0.8, IoU 80/120 = 0.67). C (0.7) has IoU 70/130 = 0.54
    # with B but only 50/150 = 0.33 with A. Greedy suppression compares against
    # kept detections only, so C survives. D (Button) coincides with A.
    a = det(0, 'ImageView', 0.9, [0, 0, 10, 10])
    b = det(1, 'ImageView', 0.8, [2, 0, 12, 10])
    c = det(2, 'ImageView', 0.7, [5, 0, 15, 10])
    d = det(3, 'Button', 0.6, [0, 0, 10, 10])
    assert iou(b['bbox'], c['bbox']) >= 0.5 > iou(a['bbox'], c['bbox'])
    kept, removed = deduplicate_detections([a, b, c, d])
    assert ids(kept) == [0, 2]
    assert ids(removed) == [1, 3]
    assert all(r['removed_by'] == 0 for r in removed)
    assert removed[1]['dedup_rule'] == RULE_CROSS_CLASS_IOU


def test_kept_preserves_input_order_and_original_ids():
    dets = [
        det(7, 'TextView', 0.40, [0, 300, 100, 330]),
        det(3, 'ImageView', 0.95, [0, 0, 100, 100]),
        det(5, 'ImageView', 0.30, [2, 2, 100, 100]),   # duplicate of id 3
        det(9, 'Button', 0.80, [0, 400, 100, 450]),
    ]
    kept, removed = deduplicate_detections(dets)
    assert ids(kept) == [7, 3, 9]
    assert ids(removed) == [5]
    assert removed[0]['removed_by'] == 3


def test_removed_lists_in_input_order():
    dets = [
        det(0, 'TextView', 0.30, [0, 0, 100, 20]),
        det(1, 'TextView', 0.95, [0, 0, 100, 20]),
        det(2, 'TextView', 0.40, [0, 0, 100, 20]),
    ]
    kept, removed = deduplicate_detections(dets)
    assert ids(kept) == [1]
    assert ids(removed) == [0, 2]


def test_confidence_tie_is_broken_by_lower_id():
    a = det(4, 'Button', 0.5, [0, 0, 10, 10])
    b = det(2, 'Button', 0.5, [0, 0, 10, 10])
    for order in ([a, b], [b, a]):
        kept, removed = deduplicate_detections(order)
        assert ids(kept) == [2]
        assert removed[0]['id'] == 4
        assert removed[0]['removed_by'] == 2


def test_deterministic_under_input_permutation():
    dets = [
        det(0, 'ImageView', 0.375, [70, 482, 954, 1087]),
        det(1, 'ImageView', 0.307, [322, 538, 868, 1107]),
        det(2, 'TextView', 0.551, [390, 1254, 807, 1306]),
        det(3, 'TextView', 0.414, [33, 1255, 1171, 1308]),
        det(4, 'TextView', 0.411, [20, 1450, 1182, 1758]),
        det(5, 'RecyclerView', 0.744, [12, 1470, 1189, 1920]),
        det(6, 'Button', 0.462, [6, 1731, 205, 1827]),
    ]
    reference_kept, reference_removed = deduplicate_detections(dets)
    for perm in ([6, 5, 4, 3, 2, 1, 0], [3, 0, 6, 1, 5, 2, 4]):
        shuffled = [dets[i] for i in perm]
        kept, removed = deduplicate_detections(shuffled)
        assert sorted(ids(kept)) == sorted(ids(reference_kept))
        assert sorted((r['id'], r['removed_by'], r['dedup_rule']) for r in removed) == \
            sorted((r['id'], r['removed_by'], r['dedup_rule']) for r in reference_removed)
    assert sorted(ids(reference_kept)) == [0, 2, 5, 6]
    assert {r['id']: r['dedup_rule'] for r in reference_removed} == {
        1: RULE_SAME_CLASS_IOU, 3: RULE_SAME_CLASS_NESTED, 4: RULE_CROSS_CLASS_IOU}


def test_inputs_are_not_modified_and_removed_items_are_copies():
    dets = [det(0, 'Button', 0.9, [0, 0, 10, 10]),
            det(1, 'Button', 0.8, [0, 0, 10, 10])]
    snapshot = copy.deepcopy(dets)
    kept, removed = deduplicate_detections(dets)
    assert dets == snapshot
    assert kept[0] is dets[0]
    assert removed[0] is not dets[1]
    assert 'dedup_rule' not in dets[1]
    removed[0]['bbox'][0] = -1.0
    assert dets[1]['bbox'][0] == 0.0


def test_extra_keys_are_preserved_on_removed_copies():
    a = det(0, 'Button', 0.9, [0, 0, 10, 10])
    b = dict(det(1, 'Button', 0.8, [0, 0, 10, 10]), img_w=1200, img_h=1920)
    _, removed = deduplicate_detections([a, b])
    assert removed[0]['img_w'] == 1200 and removed[0]['img_h'] == 1920


def test_first_rule_order_same_class_iou_before_nested():
    # Both rule (a) and rule (b) hold for identical TextView boxes;
    # the reported rule is the first one in the documented order.
    a = det(0, 'TextView', 0.9, [0, 0, 10, 10])
    b = det(1, 'TextView', 0.8, [0, 0, 10, 10])
    _, removed = deduplicate_detections([a, b])
    assert removed[0]['dedup_rule'] == RULE_SAME_CLASS_IOU


def test_suppressor_is_highest_confidence_triggering_detection():
    # Both kept detections 0 and 1 would remove detection 2; the reported
    # suppressor is the one kept first (higher confidence).
    a = det(0, 'ImageView', 0.95, [0, 0, 100, 100])
    b = det(1, 'Button', 0.90, [200, 0, 300, 100])
    c = det(2, 'TextView', 0.30, [0, 0, 100, 100])
    kept, removed = deduplicate_detections([b, c, a])
    assert ids(kept) == [1, 0]
    assert removed[0]['removed_by'] == 0
    c2 = det(2, 'TextView', 0.30, [0, 0, 300, 100])  # overlaps a and b equally
    kept, removed = deduplicate_detections([b, c2, a], cross_class_iou=0.3)
    assert removed[0]['removed_by'] == 0


def test_disabling_rules_with_none():
    tv_tight = det(0, 'TextView', 0.9, [100, 0, 200, 20])
    tv_wide = det(1, 'TextView', 0.5, [0, 0, 1000, 20])
    img = det(2, 'ImageView', 0.9, [0, 100, 100, 200])
    btn = det(3, 'Button', 0.5, [0, 100, 100, 200])
    dets = [tv_tight, tv_wide, img, btn]
    kept, _ = deduplicate_detections(dets, same_class_iomin=None)
    assert ids(kept) == [0, 1, 2]
    kept, _ = deduplicate_detections(dets, cross_class_iou=None)
    assert ids(kept) == [0, 2, 3]
    kept, _ = deduplicate_detections(dets, same_class_iou=None,
                                     same_class_iomin=None, cross_class_iou=None)
    assert ids(kept) == [0, 1, 2, 3]


@pytest.mark.parametrize('kwargs', [
    {'same_class_iou': 0.0}, {'same_class_iou': 1.5},
    {'same_class_iomin': -0.1}, {'cross_class_iou': 0.0},
])
def test_invalid_thresholds_raise(kwargs):
    with pytest.raises(ValueError):
        deduplicate_detections([], **kwargs)


def test_missing_keys_and_duplicate_ids_raise():
    with pytest.raises(ValueError):
        deduplicate_detections([{'id': 0, 'class_name': 'Button', 'bbox': [0, 0, 1, 1]}])
    with pytest.raises(ValueError):
        deduplicate_detections([det(0, 'Button', 0.5, [0, 0, 1, 1]),
                                det(0, 'TextView', 0.5, [5, 5, 6, 6])])
    with pytest.raises(ValueError):
        deduplicate_detections([det(0, 'Button', 0.5, [0, 0, 1])])


def test_empty_input():
    kept, removed = deduplicate_detections([])
    assert kept == [] and removed == []


def test_zero_area_detections_are_neither_removed_nor_removing():
    point = det(0, 'Button', 0.9, [5, 5, 5, 5])
    point_dup = det(1, 'Button', 0.8, [5, 5, 5, 5])
    box = det(2, 'Button', 0.7, [0, 0, 10, 10])
    kept, removed = deduplicate_detections([point, point_dup, box])
    assert ids(kept) == [0, 1, 2]
    assert removed == []


def test_overlap_values_are_plain_floats():
    a = det(0, 'Button', 0.9, [0, 0, 10, 10])
    b = det(1, 'Button', 0.8, [0, 0, 10, 10])
    _, removed = deduplicate_detections([a, b])
    assert type(removed[0]['overlap']) is float
    json.dumps(removed)  # must be JSON-serialisable


# ---------------------------------------------------------------------------
# 3. apply_class_conf
# ---------------------------------------------------------------------------

def test_apply_class_conf_filters_per_class():
    dets = [det(0, 'Button', 0.46, [0, 0, 1, 1]),
            det(1, 'Button', 0.80, [0, 0, 1, 1]),
            det(2, 'TextView', 0.31, [0, 0, 1, 1]),
            det(3, 'ImageView', 0.50, [0, 0, 1, 1])]
    kept = apply_class_conf(dets, {'Button': 0.5, 'TextView': 0.31})
    # Threshold is inclusive; classes without an entry use 0 (always kept).
    assert ids(kept) == [1, 2, 3]
    assert kept[0] is dets[1]


def test_apply_class_conf_none_or_empty_keeps_everything():
    dets = [det(0, 'Button', 0.1, [0, 0, 1, 1])]
    for mapping in (None, {}):
        kept = apply_class_conf(dets, mapping)
        assert ids(kept) == [0]
        assert kept is not dets  # a new list is returned
    assert apply_class_conf([], {'Button': 0.9}) == []


def test_apply_class_conf_rejects_out_of_range_thresholds():
    with pytest.raises(ValueError):
        apply_class_conf([], {'Button': 1.2})
    with pytest.raises(ValueError):
        apply_class_conf([], {'Button': -0.1})


def test_apply_class_conf_rejects_unknown_class_names():
    # A misspelt class name would otherwise filter nothing without any warning.
    dets = [det(0, 'RecyclerView', 0.74, [0, 0, 1, 1])]
    with pytest.raises(ValueError, match=r"unknown class 'Recyclerview'.*did you mean 'RecyclerView'"):
        apply_class_conf(dets, {'Recyclerview': 0.9})
    with pytest.raises(ValueError, match="unknown class 'Card'"):
        apply_class_conf(dets, {'Card': 0.5})
    with pytest.raises(ValueError, match='must be a mapping'):
        apply_class_conf(dets, [('RecyclerView', 0.9)])
    with pytest.raises(ValueError, match='must be a number'):
        apply_class_conf(dets, {'RecyclerView': 'high'})
    # A detector with other classes: the name check can be replaced or switched off.
    assert apply_class_conf(dets, {'Card': 0.5}, known_classes=None) == dets
    assert apply_class_conf([det(1, 'Card', 0.4, [0, 0, 1, 1])], {'Card': 0.5},
                            known_classes=['Card']) == []


def test_detector_classes_equal_the_dataset_yaml_names():
    yaml = pytest.importorskip('yaml')
    from ui2compose.infer.postprocess import DETECTOR_CLASSES
    names = yaml.safe_load((REPO_ROOT / 'configs' / 'dataset.yaml').read_text(encoding='utf-8'))['names']
    assert list(DETECTOR_CLASSES) == list(names)


# ---------------------------------------------------------------------------
# 4. Detection stage with a fake model (OpenCV only, no torch/ultralytics)
# ---------------------------------------------------------------------------

class _FakeArray:
    """Minimal stand-in for a torch tensor row: supports [0], .cpu(), .numpy()."""

    def __init__(self, values):
        self._values = values

    def __getitem__(self, idx):
        value = self._values[idx]
        return _FakeArray(value) if isinstance(value, list) else value

    def cpu(self):
        return self

    def numpy(self):
        import numpy as np
        return np.asarray(self._values, dtype=float)


class _FakeBox:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = _FakeArray([list(xyxy)])
        self.cls = [float(cls)]
        self.conf = [float(conf)]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    """Records predict() keyword arguments and returns fixed boxes."""

    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [_FakeResult(self.boxes)]


def _fake_detector(boxes, class_names):
    """Builds a ComponentDetector around a fake model without loading weights.

    Input:  boxes — list of _FakeBox; class_names — list of class names
    Output: ComponentDetector instance whose model is a _FakeModel
    """
    from ui2compose.infer.detect_components import ComponentDetector
    detector = ComponentDetector.__new__(ComponentDetector)
    detector.model = _FakeModel(boxes)
    detector.class_names = list(class_names)
    return detector


def _write_blank_image(path: Path, width: int, height: int) -> str:
    cv2 = pytest.importorskip('cv2')
    np = pytest.importorskip('numpy')
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(path), img)
    return str(path)


def test_detect_passes_all_parameters_and_adds_image_size(tmp_path):
    image = _write_blank_image(tmp_path / 'screen.png', 300, 500)
    boxes = [_FakeBox([10, 200, 110, 260], 1, 0.9),
             _FakeBox([20, 20, 120, 60], 0, 0.5)]
    detector = _fake_detector(boxes, ['Button', 'TextView'])
    out = detector.detect(image, conf_threshold=0.25, iou_threshold=0.45,
                          imgsz=1024, agnostic_nms=True, max_det=50)
    call = detector.model.calls[0]
    assert call['source'] == image
    assert call['conf'] == 0.25
    assert call['iou'] == 0.45
    assert call['imgsz'] == 1024
    assert call['agnostic_nms'] is True
    assert call['max_det'] == 50
    # Sorted top to bottom and re-numbered; v1.1.0 keys plus image size.
    assert [d['class_name'] for d in out] == ['Button', 'TextView']
    assert [d['id'] for d in out] == [0, 1]
    assert set(out[0].keys()) == {'id', 'class_name', 'class_id', 'confidence',
                                  'bbox', 'bbox_norm', 'center', 'width',
                                  'height', 'img_w', 'img_h'}
    assert out[0]['img_w'] == 300 and out[0]['img_h'] == 500
    assert out[0]['bbox'] == [20.0, 20.0, 120.0, 60.0]
    assert out[0]['bbox_norm'] == pytest.approx([70 / 300, 40 / 500, 100 / 300, 40 / 500])


def test_detect_defaults_match_v110_behaviour(tmp_path):
    image = _write_blank_image(tmp_path / 'screen.png', 64, 64)
    detector = _fake_detector([], ['Button'])
    assert detector.detect(image) == []
    call = detector.model.calls[0]
    # v1.1.0 passed only conf=0.3; the other values are the ultralytics
    # predict defaults it fell back to.
    assert (call['conf'], call['iou'], call['agnostic_nms'],
            call['max_det']) == (0.3, 0.7, False, 300)
    # imgsz is not passed by default, so Ultralytics uses the checkpoint's own
    # training size (as v1.1.0 did), whatever size the checkpoint was trained at.
    assert 'imgsz' not in call


def test_effective_imgsz_is_the_argument_or_the_checkpoint_value():
    pytest.importorskip('cv2')
    detector = _fake_detector([], ['Button'])
    assert detector.effective_imgsz() is None                # fake model records no imgsz
    detector.model.overrides = {'imgsz': 416, 'task': 'detect'}
    assert detector.effective_imgsz() == 416                 # checkpoint value
    assert detector.effective_imgsz(1024) == 1024            # explicit argument wins


def test_format_label_with_and_without_id():
    pytest.importorskip('cv2')
    from ui2compose.infer.detect_components import format_label
    d = det(5, 'TextView', 0.714, [0, 0, 1, 1])
    assert format_label(d) == 'TextView 0.71'
    assert format_label(d, show_id=True) == '#5 TextView 0.71'
    assert format_label(d, removed=True, show_id=True) == '#5 TextView 0.71 removed'


def test_detect_missing_image_raises(tmp_path):
    pytest.importorskip('cv2')
    detector = _fake_detector([], ['Button'])
    with pytest.raises(FileNotFoundError):
        detector.detect(str(tmp_path / 'missing.png'))


def test_draw_detections_returns_annotated_copy():
    np = pytest.importorskip('numpy')
    pytest.importorskip('cv2')
    from ui2compose.infer.detect_components import draw_detections
    img = np.full((1920, 1200, 3), 255, dtype=np.uint8)
    dets = [det(0, 'TextView', 0.77, [100, 300, 600, 360]),
            det(1, 'ImageView', 0.5, [100, 400, 600, 900])]
    removed = [dict(det(2, 'ImageView', 0.31, [120, 420, 620, 920]),
                    removed_by=1, dedup_rule='same_class_iou', overlap=0.8)]
    out = draw_detections(img, dets, removed=removed)
    assert out.shape == img.shape
    assert (img == 255).all()  # the input image is not modified
    assert (out != 255).any()
    # Unknown (legacy) class names do not crash and get a stable colour.
    out2 = draw_detections(img, [det(0, 'Card', 0.6, [10, 10, 200, 200])])
    out3 = draw_detections(img, [det(0, 'Card', 0.6, [10, 10, 200, 200])])
    assert (out2 == out3).all()


def test_visualize_saves_file_and_draws_removed(tmp_path):
    cv2 = pytest.importorskip('cv2')
    image = _write_blank_image(tmp_path / 'screen.png', 1200, 1920)
    detector = _fake_detector([], ['Button'])
    dets = [det(0, 'Button', 0.77, [100, 100, 500, 200])]
    removed = [dict(det(1, 'Button', 0.40, [110, 105, 510, 205]),
                    removed_by=0, dedup_rule='same_class_iou', overlap=0.9)]
    out_path = tmp_path / 'viz.png'
    with_removed = detector.visualize(image, dets, str(out_path), removed=removed)
    without_removed = detector.visualize(image, dets)
    assert out_path.exists()
    assert (cv2.imread(str(out_path)) == with_removed).all()
    assert (with_removed != without_removed).any()
    # Labels carry the detection ids by default ("#0 Button 0.77"); without ids they differ.
    without_ids = detector.visualize(image, dets, removed=removed, show_ids=False)
    assert (with_removed != without_ids).any()
    with pytest.raises(FileNotFoundError):
        detector.visualize(str(tmp_path / 'missing.png'), dets)


def test_label_style_scales_with_image_width():
    pytest.importorskip('cv2')
    from ui2compose.infer.detect_components import label_style
    style_1200 = label_style(1200)
    assert style_1200['font_scale'] == pytest.approx(0.9)
    assert style_1200['text_thickness'] >= 2
    assert style_1200['box_thickness'] >= 2
    assert label_style(2400)['font_scale'] == pytest.approx(1.8)
    assert label_style(100)['text_thickness'] >= 2


def test_class_colours_are_distinct_and_text_contrasts():
    pytest.importorskip('cv2')
    from ui2compose.infer.detect_components import (
        CLASS_COLORS_BGR, REMOVED_COLOR_BGR, class_color, text_color_for)
    colours = list(CLASS_COLORS_BGR.values())
    assert len(set(colours)) == len(colours) == 13
    assert REMOVED_COLOR_BGR not in colours
    assert class_color('Button') == CLASS_COLORS_BGR['Button']
    assert class_color('Card') == class_color('Card')
    assert text_color_for((255, 255, 255)) == (0, 0, 0)
    assert text_color_for((0, 0, 0)) == (255, 255, 255)


def test_labels_of_boxes_with_same_corner_do_not_overlap():
    pytest.importorskip('cv2')
    from ui2compose.infer.detect_components import layout_labels
    dets = [det(0, 'RecyclerView', 0.74, [12, 1470, 1189, 1920]),
            det(1, 'TextView', 0.41, [12, 1470, 1182, 1758]),
            det(2, 'Button', 0.46, [12, 1471, 205, 1827])]
    rects = layout_labels(dets, img_w=1200, img_h=1920)
    assert len(rects) == 3
    for i in range(3):
        x1, y1, x2, y2 = rects[i]
        assert 0 <= x1 < x2 <= 1200 and 0 <= y1 < y2 <= 1920
        for j in range(i + 1, 3):
            assert iou(rects[i], rects[j]) == 0.0


# ---------------------------------------------------------------------------
# 5. Regression against the v1.1.0 reference outputs (real model)
# ---------------------------------------------------------------------------

WEIGHTS = REPO_ROOT / 'runs' / 'oversample_5k' / 'weights' / 'best.pt'
REFERENCE_DIR = REPO_ROOT / 'results' / 'v1.1.0_examples'
SHAZAM = 'com.shazam.android-screens_screenshot_3'


def test_detect_reproduces_v110_reference_detections():
    pytest.importorskip('ultralytics')
    reference_file = REFERENCE_DIR / SHAZAM / '1_detections.json'
    image = REPO_ROOT / 'examples' / f'{SHAZAM}.png'
    require_reference_data(WEIGHTS, reference_file, image)
    from ui2compose.infer.detect_components import ComponentDetector
    detector = ComponentDetector(str(WEIGHTS), str(REPO_ROOT / 'configs' / 'dataset.yaml'))
    new = detector.detect(str(image))
    reference = json.loads(reference_file.read_text())
    assert len(new) == len(reference)
    for n, r in zip(new, reference):
        assert n['img_w'] == 1200 and n['img_h'] == 1920
        assert (n['id'], n['class_name'], n['class_id']) == \
            (r['id'], r['class_name'], r['class_id'])
        assert n['confidence'] == pytest.approx(r['confidence'], abs=1e-4)
        assert n['bbox'] == pytest.approx(r['bbox'], abs=0.05)
