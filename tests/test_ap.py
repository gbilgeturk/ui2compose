# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Tests for the evaluation tooling: ap.py, evaluate_detector.py, benchmark.py.

Sections:
  1. Pure tests of ap.py (hand-made cases: perfect / empty predictions, class
     without ground truth, background images, operating point, bootstrap
     determinism and pairing, jackknife, percentile / basic / BCa intervals
     and the bootstrap bias).
  2. Equality with Ultralytics (pytest.importorskip): box_iou,
     match_predictions, ap_per_class and the end-to-end DetMetrics result on
     random synthetic data, within 1e-6.
  3. evaluate_detector helpers and CLI with saved predictions (no model).
  4. benchmark helpers (statistics, host info) without running a model.

The module imports only NumPy/PIL code at module level.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from _reference_data import require_reference_data
from ui2compose.evaluation import ap as A
from ui2compose.evaluation import evaluate_detector as E

NAMES = ["Button", "CheckBox", "EditText", "ImageView", "TextView"]


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def synthetic_dataset(seed: int, n_images: int = 25, n_classes: int = 5,
                      max_gt: int = 8, quantize_conf: bool = False
                      ) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Random ground truth and noisy predictions (jittered GT, wrong classes, false positives).

    Input:  seed — RNG seed; n_images, n_classes, max_gt — sizes;
            quantize_conf — round confidences to 2 decimals (creates ties)
    Output: (preds_by_image, gts_by_image) in the ap.evaluate() format
    """
    rng = np.random.default_rng(seed)
    preds, gts = {}, {}
    for i in range(n_images):
        m = int(rng.integers(0, max_gt + 1))
        xy = rng.uniform(0, 900, (m, 2))
        wh = rng.uniform(20, 300, (m, 2))
        gb = np.concatenate([xy, xy + wh], 1)
        gc = rng.integers(0, n_classes - 1, m)  # last class never has ground truth
        boxes, cls, conf = [], [], []
        for b, c in zip(gb, gc):
            if rng.random() < 0.8:
                boxes.append(b + rng.normal(0, 12, 4))  # jittered true positive
                cls.append(c if rng.random() < 0.85 else int(rng.integers(0, n_classes)))
                conf.append(rng.uniform(0.2, 1.0))
            if rng.random() < 0.3:  # duplicate detection of the same object
                boxes.append(b + rng.normal(0, 25, 4))
                cls.append(c)
                conf.append(rng.uniform(0.0, 0.6))
        for _ in range(int(rng.integers(0, 4))):  # false positives
            p = rng.uniform(0, 900, 2)
            boxes.append(np.r_[p, p + rng.uniform(20, 200, 2)])
            cls.append(int(rng.integers(0, n_classes)))
            conf.append(rng.uniform(0.0, 0.7))
        conf_a = np.asarray(conf, np.float32)
        if quantize_conf:
            conf_a = np.round(conf_a, 2).astype(np.float32)
        img = f"img_{i:03d}"
        gts[img] = {"boxes": gb.tolist(), "cls": gc.tolist()}
        if boxes or rng.random() < 0.5:
            preds[img] = {"boxes": np.asarray(boxes, np.float32).reshape(-1, 4).tolist(),
                          "conf": conf_a.tolist(), "cls": cls}
    return preds, gts


# ---------------------------------------------------------------------------
# 1. Pure tests
# ---------------------------------------------------------------------------

def test_iou_thresholds_are_float32_linspace():
    assert A.IOU_THRESHOLDS.dtype == np.float32
    assert len(A.IOU_THRESHOLDS) == 10
    assert A.IOU_THRESHOLDS[0] == np.float32(0.5) and A.IOU_THRESHOLDS[-1] == np.float32(0.95)


def test_box_iou_basic_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    iou = A.box_iou(a, b)
    assert iou.dtype == np.float32 and iou.shape == (1, 3)
    np.testing.assert_allclose(iou[0], [1.0, 50 / 150, 0.0], atol=1e-6)


def test_match_predictions_greedy_by_iou():
    # two predictions of class 0 on one label: only the higher-IoU one is a TP
    gt_boxes = np.array([[0, 0, 100, 100]])
    pred_boxes = np.array([[0, 0, 100, 90], [0, 0, 100, 60]])
    iou = A.box_iou(gt_boxes, pred_boxes)
    tp = A.match_predictions(np.array([0, 0]), np.array([0]), iou)
    assert tp[:, 0].tolist() == [True, False]
    # wrong class never matches
    tp = A.match_predictions(np.array([1, 1]), np.array([0]), iou)
    assert not tp.any()
    # IoU 0.9 is a TP up to the 0.9 threshold only
    tp = A.match_predictions(np.array([0]), np.array([0]), A.box_iou(gt_boxes, pred_boxes[:1]))
    assert tp[0].tolist() == [t <= np.float32(0.9) for t in A.IOU_THRESHOLDS]


def test_perfect_predictions_follow_ultralytics_0995():
    """Perfect predictions: P = R = F1 = 1 and AP = 0.995 (not 1.0).

    Ultralytics' compute_ap appends a sentinel (recall 1, precision 0) and
    integrates the 101-point interpolation with the trapezoid rule, so the
    last interval [0.99, 1.0] contributes 0.005 instead of 0.01. This is the
    value Ultralytics reports for a perfect detector and is reproduced here.
    """
    _, gts = synthetic_dataset(1)
    preds = {k: {"boxes": v["boxes"], "cls": v["cls"], "conf": [0.9] * len(v["cls"])}
             for k, v in gts.items()}
    r = A.evaluate(preds, gts, NAMES)
    assert r["mAP50"] == pytest.approx(0.995, abs=1e-12)
    assert r["mAP50_95"] == pytest.approx(0.995, abs=1e-12)
    assert r["precision"] == pytest.approx(1.0) and r["recall"] == pytest.approx(1.0)
    assert r["f1"] == pytest.approx(1.0)
    op = r["operating_point"]
    assert op["fp"] == 0 and op["fn"] == 0 and op["precision"] == 1.0 and op["recall"] == 1.0


def test_no_predictions_give_zero():
    _, gts = synthetic_dataset(2)
    r = A.evaluate({}, gts, NAMES)
    assert r["mAP50"] == 0.0 and r["mAP50_95"] == 0.0
    assert r["precision"] == 0.0 and r["recall"] == 0.0
    op = r["operating_point"]
    assert op["tp"] == 0 and op["precision"] is None and op["recall"] == 0.0
    assert op["fn"] == r["n_instances"] > 0


def test_no_ground_truth_at_all():
    preds = {"a": {"boxes": [[0, 0, 5, 5]], "conf": [0.9], "cls": [0]}}
    r = A.evaluate(preds, {"a": {"boxes": [], "cls": []}}, NAMES)
    assert r["mAP50"] == 0.0 and r["n_instances"] == 0
    assert all(v["ap50"] is None for v in r["per_class"].values())


def test_class_without_ground_truth_is_excluded_from_the_mean():
    gts = {"a": {"boxes": [[0, 0, 10, 10]], "cls": [0]}}
    good = {"a": {"boxes": [[0, 0, 10, 10]], "conf": [0.9], "cls": [0]}}
    extra = {"a": {"boxes": [[0, 0, 10, 10], [50, 50, 80, 80]], "conf": [0.9, 0.95], "cls": [0, 4]}}
    r1, r2 = A.evaluate(good, gts, NAMES), A.evaluate(extra, gts, NAMES)
    assert r1["mAP50"] == r2["mAP50"] == pytest.approx(0.995)
    assert r2["per_class"]["TextView"]["ap50"] is None
    assert r2["per_class"]["TextView"]["n_pred"] == 1 and r2["per_class"]["TextView"]["support"] == 0
    # ...but the operating point counts it as a false positive
    assert r2["operating_point"]["fp"] == 1


def test_background_image_false_positives_lower_ap():
    gts = {"a": {"boxes": [[0, 0, 10, 10]], "cls": [0]}, "bg": {"boxes": [], "cls": []}}
    p_ok = {"a": {"boxes": [[0, 0, 10, 10]], "conf": [0.5], "cls": [0]}}
    p_bg = dict(p_ok, bg={"boxes": [[0, 0, 10, 10]], "conf": [0.9], "cls": [0]})
    assert A.evaluate(p_bg, gts, NAMES)["mAP50"] < A.evaluate(p_ok, gts, NAMES)["mAP50"]


def test_predictions_for_unknown_image_raise():
    with pytest.raises(ValueError):
        A.evaluate({"zzz": {"boxes": [], "conf": [], "cls": []}}, {"a": {"boxes": [], "cls": []}}, NAMES)


def test_operating_point_uses_strict_threshold_and_rematches():
    gts = {"a": {"boxes": [[0, 0, 100, 100]], "cls": [0]}}
    # the low-confidence box has the higher IoU: over all predictions it would
    # take the label; at conf_op 0.3 only the high-confidence box remains and is a TP
    preds = {"a": {"boxes": [[0, 0, 100, 80], [0, 0, 100, 100]], "conf": [0.9, 0.2], "cls": [0, 0]}}
    op = A.operating_point(preds, gts, NAMES, conf=0.3)
    assert (op["tp"], op["fp"], op["fn"]) == (1, 0, 0)
    op = A.operating_point(preds, gts, NAMES, conf=0.9)  # strict '>' removes conf 0.9
    assert (op["tp"], op["fp"], op["fn"]) == (0, 0, 1)
    op = A.operating_point(preds, gts, NAMES, conf=0.1)
    assert (op["tp"], op["fp"], op["fn"]) == (1, 1, 0)
    assert op["f1"] == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_bootstrap_is_deterministic_and_brackets_the_point():
    preds, gts = synthetic_dataset(3, n_images=30)
    b1 = A.bootstrap_ci(preds, gts, NAMES, n=60, seed=7)
    b2 = A.bootstrap_ci(preds, gts, NAMES, n=60, seed=7)
    b3 = A.bootstrap_ci(preds, gts, NAMES, n=60, seed=8)
    assert json.dumps(b1, sort_keys=True) == json.dumps(b2, sort_keys=True)
    # Another seed gives other intervals (compared without the 'seed' entry itself).
    without_seed = lambda b: json.dumps({k: v for k, v in b.items() if k != "seed"}, sort_keys=True)
    assert without_seed(b1) != without_seed(b3)
    point = A.evaluate(preds, gts, NAMES)
    assert b1["mAP50"]["point"] == pytest.approx(point["mAP50"])
    # A real resampling gives a non-degenerate interval around the point estimate.
    assert b1["mAP50"]["lo"] < b1["mAP50"]["hi"]
    assert b1["mAP50"]["lo"] <= b1["mAP50"]["point"] <= b1["mAP50"]["hi"]
    assert b1["resample_unit"] == "image" and b1["n_units"] == 30
    assert b1["mAP50"]["n_valid"] == 60
    # the class that never has ground truth has no interval
    assert b1["per_class_ap50"]["TextView"]["n_valid"] == 0
    assert b1["per_class_ap50"]["TextView"]["lo"] is None


def _resampled(preds: Dict[str, dict], gts: Dict[str, dict], ids: List[str], idx) -> Tuple[dict, dict]:
    """Builds the evaluation set of one bootstrap replicate by hand (repeated images renamed).

    Input:  preds, gts — full sets; ids — sorted image ids; idx — drawn indices
    Output: (preds, gts) of the replicate, one renamed copy per draw
    """
    p, g = {}, {}
    for k, i in enumerate(idx):
        name = f"{ids[i]}#{k:04d}"
        g[name] = gts[ids[i]]
        if ids[i] in preds:
            p[name] = preds[ids[i]]
    return p, g


def test_bootstrap_replicates_are_resamples_with_replacement():
    """Each replicate equals evaluate() on len(images) images drawn with replacement by default_rng(seed)."""
    preds, gts = synthetic_dataset(11, n_images=20)
    ids = sorted(gts)
    b = A.bootstrap_ci(preds, gts, NAMES, n=25, seed=5, keep_replicates=True)
    rng = np.random.default_rng(5)
    for r in range(25):
        idx = rng.integers(0, len(ids), len(ids))
        p, g = _resampled(preds, gts, ids, idx)
        m = A.evaluate(p, g, NAMES)
        assert b["replicates"]["mAP50"][r] == pytest.approx(m["mAP50"], abs=1e-12)
        assert b["replicates"]["mAP50_95"][r] == pytest.approx(m["mAP50_95"], abs=1e-12)
    assert np.std(b["replicates"]["mAP50"]) > 0                 # the replicates vary


@pytest.mark.parametrize("alpha", [0.05, 0.1])
def test_bootstrap_interval_is_the_stated_percentile_range(alpha):
    """lo / hi are the 100*alpha/2 and 100*(1-alpha/2) percentiles of the replicates (95% for alpha 0.05)."""
    preds, gts = synthetic_dataset(12, n_images=25)
    b = A.bootstrap_ci(preds, gts, NAMES, n=200, seed=3, alpha=alpha, keep_replicates=True)
    for key in ("mAP50", "mAP50_95"):
        lo, hi = np.percentile(b["replicates"][key], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        assert (b[key]["lo"], b[key]["hi"]) == (pytest.approx(lo), pytest.approx(hi))
    with pytest.raises(ValueError, match="alpha"):
        A.bootstrap_ci(preds, gts, NAMES, n=2, alpha=1.5)


def test_bootstrap_by_group_resamples_whole_groups():
    """With groups, a replicate holds all images of each drawn group; singleton groups equal image resampling."""
    preds, gts = synthetic_dataset(13, n_images=12)
    ids = sorted(gts)
    # Singleton groups (labelled by the image id) reproduce the image-level bootstrap exactly.
    per_image = A.bootstrap_ci(preds, gts, NAMES, n=30, seed=2, keep_replicates=True)
    singletons = A.bootstrap_ci(preds, gts, NAMES, n=30, seed=2, keep_replicates=True,
                                groups={i: i for i in ids})
    for key in ("mAP50", "mAP50_95"):
        assert singletons["replicates"][key] == per_image["replicates"][key]
    assert np.array_equal(singletons["replicates"]["per_class_ap50"],
                          per_image["replicates"]["per_class_ap50"], equal_nan=True)
    assert singletons["resample_unit"] == "group" and singletons["n_units"] == 12
    # One group: every replicate is the full set, so the interval collapses onto the point.
    one = A.bootstrap_ci(preds, gts, NAMES, n=20, seed=2, groups={i: "app" for i in ids})
    assert one["mAP50"]["lo"] == one["mAP50"]["hi"] == pytest.approx(one["mAP50"]["point"])
    # Two groups of six: a replicate is AA, AB or BB, so at most three distinct values,
    # each equal to evaluate() on the drawn groups' images.
    groups = {i: ("A" if k < 6 else "B") for k, i in enumerate(ids)}
    two = A.bootstrap_ci(preds, gts, NAMES, n=40, seed=4, keep_replicates=True, groups=groups)
    members = {"A": list(range(6)), "B": list(range(6, 12))}
    rng = np.random.default_rng(4)
    for r in range(40):
        idx = [i for u in rng.integers(0, 2, 2) for i in members["AB"[u]]]
        p, g = _resampled(preds, gts, ids, idx)
        assert two["replicates"]["mAP50"][r] == pytest.approx(A.evaluate(p, g, NAMES)["mAP50"], abs=1e-12)
    assert len({round(v, 12) for v in two["replicates"]["mAP50"]}) <= 3
    with pytest.raises(ValueError, match="no group"):
        A.bootstrap_ci(preds, gts, NAMES, n=2, groups={ids[0]: "A"})


def test_bootstrap_paired_difference():
    preds, gts = synthetic_dataset(4, n_images=30)
    same = A.bootstrap_ci(preds, gts, NAMES, n=40, seed=1, other_preds=preds)
    assert same["diff"]["mAP50"]["lo"] == 0.0 and same["diff"]["mAP50"]["hi"] == 0.0
    assert same["diff"]["mAP50"]["p_diff_le_0"] == 1.0
    worse = {k: {"boxes": v["boxes"], "conf": v["conf"], "cls": [(c + 1) % 4 for c in v["cls"]]}
             for k, v in preds.items()}
    d = A.bootstrap_ci(preds, gts, NAMES, n=40, seed=1, other_preds=worse)["diff"]["mAP50"]
    assert d["point"] < 0 and d["hi"] < 0
    # the paired replicates of the base model equal the unpaired ones (same ids)
    alone = A.bootstrap_ci(preds, gts, NAMES, n=40, seed=1)
    assert alone["mAP50"] == A.bootstrap_ci(preds, gts, NAMES, n=40, seed=1, other_preds=worse)["mAP50"]


def test_kept_per_class_replicates_average_to_the_map_replicates():
    """keep_replicates also returns per-class AP50; their mean over present classes is the mAP50 replicate."""
    preds, gts = synthetic_dataset(11, n_images=20)
    b = A.bootstrap_ci(preds, gts, NAMES, n=15, seed=5, keep_replicates=True)
    pc = np.asarray(b["replicates"]["per_class_ap50"], dtype=float)
    assert pc.shape == (15, len(NAMES))
    assert np.allclose(np.nanmean(pc, axis=1), b["replicates"]["mAP50"])


def test_jackknife_leaves_out_one_unit():
    """Value k is evaluate() without the images of unit k (single images, or groups)."""
    preds, gts = synthetic_dataset(14, n_images=9)
    ids = sorted(gts)
    jk = A.jackknife(preds, gts, NAMES)
    assert jk["n_units"] == 9 and jk["resample_unit"] == "image"
    for k in (0, 4, 8):
        keep = [i for j, i in enumerate(ids) if j != k]
        m = A.evaluate({i: preds[i] for i in keep if i in preds}, {i: gts[i] for i in keep}, NAMES)
        assert jk["mAP50"][k] == pytest.approx(m["mAP50"], abs=1e-12)
        assert jk["mAP50_95"][k] == pytest.approx(m["mAP50_95"], abs=1e-12)
    groups = {i: str(j // 3) for j, i in enumerate(ids)}
    jg = A.jackknife(preds, gts, NAMES, groups)
    assert jg["n_units"] == 3 and jg["per_class_ap50"].shape == (3, len(NAMES))
    keep = ids[3:]
    m = A.evaluate({i: preds[i] for i in keep if i in preds}, {i: gts[i] for i in keep}, NAMES)
    assert jg["mAP50"][0] == pytest.approx(m["mAP50"], abs=1e-12)


def test_interval_methods_percentile_basic_bca_and_bias():
    """Symmetric replicates: all three intervals agree and the bias is 0; shifted ones: basic and BCa move down."""
    p = 0.5
    sym = p + np.linspace(-0.1, 0.1, 1001)
    r = A.interval_methods(sym, p, 0.05, jackknife_values=[-1.0, 0.0, 1.0])
    assert r["bias"] == pytest.approx(0.0, abs=1e-12) and r["share_above"] == pytest.approx(0.5, abs=1e-3)
    assert r["percentile"] == pytest.approx([p - 0.095, p + 0.095])
    assert r["basic"] == pytest.approx(r["percentile"])
    assert r["bca_z0"] == pytest.approx(0.0, abs=1e-12) and r["bca_acceleration"] == 0.0
    assert r["bca"] == pytest.approx(r["percentile"])
    shifted = sym + 0.02
    s = A.interval_methods(shifted, p, 0.05, jackknife_values=[-1.0, 0.0, 1.0])
    assert s["bias"] == pytest.approx(0.02) and s["replicate_mean"] == pytest.approx(p + 0.02)
    lo, hi = s["percentile"]
    assert s["basic"] == pytest.approx([2 * p - hi, 2 * p - lo])
    assert s["bca_z0"] < 0 and s["bca"][0] < lo and s["bca"][1] < hi
    # Skewed jackknife values give a non-zero acceleration.
    a = A.interval_methods(shifted, p, 0.05, jackknife_values=[0.0, 0.0, 0.0, 1.0])
    assert a["bca_acceleration"] != 0.0
    # Without jackknife values, or without a point, no BCa; NaN replicates are skipped.
    assert A.interval_methods(shifted, p)["bca"] is None
    assert A.interval_methods([np.nan, 0.4, 0.6], p)["n_valid"] == 2
    none = A.interval_methods(shifted, None)
    assert none["percentile"] is None and none["n_valid"] == 1001
    # All replicates above the point: z0 is infinite, BCa undefined.
    assert A.interval_methods(p + 1 + sym, p, jackknife_values=[0.0, 1.0])["bca"] is None
    # The percentile endpoints are those of bootstrap_ci.
    preds, gts = synthetic_dataset(12, n_images=25)
    b = A.bootstrap_ci(preds, gts, NAMES, n=100, seed=3, keep_replicates=True)
    iv = A.interval_methods(b["replicates"]["mAP50"], b["mAP50"]["point"])
    assert iv["percentile"] == [b["mAP50"]["lo"], b["mAP50"]["hi"]]


def test_class_ids_outside_names_raise():
    gts = {"a": {"boxes": [[0, 0, 10, 10]], "cls": [len(NAMES)]}}
    with pytest.raises(ValueError, match="class id"):
        A.evaluate({}, gts, NAMES)
    gts = {"a": {"boxes": [[0, 0, 10, 10]], "cls": [0]}}
    with pytest.raises(ValueError, match="class id"):
        A.bootstrap_ci({"a": {"boxes": [[0, 0, 1, 1]], "conf": [0.5], "cls": [-1]}}, gts, NAMES, n=2)


def test_inconsistent_entry_raises():
    with pytest.raises(ValueError):
        A.evaluate({"a": {"boxes": [[0, 0, 1, 1]], "conf": [0.5, 0.4], "cls": [0]}},
                   {"a": {"boxes": [], "cls": []}}, NAMES)


# ---------------------------------------------------------------------------
# 2. Equality with Ultralytics
# ---------------------------------------------------------------------------

def _ultralytics_stats(preds: Dict[str, dict], gts: Dict[str, dict]) -> List[dict]:
    """Per-image statistics computed with Ultralytics' own functions.

    Input:  preds, gts — synthetic dataset
    Output: list of stat dicts as DetectionValidator.update_metrics builds them
    """
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import box_iou as u_box_iou

    fake = SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10))
    out = []
    for img in sorted(gts):
        g, p = gts[img], preds.get(img, {"boxes": [], "conf": [], "cls": []})
        gb = torch.tensor(g["boxes"], dtype=torch.float32).reshape(-1, 4)
        gc = torch.tensor(g["cls"], dtype=torch.float32).reshape(-1)
        pb = torch.tensor(p["boxes"], dtype=torch.float32).reshape(-1, 4)
        pc = torch.tensor(p["cls"], dtype=torch.float32).reshape(-1)
        pconf = torch.tensor(p["conf"], dtype=torch.float32).reshape(-1)
        if gc.shape[0] == 0 or pc.shape[0] == 0:
            tp = np.zeros((pc.shape[0], 10), dtype=bool)
        else:
            tp = BaseValidator.match_predictions(fake, pc, gc, u_box_iou(gb, pb)).cpu().numpy()
        out.append({"tp": tp, "conf": pconf.numpy(), "pred_cls": pc.numpy(),
                    "target_cls": gc.numpy(), "target_img": np.unique(gc.numpy())})
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_box_iou_equals_ultralytics(seed):
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from ultralytics.utils.metrics import box_iou as u_box_iou

    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 500, (40, 2)).astype(np.float32)
    b1 = np.concatenate([xy, xy + rng.uniform(1, 200, (40, 2))], 1).astype(np.float32)
    xy = rng.uniform(0, 500, (30, 2)).astype(np.float32)
    b2 = np.concatenate([xy, xy + rng.uniform(1, 200, (30, 2))], 1).astype(np.float32)
    ours = A.box_iou(b1, b2)
    ref = u_box_iou(torch.from_numpy(b1), torch.from_numpy(b2)).numpy()
    np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_match_predictions_equals_ultralytics(seed):
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from types import SimpleNamespace

    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import box_iou as u_box_iou

    preds, gts = synthetic_dataset(seed)
    fake = SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10))
    for img in sorted(gts):
        if img not in preds or not gts[img]["cls"] or not preds[img]["cls"]:
            continue
        gb = torch.tensor(gts[img]["boxes"], dtype=torch.float32)
        pb = torch.tensor(preds[img]["boxes"], dtype=torch.float32)
        gc = torch.tensor(gts[img]["cls"], dtype=torch.float32)
        pc = torch.tensor(preds[img]["cls"], dtype=torch.float32)
        iou = u_box_iou(gb, pb)
        ref = BaseValidator.match_predictions(fake, pc, gc, iou).cpu().numpy()
        ours = A.match_predictions(pc.numpy(), gc.numpy(), iou.numpy())
        np.testing.assert_array_equal(ours, ref)


@pytest.mark.parametrize("seed,quantize", [(0, False), (1, False), (2, True), (3, True)])
def test_ap_per_class_equals_ultralytics(seed, quantize):
    pytest.importorskip("ultralytics")
    from ultralytics.utils.metrics import ap_per_class as u_ap_per_class

    preds, gts = synthetic_dataset(seed, n_images=40, quantize_conf=quantize)
    st = _ultralytics_stats(preds, gts)
    tp, conf, pc, tc = (np.concatenate([s[k] for s in st], 0)
                        for k in ("tp", "conf", "pred_cls", "target_cls"))
    ref = u_ap_per_class(tp, conf, pc, tc)
    ours = A.ap_per_class(tp, conf, pc, tc)
    keys = ["tp", "fp", "p", "r", "f1", "ap", "unique_classes", "p_curve", "r_curve",
            "f1_curve", "x", "prec_values"]
    for k, r in zip(keys, ref):
        np.testing.assert_allclose(np.asarray(ours[k], float), np.asarray(r, float),
                                   rtol=0, atol=1e-6, err_msg=k)


@pytest.mark.parametrize("seed,quantize", [(10, False), (11, True), (12, False)])
def test_evaluate_equals_ultralytics_detmetrics(seed, quantize):
    pytest.importorskip("ultralytics")
    from ultralytics.utils.metrics import DetMetrics

    preds, gts = synthetic_dataset(seed, n_images=50, quantize_conf=quantize)
    dm = DetMetrics(names={i: n for i, n in enumerate(NAMES)})
    for s in _ultralytics_stats(preds, gts):
        dm.update_stats(s)
    dm.process()
    ours = A.evaluate(preds, gts, NAMES)
    assert ours["mAP50"] == pytest.approx(dm.box.map50, abs=1e-6)
    assert ours["mAP50_95"] == pytest.approx(dm.box.map, abs=1e-6)
    assert ours["precision"] == pytest.approx(dm.box.mp, abs=1e-6)
    assert ours["recall"] == pytest.approx(dm.box.mr, abs=1e-6)
    for k, c in enumerate(dm.box.ap_class_index):
        assert ours["per_class"][NAMES[c]]["ap50"] == pytest.approx(dm.box.ap50[k], abs=1e-6)
        assert ours["per_class"][NAMES[c]]["ap50_95"] == pytest.approx(dm.box.ap[k], abs=1e-6)
    # the class without ground truth is absent from Ultralytics' class index too
    assert NAMES.index("TextView") not in list(dm.box.ap_class_index)
    assert ours["per_class"]["TextView"]["ap50"] is None


# ---------------------------------------------------------------------------
# 3. evaluate_detector helpers and CLI (saved predictions, no model)
# ---------------------------------------------------------------------------

def _write_png(path: Path, w: int = 100, h: int = 160) -> None:
    """Writes a blank PNG of the given size.

    Input:  path — output path; w, h — size
    Output: none
    """
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), (200, 200, 200)).save(path)


def test_read_image_list_dir_and_txt(tmp_path):
    for n in ("b.png", "a.jpg", "notes.txt"):
        if n.endswith("txt"):
            (tmp_path / "imgs" / n).write_text("x")
        else:
            _write_png(tmp_path / "imgs" / n)
    assert [p.name for p in E.read_image_list(tmp_path / "imgs")] == ["a.jpg", "b.png"]
    lst = tmp_path / "list.txt"
    lst.write_text(f"./imgs/b.png\n# comment\n\n{tmp_path / 'imgs' / 'a.jpg'}\n")
    got = E.read_image_list(lst)
    assert sorted(p.name for p in got) == ["a.jpg", "b.png"]
    assert all(p.exists() for p in got)
    with pytest.raises(FileNotFoundError):
        E.read_image_list(tmp_path / "nope")


def test_load_yolo_label_checks_like_ultralytics(tmp_path):
    lab = tmp_path / "x.txt"
    lab.write_text("0 0.5 0.5 0.2 0.1\n0 0.5 0.5 0.2 0.1\n3 0.25 0.25 0.5 0.5\n")
    entry, status = E.load_yolo_label(lab, 100, 200, 5)
    assert status == "duplicates_removed:1" and sorted(entry["cls"]) == [0, 3]
    i0 = entry["cls"].index(0)
    np.testing.assert_allclose(entry["boxes"][i0], [40, 90, 60, 110], atol=1e-4)
    assert E.load_yolo_label(tmp_path / "missing.txt", 1, 1, 5) == ({"boxes": [], "cls": []}, "missing")
    lab.write_text("")
    assert E.load_yolo_label(lab, 1, 1, 5)[1] == "empty"
    for bad in ("7 0.5 0.5 0.1 0.1\n", "0 1.5 0.5 0.1 0.1\n", "0 0.5 0.5 0.1\n", "0 -0.5 0.5 0.1 0.1\n"):
        lab.write_text(bad)
        entry, status = E.load_yolo_label(lab, 10, 10, 5)
        assert entry is None and status.startswith("corrupt:")


def test_model_tag():
    assert E.model_tag("runs/oversample_5k/weights/best.pt") == "oversample_5k"
    assert E.model_tag("foo/model.pt") == "model"
    assert E.model_tag("runs/oversample_2k/weights/last.pt") == "oversample_2k"
    assert E.model_tag("data/weights/imgsz_640.pt") == "imgsz_640"


def test_evaluate_detector_cli_with_saved_predictions(tmp_path):
    preds, gts = synthetic_dataset(5, n_images=12)
    img_dir, lab_dir = tmp_path / "images", tmp_path / "labels"
    W, H = 1000, 1600
    for img, g in gts.items():
        _write_png(img_dir / f"{img}.png", W, H)
        lines = []
        for (x1, y1, x2, y2), c in zip(g["boxes"], g["cls"]):
            x1, x2 = max(0.0, x1), min(float(W), x2)
            y1, y2 = max(0.0, y1), min(float(H), y2)
            lines.append(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                         f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
        lab_dir.mkdir(exist_ok=True)
        (lab_dir / f"{img}.txt").write_text("".join(lines))
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text("names:\n" + "".join(f"  - {n}\n" for n in NAMES))
    p1, p2 = tmp_path / "p1.json", tmp_path / "p2.json"
    E.save_predictions(p1, preds, {"weights": "a.pt"})
    worse = {k: dict(v, cls=[(c + 1) % 4 for c in v["cls"]]) for k, v in preds.items()}
    E.save_predictions(p2, worse, {"weights": "b.pt"})
    out = tmp_path / "res.json"
    rc = E.main(["--preds", str(p1), "--preds", str(p2), "--images", str(img_dir),
                 "--labels", str(lab_dir), "--dataset-yaml", str(yaml_path), "--out", str(out),
                 "--bootstrap", "20"])
    assert rc == 0
    res = json.loads(out.read_text())
    assert [m["name"] for m in res["models"]] == ["p1", "p2"]
    assert res["images"]["evaluated"] == 12
    # metrics equal a direct evaluate() on the label files read back
    gts_read, _ = E.load_ground_truth(E.read_image_list(img_dir), lab_dir, len(NAMES))
    direct = A.evaluate(preds, gts_read, NAMES)
    assert res["models"][0]["metrics"]["mAP50"] == pytest.approx(direct["mAP50"])
    assert res["models"][0]["bootstrap"]["n"] == 20
    assert res["paired"][0]["base"] == "p1" and res["paired"][0]["other"] == "p2"
    assert res["paired"][0]["diff"]["mAP50"]["point"] < 0
    # Saved AP predictions without single-label predict() metadata cannot give the
    # pipeline's operating point; the reason is reported instead of a wrong number.
    op = res["models"][0]["operating_point_pipeline"]
    assert op["predictions"] is None and "not predict()" in op["reason"]


def _cli_dataset(tmp_path: Path, seed: int = 5, n_images: int = 12):
    """Writes images, YOLO labels and a dataset yaml for a synthetic set.

    Input:  tmp_path — scratch directory; seed, n_images — synthetic_dataset() arguments
    Output: (preds, img_dir, lab_dir, yaml_path)
    """
    preds, gts = synthetic_dataset(seed, n_images=n_images)
    img_dir, lab_dir = tmp_path / "images", tmp_path / "labels"
    W, H = 1000, 1600
    lab_dir.mkdir(parents=True, exist_ok=True)
    for img, g in gts.items():
        _write_png(img_dir / f"{img}.png", W, H)
        lines = []
        for (x1, y1, x2, y2), c in zip(g["boxes"], g["cls"]):
            x1, x2 = max(0.0, x1), min(float(W), x2)
            y1, y2 = max(0.0, y1), min(float(H), y2)
            lines.append(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                         f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
        (lab_dir / f"{img}.txt").write_text("".join(lines))
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text("names:\n" + "".join(f"  - {n}\n" for n in NAMES))
    return preds, img_dir, lab_dir, yaml_path


def test_evaluate_detector_defaults():
    """The AP predictions come from YOLO.val() by default; imgsz defaults to the checkpoint's."""
    args = E.parse_args(["--model", "m.pt", "--images", "i", "--labels", "l",
                         "--dataset-yaml", "d.yaml", "--out", "o.json"])
    assert args.pred_source == "val" == E.DEFAULT_PRED_SOURCE
    assert args.imgsz is None and args.groups is None and args.expect_metrics is None
    assert args.preds_dir is None


def test_evaluate_detector_preds_dir(tmp_path, monkeypatch):
    """--preds-dir puts the raw prediction files there instead of next to --out (model calls faked)."""
    preds, img_dir, lab_dir, yaml_path = _cli_dataset(tmp_path)
    weights = tmp_path / "runs" / "m" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"not a real checkpoint")

    def fake_run_val(model_path, images, *args, **kwargs):
        """Stands in for run_val: the synthetic predictions as YOLO.val() output."""
        return ({"settings": {"conf": 0.001}}, {p.stem: preds[p.stem] for p in images})

    def fake_predict(model_path, images, conf=0.001, *args, **kwargs):
        """Stands in for predict_images: the synthetic predictions above conf."""
        kept = {p.stem: {k: [v for v, c in zip(preds[p.stem][k], preds[p.stem]["conf"]) if c > conf]
                         for k in ("boxes", "conf", "cls")} for p in images}
        return kept, {"settings": {"conf": conf, "multi_label": False}}

    monkeypatch.setattr(E, "run_val", fake_run_val)
    monkeypatch.setattr(E, "predict_images", fake_predict)
    out = tmp_path / "summary" / "res.json"
    pdir = tmp_path / "elsewhere" / "preds"
    assert E.main(["--model", str(weights), "--images", str(img_dir), "--labels", str(lab_dir),
                   "--dataset-yaml", str(yaml_path), "--out", str(out), "--bootstrap", "0",
                   "--preds-dir", str(pdir)]) == 0
    assert sorted(p.name for p in pdir.iterdir()) == ["res_preds_m.json", "res_preds_m_pipeline.json"]
    assert sorted(p.name for p in out.parent.iterdir()) == ["res.json"]
    m = json.loads(out.read_text())["models"][0]
    assert m["predictions_file"] == str(pdir / "res_preds_m.json")
    saved, _ = E.load_predictions(pdir / "res_preds_m.json")
    assert saved == json.loads(json.dumps(preds, default=E._json_default))


def test_evaluate_detector_cli_pipeline_operating_point_and_groups(tmp_path):
    """Single-label predict() predictions give the pipeline operating point; --groups adds a group bootstrap."""
    preds, img_dir, lab_dir, yaml_path = _cli_dataset(tmp_path)
    p = tmp_path / "pipe.json"
    E.save_predictions(p, preds, {"source": "predict", "settings": {"multi_label": False, "conf": 0.3}})
    manifest = tmp_path / "manifest.csv"
    stems = sorted(q.stem for q in img_dir.iterdir())
    manifest.write_text("stem,package,group_id\n" + "".join(
        f"{st},app{k // 3},{k // 3}\n" for k, st in enumerate(stems)))
    out = tmp_path / "res.json"
    assert E.main(["--preds", str(p), "--images", str(img_dir), "--labels", str(lab_dir),
                   "--dataset-yaml", str(yaml_path), "--out", str(out), "--bootstrap", "30",
                   "--groups", str(manifest)]) == 0
    m = json.loads(out.read_text())["models"][0]
    gts_read, _ = E.load_ground_truth(E.read_image_list(img_dir), lab_dir, len(NAMES))
    direct = A.operating_point(preds, gts_read, NAMES, 0.3)
    op = m["operating_point_pipeline"]
    assert (op["tp"], op["fp"], op["fn"]) == (direct["tp"], direct["fp"], direct["fn"])
    assert m["bootstrap"]["resample_unit"] == "image" and m["bootstrap"]["n_units"] == 12
    assert m["bootstrap_by_group"]["resample_unit"] == "group" and m["bootstrap_by_group"]["n_units"] == 4
    assert m["bootstrap_by_group"]["mAP50"] == A.bootstrap_ci(
        preds, gts_read, NAMES, 30, 0, groups={st: str(k // 3) for k, st in enumerate(stems)})["mAP50"]
    # Predictions made at a higher threshold than conf-op cannot be used for it.
    E.save_predictions(p, preds, {"source": "predict", "settings": {"multi_label": False, "conf": 0.5}})
    assert E.main(["--preds", str(p), "--images", str(img_dir), "--labels", str(lab_dir),
                   "--dataset-yaml", str(yaml_path), "--out", str(out), "--bootstrap", "0"]) == 0
    assert "conf 0.5 > 0.3" in json.loads(out.read_text())["models"][0]["operating_point_pipeline"]["reason"]
    # A screen missing from the manifest is an error (every image needs its group).
    manifest.write_text("stem,package,group_id\n" + f"{stems[0]},a,0\n")
    assert E.main(["--preds", str(p), "--images", str(img_dir), "--labels", str(lab_dir),
                   "--dataset-yaml", str(yaml_path), "--out", str(out), "--groups", str(manifest)]) == 1


def test_evaluate_detector_expect_metrics(tmp_path):
    """--expect-metrics passes (status 0) when the published values are reproduced and fails (3) otherwise."""
    preds, img_dir, lab_dir, yaml_path = _cli_dataset(tmp_path)
    p = tmp_path / "p.json"
    E.save_predictions(p, preds, {"weights": "a.pt"})
    gts_read, _ = E.load_ground_truth(E.read_image_list(img_dir), lab_dir, len(NAMES))
    m = A.evaluate(preds, gts_read, NAMES)
    per_class = {n: v["ap50"] for n, v in m["per_class"].items() if v["ap50"] is not None}
    nested = {"run": {"overall": {"mAP50": m["mAP50"] + 0.004, "mAP50_95": m["mAP50_95"]},
                      "per_class_ap50": per_class}}
    (tmp_path / "nested.json").write_text(json.dumps(nested))
    (tmp_path / "flat.json").write_text(json.dumps({"mAP50": m["mAP50"] - 0.01, "mAP50_95": m["mAP50_95"]}))
    base = ["--preds", str(p), "--images", str(img_dir), "--labels", str(lab_dir),
            "--dataset-yaml", str(yaml_path), "--out", str(tmp_path / "r.json"), "--bootstrap", "0"]
    assert E.main(base + ["--expect-metrics", str(tmp_path / "nested.json"), "--expect-key", "run"]) == 0
    rep = json.loads((tmp_path / "r.json").read_text())["models"][0]["reproduction"]
    assert rep["ok"] and rep["overall"]["mAP50"]["abs_diff"] == pytest.approx(0.004)
    assert set(rep["per_class_ap50"]) == set(per_class)
    assert E.main(base + ["--expect-metrics", str(tmp_path / "flat.json")]) == 3
    rep = json.loads((tmp_path / "r.json").read_text())["models"][0]["reproduction"]
    assert not rep["overall"]["mAP50"]["ok"] and rep["overall"]["mAP50_95"]["ok"]
    assert E.main(base + ["--expect-metrics", str(tmp_path / "flat.json"), "--expect-tol", "0.02"]) == 0
    # A per-class value outside the class tolerance fails the check.
    bad = json.loads(json.dumps(nested))
    first = sorted(per_class)[0]
    bad["run"]["per_class_ap50"][first] += 0.05
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    assert E.main(base + ["--expect-metrics", str(tmp_path / "bad.json"), "--expect-key", "run"]) == 3
    # The number of listed images is checked as well.
    assert E.main(base + ["--expect-images", "12"]) == 0
    assert json.loads((tmp_path / "r.json").read_text())["images"]["expected_listed"] == 12
    assert E.main(base + ["--expect-images", "13"]) == 3
    # The published files of this repository have the layouts load_expected_metrics reads.
    repo = Path(__file__).resolve().parents[1]
    test_ref = E.load_expected_metrics(repo / "results" / "perclass_ap_results.json", "oversample_5k")
    assert test_ref["mAP50"] == pytest.approx(0.48699, abs=1e-5) and len(test_ref["per_class_ap50"]) == 13
    val_ref = E.load_expected_metrics(repo / "results" / "oversample_5k_metrics.json")
    assert val_ref["mAP50"] == pytest.approx(0.51823) and val_ref["per_class_ap50"] == {}
    with pytest.raises(ValueError):
        E.load_expected_metrics(repo / "results" / "perclass_ap_results.json", "missing")


def test_corrupt_label_file_excludes_the_image_from_the_evaluation(tmp_path):
    """An image with a corrupt label file is left out (as in Ultralytics), so its predictions are not false positives."""
    preds, img_dir, lab_dir, yaml_path = _cli_dataset(tmp_path, seed=6, n_images=8)
    bad = sorted(q.stem for q in img_dir.iterdir())[0]
    (lab_dir / f"{bad}.txt").write_text("0 0.5 0.5 1.8 0.2\n")       # width > 1.01: corrupt
    images = E.read_image_list(img_dir)
    gts, report = E.load_ground_truth(images, lab_dir, len(NAMES))
    assert bad not in gts and len(gts) == 7
    assert report["corrupt"] == [{"image": bad, "reason": "non-normalized or out of bounds coordinates"}]
    p = tmp_path / "p.json"
    preds[bad] = {"boxes": [[0, 0, 50, 50]] * 3, "conf": [0.9, 0.8, 0.7], "cls": [0, 1, 2]}
    E.save_predictions(p, preds, {"weights": "a.pt"})
    out = tmp_path / "r.json"
    assert E.main(["--preds", str(p), "--images", str(img_dir), "--labels", str(lab_dir),
                   "--dataset-yaml", str(yaml_path), "--out", str(out), "--bootstrap", "0"]) == 0
    res = json.loads(out.read_text())
    assert res["images"]["evaluated"] == 7 and res["images"]["labels"]["corrupt"][0]["image"] == bad
    expected = A.evaluate({k: v for k, v in preds.items() if k != bad}, gts, NAMES)
    assert res["models"][0]["metrics"]["mAP50"] == pytest.approx(expected["mAP50"])
    assert res["models"][0]["metrics"]["n_images"] == 7


def test_load_ground_truth_rejects_duplicate_stems(tmp_path):
    _write_png(tmp_path / "a" / "x.png")
    _write_png(tmp_path / "b" / "x.png")
    with pytest.raises(ValueError):
        E.load_ground_truth([tmp_path / "a" / "x.png", tmp_path / "b" / "x.png"], tmp_path, 5)


def _smoke_setup(tmp_path: Path, n: int = 2):
    """Two bundled examples with made-up labels (the model's own v1.1.0 detections).

    Input:  tmp_path — scratch directory; n — number of examples
    Output: (weights, images, labels_dir, names); skips if assets are missing
    """
    pytest.importorskip("ultralytics")
    repo = Path(__file__).resolve().parents[1]
    weights = repo / "runs" / "oversample_5k_appsplit" / "weights" / "best.pt"   # pipeline default
    ref = repo / "results" / "v1.1.0_examples"
    require_reference_data(weights, ref)
    images = sorted((repo / "examples").glob("*.png"))[:n]
    lab_dir = tmp_path / "labels"
    lab_dir.mkdir()
    for img in images:
        dets = json.loads((ref / img.stem / "1_detections.json").read_text())
        W, H = E.exif_size(img)
        (lab_dir / f"{img.stem}.txt").write_text("".join(
            f"{d['class_id']} {((d['bbox'][0] + d['bbox'][2]) / 2) / W:.6f} "
            f"{((d['bbox'][1] + d['bbox'][3]) / 2) / H:.6f} {(d['bbox'][2] - d['bbox'][0]) / W:.6f} "
            f"{(d['bbox'][3] - d['bbox'][1]) / H:.6f}\n" for d in dets))
    return weights, images, lab_dir, E.load_names(repo / "configs" / "dataset.yaml")


def test_evaluate_detector_cli_val_source_smoke(tmp_path):
    """CLI with --pred-source val --cross-check-val: ap.py reproduces YOLO.val exactly."""
    weights, images, lab_dir, _ = _smoke_setup(tmp_path)
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for p in images:
        (img_dir / p.name).symlink_to(p)
    out = tmp_path / "res.json"
    repo = Path(__file__).resolve().parents[1]
    assert E.main(["--model", str(weights), "--images", str(img_dir), "--labels", str(lab_dir),
                   "--dataset-yaml", str(repo / "configs" / "dataset.yaml"), "--out", str(out),
                   "--bootstrap", "5", "--device", "cpu", "--pred-source", "val",
                   "--cross-check-val", "--val-batch", "2"]) == 0
    res = json.loads(out.read_text())
    m = res["models"][0]
    assert m["meta"]["source"] == "val" and res["settings"]["pred_source"] == "val"
    for k in ("mAP50", "mAP50_95", "precision", "recall"):
        assert m["val_cross_check"]["abs_diff"][k] < 1e-6
        assert m["val_cross_check"]["abs_diff_metric_port"][k] < 1e-6
    saved, _ = E.load_predictions(out.with_name("res_preds_oversample_5k_appsplit.json"))
    assert set(saved) == {p.stem for p in images}
    # The pipeline operating point is computed from exactly the pipeline's detections
    # (the golden raw detections of these screens), not from the multi-label AP predictions.
    pipe, pipe_meta = E.load_predictions(out.with_name("res_preds_oversample_5k_appsplit_pipeline.json"))
    assert pipe_meta["settings"]["multi_label"] is False and pipe_meta["settings"]["conf"] == 0.3
    assert pipe_meta["settings"]["imgsz"] == 640                     # the checkpoint's own size
    for p in images:
        golden = json.loads((repo / "tests" / "golden" / p.stem / "1_detections_raw.json").read_text())
        assert sorted(pipe[p.stem]["cls"]) == sorted(d["class_id"] for d in golden), p.stem
    op = m["operating_point_pipeline"]
    assert op["tp"] + op["fp"] == sum(len(pipe[p.stem]["cls"]) for p in images)


def test_val_capture_matches_ultralytics_and_predict_smoke(tmp_path):
    """run_val / predict_images on two bundled examples with made-up labels.

    The labels are the released model's own v1.1.0 detections, so the metric
    values are meaningless and are not reported anywhere. What is checked:
    ap.py applied to the predictions captured from YOLO.val() reproduces
    YOLO.val()'s own numbers (the metric port on real model output), and
    predict()-based metrics are close to them.
    """
    weights, images, lab_dir, names = _smoke_setup(tmp_path)
    preds, meta = E.predict_images(weights, images, device="cpu")
    assert set(preds) == {p.stem for p in images} and meta["settings"]["multi_label"]
    gts, _ = E.load_ground_truth(images, lab_dir, len(names))
    ours = A.evaluate(preds, gts, names)
    val, val_preds = E.run_val(weights, images, lab_dir, names, device="cpu", batch=2)
    assert set(val_preds) == set(preds)
    port = E.compare_with_val(A.evaluate(val_preds, gts, names), val)
    for k in ("mAP50", "mAP50_95", "precision", "recall", "max_abs_class_ap50"):
        assert port[k] < 1e-6, (k, port[k])
    assert E.compare_with_val(ours, val)["mAP50"] < 0.02


# ---------------------------------------------------------------------------
# 4. benchmark helpers
# ---------------------------------------------------------------------------

def test_benchmark_summary_statistics():
    from ui2compose.evaluation import benchmark as B

    s = B.summarize([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s["n"] == 5 and s["median"] == 3.0
    assert s["q1"] == 2.0 and s["q3"] == 4.0 and s["iqr"] == 2.0
    assert B.summarize([])["n"] == 0


def test_benchmark_host_info_has_required_fields():
    from ui2compose.evaluation import benchmark as B

    info = B.host_info()
    for key in ("platform", "cpu_model", "logical_cores", "physical_cores", "ram_bytes",
                "python", "torch", "ultralytics"):
        assert key in info
    assert info["logical_cores"] is None or info["logical_cores"] >= 1
    tag = B.host_tag(info)
    assert tag and all(ch.isalnum() or ch == "-" for ch in tag)
    assert B.host_tag({"system": "Linux", "machine": "x86_64", "cpu_model": "Intel(R) Xeon(R)"}) \
        == "linux-x86-64-intel-r-xeon-r"


def test_run_benchmark_smoke(tmp_path):
    """One image, one repeat: every stage is timed and the report is complete."""
    pytest.importorskip("ultralytics")
    from ui2compose.evaluation import benchmark as B

    repo = Path(__file__).resolve().parents[1]
    weights = repo / "runs" / "oversample_5k" / "weights" / "best.pt"
    require_reference_data(weights)
    img = sorted((repo / "examples").glob("*.png"))[0]
    res = B.run_benchmark([img], str(weights), str(repo / "configs" / "dataset.yaml"), repeats=1)
    st = res["stages"]
    assert st["model_load"]["first"] > 0 and st["detect"]["first"] > 0
    assert st["import_in_process"] >= 0 and st["import_subprocess"]["all"]["n"] == 1
    for k in ("graph", "codegen"):
        assert st[k]["first"] is not None and st[k]["warm"]["n"] == 0
    assert res["host"]["device"] is not None
    la = res["host"]["load_average_1_5_15min"]
    assert set(la) == {"start", "end"}
    assert all(v is None or (len(v) == 3 and min(v) >= 0) for v in la.values())
    with pytest.raises(ValueError):
        B.run_benchmark([img], str(weights), "x", repeats=0)
