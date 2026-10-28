# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Detection AP exactly as Ultralytics computes it, in pure NumPy, plus bootstrap CIs.

Ported from ultralytics 8.4.24 (the version the published results were
produced with):
  * box_iou()           <- ultralytics.utils.metrics.box_iou (float32, eps 1e-7)
  * match_predictions() <- ultralytics.engine.validator.BaseValidator.match_predictions
                           (greedy by IoU, non-SciPy path), IoU thresholds
                           0.50:0.05:0.95 as float32 (torch.linspace(0.5, 0.95, 10))
  * compute_ap(), smooth(), ap_per_class()
                        <- ultralytics.utils.metrics (101-point interpolated AP,
                           P/R/F1 at the confidence that maximises the
                           0.1-smoothed mean F1 curve)
tests/test_ap.py checks equality with the Ultralytics functions.

Conventions inherited from Ultralytics (and kept on purpose):
  * mAP is the mean over the classes that have at least one ground-truth box
    in the evaluated set; predictions of a class without ground truth do not
    enter any AP. A class with ground truth but no prediction has AP 0.
  * precision / recall / f1 are means over those classes at the single
    confidence that maximises the smoothed mean F1 curve.

Differences to running YOLO.val() itself (see evaluate_detector.py): this
module receives boxes in original-image pixels, whereas Ultralytics matches in
letterboxed network-input coordinates (IoU is invariant to that uniform
scaling up to float rounding, but predictions are clipped to the image only in
the former); and the order of predictions with tied confidences can differ
because NumPy's default sort is not stable and Ultralytics concatenates images
in data-loader order.

Additions (not in Ultralytics): evaluate() reports an operating point (P/R/F1
with a fixed confidence threshold, micro-averaged over all boxes, IoU 0.5) of
the predictions it is given, and bootstrap_ci() gives percentile bootstrap
confidence intervals, optionally paired between two models. The bootstrap
resamples images, or, with a groups mapping (e.g. app groups from
app_split's manifest.csv), whole groups of images: screens of the same app are
not independent, so the group (cluster) bootstrap is the appropriate interval
when a test set holds several screens per app.

AP is not a mean over images, and its bootstrap replicates are not centred on
the point estimate: on the test split, resampling images or package+MD5
groups, the replicate mean of mAP50 lies 0.008-0.009 above it, and 0.02-0.04
above it on the unseen subsets, where classes with a few instances drop in
and out of the per-replicate class mean. interval_methods()
therefore gives, from the same replicates, the bootstrap bias, the basic
(reverse percentile) interval and, with leave-one-unit-out values from
jackknife(), the BCa interval next to the percentile interval.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# IoU thresholds 0.50:0.05:0.95, bit-identical to torch.linspace(0.5, 0.95, 10).
IOU_THRESHOLDS: np.ndarray = np.linspace(0.5, 0.95, 10, dtype=np.float32)

# Confidence threshold of the operating point (the pipeline/demo threshold).
DEFAULT_CONF_OP = 0.3
# IoU threshold of the operating point (column 0 of IOU_THRESHOLDS).
DEFAULT_IOU_OP = 0.5
# Bootstrap defaults: number of resamples, RNG seed, two-sided alpha.
DEFAULT_BOOTSTRAP_N = 1000
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_ALPHA = 0.05

# np.trapezoid (NumPy >= 2.0) or np.trapz (older), as in Ultralytics' compute_ap
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# Ultralytics ports
# ---------------------------------------------------------------------------

def box_iou(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Pairwise IoU of two sets of xyxy boxes (float32, as ultralytics.utils.metrics.box_iou).

    Input:  box1 — (N, 4) boxes; box2 — (M, 4) boxes, both x1, y1, x2, y2;
            eps — added to the union
    Output: (N, M) float32 IoU matrix
    """
    b1 = np.asarray(box1, dtype=np.float32).reshape(-1, 4)
    b2 = np.asarray(box2, dtype=np.float32).reshape(-1, 4)
    a1, a2 = b1[:, None, :2], b1[:, None, 2:]
    c1, c2 = b2[None, :, :2], b2[None, :, 2:]
    inter = np.clip(np.minimum(a2, c2) - np.maximum(a1, c1), 0, None).prod(2)
    return inter / ((a2 - a1).prod(2) + (c2 - c1).prod(2) - inter + np.float32(eps))


def match_predictions(pred_classes: np.ndarray, true_classes: np.ndarray, iou: np.ndarray,
                      iou_thresholds: np.ndarray = IOU_THRESHOLDS) -> np.ndarray:
    """Marks predictions as true positives per IoU threshold (Ultralytics' greedy matching).

    For every threshold, all (label, prediction) pairs of the same class with
    IoU >= threshold are sorted by IoU (descending); each prediction keeps its
    best label, then each label keeps its best prediction.

    Input:  pred_classes — (N,) predicted class ids; true_classes — (M,)
            ground-truth class ids; iou — (M, N) IoU matrix (labels x
            predictions); iou_thresholds — thresholds (float32)
    Output: (N, T) bool matrix of correct predictions
    """
    pred_classes = np.asarray(pred_classes).astype(np.int64)
    true_classes = np.asarray(true_classes).astype(np.int64)
    correct = np.zeros((pred_classes.shape[0], len(iou_thresholds)), dtype=bool)
    correct_class = true_classes[:, None] == pred_classes
    iou = np.asarray(iou, dtype=np.float32) * correct_class  # zero out wrong classes
    for i, threshold in enumerate(np.asarray(iou_thresholds, dtype=np.float32).tolist()):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """Box filter of fraction f (ultralytics.utils.metrics.smooth).

    Input:  y — 1-D array; f — filter width as a fraction of len(y)
    Output: smoothed array of the same length
    """
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (odd)
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """101-point interpolated average precision (ultralytics.utils.metrics.compute_ap).

    Input:  recall, precision — curves ordered by decreasing confidence
    Output: (ap, precision envelope, recall with sentinels)
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    ap = _trapezoid(np.interp(x, mrec, mpre), x)
    return ap, mpre, mrec


def ap_per_class(tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray,
                 target_cls: np.ndarray, eps: float = 1e-16) -> Dict[str, np.ndarray]:
    """Per-class AP, P, R, F1 (ultralytics.utils.metrics.ap_per_class without plotting).

    Input:  tp — (N, T) bool true-positive matrix; conf — (N,) confidences;
            pred_cls — (N,) predicted classes; target_cls — (M,) ground-truth
            classes of all images; eps — numerical guard
    Output: dict with 'tp', 'fp', 'p', 'r', 'f1' (per class at the max-F1
            confidence), 'ap' (classes x T), 'unique_classes' (classes with
            ground truth), 'p_curve', 'r_curve', 'f1_curve' (classes x 1000),
            'x' (confidence grid), 'prec_values', 'max_f1_index'
    """
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]

    x, prec_values = np.linspace(0, 1, 1000), []
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]
        n_p = i.sum()
        if n_p == 0 or n_l == 0:
            continue
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        recall = tpc / (n_l + eps)
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)
        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))

    prec_values = np.array(prec_values) if prec_values else np.zeros((1, 1000))
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    with warnings.catch_warnings():  # mean of an empty class axis when nc == 0
        warnings.simplefilter("ignore", RuntimeWarning)
        i = int(smooth(f1_curve.mean(0), 0.1).argmax())
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]
    tp_n = (r * nt).round()
    fp_n = (tp_n / (p + eps) - tp_n).round()
    return {"tp": tp_n, "fp": fp_n, "p": p, "r": r, "f1": f1, "ap": ap,
            "unique_classes": unique_classes.astype(int), "p_curve": p_curve,
            "r_curve": r_curve, "f1_curve": f1_curve, "x": x,
            "prec_values": prec_values, "max_f1_index": i}


# ---------------------------------------------------------------------------
# Per-image statistics
# ---------------------------------------------------------------------------

@dataclass
class ImageStats:
    """Per-image matching result, the unit that the bootstrap resamples.

    tp         — (N, T) bool, prediction correct at each IoU threshold
    conf       — (N,) float32 confidences
    pred_cls   — (N,) int64 predicted classes
    target_cls — (M,) int64 ground-truth classes
    """
    tp: np.ndarray
    conf: np.ndarray
    pred_cls: np.ndarray
    target_cls: np.ndarray


def _as_arrays(entry: Optional[Mapping[str, Any]], with_conf: bool
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts one image entry {'boxes', 'cls'[, 'conf']} to arrays.

    Input:  entry — mapping or None (no boxes); with_conf — read 'conf'
    Output: (boxes (K, 4) float32, cls (K,) int64, conf (K,) float32)
    """
    if not entry:
        return (np.zeros((0, 4), np.float32), np.zeros(0, np.int64), np.zeros(0, np.float32))
    boxes = np.asarray(entry.get("boxes", []), dtype=np.float32).reshape(-1, 4)
    cls = np.asarray(entry.get("cls", []), dtype=np.float64).reshape(-1).astype(np.int64)
    conf = (np.asarray(entry.get("conf", []), dtype=np.float32).reshape(-1)
            if with_conf else np.ones(len(cls), np.float32))
    if not (len(boxes) == len(cls) == len(conf)):
        raise ValueError(f"inconsistent lengths: boxes {len(boxes)}, cls {len(cls)}, conf {len(conf)}")
    return boxes, cls, conf


def image_stats(pred: Optional[Mapping[str, Any]], gt: Optional[Mapping[str, Any]],
                iou_thresholds: np.ndarray = IOU_THRESHOLDS,
                conf_min: Optional[float] = None) -> ImageStats:
    """Matches one image's predictions to its ground truth (DetectionValidator._process_batch).

    Input:  pred — {'boxes': (N, 4) xyxy, 'conf': (N,), 'cls': (N,)} or None;
            gt — {'boxes': (M, 4) xyxy, 'cls': (M,)} or None;
            iou_thresholds — thresholds; conf_min — keep only predictions
            with conf > conf_min (None: keep all)
    Output: ImageStats
    """
    pb, pc, pconf = _as_arrays(pred, True)
    gb, gc, _ = _as_arrays(gt, False)
    if conf_min is not None:
        keep = pconf > np.float32(conf_min)
        pb, pc, pconf = pb[keep], pc[keep], pconf[keep]
    if len(gc) == 0 or len(pc) == 0:
        tp = np.zeros((len(pc), len(iou_thresholds)), dtype=bool)
    else:
        tp = match_predictions(pc, gc, box_iou(gb, pb), iou_thresholds)
    return ImageStats(tp, pconf, pc, gc)


def _image_ids(preds_by_image: Mapping[str, Any], gts_by_image: Mapping[str, Any]) -> List[str]:
    """Returns the evaluation image ids (sorted keys of the ground truth).

    Input:  preds_by_image, gts_by_image — per-image mappings
    Output: sorted list of image ids; raises ValueError if a prediction
            refers to an image that has no ground-truth entry
    """
    extra = sorted(set(preds_by_image) - set(gts_by_image))
    if extra:
        raise ValueError(f"{len(extra)} predicted image(s) have no ground-truth entry, "
                         f"e.g. {extra[:3]} (use an empty entry for background images)")
    return sorted(gts_by_image)


def _concat(stats: Sequence[ImageStats], idx: Optional[np.ndarray] = None
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenates per-image statistics (optionally a resampled list of images).

    Input:  stats — per-image statistics; idx — image indices to use (with repetition)
    Output: (tp, conf, pred_cls, target_cls) arrays
    """
    sel = stats if idx is None else [stats[i] for i in idx]
    n_t = stats[0].tp.shape[1] if stats else len(IOU_THRESHOLDS)
    if not sel:
        return (np.zeros((0, n_t), bool), np.zeros(0, np.float32),
                np.zeros(0, np.int64), np.zeros(0, np.int64))
    return (np.concatenate([s.tp for s in sel], 0), np.concatenate([s.conf for s in sel]),
            np.concatenate([s.pred_cls for s in sel]), np.concatenate([s.target_cls for s in sel]))


def _summarise(tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray, target_cls: np.ndarray
               ) -> Dict[str, Any]:
    """Runs ap_per_class and the Ultralytics-style means.

    Input:  concatenated statistics
    Output: dict with mAP50, mAP50_95, precision, recall, f1, max_f1_conf and
            the raw ap_per_class result under 'raw'
    """
    res = ap_per_class(tp, conf, pred_cls, target_cls)
    ap = res["ap"]
    has = len(ap) > 0
    return {
        "mAP50": float(ap[:, 0].mean()) if has else 0.0,
        "mAP50_95": float(ap.mean()) if has else 0.0,
        "precision": float(res["p"].mean()) if has else 0.0,
        "recall": float(res["r"].mean()) if has else 0.0,
        "f1": float(res["f1"].mean()) if has else 0.0,
        "max_f1_conf": float(res["x"][res["max_f1_index"]]),
        "raw": res,
    }


def operating_point(preds_by_image: Mapping[str, Any], gts_by_image: Mapping[str, Any],
                    class_names: Sequence[str], conf: float = DEFAULT_CONF_OP,
                    iou: float = DEFAULT_IOU_OP) -> Dict[str, Any]:
    """Precision / recall / F1 of the detections kept at a fixed confidence threshold.

    Predictions with confidence > conf (strict, like Ultralytics' NMS filter)
    are matched per image with match_predictions() at the IoU threshold; TP,
    FP and FN are summed over all images and classes (micro average) and per
    class.

    Input:  preds_by_image, gts_by_image — see evaluate(); class_names —
            names indexed by class id; conf — confidence threshold; iou — IoU
    Output: {'conf', 'iou', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1',
             'per_class': {name: {'tp', 'fp', 'fn', 'precision', 'recall'}}};
            undefined ratios (0/0) are None
    """
    thr = np.asarray([iou], dtype=np.float32)
    nc = len(class_names)
    tp_c, fp_c, fn_c = np.zeros(nc, int), np.zeros(nc, int), np.zeros(nc, int)
    for img in _image_ids(preds_by_image, gts_by_image):
        st = image_stats(preds_by_image.get(img), gts_by_image.get(img), thr, conf_min=conf)
        correct = st.tp[:, 0]
        for c in range(nc):
            is_c = st.pred_cls == c
            tp = int(correct[is_c].sum())
            tp_c[c] += tp
            fp_c[c] += int(is_c.sum()) - tp
            fn_c[c] += int((st.target_cls == c).sum()) - tp

    def ratio(a: int, b: int) -> Optional[float]:
        """Returns a / b, or None when b == 0.

        Input:  a, b — counts
        Output: float or None
        """
        return float(a / b) if b else None

    TP, FP, FN = int(tp_c.sum()), int(fp_c.sum()), int(fn_c.sum())
    p, r = ratio(TP, TP + FP), ratio(TP, TP + FN)
    f1 = (2 * p * r / (p + r)) if (p is not None and r is not None and p + r > 0) else None
    return {
        "conf": float(conf), "iou": float(iou), "tp": TP, "fp": FP, "fn": FN,
        "precision": p, "recall": r, "f1": f1,
        "per_class": {class_names[c]: {"tp": int(tp_c[c]), "fp": int(fp_c[c]), "fn": int(fn_c[c]),
                                       "precision": ratio(tp_c[c], tp_c[c] + fp_c[c]),
                                       "recall": ratio(tp_c[c], tp_c[c] + fn_c[c])}
                      for c in range(nc)},
    }


def _check_class_ids(stats: Sequence[ImageStats], n_classes: int) -> None:
    """Rejects class ids outside [0, n_classes).

    Input:  stats — per-image statistics; n_classes — number of class names
    Output: none; raises ValueError on an invalid id
    """
    for st in stats:
        for arr in (st.pred_cls, st.target_cls):
            if arr.size and (arr.min() < 0 or arr.max() >= n_classes):
                raise ValueError(f"class id outside [0, {n_classes}): {sorted(set(arr.tolist()))}")


def compute_image_stats(preds_by_image: Mapping[str, Any], gts_by_image: Mapping[str, Any]
                        ) -> Tuple[List[str], List[ImageStats]]:
    """Matches all images once (all ten IoU thresholds).

    Input:  preds_by_image, gts_by_image — see evaluate()
    Output: (sorted image ids, per-image statistics in that order)
    """
    ids = _image_ids(preds_by_image, gts_by_image)
    return ids, [image_stats(preds_by_image.get(i), gts_by_image.get(i)) for i in ids]


def evaluate(preds_by_image: Mapping[str, Any], gts_by_image: Mapping[str, Any],
             class_names: Sequence[str], conf_op: float = DEFAULT_CONF_OP) -> Dict[str, Any]:
    """Ultralytics-equivalent detection metrics plus a fixed-confidence operating point.

    The evaluation set is the set of keys of gts_by_image (an image without
    boxes is a background image: all its predictions are false positives);
    images without a predictions entry have no predictions.

    Input:  preds_by_image — {image_id: {'boxes': [[x1, y1, x2, y2], ...],
                'conf': [...], 'cls': [...]}} in original-image pixels;
            gts_by_image — {image_id: {'boxes': [...], 'cls': [...]}};
            class_names — names indexed by class id;
            conf_op — confidence threshold of the operating point
    Output: {'n_images', 'n_instances', 'mAP50', 'mAP50_95', 'precision',
             'recall', 'f1', 'max_f1_conf',
             'per_class': {name: {'ap50', 'ap50_95', 'precision', 'recall',
                                  'support', 'n_images', 'n_pred'}},
             'operating_point': {...}} — per-class precision / recall are taken
            at the max-F1 confidence; per-class values are None for classes
            without ground truth (excluded from the means, as in Ultralytics);
            raises ValueError for class ids outside [0, len(class_names))
    """
    ids, stats = compute_image_stats(preds_by_image, gts_by_image)
    _check_class_ids(stats, len(class_names))
    tp, conf, pred_cls, target_cls = _concat(stats)
    s = _summarise(tp, conf, pred_cls, target_cls)
    res = s.pop("raw")
    row = {int(c): k for k, c in enumerate(res["unique_classes"])}
    support = np.bincount(target_cls, minlength=len(class_names)) if len(target_cls) else \
        np.zeros(len(class_names), int)
    n_img = np.zeros(len(class_names), int)
    for st in stats:
        for c in np.unique(st.target_cls):
            n_img[int(c)] += 1
    n_pred = np.bincount(pred_cls, minlength=len(class_names)) if len(pred_cls) else \
        np.zeros(len(class_names), int)
    per_class = {}
    for c, name in enumerate(class_names):
        k = row.get(c)
        per_class[name] = {
            "ap50": float(res["ap"][k, 0]) if k is not None else None,
            "ap50_95": float(res["ap"][k].mean()) if k is not None else None,
            "precision": float(res["p"][k]) if k is not None else None,
            "recall": float(res["r"][k]) if k is not None else None,
            "support": int(support[c]), "n_images": int(n_img[c]), "n_pred": int(n_pred[c]),
        }
    return {"n_images": len(ids), "n_instances": int(len(target_cls)), **s,
            "per_class": per_class,
            "operating_point": operating_point(preds_by_image, gts_by_image, class_names, conf_op)}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _resample_metrics(stats: Sequence[ImageStats], idx: np.ndarray, n_classes: int
                      ) -> Tuple[float, float, np.ndarray]:
    """Computes mAP50, mAP50-95 and per-class AP50 on a resampled image list.

    Input:  stats — per-image statistics; idx — resampled image indices;
            n_classes — number of class ids
    Output: (mAP50, mAP50_95, per-class AP50 array with NaN for classes
             without ground truth in the resample)
    """
    tp, conf, pred_cls, target_cls = _concat(stats, idx)
    s = _summarise(tp, conf, pred_cls, target_cls)
    res = s["raw"]
    ap50 = np.full(n_classes, np.nan)
    for k, c in enumerate(res["unique_classes"]):
        if 0 <= c < n_classes:
            ap50[c] = res["ap"][k, 0]
    return s["mAP50"], s["mAP50_95"], ap50


def _interval(values: np.ndarray, point: float, alpha: float) -> Dict[str, Any]:
    """Percentile interval of bootstrap replicates.

    Input:  values — replicate values (NaN = undefined in that resample);
            point — estimate on the full set; alpha — two-sided level
    Output: {'point', 'lo', 'hi', 'n_valid'} (lo/hi None if no valid replicate)
    """
    v = values[~np.isnan(values)]
    if v.size == 0:
        return {"point": point, "lo": None, "hi": None, "n_valid": 0}
    lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_valid": int(v.size)}


def _resampling_units(ids: Sequence[str], groups: Optional[Mapping[str, Any]]
                      ) -> Tuple[List[np.ndarray], List[str]]:
    """Splits the image indices into the units the bootstrap draws.

    Input:  ids — sorted evaluation image ids; groups — mapping image id ->
            group label, or None (every image is its own unit)
    Output: (list of index arrays, one per unit, ordered by str(group label)
            [by image id without groups]; the unit labels); raises ValueError
            if groups lacks an image
    """
    if groups is None:
        return [np.asarray([i]) for i in range(len(ids))], list(ids)
    missing = [i for i in ids if i not in groups]
    if missing:
        raise ValueError(f"{len(missing)} evaluation image(s) have no group, e.g. {missing[:3]}")
    members: Dict[str, List[int]] = {}
    for k, img in enumerate(ids):
        members.setdefault(str(groups[img]), []).append(k)
    labels = sorted(members)
    return [np.asarray(members[g]) for g in labels], labels


def bootstrap_ci(preds_by_image: Mapping[str, Any], gts_by_image: Mapping[str, Any],
                 class_names: Sequence[str], n: int = DEFAULT_BOOTSTRAP_N,
                 seed: int = DEFAULT_BOOTSTRAP_SEED,
                 other_preds: Optional[Mapping[str, Any]] = None,
                 alpha: float = DEFAULT_ALPHA,
                 groups: Optional[Mapping[str, Any]] = None,
                 keep_replicates: bool = False) -> Dict[str, Any]:
    """Percentile bootstrap CIs over images (or groups of images) for mAP50, mAP50-95 and per-class AP50.

    Without groups, each replicate draws len(images) image ids with
    replacement (NumPy default_rng(seed)); with groups (a cluster bootstrap),
    it draws as many groups as there are with replacement and takes all
    images of every drawn group. The metrics are recomputed with
    ap_per_class on the concatenated per-image statistics of the drawn images
    (an image drawn twice counts twice). As in Ultralytics, the mean is over
    the classes present in the replicate's ground truth; per-class AP50 is
    undefined (skipped, see 'n_valid') in replicates without that class. The
    interval is the [100*alpha/2, 100*(1-alpha/2)] percentile range of the
    valid replicates (np.percentile, linear interpolation). With other_preds,
    the SAME resampled ids are used for both models and the intervals of the
    paired differences (other minus preds) are reported.

    Input:  preds_by_image, gts_by_image, class_names — see evaluate();
            n — number of replicates; seed — RNG seed; other_preds —
            predictions of a second model on the same images (optional);
            alpha — two-sided level in (0, 1) (0.05 -> 95% intervals);
            groups — optional mapping image id -> group label (e.g. the
            group_id column of app_split's manifest.csv); every evaluation
            image needs a group; keep_replicates — also return the replicate
            values of mAP50, mAP50_95 and per-class AP50 of the first model
    Output: {'n', 'seed', 'alpha', 'n_images', 'resample_unit' ('image' or
             'group'), 'n_units', 'mAP50': interval, 'mAP50_95': interval,
             'per_class_ap50': {name: interval}} and, with other_preds,
            'other': same for the second model and 'diff': intervals of
            other - preds plus 'p_diff_le_0' (fraction of replicates with
            difference <= 0) for mAP50 and mAP50_95; with keep_replicates,
            'replicates': {'mAP50': [...], 'mAP50_95': [...],
            'per_class_ap50': [[one value per class id, NaN = absent], ...]};
            interval = {'point', 'lo', 'hi', 'n_valid'}
    """
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    ids, stats = compute_image_stats(preds_by_image, gts_by_image)
    units, _ = _resampling_units(ids, groups)
    nc = len(class_names)
    rng = np.random.default_rng(seed)
    full = np.arange(len(ids))
    models = [stats]
    if other_preds is not None:
        models.append(compute_image_stats(other_preds, gts_by_image)[1])
    for m in models:
        _check_class_ids(m, nc)
    points = [_resample_metrics(m, full, nc) for m in models]
    reps = [(np.empty(n), np.empty(n), np.empty((n, nc))) for _ in models]
    n_units = len(units)
    for b in range(n):
        if not n_units:
            idx = full
        elif groups is None:      # image level: len(images) draws of single images
            idx = rng.integers(0, len(ids), len(ids))
        else:                     # group level: all images of every drawn group
            idx = np.concatenate([units[u] for u in rng.integers(0, n_units, n_units)])
        for m, rep in zip(models, reps):
            rep[0][b], rep[1][b], rep[2][b] = _resample_metrics(m, idx, nc)

    def block(k: int) -> Dict[str, Any]:
        """Intervals of model k.

        Input:  k — model index (0 = preds, 1 = other_preds)
        Output: dict of intervals
        """
        m50, m5095, ap50 = reps[k]
        p50, p5095, pap = points[k]
        return {"mAP50": _interval(m50, p50, alpha), "mAP50_95": _interval(m5095, p5095, alpha),
                "per_class_ap50": {class_names[c]: _interval(ap50[:, c],
                                                             None if np.isnan(pap[c]) else float(pap[c]),
                                                             alpha) for c in range(nc)}}

    out: Dict[str, Any] = {"n": n, "seed": seed, "alpha": alpha, "n_images": len(ids),
                           "resample_unit": "image" if groups is None else "group",
                           "n_units": n_units, **block(0)}
    if keep_replicates:
        out["replicates"] = {"mAP50": reps[0][0].tolist(), "mAP50_95": reps[0][1].tolist(),
                             "per_class_ap50": reps[0][2].tolist()}
    if other_preds is not None:
        out["other"] = block(1)
        diff: Dict[str, Any] = {}
        for key, j in (("mAP50", 0), ("mAP50_95", 1)):
            d = reps[1][j] - reps[0][j]
            diff[key] = _interval(d, points[1][j] - points[0][j], alpha)
            diff[key]["p_diff_le_0"] = float(np.mean(d <= 0)) if n else None
        dap = reps[1][2] - reps[0][2]
        pdap = points[1][2] - points[0][2]
        diff["per_class_ap50"] = {class_names[c]: _interval(dap[:, c],
                                                            None if np.isnan(pdap[c]) else float(pdap[c]),
                                                            alpha) for c in range(nc)}
        out["diff"] = diff
    return out


# ---------------------------------------------------------------------------
# Bias of the bootstrap and bias-corrected intervals
# ---------------------------------------------------------------------------

def jackknife(preds_by_image: Mapping[str, Any], gts_by_image: Mapping[str, Any],
              class_names: Sequence[str], groups: Optional[Mapping[str, Any]] = None
              ) -> Dict[str, Any]:
    """Leave-one-unit-out values of mAP50, mAP50-95 and per-class AP50 (input of the BCa acceleration).

    The units are those bootstrap_ci resamples (single images, or the groups
    of `groups`); value k is the metric on every image except those of unit k.

    Input:  preds_by_image, gts_by_image, class_names — see evaluate();
            groups — optional mapping image id -> group label
    Output: {'n_units', 'resample_unit', 'mAP50' (n_units,) array,
             'mAP50_95' (n_units,) array, 'per_class_ap50' (n_units,
             n_classes) array (NaN = class absent without that unit)}
    """
    ids, stats = compute_image_stats(preds_by_image, gts_by_image)
    nc = len(class_names)
    _check_class_ids(stats, nc)
    units, _ = _resampling_units(ids, groups)
    m50, m5095, ap50 = np.empty(len(units)), np.empty(len(units)), np.empty((len(units), nc))
    member = np.empty(len(ids), dtype=int)
    for u, idx in enumerate(units):
        member[idx] = u
    for u in range(len(units)):
        m50[u], m5095[u], ap50[u] = _resample_metrics(stats, np.flatnonzero(member != u), nc)
    return {"n_units": len(units), "resample_unit": "image" if groups is None else "group",
            "mAP50": m50, "mAP50_95": m5095, "per_class_ap50": ap50}


def interval_methods(replicates: Sequence[float], point: Optional[float],
                     alpha: float = DEFAULT_ALPHA,
                     jackknife_values: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """Percentile, basic and BCa intervals from the same bootstrap replicates, with the bootstrap bias.

    * percentile: [q(alpha/2), q(1 - alpha/2)] of the valid replicates (as
      bootstrap_ci; np.percentile, linear interpolation);
    * basic (reverse percentile): [2 point - q(1 - alpha/2), 2 point - q(alpha/2)],
      which corrects a percentile interval for a shift of the replicate
      distribution against the point estimate;
    * BCa (Efron 1987), only with jackknife values: bias correction
      z0 = Phi^-1((#{rep < point} + #{rep = point} / 2) / B), acceleration
      a = sum(d^3) / (6 (sum d^2)^1.5) with d = mean(jack) - jack, endpoints
      = replicate quantiles at Phi(z0 + (z0 + z) / (1 - a (z0 + z))) for
      z = Phi^-1(alpha/2) and Phi^-1(1 - alpha/2); None when z0 is not finite
      or the jackknife values have no spread;
    * bias = mean(replicates) - point (the bootstrap estimate of the bias of
      the estimator), share_above = fraction of replicates > point.

    Input:  replicates — bootstrap values (NaN = undefined, skipped); point
            — estimate on the evaluated set (None: nothing but the counts);
            alpha — two-sided level; jackknife_values — leave-one-unit-out
            estimates (NaN skipped), or None (no BCa)
    Output: {'point', 'n_valid', 'replicate_mean', 'replicate_median', 'bias',
             'share_above', 'percentile' [lo, hi], 'basic' [lo, hi], 'bca'
             [lo, hi] or None, 'bca_z0', 'bca_acceleration', 'bca_levels'}
    """
    from statistics import NormalDist

    nd = NormalDist()
    v = np.asarray(replicates, dtype=np.float64)
    v = v[np.isfinite(v)]
    out: Dict[str, Any] = {"point": point, "n_valid": int(v.size), "replicate_mean": None,
                           "replicate_median": None, "bias": None, "share_above": None,
                           "percentile": None, "basic": None, "bca": None, "bca_z0": None,
                           "bca_acceleration": None, "bca_levels": None}
    if not v.size or point is None:
        return out
    lo, hi = (float(x) for x in np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
    out.update({"replicate_mean": float(v.mean()), "replicate_median": float(np.median(v)),
                "bias": float(v.mean() - point), "share_above": float(np.mean(v > point)),
                "percentile": [lo, hi], "basic": [2 * point - hi, 2 * point - lo]})
    if jackknife_values is None:
        return out
    j = np.asarray(jackknife_values, dtype=np.float64)
    j = j[np.isfinite(j)]
    frac = (np.sum(v < point) + 0.5 * np.sum(v == point)) / v.size
    if j.size < 2 or not 0.0 < frac < 1.0:
        return out
    d = j.mean() - j
    ss = float(np.sum(d ** 2))
    if ss <= 0:
        return out
    z0 = nd.inv_cdf(float(frac))
    a = float(np.sum(d ** 3) / (6.0 * ss ** 1.5))
    levels = []
    for q in (alpha / 2, 1 - alpha / 2):
        z = z0 + nd.inv_cdf(q)
        levels.append(nd.cdf(z0 + z / (1.0 - a * z)))
    b_lo, b_hi = (float(x) for x in np.percentile(v, [100 * levels[0], 100 * levels[1]]))
    out.update({"bca": [b_lo, b_hi], "bca_z0": float(z0), "bca_acceleration": a,
                "bca_levels": [float(x) for x in levels]})
    return out
