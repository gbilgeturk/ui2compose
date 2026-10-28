# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Summary tables of saved detector evaluations: CIs over several groupings, and seen vs. unseen apps.

Works on the files ui2compose-evaluate-detector writes (the metrics JSON and
the raw AP / pipeline predictions), so the model is not needed. Two commands:

  summary      One row per evaluation (e.g. detector_<split>_<subset>.json):
               mAP50, mAP50-95 and per-class AP50 with percentile bootstrap
               intervals over images and over every grouping given with
               --groups NAME=CSV (cluster bootstrap, ap.bootstrap_ci), and the
               pipeline operating point (best-class NMS, conf > 0.3, IoU 0.5,
               micro-averaged P / R / F1) with intervals from the same
               resampling schemes (operating_point_bootstrap). Intervals
               already in the evaluation file are copied, not recomputed.
               Per-class support (ground-truth instances) is reported and
               classes with fewer than --min-support instances are flagged
               unreliable.
  seen-unseen  For one evaluation of a full split (e.g. detector_test_full):
               the same metrics on the screens whose app group occurs in
               train ("seen", manifest column seen_in_train = 1) and on the
               others ("unseen"), per class with support, plus the difference
               unseen - seen with a stratified bootstrap interval (groups
               resampled independently within each subset).
  intervals    The bias of those percentile intervals (added after the
               review of the revision): AP is not a mean over images, and
               its bootstrap replicates are shifted against the point
               estimate (upwards on these data). For every evaluation and
               scheme, the same replicates as above (same n, seed and
               units, so the percentile intervals are reproduced) give the
               bootstrap bias (replicate mean - point), the share of
               replicates above the point, the basic (reverse percentile)
               interval and the BCa interval (acceleration from
               leave-one-unit-out values, ap.jackknife), for mAP50,
               mAP50-95 and the mean AP50 over the classes with at least
               --min-support instances in the set; with --seen-unseen, the
               same (without BCa) for the unseen - seen differences.

Why several groupings: app_split's full grouping (package OR MD5 OR dHash
<= 4, transitively closed) can chain thousands of screens of different apps
into one group through near-duplicate dHashes (in the ReDraw manifest one
group holds 9,071 of 18,694 screens, about half of val and test). A cluster
bootstrap over such a grouping resamples half of the evaluation set as a
single unit; the package+MD5 grouping (app_split --near-dup-dist -1) keeps
app clusters without that chaining. Both are reported; which one an interval
used is always named.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ui2compose.evaluation import ap as apm

# Classes with fewer ground-truth instances than this in an evaluated set are
# flagged unreliable (their AP50 moves by several points with a few boxes).
DEFAULT_MIN_SUPPORT = 20
# Name of the image-level resampling scheme in the outputs.
IMAGE_SCHEME = "image"
# Name of the intervals copied from evaluate_detector's 'bootstrap_by_group'
# (grouping of its --groups manifest).
EVAL_GROUP_SCHEME = "eval_groups"
_RESULT_NAME = re.compile(r"^detector_(?P<split>[a-z]+)_(?P<subset>.+)\.json$")


# ---------------------------------------------------------------------------
# Operating point with bootstrap intervals
# ---------------------------------------------------------------------------

def operating_point_counts(preds_by_image: Mapping[str, Any], gts_by_image: Mapping[str, Any],
                           n_classes: int, conf: float = apm.DEFAULT_CONF_OP,
                           iou: float = apm.DEFAULT_IOU_OP) -> Tuple[List[str], np.ndarray]:
    """Per-image, per-class TP / FP / FN of the detections with confidence > conf (as ap.operating_point).

    Input:  preds_by_image, gts_by_image — see ap.evaluate(); n_classes —
            number of class ids; conf — confidence threshold (strict);
            iou — IoU threshold of a match
    Output: (sorted image ids, int array (n_images, n_classes, 3) of TP, FP, FN)
    """
    ids = apm._image_ids(preds_by_image, gts_by_image)
    thr = np.asarray([iou], dtype=np.float32)
    counts = np.zeros((len(ids), n_classes, 3), dtype=np.int64)
    for k, img in enumerate(ids):
        st = apm.image_stats(preds_by_image.get(img), gts_by_image.get(img), thr, conf_min=conf)
        correct = st.tp[:, 0]
        tp = np.bincount(st.pred_cls[correct], minlength=n_classes)[:n_classes]
        n_pred = np.bincount(st.pred_cls, minlength=n_classes)[:n_classes]
        n_gt = np.bincount(st.target_cls, minlength=n_classes)[:n_classes]
        counts[k, :, 0], counts[k, :, 1], counts[k, :, 2] = tp, n_pred - tp, n_gt - tp
    return ids, counts


def _prf(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision, recall, F1 from counts (NaN where undefined).

    Input:  tp, fp, fn — arrays of equal shape
    Output: (precision, recall, f1) arrays
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), np.nan)
        r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), np.nan)
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), np.nan)
    return p, r, f1


def operating_point_bootstrap(preds_by_image: Mapping[str, Any], gts_by_image: Mapping[str, Any],
                              class_names: Sequence[str], conf: float = apm.DEFAULT_CONF_OP,
                              iou: float = apm.DEFAULT_IOU_OP, n: int = apm.DEFAULT_BOOTSTRAP_N,
                              seed: int = apm.DEFAULT_BOOTSTRAP_SEED,
                              groups: Optional[Mapping[str, Any]] = None,
                              alpha: float = apm.DEFAULT_ALPHA) -> Dict[str, Any]:
    """Micro-averaged P / R / F1 at a fixed threshold with percentile bootstrap intervals.

    The resampling is that of ap.bootstrap_ci (default_rng(seed); images, or
    whole groups with groups), so intervals are comparable with the AP intervals.

    Input:  preds_by_image, gts_by_image, class_names — see ap.evaluate();
            conf, iou — operating point; n — replicates; seed — RNG seed;
            groups — optional image id -> group label; alpha — two-sided level
    Output: {'conf', 'iou', 'n', 'seed', 'resample_unit', 'n_units', 'tp',
             'fp', 'fn', 'precision', 'recall', 'f1' (intervals {'point',
             'lo', 'hi', 'n_valid'}), 'per_class': {name: {'tp', 'fp', 'fn',
             'precision', 'recall'}} (points)}
    """
    nc = len(class_names)
    ids, counts = operating_point_counts(preds_by_image, gts_by_image, nc, conf, iou)
    units, _ = apm._resampling_units(ids, groups)
    tot = counts.sum(0)                                   # (nc, 3)
    p0, r0, f0 = _prf(*(tot[:, j].sum() for j in range(3)))
    per_img = counts.sum(1)                               # (n_images, 3) micro
    rng = np.random.default_rng(seed)
    reps = np.full((n, 3), np.nan)
    n_units = len(units)
    for b in range(n):
        if not n_units:
            idx = np.arange(len(ids))
        elif groups is None:
            idx = rng.integers(0, len(ids), len(ids))
        else:
            idx = np.concatenate([units[u] for u in rng.integers(0, n_units, n_units)])
        s = per_img[idx].sum(0)
        reps[b] = _prf(s[0], s[1], s[2])
    pc_p, pc_r, _ = _prf(tot[:, 0], tot[:, 1], tot[:, 2])

    def point(v: Any) -> Optional[float]:
        """Returns a float, or None for NaN.

        Input:  v — scalar
        Output: float or None
        """
        return None if np.isnan(v) else float(v)

    return {"conf": float(conf), "iou": float(iou), "n": n, "seed": seed,
            "resample_unit": "image" if groups is None else "group", "n_units": n_units,
            "tp": int(tot[:, 0].sum()), "fp": int(tot[:, 1].sum()), "fn": int(tot[:, 2].sum()),
            "precision": apm._interval(reps[:, 0], point(p0), alpha),
            "recall": apm._interval(reps[:, 1], point(r0), alpha),
            "f1": apm._interval(reps[:, 2], point(f0), alpha),
            "per_class": {class_names[c]: {"tp": int(tot[c, 0]), "fp": int(tot[c, 1]),
                                           "fn": int(tot[c, 2]), "precision": point(pc_p[c]),
                                           "recall": point(pc_r[c])} for c in range(nc)}}


# ---------------------------------------------------------------------------
# Difference between two disjoint subsets
# ---------------------------------------------------------------------------

def subset_difference_bootstrap(preds_by_image: Mapping[str, Any], gts_by_image: Mapping[str, Any],
                                class_names: Sequence[str], stems_a: Sequence[str],
                                stems_b: Sequence[str], n: int = apm.DEFAULT_BOOTSTRAP_N,
                                seed: int = apm.DEFAULT_BOOTSTRAP_SEED,
                                groups: Optional[Mapping[str, Any]] = None,
                                alpha: float = apm.DEFAULT_ALPHA,
                                classes: Optional[Sequence[str]] = None,
                                keep_replicates: bool = False) -> Dict[str, Any]:
    """Interval of metric(B) - metric(A) for two disjoint image subsets (stratified bootstrap).

    Every replicate resamples A and B independently (images, or whole groups
    with groups; a group split between A and B is resampled within each),
    recomputes mAP50, mAP50-95 and per-class AP50 on each and takes the
    difference B - A. mAP is the mean over the classes present in a set, so
    two sets with different class composition average over different
    classes; with classes, the mean AP50 over that fixed class list
    (classes absent from a replicate skipped) is reported as well, for A, B
    and B - A.

    Input:  preds_by_image, gts_by_image, class_names — see ap.evaluate();
            stems_a, stems_b — disjoint image ids (both subsets of gts_by_image);
            n, seed, groups, alpha — as ap.bootstrap_ci; classes — optional
            class names of the fixed-class mean; keep_replicates — also
            return the replicate values
    Output: {'n', 'seed', 'alpha', 'resample_unit', 'n_units_a', 'n_units_b',
             'mAP50', 'mAP50_95' (intervals plus 'p_diff_le_0'),
             'per_class_ap50': {name: interval}} and, with classes,
            'mAP50_classes': {'classes', 'a', 'b', 'diff' (intervals; diff
            plus 'p_diff_le_0')}; with keep_replicates, 'replicates':
            {'mAP50', 'mAP50_95' (differences B - A), 'mAP50_classes_a',
            'mAP50_classes_b', 'mAP50_classes_diff'} (lists, NaN =
            undefined); raises ValueError if the subsets overlap or a class
            name is unknown
    """
    a, b = sorted(set(stems_a)), sorted(set(stems_b))
    if set(a) & set(b):
        raise ValueError("the two subsets overlap")
    nc = len(class_names)

    def prep(ids: Sequence[str]):
        """Image statistics and resampling units of one subset.

        Input:  ids — image ids
        Output: (stats, units)
        """
        sub_g = {k: gts_by_image[k] for k in ids}
        sub_p = {k: v for k, v in preds_by_image.items() if k in sub_g}
        ids_s, stats = apm.compute_image_stats(sub_p, sub_g)
        apm._check_class_ids(stats, nc)
        return stats, apm._resampling_units(ids_s, groups)[0]

    (sa, ua), (sb, ub) = prep(a), prep(b)
    pa = apm._resample_metrics(sa, np.arange(len(sa)), nc)
    pb = apm._resample_metrics(sb, np.arange(len(sb)), nc)
    unknown = sorted(set(classes or []) - set(class_names))
    if unknown:
        raise ValueError(f"unknown class names: {unknown}")
    cidx = [list(class_names).index(c) for c in (classes or [])]
    rng = np.random.default_rng(seed)
    d50, d5095, dap = np.empty(n), np.empty(n), np.empty((n, nc))
    ra, rb = np.full(n, np.nan), np.full(n, np.nan)

    def fixed_mean(ap50: np.ndarray) -> float:
        """Mean AP50 over the fixed classes present (NaN if none is).

        Input:  ap50 — per-class AP50 array (NaN = class absent)
        Output: mean or NaN
        """
        v = ap50[cidx]
        v = v[~np.isnan(v)]
        return float(v.mean()) if v.size else float("nan")

    def draw(units: List[np.ndarray], size: int) -> np.ndarray:
        """One resample of a subset.

        Input:  units — resampling units; size — number of images
        Output: image indices
        """
        if groups is None:
            return rng.integers(0, size, size)
        return np.concatenate([units[u] for u in rng.integers(0, len(units), len(units))])

    for r in range(n):
        ma = apm._resample_metrics(sa, draw(ua, len(sa)), nc)
        mb = apm._resample_metrics(sb, draw(ub, len(sb)), nc)
        d50[r], d5095[r], dap[r] = mb[0] - ma[0], mb[1] - ma[1], mb[2] - ma[2]
        if cidx:
            ra[r], rb[r] = fixed_mean(ma[2]), fixed_mean(mb[2])
    out: Dict[str, Any] = {"n": n, "seed": seed, "alpha": alpha,
                           "resample_unit": "image" if groups is None else "group",
                           "n_units_a": len(ua), "n_units_b": len(ub)}
    for key, vals, j in (("mAP50", d50, 0), ("mAP50_95", d5095, 1)):
        out[key] = apm._interval(vals, float(pb[j] - pa[j]), alpha)
        out[key]["p_diff_le_0"] = float(np.mean(vals <= 0)) if n else None
    pd = pb[2] - pa[2]
    out["per_class_ap50"] = {class_names[c]: apm._interval(dap[:, c], None if np.isnan(pd[c])
                                                           else float(pd[c]), alpha)
                             for c in range(nc)}
    if cidx:
        fa, fb = fixed_mean(pa[2]), fixed_mean(pb[2])

        def pt(v: float) -> Optional[float]:
            """None for NaN.

            Input:  v — float
            Output: v or None
            """
            return None if np.isnan(v) else v

        dd = rb - ra
        out["mAP50_classes"] = {"classes": list(classes), "a": apm._interval(ra, pt(fa), alpha),
                                "b": apm._interval(rb, pt(fb), alpha),
                                "diff": apm._interval(dd, pt(fb - fa), alpha)}
        valid = dd[~np.isnan(dd)]
        out["mAP50_classes"]["diff"]["p_diff_le_0"] = float(np.mean(valid <= 0)) if valid.size else None
    if keep_replicates:
        out["replicates"] = {"mAP50": d50.tolist(), "mAP50_95": d5095.tolist(),
                             "mAP50_classes_a": ra.tolist(), "mAP50_classes_b": rb.tolist(),
                             "mAP50_classes_diff": (rb - ra).tolist()}
    return out


# ---------------------------------------------------------------------------
# Loading saved evaluations
# ---------------------------------------------------------------------------

def parse_groups_arg(specs: Sequence[str]) -> Dict[str, Path]:
    """Parses --groups NAME=CSV arguments.

    Input:  specs — strings 'name=path'
    Output: {name: Path}; raises ValueError for a malformed spec or a duplicate name
    """
    out: Dict[str, Path] = {}
    for s in specs:
        name, sep, path = s.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--groups expects NAME=CSV, got {s!r}")
        if name in out or name in (IMAGE_SCHEME, EVAL_GROUP_SCHEME):
            raise ValueError(f"grouping name {name!r} used twice or reserved")
        out[name] = Path(path)
    return out


def load_manifest(path: str | Path) -> Dict[str, Dict[str, str]]:
    """Reads app_split's manifest.csv.

    Input:  path — manifest with at least the columns 'stem' and 'seen_in_train'
    Output: {stem: row}; raises ValueError if a column is missing
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"stem", "seen_in_train"} <= set(reader.fieldnames):
            raise ValueError(f"{path}: needs the columns 'stem' and 'seen_in_train'")
        return {row["stem"]: row for row in reader}


def split_seen_unseen(stems: Sequence[str], manifest: Mapping[str, Mapping[str, str]]
                      ) -> Tuple[List[str], List[str]]:
    """Splits screens into seen (seen_in_train = 1) and unseen (0).

    Input:  stems — image ids; manifest — load_manifest() result
    Output: (seen, unseen) sorted lists; raises ValueError for a stem missing
            from the manifest or with another value
    """
    seen, unseen = [], []
    for s in sorted(stems):
        if s not in manifest:
            raise ValueError(f"{s} is not in the manifest")
        v = str(manifest[s]["seen_in_train"]).strip()
        if v not in ("0", "1"):
            raise ValueError(f"{s}: seen_in_train = {v!r}")
        (seen if v == "1" else unseen).append(s)
    return seen, unseen


def load_evaluation(result_path: str | Path, base_dir: str | Path = "."
                    ) -> Dict[str, Any]:
    """Loads an evaluate_detector result with its ground truth and saved predictions.

    Paths recorded in the result (image list, labels, predictions) are
    resolved against base_dir (the directory the evaluation was run from).
    The pipeline predictions are the file next to the AP predictions with the
    suffix '_pipeline' (evaluate_detector's naming).

    Input:  result_path — metrics JSON; base_dir — directory of relative paths
    Output: {'result' (the JSON), 'model' (its first model entry), 'names',
             'gts', 'ap_preds', 'pipe_preds' (None if the file is missing),
             'ap_preds_file', 'pipe_preds_file'}
    """
    from ui2compose.evaluation import evaluate_detector as E

    base = Path(base_dir)
    res = json.loads(Path(result_path).read_text(encoding="utf-8"))
    model = res["models"][0]

    def resolve(p: str) -> Path:
        """Resolves a recorded path against base_dir.

        Input:  p — path string
        Output: Path
        """
        q = Path(p)
        return q if q.is_absolute() else base / q

    ap_file = resolve(model["predictions_file"])
    ap_preds, _ = E.load_predictions(ap_file)
    names = model["metrics"]["per_class"].keys()
    names = list(names)
    images = E.read_image_list(resolve(res["images"]["spec"]))
    gts, _ = E.load_ground_truth(images, resolve(res["images"]["labels_dir"]), len(names))
    pipe_file = ap_file.with_name(f"{ap_file.stem}_pipeline.json")
    pipe_preds = E.load_predictions(pipe_file)[0] if pipe_file.is_file() else None
    keep = set(gts)
    return {"result": res, "model": model, "names": names, "gts": gts,
            "ap_preds": {k: v for k, v in ap_preds.items() if k in keep},
            "pipe_preds": None if pipe_preds is None else
            {k: v for k, v in pipe_preds.items() if k in keep},
            "ap_preds_file": str(ap_file), "pipe_preds_file": str(pipe_file)}


def _ci(interval: Optional[Mapping[str, Any]]) -> Optional[List[Optional[float]]]:
    """[lo, hi] of an interval dict.

    Input:  interval — {'lo', 'hi', ...} or None
    Output: [lo, hi] or None
    """
    return None if interval is None else [interval.get("lo"), interval.get("hi")]


def precomputed_intervals(ev: Mapping[str, Any], groupings: Mapping[str, Path],
                          base_dir: str | Path, n: int, seed: int) -> Dict[str, Mapping[str, Any]]:
    """Bootstrap results already in an evaluation file that match the requested settings.

    The image-level 'bootstrap' is reused when its n and seed equal the
    requested ones; 'bootstrap_by_group' likewise, named after the grouping
    whose CSV is the evaluation's --groups file (same resolved path), or
    EVAL_GROUP_SCHEME if none is.

    Input:  ev — load_evaluation() result; groupings — {name: CSV};
            base_dir — directory of relative paths; n, seed — bootstrap settings
    Output: {scheme: ap.bootstrap_ci result} (possibly empty)
    """
    model, res = ev["model"], ev["result"]
    b = model.get("bootstrap")
    if not b or b.get("n") != n or b.get("seed") != seed:
        return {}
    out: Dict[str, Mapping[str, Any]] = {IMAGE_SCHEME: b}
    if "bootstrap_by_group" in model and res["settings"].get("groups"):
        g = Path(res["settings"]["groups"])
        g = (g if g.is_absolute() else Path(base_dir) / g).resolve()
        name = next((k for k, v in groupings.items() if Path(v).resolve() == g), EVAL_GROUP_SCHEME)
        out[name] = model["bootstrap_by_group"]
    return out


def _subset_block(ap_preds: Mapping[str, Any], pipe_preds: Optional[Mapping[str, Any]],
                  gts: Mapping[str, Any], names: Sequence[str],
                  groupings: Mapping[str, Mapping[str, str]], n: int, seed: int,
                  min_support: int, conf_op: float,
                  precomputed: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Any]:
    """Metrics and intervals of one evaluated set under several resampling schemes.

    Input:  ap_preds — AP predictions; pipe_preds — pipeline predictions or
            None; gts — ground truth (defines the set); names — class names;
            groupings — {scheme name: image id -> group}; n, seed — bootstrap;
            min_support — reliability threshold; conf_op — operating point;
            precomputed — {scheme: ap.bootstrap_ci result} reused instead of
            recomputed (e.g. intervals already in the evaluation file)
    Output: {'n_images', 'n_instances', 'n_units' {scheme: n}, 'mAP50',
             'mAP50_95', 'precision', 'recall' (points, max-F1 confidence),
             'mAP50_reliable_classes' {'classes', 'point'} (mean AP50 over the
             classes with >= min_support instances in this set; no interval),
             'ci' {scheme: {'mAP50': [lo, hi], 'mAP50_95': [lo, hi]}},
             'per_class' {name: {'support', 'n_images', 'ap50', 'ap50_95',
             'reliable', 'ci_ap50' {scheme: [lo, hi]}}},
             'operating_point_pipeline' {'tp', 'fp', 'fn', 'precision',
             'recall', 'f1' (points), 'ci' {scheme: {'precision', 'recall',
             'f1'}}, 'per_class'} or None}
    """
    m = apm.evaluate(ap_preds, gts, names, conf_op)
    schemes: Dict[str, Optional[Mapping[str, str]]] = {IMAGE_SCHEME: None, **groupings}
    boot: Dict[str, Mapping[str, Any]] = dict(precomputed or {})
    for s, g in schemes.items():
        if s not in boot:
            boot[s] = apm.bootstrap_ci(ap_preds, gts, names, n, seed,
                                       groups=None if g is None else {k: g[k] for k in gts})
    out: Dict[str, Any] = {
        "n_images": m["n_images"], "n_instances": m["n_instances"],
        "n_units": {s: b.get("n_units") for s, b in boot.items()},
        "mAP50": m["mAP50"], "mAP50_95": m["mAP50_95"],
        "precision": m["precision"], "recall": m["recall"],
        "mAP50_reliable_classes": None,
        "ci": {s: {"mAP50": _ci(b["mAP50"]), "mAP50_95": _ci(b["mAP50_95"])}
               for s, b in boot.items()},
        "per_class": {}}
    for c in names:
        pc = m["per_class"][c]
        out["per_class"][c] = {
            "support": pc["support"], "n_images": pc["n_images"], "ap50": pc["ap50"],
            "ap50_95": pc["ap50_95"], "reliable": pc["support"] >= min_support,
            "ci_ap50": {s: _ci(b["per_class_ap50"][c]) for s, b in boot.items()}}
    rel = [c for c in names if out["per_class"][c]["reliable"] and out["per_class"][c]["ap50"] is not None]
    out["mAP50_reliable_classes"] = {
        "classes": rel, "point": float(np.mean([out["per_class"][c]["ap50"] for c in rel])) if rel else None}
    if pipe_preds is None:
        out["operating_point_pipeline"] = None
        return out
    op_ci = {s: operating_point_bootstrap(pipe_preds, gts, names, conf_op, n=n, seed=seed,
                                          groups=None if g is None else {k: g[k] for k in gts})
             for s, g in schemes.items()}
    first = op_ci[IMAGE_SCHEME]
    out["operating_point_pipeline"] = {
        "conf": conf_op, "tp": first["tp"], "fp": first["fp"], "fn": first["fn"],
        "precision": first["precision"]["point"], "recall": first["recall"]["point"],
        "f1": first["f1"]["point"],
        "ci": {s: {k: _ci(o[k]) for k in ("precision", "recall", "f1")} for s, o in op_ci.items()},
        "per_class": first["per_class"]}
    return out


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def summarize_evaluations(result_paths: Sequence[str | Path], groupings: Mapping[str, Path],
                          base_dir: str | Path = ".", n: int = apm.DEFAULT_BOOTSTRAP_N,
                          seed: int = apm.DEFAULT_BOOTSTRAP_SEED,
                          min_support: int = DEFAULT_MIN_SUPPORT) -> Dict[str, Any]:
    """Summary of several evaluate_detector results (see the 'summary' command).

    Input:  result_paths — metrics JSONs named detector_<split>_<subset>.json
                (other names use the file stem as the subset); groupings —
                {name: manifest CSV with stem, group_id}; base_dir — directory
                the evaluations were run from; n, seed — bootstrap;
                min_support — reliability threshold
    Output: {'settings', 'evaluations': [{'file', 'split', 'subset',
             'predictions', 'cross_check', 'reproduction', **_subset_block}]}
    """
    from ui2compose.evaluation.evaluate_detector import load_groups

    group_maps = {name: load_groups(p) for name, p in groupings.items()}
    rows = []
    for rp in result_paths:
        ev = load_evaluation(rp, base_dir)
        res, model = ev["result"], ev["model"]
        mt = _RESULT_NAME.match(Path(rp).name)
        split, subset = (mt["split"], mt["subset"]) if mt else (None, Path(rp).stem)
        missing = {g: sorted(set(ev["gts"]) - set(m))[:3] for g, m in group_maps.items()
                   if set(ev["gts"]) - set(m)}
        if missing:
            raise ValueError(f"{rp}: images without a group: {missing}")
        pre = precomputed_intervals(ev, groupings, base_dir, n, seed)
        block = _subset_block(ev["ap_preds"], ev["pipe_preds"], ev["gts"], ev["names"],
                              group_maps, n, seed, min_support,
                              res["settings"].get("conf_op", apm.DEFAULT_CONF_OP),
                              precomputed=pre)
        block["reused_from_evaluation_file"] = sorted(pre)
        if EVAL_GROUP_SCHEME in block["ci"]:
            block["eval_groups_source"] = res["settings"].get("groups")
        vc = model.get("val_cross_check")
        rows.append({
            "file": str(rp), "split": split, "subset": subset,
            "images_spec": res["images"]["spec"], "images_listed": res["images"]["listed"],
            "predictions": {"ap": ev["ap_preds_file"], "pipeline": ev["pipe_preds_file"],
                            "settings": model["meta"].get("settings")},
            "cross_check": None if vc is None else {
                "ultralytics_val_mAP50": vc["ultralytics_val"]["mAP50"],
                "ultralytics_val_mAP50_95": vc["ultralytics_val"]["mAP50_95"],
                "abs_diff_metric_port": vc["abs_diff_metric_port"]},
            "reproduction": model.get("reproduction"),
            **block})
    return {"settings": {"bootstrap": n, "seed": seed, "min_support": min_support,
                         "groupings": {k: str(v) for k, v in groupings.items()},
                         "schemes": [IMAGE_SCHEME, EVAL_GROUP_SCHEME, *groupings],
                         "interval": "percentile, 95%"},
            "evaluations": rows}


def seen_unseen(result_path: str | Path, manifest_path: str | Path,
                groupings: Mapping[str, Path], base_dir: str | Path = ".",
                n: int = apm.DEFAULT_BOOTSTRAP_N, seed: int = apm.DEFAULT_BOOTSTRAP_SEED,
                min_support: int = DEFAULT_MIN_SUPPORT,
                diff_scheme: Optional[str] = None) -> Dict[str, Any]:
    """Seen vs. unseen breakdown of one full-split evaluation (see the 'seen-unseen' command).

    Input:  result_path — metrics JSON of the full split; manifest_path —
                app_split manifest (seen_in_train); groupings — {name: CSV};
                base_dir, n, seed, min_support — as summarize_evaluations;
                diff_scheme — grouping of the difference interval (default:
                the first of groupings, else image level)
    Output: {'settings', 'counts', 'subsets': {'all', 'seen', 'unseen':
             _subset_block}, 'difference_unseen_minus_seen':
             subset_difference_bootstrap result, 'rows': table rows (one per
             class plus 'ALL')}
    """
    from ui2compose.evaluation.evaluate_detector import load_groups

    ev = load_evaluation(result_path, base_dir)
    names, gts = ev["names"], ev["gts"]
    seen, unseen = split_seen_unseen(list(gts), load_manifest(manifest_path))
    group_maps = {name: load_groups(p) for name, p in groupings.items()}
    conf_op = ev["result"]["settings"].get("conf_op", apm.DEFAULT_CONF_OP)
    pre = precomputed_intervals(ev, groupings, base_dir, n, seed)
    pre.pop(EVAL_GROUP_SCHEME, None)
    subsets: Dict[str, Any] = {}
    for label, stems in (("all", sorted(gts)), ("seen", seen), ("unseen", unseen)):
        sel = set(stems)
        subsets[label] = _subset_block(
            {k: v for k, v in ev["ap_preds"].items() if k in sel},
            None if ev["pipe_preds"] is None else
            {k: v for k, v in ev["pipe_preds"].items() if k in sel},
            {k: gts[k] for k in stems}, names, group_maps, n, seed, min_support, conf_op,
            precomputed=pre if label == "all" else None)
    diff_scheme = diff_scheme or (next(iter(group_maps)) if group_maps else None)
    # Classes with enough instances in both subsets (a property of the ground truth only).
    reliable = [c for c in names if subsets["seen"]["per_class"][c]["support"] >= min_support
                and subsets["unseen"]["per_class"][c]["support"] >= min_support]
    diff = subset_difference_bootstrap(ev["ap_preds"], gts, names, seen, unseen, n, seed,
                                       groups=None if diff_scheme is None else group_maps[diff_scheme],
                                       classes=reliable or None)
    diff["scheme"] = diff_scheme or IMAGE_SCHEME
    schemes = [IMAGE_SCHEME, *group_maps]
    rows = []
    for c in names + ["ALL"]:
        row: Dict[str, Any] = {"class": c}
        for label in ("all", "seen", "unseen"):
            b = subsets[label]
            if c == "ALL":
                row[f"support_{label}"] = b["n_instances"]
                row[f"images_{label}"] = b["n_images"]
                row[f"ap50_{label}"] = b["mAP50"]
                row[f"ap50_95_{label}"] = b["mAP50_95"]
                for s in schemes:
                    row[f"ap50_{label}_ci_{s}"] = b["ci"][s]["mAP50"]
                op = b["operating_point_pipeline"]
                row[f"op_recall_{label}"] = None if op is None else op["recall"]
                row[f"op_precision_{label}"] = None if op is None else op["precision"]
            else:
                pc = b["per_class"][c]
                row[f"support_{label}"] = pc["support"]
                row[f"images_{label}"] = pc["n_images"]
                row[f"ap50_{label}"] = pc["ap50"]
                row[f"ap50_95_{label}"] = pc["ap50_95"]
                for s in schemes:
                    row[f"ap50_{label}_ci_{s}"] = pc["ci_ap50"][s]
                op = b["operating_point_pipeline"]
                row[f"op_recall_{label}"] = None if op is None else op["per_class"][c]["recall"]
                row[f"op_precision_{label}"] = None if op is None else op["per_class"][c]["precision"]
            row[f"reliable_{label}"] = row[f"support_{label}"] >= min_support
        d = diff["mAP50"] if c == "ALL" else diff["per_class_ap50"][c]
        row["ap50_diff_unseen_minus_seen"] = d["point"]
        row[f"ap50_diff_ci_{diff['scheme']}"] = [d["lo"], d["hi"]]
        rows.append(row)
    if "mAP50_classes" in diff:
        fc = diff["mAP50_classes"]
        row = {"class": "MEAN_RELIABLE_CLASSES"}
        for label, key in (("seen", "a"), ("unseen", "b")):
            row[f"support_{label}"] = sum(subsets[label]["per_class"][c]["support"]
                                          for c in fc["classes"])
            row[f"ap50_{label}"] = fc[key]["point"]
            row[f"ap50_{label}_ci_{diff['scheme']}"] = [fc[key]["lo"], fc[key]["hi"]]
            row[f"reliable_{label}"] = True
        row["ap50_diff_unseen_minus_seen"] = fc["diff"]["point"]
        row[f"ap50_diff_ci_{diff['scheme']}"] = [fc["diff"]["lo"], fc["diff"]["hi"]]
        row["note"] = "mean AP50 over " + ", ".join(fc["classes"])
        rows.append(row)
    return {"settings": {"result": str(result_path), "manifest": str(manifest_path),
                         "groupings": {k: str(v) for k, v in groupings.items()},
                         "bootstrap": n, "seed": seed, "min_support": min_support,
                         "conf_op": conf_op, "difference_scheme": diff["scheme"],
                         "seen_definition": "manifest seen_in_train = 1 (the screen's app_split "
                                            "group contains a train screen)"},
            "counts": {"all": len(gts), "seen": len(seen), "unseen": len(unseen)},
            "subsets": subsets, "difference_unseen_minus_seen": diff, "rows": rows}


# ---------------------------------------------------------------------------
# Bias of the bootstrap: percentile, basic and BCa intervals side by side
# ---------------------------------------------------------------------------

def _fixed_class_mean(ap50: np.ndarray, cidx: Sequence[int]) -> np.ndarray:
    """Mean AP50 over a fixed list of classes, per row (classes absent from a row skipped).

    Input:  ap50 — (..., n_classes) per-class AP50 (NaN = class absent);
            cidx — class ids of the fixed list
    Output: (...) array of means (NaN where none of the classes is present)
    """
    a = np.asarray(ap50, dtype=np.float64)[..., list(cidx)]
    ok = np.isfinite(a)
    cnt = ok.sum(-1)
    tot = np.where(ok, a, 0.0).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def _support(gts: Mapping[str, Any], stems: Sequence[str], n_classes: int) -> np.ndarray:
    """Ground-truth instances per class over a list of images.

    Input:  gts — ground truth per image; stems — image ids; n_classes — number of classes
    Output: (n_classes,) int array
    """
    cls = [np.asarray(gts[s].get("cls", []), dtype=np.float64).astype(np.int64) for s in stems]
    cat = np.concatenate(cls) if cls else np.zeros(0, np.int64)
    return np.bincount(cat, minlength=n_classes)[:n_classes]


def interval_check_set(ap_preds: Mapping[str, Any], gts: Mapping[str, Any], names: Sequence[str],
                       scheme: str, groups: Optional[Mapping[str, str]],
                       n: int = apm.DEFAULT_BOOTSTRAP_N, seed: int = apm.DEFAULT_BOOTSTRAP_SEED,
                       min_support: int = DEFAULT_MIN_SUPPORT, alpha: float = apm.DEFAULT_ALPHA,
                       bca: bool = True, classes: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Percentile, basic and BCa intervals of mAP50, mAP50-95 and a fixed-class mean AP50 of one set.

    The replicates are those of ap.bootstrap_ci with the same n, seed and
    resampling units, so the percentile interval equals the one the
    evaluation files report ('matches_bootstrap_ci' checks it). The fixed
    classes are the classes with at least min_support instances in the set
    (as 'mAP50_reliable_classes' of the summary) unless given.

    Input:  ap_preds, gts, names — see ap.evaluate(); scheme — name of the
            resampling scheme; groups — image id -> group (None: images);
            n, seed, alpha — bootstrap; min_support — fixed-class threshold;
            bca — also compute the jackknife and the BCa interval; classes —
            optional fixed class list
    Output: {'scheme', 'n_units', 'classes', 'matches_bootstrap_ci',
             'mAP50' / 'mAP50_95' / 'mAP50_classes': ap.interval_methods result}
    """
    nc = len(names)
    m = apm.evaluate(ap_preds, gts, names)
    fixed = list(classes) if classes else [c for c in names
                                           if m["per_class"][c]["support"] >= min_support]
    cidx = [list(names).index(c) for c in fixed]
    pc = [m["per_class"][c]["ap50"] for c in fixed]
    points = {"mAP50": m["mAP50"], "mAP50_95": m["mAP50_95"],
              "mAP50_classes": float(np.mean([v for v in pc if v is not None])) if pc else None}
    sub_g = None if groups is None else {k: groups[k] for k in gts}
    b = apm.bootstrap_ci(ap_preds, gts, names, n, seed, alpha=alpha, groups=sub_g,
                         keep_replicates=True)
    reps = {"mAP50": np.asarray(b["replicates"]["mAP50"], dtype=np.float64),
            "mAP50_95": np.asarray(b["replicates"]["mAP50_95"], dtype=np.float64),
            "mAP50_classes": _fixed_class_mean(
                np.asarray(b["replicates"]["per_class_ap50"], dtype=np.float64).reshape(-1, nc),
                cidx)}
    jack: Optional[Dict[str, np.ndarray]] = None
    if bca:
        jk = apm.jackknife(ap_preds, gts, names, sub_g)
        jack = {"mAP50": jk["mAP50"], "mAP50_95": jk["mAP50_95"],
                "mAP50_classes": _fixed_class_mean(jk["per_class_ap50"], cidx)}
    out: Dict[str, Any] = {"scheme": scheme, "n_units": b["n_units"], "n": n, "seed": seed,
                           "classes": fixed}
    for k in ("mAP50", "mAP50_95", "mAP50_classes"):
        out[k] = apm.interval_methods(reps[k], points[k], alpha, None if jack is None else jack[k])
    out["matches_bootstrap_ci"] = all(
        out[k]["percentile"] is not None
        and abs(out[k]["percentile"][0] - b[k]["lo"]) < 1e-12
        and abs(out[k]["percentile"][1] - b[k]["hi"]) < 1e-12 for k in ("mAP50", "mAP50_95"))
    return out


def interval_check_difference(ev: Mapping[str, Any], manifest_path: str | Path,
                              scheme: str, groups: Optional[Mapping[str, str]],
                              n: int = apm.DEFAULT_BOOTSTRAP_N,
                              seed: int = apm.DEFAULT_BOOTSTRAP_SEED,
                              min_support: int = DEFAULT_MIN_SUPPORT,
                              alpha: float = apm.DEFAULT_ALPHA) -> Dict[str, Any]:
    """Percentile and basic intervals (with the bootstrap bias) of the unseen - seen differences.

    Same replicates as the seen-unseen command (subset_difference_bootstrap:
    seen and unseen resampled independently, same n, seed and grouping), so
    its percentile intervals are reproduced. No BCa: the jackknife of a
    two-sample statistic is not implemented.

    Input:  ev — load_evaluation() result of a full split; manifest_path —
            app_split manifest (seen_in_train); scheme, groups — resampling
            scheme name and image id -> group (None: images); n, seed,
            min_support, alpha — as the seen-unseen command
    Output: {'scheme', 'counts', 'classes', 'mAP50', 'mAP50_95' (unseen -
             seen), 'mAP50_classes_seen', 'mAP50_classes_unseen',
             'mAP50_classes' (unseen - seen): ap.interval_methods results}
    """
    names, gts = ev["names"], ev["gts"]
    seen, unseen = split_seen_unseen(list(gts), load_manifest(manifest_path))
    sup_s, sup_u = _support(gts, seen, len(names)), _support(gts, unseen, len(names))
    fixed = [c for k, c in enumerate(names) if sup_s[k] >= min_support and sup_u[k] >= min_support]
    d = subset_difference_bootstrap(ev["ap_preds"], gts, names, seen, unseen, n, seed,
                                    groups=groups, alpha=alpha, classes=fixed or None,
                                    keep_replicates=True)
    r = d["replicates"]
    out: Dict[str, Any] = {"scheme": scheme, "counts": {"seen": len(seen), "unseen": len(unseen)},
                           "classes": fixed, "n": n, "seed": seed,
                           "mAP50": apm.interval_methods(r["mAP50"], d["mAP50"]["point"], alpha),
                           "mAP50_95": apm.interval_methods(r["mAP50_95"], d["mAP50_95"]["point"],
                                                            alpha)}
    if fixed:
        fc = d["mAP50_classes"]
        out["mAP50_classes_seen"] = apm.interval_methods(r["mAP50_classes_a"], fc["a"]["point"], alpha)
        out["mAP50_classes_unseen"] = apm.interval_methods(r["mAP50_classes_b"], fc["b"]["point"],
                                                           alpha)
        out["mAP50_classes"] = apm.interval_methods(r["mAP50_classes_diff"], fc["diff"]["point"],
                                                    alpha)
    out["matches_seen_unseen"] = all(
        abs(out[k]["percentile"][0] - d[k]["lo"]) < 1e-12
        and abs(out[k]["percentile"][1] - d[k]["hi"]) < 1e-12 for k in ("mAP50", "mAP50_95"))
    return out


def _interval_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    """One unit of work of interval_check (a set under one scheme, or a seen-unseen difference).

    Input:  task — {'kind' ('set' | 'difference'), 'result', 'base_dir',
            'scheme', 'groups_csv' (or None), 'manifest', 'n', 'seed',
            'min_support', 'bca'}
    Output: {'task': task without the large fields, 'value': result}
    """
    from ui2compose.evaluation.evaluate_detector import load_groups

    ev = load_evaluation(task["result"], task["base_dir"])
    groups = None if task["groups_csv"] is None else load_groups(task["groups_csv"])
    if task["kind"] == "set":
        value = interval_check_set(ev["ap_preds"], ev["gts"], ev["names"], task["scheme"], groups,
                                   task["n"], task["seed"], task["min_support"],
                                   bca=task["bca"])
        value.update({"n_images": len(ev["gts"])})
    else:
        value = interval_check_difference(ev, task["manifest"], task["scheme"], groups, task["n"],
                                          task["seed"], task["min_support"])
    return {"task": {k: str(v) if isinstance(v, Path) else v for k, v in task.items()},
            "value": value}


def interval_check(result_paths: Sequence[str | Path], groupings: Mapping[str, Path],
                   base_dir: str | Path = ".", n: int = apm.DEFAULT_BOOTSTRAP_N,
                   seed: int = apm.DEFAULT_BOOTSTRAP_SEED, min_support: int = DEFAULT_MIN_SUPPORT,
                   bca: bool = True, seen_unseen_results: Sequence[str | Path] = (),
                   manifest: Optional[str | Path] = None, diff_scheme: Optional[str] = None,
                   workers: int = 1) -> Dict[str, Any]:
    """Bias of the percentile bootstrap of AP, with basic and BCa intervals (the 'intervals' command).

    For every evaluation and resampling scheme (images and every grouping):
    interval_check_set; for every full-split evaluation in
    seen_unseen_results: interval_check_difference under diff_scheme.

    Input:  result_paths — evaluate_detector JSONs (detector_<split>_<subset>.json);
            groupings — {name: manifest CSV}; base_dir — directory the
            evaluations were run from; n, seed — bootstrap (use those of the
            evaluation files to reproduce their percentile intervals);
            min_support — fixed-class threshold; bca — compute BCa;
            seen_unseen_results — full-split evaluations for the seen /
            unseen difference; manifest — app_split manifest (seen_in_train);
            diff_scheme — grouping name of the difference (default: the
            first grouping); workers — processes
    Output: {'settings', 'sets': [{'file', 'split', 'subset', 'n_images',
             'schemes': {scheme: interval_check_set}}], 'differences':
             [{'file', 'split', ...interval_check_difference}]}
    """
    tasks: List[Dict[str, Any]] = []
    common = {"base_dir": str(base_dir), "n": n, "seed": seed, "min_support": min_support,
              "bca": bca, "manifest": None if manifest is None else str(manifest)}
    for rp in result_paths:
        for scheme, csv_path in {IMAGE_SCHEME: None, **groupings}.items():
            tasks.append({"kind": "set", "result": str(rp), "scheme": scheme,
                          "groups_csv": None if csv_path is None else str(csv_path), **common})
    if seen_unseen_results:
        if manifest is None:
            raise ValueError("the seen / unseen difference needs the manifest")
        diff_scheme = diff_scheme or (next(iter(groupings)) if groupings else IMAGE_SCHEME)
        csv_path = None if diff_scheme == IMAGE_SCHEME else groupings[diff_scheme]
        for rp in seen_unseen_results:
            tasks.append({"kind": "difference", "result": str(rp), "scheme": diff_scheme,
                          "groups_csv": None if csv_path is None else str(csv_path), **common})
    if workers > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(workers) as pool:
            done = pool.map(_interval_task, tasks, chunksize=1)
    else:
        done = [_interval_task(t) for t in tasks]
    sets: Dict[str, Dict[str, Any]] = {}
    diffs: List[Dict[str, Any]] = []
    for d in done:
        t, v = d["task"], d["value"]
        mt = _RESULT_NAME.match(Path(t["result"]).name)
        split, subset = (mt["split"], mt["subset"]) if mt else (None, Path(t["result"]).stem)
        if t["kind"] == "set":
            row = sets.setdefault(t["result"], {"file": t["result"], "split": split,
                                                "subset": subset, "n_images": v["n_images"],
                                                "schemes": {}})
            row["schemes"][t["scheme"]] = v
        else:
            diffs.append({"file": t["result"], "split": split, **v})
    return {"settings": {"bootstrap": n, "seed": seed, "min_support": min_support, "bca": bca,
                         "alpha": apm.DEFAULT_ALPHA,
                         "groupings": {k: str(v) for k, v in groupings.items()},
                         "schemes": [IMAGE_SCHEME, *groupings], "difference_scheme": diff_scheme,
                         "manifest": None if manifest is None else str(manifest),
                         "methods": {
                             "percentile": "[q(2.5 %), q(97.5 %)] of the replicates (as the "
                                           "evaluation files)",
                             "basic": "[2 point - q(97.5 %), 2 point - q(2.5 %)]",
                             "bca": "Efron's BCa: bias correction from the share of replicates "
                                    "below the point, acceleration from leave-one-unit-out "
                                    "(jackknife) values over the same resampling units; sets only",
                             "bias": "mean of the replicates - point"}},
            "sets": list(sets.values()), "differences": diffs}


def interval_rows(check: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Long-format table of an interval_check() result.

    Input:  check — interval_check() output
    Output: rows {'split', 'subset', 'scheme', 'metric', 'classes', 'point',
             'replicate_mean', 'bias', 'share_above', 'percentile', 'basic',
             'bca', 'bca_z0', 'bca_acceleration', 'n_units'}
    """
    rows: List[Dict[str, Any]] = []

    def add(base: Mapping[str, Any], metric: str, iv: Mapping[str, Any], classes: Sequence[str]
            ) -> None:
        """Appends one row.

        Input:  base — leading columns; metric — name; iv — interval_methods
                result; classes — fixed classes of the metric
        Output: none
        """
        rows.append({**base, "metric": metric,
                     "classes": "+".join(classes) if metric.startswith("mAP50_classes") else "ALL",
                     **{k: iv.get(k) for k in ("point", "replicate_mean", "bias", "share_above",
                                               "percentile", "basic", "bca", "bca_z0",
                                               "bca_acceleration")}})

    for s in check["sets"]:
        for scheme, v in s["schemes"].items():
            base = {"split": s["split"], "subset": s["subset"], "n_images": s["n_images"],
                    "scheme": scheme, "n_units": v["n_units"], "quantity": "set"}
            for metric in ("mAP50", "mAP50_95", "mAP50_classes"):
                add(base, metric, v[metric], v["classes"])
    for d in check["differences"]:
        base = {"split": d["split"], "subset": "unseen - seen", "n_images":
                d["counts"]["seen"] + d["counts"]["unseen"], "scheme": d["scheme"],
                "n_units": None, "quantity": "difference"}
        for metric in ("mAP50", "mAP50_95", "mAP50_classes_seen", "mAP50_classes_unseen",
                       "mAP50_classes"):
            if metric in d:
                add(base, metric, d[metric], d["classes"])
    return rows


def _fmt(v: Any) -> Any:
    """Formats a value for CSV (floats rounded to 5 decimals, [lo, hi] as 'lo;hi').

    Input:  v — value
    Output: CSV cell value
    """
    if isinstance(v, float):
        return f"{v:.5f}"
    if isinstance(v, list):
        return ";".join("" if x is None else f"{x:.5f}" for x in v)
    return "" if v is None else v


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Writes table rows (keys of the first row as columns, in order).

    Input:  path — CSV file; rows — dicts
    Output: none (file written)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols: List[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in cols})


def summary_rows(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Long-format table of a summarize_evaluations() result.

    Input:  summary — summarize_evaluations() output
    Output: rows {'split', 'subset', 'metric', 'class', 'point', 'support',
             'reliable', 'ci_<scheme>' ...} for mAP50, mAP50_95, AP50 per class
            and the pipeline operating point (precision, recall, f1)
    """
    rows: List[Dict[str, Any]] = []
    for ev in summary["evaluations"]:
        base = {"split": ev["split"], "subset": ev["subset"], "n_images": ev["n_images"]}
        for metric in ("mAP50", "mAP50_95"):
            rows.append({**base, "metric": metric, "class": "ALL", "point": ev[metric],
                         "support": ev["n_instances"], "reliable": True,
                         **{f"ci_{s}": v[metric] for s, v in ev["ci"].items()}})
        rc = ev.get("mAP50_reliable_classes")
        if rc and rc["point"] is not None:
            rows.append({**base, "metric": "mAP50_reliable_classes", "class": "+".join(rc["classes"]),
                         "point": rc["point"],
                         "support": sum(ev["per_class"][c]["support"] for c in rc["classes"]),
                         "reliable": True})
        for c, pc in ev["per_class"].items():
            rows.append({**base, "metric": "AP50", "class": c, "point": pc["ap50"],
                         "support": pc["support"], "reliable": pc["reliable"],
                         **{f"ci_{s}": v for s, v in pc["ci_ap50"].items()}})
        op = ev["operating_point_pipeline"]
        if op is not None:
            for metric in ("precision", "recall", "f1"):
                rows.append({**base, "metric": f"op{op['conf']:g}_{metric}", "class": "ALL",
                             "point": op[metric], "support": ev["n_instances"], "reliable": True,
                             **{f"ci_{s}": v[metric] for s, v in op["ci"].items()}})
    return rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Defines and reads the command-line arguments.

    Input:  argv — argument list (default sys.argv[1:])
    Output: argparse namespace
    """
    ap = argparse.ArgumentParser(description="Summaries of saved detector evaluations")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("summary", "seen-unseen", "intervals"):
        p = sub.add_parser(name)
        p.add_argument("--groups", action="append", default=[], metavar="NAME=CSV",
                       help="grouping for a cluster bootstrap (repeatable): manifest with "
                            "stem, group_id")
        p.add_argument("--base-dir", default=".",
                       help="directory the evaluations were run from (relative paths)")
        p.add_argument("--bootstrap", type=int, default=apm.DEFAULT_BOOTSTRAP_N)
        p.add_argument("--seed", type=int, default=apm.DEFAULT_BOOTSTRAP_SEED)
        p.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT)
        p.add_argument("--out-json", required=True)
        p.add_argument("--out-csv", required=True)
        if name == "summary":
            p.add_argument("results", nargs="+", help="evaluate_detector metrics JSONs")
        elif name == "intervals":
            p.add_argument("results", nargs="+", help="evaluate_detector metrics JSONs")
            p.add_argument("--seen-unseen", action="append", default=[], metavar="RESULT",
                           help="full-split metrics JSON whose unseen - seen differences are "
                                "checked as well (repeatable; needs --manifest)")
            p.add_argument("--manifest", default=None, help="app_split manifest (seen_in_train)")
            p.add_argument("--diff-groups", default=None,
                           help="grouping NAME of the differences (default: first --groups)")
            p.add_argument("--no-bca", action="store_true", help="skip the jackknife and BCa")
            p.add_argument("--workers", type=int, default=1)
        else:
            p.add_argument("result", help="evaluate_detector metrics JSON of a full split")
            p.add_argument("--manifest", required=True, help="app_split manifest (seen_in_train)")
            p.add_argument("--diff-groups", default=None,
                           help="grouping NAME of the difference interval (default: first --groups)")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Runs one command and writes its JSON and CSV.

    Input:  argv — argument list (default sys.argv[1:])
    Output: exit code 0; files --out-json and --out-csv
    """
    args = parse_args(argv)
    groupings = parse_groups_arg(args.groups)
    if args.command == "summary":
        res = summarize_evaluations(args.results, groupings, args.base_dir, args.bootstrap,
                                    args.seed, args.min_support)
        rows = summary_rows(res)
    elif args.command == "intervals":
        res = interval_check(args.results, groupings, args.base_dir, args.bootstrap, args.seed,
                             args.min_support, not args.no_bca, args.seen_unseen, args.manifest,
                             args.diff_groups, args.workers)
        rows = interval_rows(res)
    else:
        res = seen_unseen(args.result, args.manifest, groupings, args.base_dir, args.bootstrap,
                          args.seed, args.min_support, args.diff_groups)
        rows = res["rows"]
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    write_csv(args.out_csv, rows)
    print(f"Saved: {args.out_json}, {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
