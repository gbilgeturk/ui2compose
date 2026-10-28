# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Oversampling ablation of the detector on the same evaluation sets, with paired differences.

Added for the revision (R2.5 / R2.6: does oversampling still help on screens
of apps never seen in training?). Works on evaluate_detector outputs of
several checkpoints on the same image lists, e.g.

    results/eval/ablation/imgsz_640/detector_test_unseen_full.json   (no oversampling)
    results/eval/ablation/oversample_2k/detector_test_unseen_full.json
    results/eval/detector_test_unseen_full.json                       (released 5k weights)

so the models are not needed. For every set (split x subset) and every
resampling scheme (one per --groups grouping):

  * Replicates: those of ap.bootstrap_ci (default_rng(seed), the same units:
    whole app groups). They depend only on the image list, the grouping, n
    and seed, not on the predictions, so the replicates of the checkpoints
    are paired (each replicate draws the same screens for every checkpoint),
    and each checkpoint's percentile intervals are those of its evaluation
    file and of `detector_breakdown intervals` (checked, see below).
  * Metrics: mAP50 and mAP50-95 (Ultralytics' convention: the mean over the
    classes present), the mean AP50 over a fixed list of classes (the
    classes with at least --min-support ground-truth instances in the set;
    one list for all checkpoints, since the ground truth is the same),
    per-class AP50 with its support, and the pipeline operating point:
    micro precision / recall / F1 at IoU 0.5 of the pipeline's own
    detections (best-class-only NMS, conf > 0.3; evaluate_detector's
    '_pipeline' predictions).
  * Per checkpoint: ap.interval_methods (percentile, basic and BCa with the
    leave-one-unit-out jackknife over the same units, and the bootstrap
    bias) — the computation of the 'intervals' command.
  * Per pair A - B: the same on the per-replicate differences. The
    jackknife value of the difference without unit k is the difference of
    the two checkpoints' values without unit k (the same unit is left out),
    so the BCa interval applies to the paired difference as well. P(diff <=
    0) is the share of replicates with a difference <= 0.

Checks written to the output ('checks'): the checkpoints' evaluation files
list the same images and labels and were made with the same settings (AP
predictions: YOLO.val, conf 0.001, NMS IoU 0.7, max_det 300, the checkpoint's
imgsz; operating point at conf 0.3); each checkpoint's percentile intervals of
mAP50 / mAP50-95 equal those of its evaluation file under the file's own
grouping; the operating-point intervals equal those of
detector_breakdown.operating_point_bootstrap (same draws); and, with
--intervals-reference, every interval of a checkpoint whose evaluation file
the reference lists equals the reference's (percentile, basic and BCa).

Usage (from the repository root; EVAL_BASE as in scripts/run_evaluation.sh):
    python -m ui2compose.evaluation.detector_ablation --base-dir "$EVAL_BASE" \
        --model baseline=results/eval/ablation/imgsz_640 \
        --model oversample_2k=results/eval/ablation/oversample_2k \
        --model oversample_5k=results/eval \
        --groups package_md5=$PKG/splits/package_md5/manifest.csv \
        --groups full_grouping=$PKG/splits/full/manifest.csv \
        --intervals-reference results/eval/detector_intervals.json \
        --out-json results/eval/detector_ablation.json --out-csv ... --out-md ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ui2compose.evaluation import ap as apm
from ui2compose.evaluation import detector_breakdown as DB

# Evaluation sets (detector_<set>.json in every --model directory), in table order.
DEFAULT_SETS = ("test_full", "test_unseen_full", "test_unseen_package_only",
                "val_full", "val_unseen_full", "val_unseen_package_only")
# Pairs A-B (A minus B) of --model names.
DEFAULT_PAIRS = ("oversample_5k-baseline", "oversample_5k-oversample_2k", "oversample_2k-baseline")
SET_LABELS = {"full": "all screens", "unseen_full": "unseen apps (full grouping)",
              "unseen_package_md5": "unseen apps (package+MD5)",
              "unseen_package_only": "unseen apps (package only)"}
MODEL_LABELS = {"baseline": "no oversampling", "oversample_2k": "oversampled to 2k",
                "oversample_5k": "oversampled to 5k (released)"}
# Settings that must agree between the checkpoints' evaluation files.
_SETTING_KEYS = ("conf", "iou", "imgsz", "max_det", "pred_source", "multi_label", "conf_op")
# Scalar metrics of every checkpoint and pair (the per-class AP50 are added as 'AP50:<class>').
SCALAR_METRICS = ("mAP50_classes", "mAP50", "mAP50_95", "precision", "recall", "f1")
# Bumped whenever model_task's output changes (invalidates --cache-dir entries).
CACHE_VERSION = 1


def oversampled_classes() -> List[str]:
    """Names of the classes the training sets were oversampled for (oversample_dataset.RARE_CLASSES).

    Input:  none
    Output: class names in class-id order
    """
    from ui2compose.preprocess.oversample_dataset import CLASS_NAMES, RARE_CLASSES

    return [CLASS_NAMES[c] for c in sorted(RARE_CLASSES)]


def parse_named(specs: Sequence[str], what: str) -> Dict[str, str]:
    """Parses repeatable NAME=VALUE arguments, keeping their order.

    Input:  specs — strings 'NAME=VALUE'; what — option name for messages
    Output: {name: value}; raises ValueError on a malformed or repeated name
    """
    out: Dict[str, str] = {}
    for s in specs:
        name, sep, value = s.partition("=")
        if not sep or not name or not value:
            raise ValueError(f"{what}: expected NAME=VALUE, got {s!r}")
        if name in out:
            raise ValueError(f"{what}: {name!r} given twice")
        out[name] = value
    return out


def parse_pairs(specs: Sequence[str], models: Sequence[str]) -> List[Tuple[str, str]]:
    """Parses 'A-B' pair specifications (A minus B) against the model names.

    Input:  specs — strings 'A-B'; models — known model names
    Output: [(A, B), ...]; raises ValueError for an unknown name
    """
    pairs = []
    for s in specs:
        for k in range(1, len(s)):
            a, b = s[:k], s[k + 1:]
            if s[k] == "-" and a in models and b in models:
                pairs.append((a, b))
                break
        else:
            raise ValueError(f"pair {s!r}: expected A-B with A, B in {list(models)}")
    return pairs


def _draws(n_ids: int, units: Sequence[np.ndarray], grouped: bool, n: int, seed: int
           ) -> List[np.ndarray]:
    """The resampled image indices of ap.bootstrap_ci (same RNG calls in the same order).

    Input:  n_ids — number of images; units — resampling units
            (ap._resampling_units); grouped — True for a cluster bootstrap;
            n — replicates; seed — RNG seed
    Output: list of n index arrays
    """
    rng = np.random.default_rng(seed)
    full = np.arange(n_ids)
    out = []
    for _ in range(n):
        if not units:
            out.append(full)
        elif not grouped:
            out.append(rng.integers(0, n_ids, n_ids))
        else:
            out.append(np.concatenate([units[u] for u in rng.integers(0, len(units), len(units))]))
    return out


def _prf_rows(counts: np.ndarray) -> np.ndarray:
    """Micro precision / recall / F1 of rows of summed (TP, FP, FN) counts.

    Input:  counts — (..., 3) array
    Output: (..., 3) array of P, R, F1 (NaN where undefined)
    """
    p, r, f1 = DB._prf(counts[..., 0].astype(float), counts[..., 1].astype(float),
                       counts[..., 2].astype(float))
    return np.stack([p, r, f1], -1)


def _sha256(path: str | Path) -> str:
    """SHA-256 of a file.

    Input:  path — file
    Output: hex digest
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def task_cache_key(task: Mapping[str, Any]) -> str:
    """Key of a model_task result: its inputs (evaluation file, predictions, grouping, settings).

    The evaluation JSON and the grouping CSV enter by content, the two
    prediction files by size and modification time.

    Input:  task — model_task() argument
    Output: hex digest
    """
    res = json.loads(Path(task["result"]).read_text(encoding="utf-8"))
    ap_file = _resolved(res["models"][0]["predictions_file"], Path(task["base_dir"]))
    pipe_file = ap_file.with_name(f"{ap_file.stem}_pipeline.json")
    parts = {"version": CACHE_VERSION, "result": _sha256(task["result"]),
             "groups": None if task["groups_csv"] is None else _sha256(task["groups_csv"]),
             **{k: task[k] for k in ("set", "model", "scheme", "n", "seed", "conf_op")},
             "preds": [[str(f), f.stat().st_size, f.stat().st_mtime_ns] if f.is_file() else None
                       for f in (ap_file, pipe_file)]}
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def cached_model_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    """model_task() with an optional on-disk cache (task['cache_dir']; None: no cache).

    The cache holds this module's own results (pickle), keyed by
    task_cache_key(), so that the report can be rewritten, or a checkpoint
    added, without recomputing the replicates of the others.

    Input:  task — model_task() argument plus 'cache_dir'
    Output: model_task() result (with 'cached': True when read from the cache)
    """
    import pickle

    cache_dir = task.get("cache_dir")
    if not cache_dir:
        return model_task(task)
    key = task_cache_key(task)
    f = Path(cache_dir) / f"{task['set']}__{task['model']}__{task['scheme']}.pkl"
    if f.is_file():
        with open(f, "rb") as fh:
            stored = pickle.load(fh)
        if stored.get("key") == key:
            return {**stored["value"], "cached": True}
    value = model_task(task)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump({"key": key, "value": value}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(f)
    return value


def model_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Replicates and jackknife values of one checkpoint on one set under one scheme.

    Input:  task — {'set', 'model', 'result' (evaluation JSON), 'base_dir',
            'scheme', 'groups_csv' (None: images), 'n', 'seed', 'conf_op'}
    Output: {'set', 'model', 'scheme', 'ids' (sorted image ids), 'labels_hash',
             'n_instances', 'support', 'n_images_with_class', 'n_units',
             'points': {'mAP50', 'mAP50_95', 'ap50' (per class, NaN = absent),
             'op' [P, R, F1], 'op_counts' [TP, FP, FN]}, 'reps': {'mAP50',
             'mAP50_95', 'ap50' (n, nc), 'op' (n, 3)}, 'jack': same keys over
             the units, 'eval_meta', 'eval_settings', 'eval_groups',
             'eval_bootstrap_by_group', 'checks'}
    """
    from ui2compose.evaluation.evaluate_detector import load_groups

    ev = DB.load_evaluation(task["result"], task["base_dir"])
    names, gts = ev["names"], ev["gts"]
    nc = len(names)
    if ev["pipe_preds"] is None:
        raise FileNotFoundError(f"{task['result']}: pipeline predictions missing "
                                f"({ev['pipe_preds_file']})")
    groups = None
    if task["groups_csv"] is not None:
        allg = load_groups(task["groups_csv"])
        groups = {k: allg[k] for k in gts}
    n, seed = task["n"], task["seed"]
    b = apm.bootstrap_ci(ev["ap_preds"], gts, names, n, seed, groups=groups, keep_replicates=True)
    jk = apm.jackknife(ev["ap_preds"], gts, names, groups)
    ids, counts = DB.operating_point_counts(ev["pipe_preds"], gts, nc, task["conf_op"])
    units, _ = apm._resampling_units(ids, groups)
    per_img = counts.sum(1)                                   # (n_images, 3), micro
    tot = per_img.sum(0)
    op_reps = _prf_rows(np.stack([per_img[idx].sum(0) for idx in
                                  _draws(len(ids), units, groups is not None, n, seed)]))
    member = np.empty(len(ids), dtype=int)
    for u, idx in enumerate(units):
        member[idx] = u
    unit_counts = np.zeros((len(units), 3), dtype=np.int64)
    np.add.at(unit_counts, member, per_img)
    op_jack = _prf_rows(tot[None, :] - unit_counts)
    # Same draws as operating_point_bootstrap (its percentile intervals must be reproduced).
    opb = DB.operating_point_bootstrap(ev["pipe_preds"], gts, names, task["conf_op"], n=n,
                                       seed=seed, groups=groups)
    op_match = all(
        opb[k]["lo"] is not None
        and np.isclose(np.percentile(op_reps[np.isfinite(op_reps[:, j]), j], 2.5), opb[k]["lo"],
                       rtol=0, atol=1e-12)
        and np.isclose(np.percentile(op_reps[np.isfinite(op_reps[:, j]), j], 97.5), opb[k]["hi"],
                       rtol=0, atol=1e-12)
        for j, k in enumerate(("precision", "recall", "f1")))
    ap50_point = np.array([np.nan if b["per_class_ap50"][c]["point"] is None
                           else b["per_class_ap50"][c]["point"] for c in names], dtype=float)
    support = DB._support(gts, ids, nc)
    n_img_cls = np.zeros(nc, dtype=int)
    for s in ids:
        for c in set(int(x) for x in np.asarray(gts[s]["cls"], dtype=float)):
            n_img_cls[c] += 1
    # Fingerprint of the ground truth (labels) of the set, to check it is the same for every model.
    lab = json.dumps([[s, np.round(np.asarray(gts[s]["boxes"], dtype=float), 3).tolist(),
                       list(map(int, gts[s]["cls"]))] for s in ids])
    res, model = ev["result"], ev["model"]
    return {"set": task["set"], "model": task["model"], "scheme": task["scheme"],
            "names": list(names), "ids": ids, "labels_hash": hashlib.sha256(lab.encode()).hexdigest(),
            "n_instances": int(support.sum()), "support": support.tolist(),
            "n_images_with_class": n_img_cls.tolist(), "n_units": len(units),
            "points": {"mAP50": b["mAP50"]["point"], "mAP50_95": b["mAP50_95"]["point"],
                       "ap50": ap50_point, "op": _prf_rows(tot), "op_counts": tot.tolist()},
            "reps": {"mAP50": np.asarray(b["replicates"]["mAP50"], dtype=float),
                     "mAP50_95": np.asarray(b["replicates"]["mAP50_95"], dtype=float),
                     "ap50": np.asarray(b["replicates"]["per_class_ap50"], dtype=float).reshape(-1, nc),
                     "op": op_reps},
            "jack": {"mAP50": jk["mAP50"], "mAP50_95": jk["mAP50_95"],
                     "ap50": jk["per_class_ap50"], "op": op_jack},
            "percentile": {"mAP50": [b["mAP50"]["lo"], b["mAP50"]["hi"]],
                           "mAP50_95": [b["mAP50_95"]["lo"], b["mAP50_95"]["hi"]]},
            "eval_meta": {k: model["meta"].get(k) for k in ("weights", "weights_md5",
                                                            "ultralytics_version", "source")},
            "eval_imgsz": (model["meta"].get("settings") or {}).get("imgsz"),
            "eval_settings": {k: res["settings"].get(k) for k in _SETTING_KEYS},
            "eval_groups": res["settings"].get("groups"),
            "eval_bootstrap_by_group": {k: [model["bootstrap_by_group"][k]["lo"],
                                            model["bootstrap_by_group"][k]["hi"]]
                                        for k in ("mAP50", "mAP50_95")}
            if "bootstrap_by_group" in model else None,
            "eval_seed_n": [res["settings"].get("seed"), res["settings"].get("bootstrap")],
            "pipe_meta": {"file": ev["pipe_preds_file"]},
            "reproduction": model.get("reproduction"),
            "checks": {"op_matches_operating_point_bootstrap": bool(op_match)}}


def _metric_arrays(r: Mapping[str, Any], cidx: Sequence[int], names: Sequence[str]
                   ) -> Dict[str, Tuple[Any, np.ndarray, np.ndarray]]:
    """(point, replicates, jackknife values) of every metric of one model_task result.

    Input:  r — model_task() result; cidx — fixed class ids; names — class names
    Output: {metric: (point or None, replicate array, jackknife array)}
    """
    p, reps, jack = r["points"], r["reps"], r["jack"]
    fixed_point = (float(np.mean([p["ap50"][c] for c in cidx if np.isfinite(p["ap50"][c])]))
                   if any(np.isfinite(p["ap50"][c]) for c in cidx) else None)
    out = {"mAP50_classes": (fixed_point, DB._fixed_class_mean(reps["ap50"], cidx),
                             DB._fixed_class_mean(jack["ap50"], cidx)),
           "mAP50": (p["mAP50"], reps["mAP50"], jack["mAP50"]),
           "mAP50_95": (p["mAP50_95"], reps["mAP50_95"], jack["mAP50_95"])}
    for j, k in enumerate(("precision", "recall", "f1")):
        v = p["op"][j]
        out[k] = (None if not np.isfinite(v) else float(v), reps["op"][:, j], jack["op"][:, j])
    for c, name in enumerate(names):
        v = p["ap50"][c]
        out[f"AP50:{name}"] = (None if not np.isfinite(v) else float(v), reps["ap50"][:, c],
                               jack["ap50"][:, c])
    return out


def _with_p(iv: Dict[str, Any], reps: np.ndarray) -> Dict[str, Any]:
    """Adds P(diff <= 0) (share of the valid replicates <= 0) to an interval_methods result.

    Input:  iv — interval_methods() result; reps — replicate differences
    Output: iv with 'p_le_0'
    """
    v = reps[np.isfinite(reps)]
    iv["p_le_0"] = float(np.mean(v <= 0)) if v.size else None
    return iv


def combine(results: Sequence[Mapping[str, Any]], models: Sequence[str],
            pairs: Sequence[Tuple[str, str]], min_support: int, alpha: float,
            focus_classes: Sequence[str]) -> Dict[str, Any]:
    """Intervals of every checkpoint and every pair on one set under one scheme.

    Input:  results — model_task() results of the same set and scheme, one per
            model; models — model order; pairs — (A, B) pairs (A - B);
            min_support — fixed-class threshold; alpha — two-sided level;
            focus_classes — classes reported per class (e.g. the oversampled ones)
    Output: {'n_units', 'classes_fixed', 'models': {name: {metric: interval}},
             'pairs': {'A - B': {metric: interval + p_le_0}}, 'checks'}
    """
    by = {r["model"]: r for r in results}
    ref = by[models[0]]
    names = ref["names"]
    support = ref["support"]
    fixed = [c for k, c in enumerate(names) if support[k] >= min_support]
    cidx = [names.index(c) for c in fixed]
    keep = list(SCALAR_METRICS) + [f"AP50:{c}" for c in focus_classes]
    arrays = {m: _metric_arrays(by[m], cidx, names) for m in models}
    out_models: Dict[str, Any] = {}
    for m in models:
        out_models[m] = {}
        for k in keep:
            point, reps, jack = arrays[m][k]
            out_models[m][k] = apm.interval_methods(reps, point, alpha, jack)
    out_pairs: Dict[str, Any] = {}
    for a, b in pairs:
        d: Dict[str, Any] = {}
        for k in keep:
            pa, ra, ja = arrays[a][k]
            pb, rb, jb = arrays[b][k]
            point = None if pa is None or pb is None else pa - pb
            d[k] = _with_p(apm.interval_methods(ra - rb, point, alpha, ja - jb), ra - rb)
        out_pairs[f"{a} - {b}"] = d
    checks = {
        "same_images": all(by[m]["ids"] == ref["ids"] for m in models),
        "same_labels": all(by[m]["labels_hash"] == ref["labels_hash"] for m in models),
        "same_units": all(by[m]["n_units"] == ref["n_units"] for m in models),
        "op_matches_operating_point_bootstrap": all(
            by[m]["checks"]["op_matches_operating_point_bootstrap"] for m in models)}
    return {"n_units": ref["n_units"], "classes_fixed": fixed, "models": out_models,
            "pairs": out_pairs, "checks": checks}


def _resolved(p: Optional[str], base: Path) -> Optional[Path]:
    """Resolves a recorded path against a base directory.

    Input:  p — path string or None; base — base directory
    Output: resolved Path or None
    """
    if p is None:
        return None
    q = Path(p)
    return (q if q.is_absolute() else base / q).resolve()


def _reference_checks(reference: Mapping[str, Any], model_files: Mapping[Tuple[str, str], str],
                      sets: Mapping[str, Any], tol: float = 1e-12) -> List[Dict[str, Any]]:
    """Compares the checkpoints' intervals with those of an 'intervals' command output.

    Input:  reference — detector_breakdown intervals JSON; model_files —
            {(set, model): evaluation file as given}; sets — this run's
            sets ({set: {'schemes': {scheme: combine()}}}); tol — tolerance
    Output: rows {'set', 'model', 'scheme', 'metric', 'ok', 'max_abs_diff'}
            for every (file, scheme) the reference holds
    """
    ref_sets = {Path(s["file"]).as_posix(): s for s in reference.get("sets", [])}
    rows = []
    for (set_key, model), f in model_files.items():
        rs = ref_sets.get(Path(f).as_posix())
        if rs is None:
            continue
        for scheme, comb in sets[set_key]["schemes"].items():
            rv = rs["schemes"].get(scheme)
            if rv is None:
                continue
            if rv["classes"] != comb["classes_fixed"]:
                rows.append({"set": set_key, "model": model, "scheme": scheme,
                             "metric": "classes", "ok": False, "max_abs_diff": None})
                continue
            for metric in ("mAP50", "mAP50_95", "mAP50_classes"):
                mine, theirs = comb["models"][model][metric], rv[metric]
                diffs = []
                for k in ("point", "percentile", "basic", "bca"):
                    a, b = mine.get(k), theirs.get(k)
                    if a is None or b is None:
                        diffs.append(0.0 if a is None and b is None else np.inf)
                        continue
                    diffs.append(float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float)))))
                rows.append({"set": set_key, "model": model, "scheme": scheme, "metric": metric,
                             "ok": max(diffs) <= tol, "max_abs_diff": max(diffs)})
    return rows


def run_ablation(models: Mapping[str, str], sets: Sequence[str], groupings: Mapping[str, Path],
                 pairs: Sequence[Tuple[str, str]], base_dir: str | Path = ".",
                 n: int = apm.DEFAULT_BOOTSTRAP_N, seed: int = apm.DEFAULT_BOOTSTRAP_SEED,
                 min_support: int = DB.DEFAULT_MIN_SUPPORT, alpha: float = apm.DEFAULT_ALPHA,
                 conf_op: float = apm.DEFAULT_CONF_OP, focus_classes: Optional[Sequence[str]] = None,
                 workers: int = 1, reference: Optional[Mapping[str, Any]] = None,
                 primary_scheme: Optional[str] = None, cache_dir: Optional[str | Path] = None
                 ) -> Dict[str, Any]:
    """The ablation over every set, scheme and checkpoint (see the module docstring).

    Input:  models — {name: directory holding detector_<set>.json} in table
            order; sets — set keys ('<split>_<subset>'); groupings — {scheme:
            manifest CSV}; pairs — (A, B) pairs; base_dir — directory the
            evaluations were run from; n, seed, alpha — bootstrap (use those
            of the evaluation files); min_support — fixed-class threshold;
            conf_op — operating threshold; focus_classes — classes reported per
            class (default: the oversampled ones); workers — processes;
            reference — optional 'intervals' command output to check against;
            primary_scheme — scheme the report leads with (default: first);
            cache_dir — optional cache of the per-checkpoint replicates
    Output: {'settings', 'checkpoints', 'sets': {set: {...}}, 'checks'}
    """
    focus = list(focus_classes) if focus_classes else oversampled_classes()
    model_names = list(models)
    tasks, model_files = [], {}
    for s in sets:
        for m, d in models.items():
            f = str(Path(d) / f"detector_{s}.json")
            model_files[(s, m)] = f
            for scheme, csv_path in groupings.items():
                tasks.append({"set": s, "model": m, "result": f, "base_dir": str(base_dir),
                              "scheme": scheme, "groups_csv": str(csv_path), "n": n,
                              "seed": seed, "conf_op": conf_op,
                              "cache_dir": None if cache_dir is None else str(cache_dir)})
    # Heaviest first (the full splits), so that the pool is not left with one long task.
    tasks.sort(key=lambda t: (t["set"].split("_", 1)[1] != "full", t["set"]))
    if workers > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(workers) as pool:
            done = pool.map(cached_model_task, tasks, chunksize=1)
    else:
        done = [cached_model_task(t) for t in tasks]
    base = Path(base_dir)
    out_sets: Dict[str, Any] = {}
    checkpoints: Dict[str, Any] = {}
    settings_seen: Dict[str, Dict[str, Any]] = {}
    file_checks: List[Dict[str, Any]] = []
    for s in sets:
        split, subset = s.split("_", 1)
        entry: Dict[str, Any] = {"split": split, "subset": subset,
                                 "label": SET_LABELS.get(subset, subset), "schemes": {}}
        for scheme, csv_path in groupings.items():
            rs = [r for r in done if r["set"] == s and r["scheme"] == scheme]
            rs.sort(key=lambda r: model_names.index(r["model"]))
            ref = rs[0]
            entry.update({"n_images": len(ref["ids"]), "n_instances": ref["n_instances"],
                          "support": dict(zip(ref["names"], ref["support"])),
                          "n_images_with_class": dict(zip(ref["names"],
                                                          ref["n_images_with_class"]))})
            comb = combine(rs, model_names, pairs, min_support, alpha, focus)
            comb["op_counts"] = {r["model"]: dict(zip(("tp", "fp", "fn"), r["points"]["op_counts"]))
                                 for r in rs}
            entry["schemes"][scheme] = comb
            for r in rs:
                settings_seen.setdefault(f"{s}:{r['model']}",
                                         {**r["eval_settings"], "imgsz_effective": r["eval_imgsz"],
                                          "seed_n": r["eval_seed_n"]})
                ck = checkpoints.setdefault(r["model"], {"label": MODEL_LABELS.get(r["model"]),
                                                         "dir": models[r["model"]],
                                                         **r["eval_meta"], "reproduction": {}})
                if r["eval_meta"]["weights_md5"] != ck["weights_md5"]:
                    ck.setdefault("md5_mismatch", []).append(s)
                if subset == "full" and r["reproduction"] is not None:
                    rep = r["reproduction"]
                    ck["reproduction"][split] = {
                        "source": rep["source"], "ok": rep["ok"], "tolerance": rep["tolerance"],
                        **{k: rep["overall"][k] for k in ("mAP50", "mAP50_95")},
                        "max_abs_class_ap50_diff": max(
                            (v["abs_diff"] for v in rep["per_class_ap50"].values()
                             if v["abs_diff"] is not None), default=None)}
                # Percentile intervals of the evaluation file (its own grouping) reproduced?
                eg = _resolved(r["eval_groups"], base)
                if r["eval_bootstrap_by_group"] is not None and eg == Path(csv_path).resolve() \
                        and r["eval_seed_n"] == [seed, n]:
                    diff = max(abs(a - b) for k in ("mAP50", "mAP50_95")
                               for a, b in zip(r["percentile"][k], r["eval_bootstrap_by_group"][k]))
                    file_checks.append({"set": s, "model": r["model"], "scheme": scheme,
                                        "ok": diff <= 1e-12, "max_abs_diff": diff})
        out_sets[s] = entry
    distinct = {json.dumps({k: v for k, v in st.items()}, sort_keys=True)
                for st in settings_seen.values()}
    ref_rows = _reference_checks(reference, model_files, out_sets) if reference else []
    checks = {
        "settings_identical": len(distinct) == 1,
        "settings": json.loads(sorted(distinct)[0]) if len(distinct) == 1 else settings_seen,
        "same_images_labels_units": all(c for st in out_sets.values() for sc in st["schemes"].values()
                                        for c in (sc["checks"]["same_images"],
                                                  sc["checks"]["same_labels"],
                                                  sc["checks"]["same_units"])),
        "op_matches_operating_point_bootstrap": all(
            sc["checks"]["op_matches_operating_point_bootstrap"]
            for st in out_sets.values() for sc in st["schemes"].values()),
        "percentile_matches_evaluation_files": {"n": len(file_checks),
                                                "ok": all(c["ok"] for c in file_checks),
                                                "rows": file_checks},
        "matches_intervals_reference": {"n": len(ref_rows), "ok": all(c["ok"] for c in ref_rows),
                                        "rows": ref_rows},
    }
    return {"settings": {"bootstrap": n, "seed": seed, "alpha": alpha, "min_support": min_support,
                         "conf_op": conf_op, "iou_op": apm.DEFAULT_IOU_OP,
                         "groupings": {k: str(v) for k, v in groupings.items()},
                         "primary_scheme": primary_scheme or next(iter(groupings)),
                         "models": dict(models), "sets": list(sets),
                         "pairs": [f"{a} - {b}" for a, b in pairs],
                         "focus_classes": focus, "base_dir": str(base_dir),
                         "methods": {
                             "replicates": "ap.bootstrap_ci (cluster bootstrap over the grouping; "
                                           "the same draws for every checkpoint: paired)",
                             "intervals": "ap.interval_methods: percentile, basic, BCa (jackknife "
                                          "over the same units), bias = replicate mean - point",
                             "differences": "per-replicate A - B; jackknife of the difference = "
                                            "difference of the jackknife values; p_le_0 = share of "
                                            "replicates <= 0",
                             "mAP50_classes": f"mean AP50 over the classes with >= {min_support} "
                                              "ground-truth instances in the set",
                             "operating point": f"micro P / R / F1, pipeline detections "
                                                f"(best-class NMS, conf > {conf_op}), IoU "
                                                f"{apm.DEFAULT_IOU_OP}"}},
            "checkpoints": checkpoints, "sets": out_sets, "checks": checks}


def checkpoint_provenance(checkpoints: Mapping[str, str | Path],
                          configs: Optional[Mapping[str, str | Path]] = None) -> Dict[str, Any]:
    """What the checkpoints record about their training (to identify the baseline).

    Reads each checkpoint (torch.load; needs ultralytics for the pickled
    model) and reports its hashes, the Ultralytics version and date it was
    written with, the dataset, initial weights and resume entries of its
    train_args, the validation metrics stored with it, which train_args differ
    between the checkpoints, and, for every NAME in configs, whether the
    `train:` section of that configuration equals the checkpoint's train_args.

    Input:  checkpoints — {name: .pt path}; configs — optional {name: train yaml}
    Output: {'checkpoints': {name: {...}}, 'train_args_differing_keys': {key:
             {name: value}}, 'n_train_args'}
    """
    import torch
    import yaml

    from ui2compose.evaluation.evaluate_detector import md5_of

    info: Dict[str, Any] = {}
    args: Dict[str, Dict[str, Any]] = {}
    for name, path in checkpoints.items():
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        ta = dict(ck.get("train_args") or {})
        args[name] = ta
        tm = ck.get("train_metrics") or {}
        entry = {"file": str(path), "md5": md5_of(path), "sha256": _sha256(path),
                 "written_with_ultralytics": ck.get("version"), "date": ck.get("date"),
                 **{k: ta.get(k) for k in ("data", "model", "resume", "pretrained", "name",
                                           "project", "imgsz", "epochs", "seed")},
                 "stored_val_metrics": {k: tm.get(f"metrics/{k}(B)") for k in
                                        ("mAP50", "mAP50-95", "precision", "recall")}}
        if configs and name in configs:
            train = (yaml.safe_load(Path(configs[name]).read_text(encoding="utf-8")) or {})["train"]
            mism = {k: [v, ta.get(k, "<absent>")] for k, v in train.items()
                    if ta.get(k, "<absent>") != v}
            entry["config"] = {"file": str(configs[name]), "n_keys": len(train),
                               "mismatches": mism, "equal": not mism}
        info[name] = entry
        del ck
    keys = sorted(set().union(*[set(a) for a in args.values()])) if args else []
    differing = {k: {n: args[n].get(k) for n in args} for k in keys
                 if len({json.dumps(args[n].get(k), default=str) for n in args}) > 1}
    return {"checkpoints": info, "train_args_differing_keys": differing,
            "n_train_args": {n: len(a) for n, a in args.items()}}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def ablation_rows(res: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Long-format table of a run_ablation() result.

    Input:  res — run_ablation() output
    Output: rows {'split', 'subset', 'n_images', 'scheme', 'n_units',
             'quantity' ('checkpoint' | 'difference'), 'name', 'metric', 'class',
             'support', 'n_images_with_class', 'reliable', 'point', 'percentile',
             'basic', 'bca', 'bias', 'p_le_0'}
    """
    rows: List[Dict[str, Any]] = []
    ms = res["settings"]["min_support"]
    for s, st in res["sets"].items():
        for scheme, sc in st["schemes"].items():
            base = {"split": st["split"], "subset": st["subset"], "n_images": st["n_images"],
                    "scheme": scheme, "n_units": sc["n_units"]}
            for quantity, block in (("checkpoint", sc["models"]), ("difference", sc["pairs"])):
                for name, metrics in block.items():
                    for metric, iv in metrics.items():
                        if metric.startswith("AP50:"):
                            cls = metric.split(":", 1)[1]
                            sup, nimg = st["support"][cls], st["n_images_with_class"][cls]
                            metric_name, reliable = "AP50", sup >= ms
                        else:
                            cls = "+".join(sc["classes_fixed"]) if metric == "mAP50_classes" else "ALL"
                            sup, nimg, metric_name, reliable = st["n_instances"], None, metric, True
                        rows.append({**base, "quantity": quantity, "name": name,
                                     "metric": metric_name, "class": cls, "support": sup,
                                     "n_images_with_class": nimg, "reliable": reliable,
                                     **{k: iv.get(k) for k in ("point", "percentile", "basic",
                                                               "bca", "bias")},
                                     "p_le_0": iv.get("p_le_0")})
    return rows


def _f(v: Optional[float], d: int = 3, signed: bool = False) -> str:
    """Formats a number for the Markdown tables ('–' for None).

    Input:  v — value; d — decimals; signed — always print the sign
    Output: string
    """
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    s = f"{v:+.{d}f}" if signed else f"{v:.{d}f}"
    return s.replace("-", "−")


def _ci(iv: Mapping[str, Any], kind: str = "bca", signed: bool = False) -> str:
    """'point [lo, hi]' of an interval_methods result.

    Input:  iv — interval_methods() result; kind — 'bca', 'percentile' or
            'basic' (BCa falls back to percentile, marked '*'); signed — signs
    Output: string
    """
    lohi, mark = iv.get(kind), ""
    if lohi is None and kind == "bca":
        lohi, mark = iv.get("percentile"), "*"
    if lohi is None:
        return _f(iv.get("point"), signed=signed)
    return (f"{_f(iv.get('point'), signed=signed)} [{_f(lohi[0], signed=signed)}, "
            f"{_f(lohi[1], signed=signed)}]{mark}")


def ablation_markdown(res: Mapping[str, Any]) -> str:
    """Markdown report: Table-2-ready rows, paired differences, oversampled classes, checks.

    Input:  res — run_ablation() output
    Output: Markdown text
    """
    st0 = res["settings"]
    sch = st0["primary_scheme"]
    models = list(st0["models"])
    ms = st0["min_support"]
    L: List[str] = []
    L.append("# Detector ablation: oversampling on all screens and on unseen apps\n")
    L.append("Written by `python -m ui2compose.evaluation.detector_ablation` "
             "(`scripts/run_evaluation.sh detector_ablation`). No retraining: the three "
             "checkpoints of the thesis ablation, evaluated with the settings of the released "
             "model (AP: YOLO.val predictions, conf 0.001, NMS IoU 0.7, max_det 300, imgsz 640; "
             f"operating point: the pipeline's detections at conf > {st0['conf_op']:g}, IoU "
             f"{st0['iou_op']:g}, micro). Intervals: 95 % BCa over `{sch}` app groups, "
             f"{st0['bootstrap']} paired replicates (seed {st0['seed']}); every checkpoint is "
             "evaluated on the same draws, so differences are paired. `*` = BCa undefined, "
             "percentile interval shown.\n")
    L.append("## Checkpoints\n")
    L.append("| name | weights (MD5) | Ultralytics (eval) | published test mAP50 → here | "
             "published val mAP50 → here |")
    L.append("|---|---|---|---|---|")
    for m in models:
        ck = res["checkpoints"].get(m, {})
        rep = ck.get("reproduction", {})

        def r(split: str) -> str:
            """Published vs reproduced mAP50 of one split.

            Input:  split — 'test' or 'val'
            Output: table cell
            """
            x = rep.get(split)
            if not x:
                return "–"
            e, g = x["mAP50"]["expected"], x["mAP50"]["got"]
            return f"{100 * e:.2f} → {100 * g:.2f} (Δ {100 * (g - e):+.2f} pt)".replace("-", "−")

        L.append(f"| {m} ({ck.get('label') or ''}) | `{ck.get('weights')}` "
                 f"({(ck.get('weights_md5') or '')[:8]}…) | {ck.get('ultralytics_version')} | "
                 f"{r('test')} | {r('val')} |")
    L.append("")
    prov = res.get("provenance")
    if prov:
        L.append("What the checkpoints store about their training (`train_args`, the "
                 "validation metrics saved with the weights):\n")
        L.append("| name | written with | train data | initial weights (`model`) | resume | "
                 "stored val mAP50 | training config equal |")
        L.append("|---|---|---|---|---|---|---|")
        for m, ck in prov["checkpoints"].items():
            cfg = ck.get("config")
            cfg_s = "–" if not cfg else (f"`{cfg['file']}`: {'yes' if cfg['equal'] else 'NO'} "
                                         f"({cfg['n_keys']} keys)")
            L.append(f"| {m} | Ultralytics {ck['written_with_ultralytics']}, {ck['date'][:10]} | "
                     f"`{ck['data']}` | `{ck['model']}` | `{ck['resume']}` | "
                     f"{_f(ck['stored_val_metrics']['mAP50'], 5)} | {cfg_s} |")
        L.append("")
        L.append("train_args that differ between the checkpoints: " + ", ".join(
            f"`{k}`" for k in prov["train_args_differing_keys"]) + " (all others equal; "
                 + ", ".join(f"{n}: {v}" for n, v in prov["n_train_args"].items()) + " keys).\n")
    L.append("## Table 2 rows (checkpoint × set)\n")
    L.append(f"Mean AP50 over the classes with ≥ {ms} instances in the set (the headline metric "
             "of Table 2), mAP50 and mAP50-95 with BCa intervals; P / R / F1 at the pipeline's "
             "operating point, F1 with its percentile interval (as in Table 2; the BCa "
             "intervals of P / R / F1 are in the JSON and CSV).\n")
    L.append(f"| set | screens (groups) | checkpoint | mean AP50, classes ≥ {ms} inst. | mAP50 | "
             "mAP50-95 | P / R / F1 (F1 interval) |")
    L.append("|---|---|---|---|---|---|---|")
    for s, st in res["sets"].items():
        sc = st["schemes"][sch]
        for k, m in enumerate(models):
            mm = sc["models"][m]
            lead = (f"{st['split']}, {st['label']} | {st['n_images']:,} ({sc['n_units']:,}) "
                    if k == 0 else " | ")
            L.append(f"| {lead}| {m} | {_ci(mm['mAP50_classes'])} ({len(sc['classes_fixed'])}) | "
                     f"{_ci(mm['mAP50'])} | {_ci(mm['mAP50_95'])} | "
                     f"{_f(mm['precision']['point'])} / {_f(mm['recall']['point'])} / "
                     f"{_ci(mm['f1'], 'percentile')} |")
    L.append("")
    L.append("## Paired differences\n")
    L.append("Point [95 % BCa], and P = share of the replicates with a difference ≤ 0.\n")
    L.append(f"| set | difference | Δ mean AP50 (classes ≥ {ms}) | P | Δ mAP50 | P | Δ mAP50-95 | "
             "Δ precision | Δ recall | Δ F1 | P |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for s, st in res["sets"].items():
        sc = st["schemes"][sch]
        for k, (pname, d) in enumerate(sc["pairs"].items()):
            lead = f"{st['split']}, {st['label']} ({st['n_images']:,})" if k == 0 else ""
            L.append(f"| {lead} | {pname} | {_ci(d['mAP50_classes'], signed=True)} | "
                     f"{_f(d['mAP50_classes']['p_le_0'], 3)} | {_ci(d['mAP50'], signed=True)} | "
                     f"{_f(d['mAP50']['p_le_0'], 3)} | {_ci(d['mAP50_95'], signed=True)} | "
                     f"{_ci(d['precision'], signed=True)} | {_ci(d['recall'], signed=True)} | "
                     f"{_ci(d['f1'], signed=True)} | {_f(d['f1']['p_le_0'], 3)} |")
    L.append("")
    L.append("## The eight oversampled classes\n")
    L.append(f"AP50 per checkpoint and the paired differences [BCa]. Support = ground-truth "
             f"instances (screens). **†** = fewer than {ms} instances: AP50 rests on a handful "
             "of boxes and its interval is not informative.\n")
    pair_names = list(next(iter(res["sets"].values()))["schemes"][sch]["pairs"])
    L.append("| set | class | support (screens) | " + " | ".join(models) + " | "
             + " | ".join(f"Δ {p}" for p in pair_names) + " |")
    L.append("|---|---|---|" + "---|" * (len(models) + len(pair_names)))
    for s, st in res["sets"].items():
        sc = st["schemes"][sch]
        for k, c in enumerate(st0["focus_classes"]):
            sup, nimg = st["support"][c], st["n_images_with_class"][c]
            flag = "" if sup >= ms else " †"
            lead = f"{st['split']}, {st['label']}" if k == 0 else ""
            vals = " | ".join(_f(sc["models"][m][f"AP50:{c}"]["point"]) for m in models)
            diffs = " | ".join(_ci(sc["pairs"][p][f"AP50:{c}"], signed=True) if sup else "–"
                               for p in pair_names)
            L.append(f"| {lead} | {c}{flag} | {sup} ({nimg}) | {vals} | {diffs} |")
    L.append("")
    ch = res["checks"]
    L.append("## Checks\n")
    L.append(f"* Evaluation settings identical for every checkpoint and set: "
             f"{ch['settings_identical']} ({json.dumps(ch['settings']) if ch['settings_identical'] else 'see JSON'}).")
    L.append(f"* Same images, labels and resampling units for every checkpoint: "
             f"{ch['same_images_labels_units']}.")
    pm = ch["percentile_matches_evaluation_files"]
    L.append(f"* Percentile intervals of mAP50 / mAP50-95 equal those of the evaluation files "
             f"(their own grouping): {pm['ok']} ({pm['n']} comparisons).")
    L.append(f"* Operating-point intervals equal `operating_point_bootstrap` (same draws): "
             f"{ch['op_matches_operating_point_bootstrap']}.")
    mr = ch["matches_intervals_reference"]
    if mr["n"]:
        L.append(f"* Point, percentile, basic and BCa intervals equal the `intervals` command's "
                 f"output for the checkpoints it covers: {mr['ok']} ({mr['n']} comparisons).")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Defines and reads the command-line arguments.

    Input:  argv — argument list (default sys.argv[1:])
    Output: argparse namespace
    """
    ap = argparse.ArgumentParser(description="Oversampling ablation of the detector: "
                                             "checkpoints x sets with paired differences")
    ap.add_argument("--model", action="append", required=True, metavar="NAME=DIR",
                    help="checkpoint name and the directory of its detector_<set>.json "
                         "(repeatable; order = table order)")
    ap.add_argument("--sets", default=",".join(DEFAULT_SETS),
                    help="comma-separated set keys <split>_<subset>")
    ap.add_argument("--pairs", default=",".join(DEFAULT_PAIRS),
                    help="comma-separated A-B pairs (A minus B) of --model names")
    ap.add_argument("--groups", action="append", required=True, metavar="NAME=CSV",
                    help="grouping of the cluster bootstrap (repeatable; the first is the "
                         "primary one of the report)")
    ap.add_argument("--base-dir", default=".",
                    help="directory the evaluations were run from (relative paths inside them)")
    ap.add_argument("--bootstrap", type=int, default=apm.DEFAULT_BOOTSTRAP_N)
    ap.add_argument("--seed", type=int, default=apm.DEFAULT_BOOTSTRAP_SEED)
    ap.add_argument("--min-support", type=int, default=DB.DEFAULT_MIN_SUPPORT)
    ap.add_argument("--conf-op", type=float, default=apm.DEFAULT_CONF_OP)
    ap.add_argument("--classes", default=None,
                    help="comma-separated classes reported per class (default: the oversampled ones)")
    ap.add_argument("--intervals-reference", default=None, metavar="JSON",
                    help="detector_breakdown intervals output to check the intervals against")
    ap.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PT",
                    help="checkpoint file of a --model, to record what it stores about its "
                         "training (repeatable; needs torch and ultralytics)")
    ap.add_argument("--train-config", action="append", default=[], metavar="NAME=YAML",
                    help="training configuration to compare with the checkpoint's train_args")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cache-dir", default=None,
                    help="cache of the per-checkpoint replicates (e.g. $OUT/ablation_cache)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Runs the ablation and writes its JSON, CSV and Markdown.

    Input:  argv — argument list (default sys.argv[1:])
    Output: exit code 0 (4 if a check failed); files --out-json, --out-csv, --out-md
    """
    args = parse_args(argv)
    models = parse_named(args.model, "--model")
    groupings = {k: Path(v) for k, v in parse_named(args.groups, "--groups").items()}
    pairs = parse_pairs([p for p in args.pairs.split(",") if p], list(models))
    sets = [s for s in args.sets.split(",") if s]
    reference = (json.loads(Path(args.intervals_reference).read_text(encoding="utf-8"))
                 if args.intervals_reference else None)
    classes = [c for c in args.classes.split(",") if c] if args.classes else None
    res = run_ablation(models, sets, groupings, pairs, args.base_dir, args.bootstrap, args.seed,
                       args.min_support, conf_op=args.conf_op, focus_classes=classes,
                       workers=args.workers, reference=reference, cache_dir=args.cache_dir)
    if args.checkpoint:
        res["provenance"] = checkpoint_provenance(parse_named(args.checkpoint, "--checkpoint"),
                                                  parse_named(args.train_config, "--train-config"))
    for p in (args.out_json, args.out_csv, args.out_md):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    from ui2compose.evaluation.evaluate_detector import _json_default

    Path(args.out_json).write_text(json.dumps(res, indent=1, default=_json_default), encoding="utf-8")
    DB.write_csv(args.out_csv, ablation_rows(res))
    Path(args.out_md).write_text(ablation_markdown(res), encoding="utf-8")
    ch = res["checks"]
    ok = (ch["settings_identical"] and ch["same_images_labels_units"]
          and ch["op_matches_operating_point_bootstrap"]
          and ch["percentile_matches_evaluation_files"]["ok"]
          and ch["matches_intervals_reference"]["ok"])
    print(("[OK] all checks passed" if ok else "[FAIL] a check failed, see 'checks' in the JSON"))
    print(f"Saved: {args.out_json}, {args.out_csv}, {args.out_md}")
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
