# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Threshold selection on the validation split: grid sweep of the v1.2.0 post-processing and layout thresholds.

Typical use (validation split only; the test split is refused):
    python -m ui2compose.evaluation.layout_sweep run --data-dir <package> --split val \
        --pred-cache <outside the repo>/pred_cache/val --out <outside the repo>/layout_sweep_val \
        --workers 6
    python -m ui2compose.evaluation.layout_sweep report --data-dir <package> --split val \
        --pred-cache <outside the repo>/pred_cache/val --sweep <outside the repo>/layout_sweep_val \
        --results-dir results/eval

The method is fixed here, before the sweep is run, and is not changed after
looking at the results.

1. Grid (GRID_*, DEDUP_SETTINGS). C4 (cached predictions + v1.2.0
   de-duplication, containment and XY-cut, layout_eval.CONDITIONS) is run
   for every combination of
     cut_tol_frac      in (0, 0.0025, 0.005, 0.01, 0.02),
     fallback          in ('y_sorted', 'least_overlap'),
     containment_ratio in (1.0, 0.95, 0.9),
     de-duplication    in off / a priori (same_class_iou 0.5, same_class_iomin
                       0.9, cross_class_iou 0.6) / stricter (0.6 / 0.95 / 0.7)
                       / looser (0.4 / 0.85 / 0.5),
   i.e. 120 configurations. 'off' disables all three rules (every threshold
   None), which equals run_pipeline(dedup=False). C1 (ground-truth boxes, no
   de-duplication) is run for the 30 combinations of cut_tol_frac, fallback
   and containment_ratio. C5 (de-duplication, then one flat Column) is run
   for the four de-duplication settings (its layout ignores the other
   thresholds); it is reported for context and not used in the selection.
   The a-priori values (A_PRIORI) are the provisional v1.2.0 defaults the
   evaluation started from (cut_tol_frac 0.005, 'y_sorted', 0.95, a-priori
   de-duplication); they are written here as literals, so the sweep does
   not change when the shipped defaults change.
2. Per screen, the reference tree is built once (layout_eval.build_reference)
   and every configuration is evaluated with layout_eval.evaluate_condition,
   i.e. with exactly the metrics, matching and exclusions of
   ui2compose-eval-layout (TEDS-g, TEDS-o, grouping F1, edit operations;
   see layout_eval). Predictions are the cached outputs of
   ComponentDetector.detect() at the pipeline operating point (conf 0.3,
   NMS IoU 0.7, imgsz 640). The per-screen records (without trees) are
   written to <out>/sweep.jsonl (one line per screen).
3. Detector check of the de-duplication settings (dedup_operating_points).
   For every de-duplication setting, the cached detections of every val
   screen are de-duplicated and scored against the ground-truth labels with
   ap.operating_point (confidence > 0.3, IoU >= 0.5, same class,
   Ultralytics' greedy matching, micro-averaged over all boxes): the same
   matching and the same numbers as the 'operating_point_pipeline' of
   ui2compose-evaluate-detector (without de-duplication the result is
   identical to it). A setting is ELIGIBLE only if its F1 is at most
   MAX_F1_DROP (0.5 F1 points) below the F1 without de-duplication.
4. Selection rule (select_config), for C4 (shipped default) and, separately,
   for C1 (ground-truth boxes; reported, not shipped):
     a. objective = mean TEDS-o over the evaluated screens of the whole val
        split (all 1,869 screens, status 'ok');
     b. among the eligible configurations, those with a mean TEDS-o within
        TIE_TOL (0.001) of the best are tied;
     c. among the tied ones, those with a mean grouping F1 (mean over
        screens with a defined F1) within TIE_TOL of the best of them are
        tied;
     d. among those, the configuration closest to the a-priori default wins:
        fewest parameters (cut_tol_frac, fallback, containment_ratio,
        de-duplication) that differ from A_PRIORI, then fewest grid steps
        from it (steps along GRID_CUT_TOL_FRAC, GRID_CONTAINMENT_RATIO,
        DEDUP_ORDER, GRID_FALLBACK), then the higher mean TEDS-o.
   TIE_TOL was chosen a priori (one tenth of a TEDS point, well below the
   width of any bootstrap interval), not tuned. The unconstrained argmax of
   the objective is reported as well.
5. Intervals. Every configuration's means get 95 % percentile bootstrap
   intervals (1,000 replicates, seed 0) with layout_eval.Bootstrap, under
   the resampling schemes of ui2compose-eval-layout: 'full' (group_id of
   splits/full/manifest.csv, primary), 'package_md5' and 'screen'. Paired
   differences (layout_eval.paired_difference; identical draws for every
   configuration) are given for every configuration against the a-priori
   configuration of its condition, and for the selected C4 configuration
   against the a-priori one. The unseen subsets of the split get point
   means only. Intervals are descriptive: the selection uses point means
   only (item 4).

6. Post-hoc check of the selection (selection_crossval, command
   'crossval'; added after the review of the revision, nothing is
   re-selected). The paired difference "selected minus a priori" of item 5
   is in-sample: it is measured on the screens on which the selected
   configuration won among 120, so it overstates what the rule gains on new
   screens (winner's curse). The check splits the app groups of the val
   split (CROSSVAL_SCHEME, each group whole) at random into two halves,
   applies the rule of item 4 to the means on one half and scores the
   configuration it selects, and the a-priori one, on the other half (both
   directions, CROSSVAL_REPEATS splits, seed 0). Every de-duplication
   setting counts as eligible (on the whole val split each raised the
   detector F1, item 3). The out-of-fold gain estimates the gain of the
   selection procedure on screens it did not see (with half the data per
   selection, so somewhat pessimistic); in-fold minus out-of-fold gain
   estimates the optimism of the in-sample number. The report also names
   the selected values that lie on the edge of their grid (grid_edges): the
   grid does not show whether a value beyond them would be better.

The test split is refused by run_sweep, report and crossval: thresholds are
chosen on the validation split only and the test split is evaluated once
afterwards with ui2compose-eval-layout.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ui2compose import __version__
from ui2compose.evaluation import ap as apm
from ui2compose.evaluation import layout_eval as LE
from ui2compose.infer.postprocess import deduplicate_detections

# ---------------------------------------------------------------------------
# Grid and selection rule (fixed before the sweep was run)
# ---------------------------------------------------------------------------

#: XY-cut tolerances (fraction of the image extent along the cut axis).
GRID_CUT_TOL_FRAC: Tuple[float, ...] = (0.0, 0.0025, 0.005, 0.01, 0.02)
#: Fallbacks when no XY cut exists.
GRID_FALLBACK: Tuple[str, ...] = ('y_sorted', 'least_overlap')
#: Containment ratios of UIGraphBuilder (1.0 = strict containment of v1.1.0).
GRID_CONTAINMENT_RATIO: Tuple[float, ...] = (1.0, 0.95, 0.9)
#: De-duplication settings: name -> deduplicate_detections thresholds
#: (same_class_iou, same_class_iomin, cross_class_iou); 'off' disables every rule.
DEDUP_SETTINGS: Dict[str, Dict[str, Optional[float]]] = {
    'off': {'same_class_iou': None, 'same_class_iomin': None, 'cross_class_iou': None},
    'a_priori': {'same_class_iou': 0.5, 'same_class_iomin': 0.9, 'cross_class_iou': 0.6},
    'stricter': {'same_class_iou': 0.6, 'same_class_iomin': 0.95, 'cross_class_iou': 0.7},
    'looser': {'same_class_iou': 0.4, 'same_class_iomin': 0.85, 'cross_class_iou': 0.5},
}
#: De-duplication settings ordered from fewest to most removals (grid steps of rule 4d).
DEDUP_ORDER: Tuple[str, ...] = ('off', 'stricter', 'a_priori', 'looser')
#: The provisional v1.2.0 defaults the sweep starts from (literals on purpose).
A_PRIORI: Dict[str, Any] = {'dedup': 'a_priori', 'containment_ratio': 0.95,
                            'cut_tol_frac': 0.005, 'fallback': 'y_sorted'}
#: Width of a tie in mean TEDS-o and in mean grouping F1 (rule 4b / 4c).
TIE_TOL: float = 0.001
#: Largest allowed drop of the detector's operating-point F1 (val, conf 0.3)
#: caused by a de-duplication setting, relative to no de-duplication.
MAX_F1_DROP: float = 0.005
#: Objective and first tie-break of the selection.
OBJECTIVE: str = 'teds_o'
TIE_BREAK: str = 'grouping_f1'
#: Confidence / IoU of the detector check (the pipeline operating point).
CHECK_CONF: float = apm.DEFAULT_CONF_OP
CHECK_IOU: float = apm.DEFAULT_IOU_OP
#: Conditions swept (layout_eval.CONDITIONS keys).
SWEPT_CONDITIONS: Tuple[str, ...] = ('C4', 'C1', 'C5')
#: Per-screen fields kept from layout_eval.evaluate_condition records.
KEEP_FIELDS: Tuple[str, ...] = (
    'status', 'teds_g', 'teds_o', 'grouping_f1', 'grouping', 'ted_g', 'ted_o', 'ops_g', 'ops_o',
    'gen_nodes', 'gen_depth', 'ref_nodes', 'n_fallback', 'n_placeholder', 'n_dedup_removed',
    'n_input', 'n_layout_input', 'n_matched', 'n_class_mismatch', 'n_lost', 'n_duplicates')
#: Name of the per-screen output file of run_sweep.
SWEEP_FILE: str = 'sweep.jsonl'
#: Decimals of the floats written to the summary files.
ROUND: int = 6
#: Post-hoc selection check (item 6): number of random two-fold splits of the
#: app groups, and the grouping whose groups are kept whole (package OR MD5;
#: the 'full' grouping would put about half of val into one fold).
CROSSVAL_REPEATS: int = 200
CROSSVAL_SCHEME: str = 'package_md5'
#: Wording of a grid edge (grid_edges) in val_sweep.md.
EDGE_TEXT: Dict[Tuple[str, str], str] = {
    ('dedup', 'first'): 'de-duplication off', ('dedup', 'last'): 'the loosest setting tried',
    ('containment_ratio', 'first'): 'strict containment',
    ('containment_ratio', 'last'): 'the smallest ratio tried',
    ('cut_tol_frac', 'first'): 'no tolerance', ('cut_tol_frac', 'last'): 'the largest tolerance tried'}
#: Name of the output file of the selection check.
CROSSVAL_FILE: str = 'val_sweep_crossval.json'


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

def sweep_configs(conditions: Sequence[str] = SWEPT_CONDITIONS) -> List[Dict[str, Any]]:
    """All configurations of the sweep (item 1 of the module docstring).

    Input:  conditions — subset of SWEPT_CONDITIONS
    Output: list of {'condition', 'dedup' (DEDUP_SETTINGS name or None),
            'containment_ratio', 'cut_tol_frac', 'fallback'}; C4 first
    """
    out: List[Dict[str, Any]] = []
    if 'C4' in conditions:
        for d in DEDUP_ORDER:
            for cr in GRID_CONTAINMENT_RATIO:
                for tol in GRID_CUT_TOL_FRAC:
                    for fb in GRID_FALLBACK:
                        out.append({'condition': 'C4', 'dedup': d, 'containment_ratio': cr,
                                    'cut_tol_frac': tol, 'fallback': fb})
    if 'C1' in conditions:
        for cr in GRID_CONTAINMENT_RATIO:
            for tol in GRID_CUT_TOL_FRAC:
                for fb in GRID_FALLBACK:
                    out.append({'condition': 'C1', 'dedup': None, 'containment_ratio': cr,
                                'cut_tol_frac': tol, 'fallback': fb})
    if 'C5' in conditions:
        for d in DEDUP_ORDER:
            out.append({'condition': 'C5', 'dedup': d,
                        'containment_ratio': A_PRIORI['containment_ratio'],
                        'cut_tol_frac': A_PRIORI['cut_tol_frac'], 'fallback': A_PRIORI['fallback']})
    return out


def config_key(cfg: Mapping[str, Any]) -> str:
    """Stable text key of a configuration.

    Input:  cfg — configuration (sweep_configs)
    Output: e.g. 'C4|dedup=a_priori|cr=0.95|tol=0.005|fb=y_sorted'
    """
    if cfg['condition'] == 'C5':
        return f"C5|dedup={cfg['dedup']}"
    head = cfg['condition'] if cfg['dedup'] is None else f"{cfg['condition']}|dedup={cfg['dedup']}"
    return (f"{head}|cr={cfg['containment_ratio']:g}|tol={cfg['cut_tol_frac']:g}"
            f"|fb={cfg['fallback']}")


def a_priori_config(condition: str) -> Dict[str, Any]:
    """The a-priori configuration of a swept condition.

    Input:  condition — 'C4', 'C1' or 'C5'
    Output: configuration dictionary
    """
    return {'condition': condition, 'dedup': None if condition == 'C1' else A_PRIORI['dedup'],
            'containment_ratio': A_PRIORI['containment_ratio'],
            'cut_tol_frac': A_PRIORI['cut_tol_frac'], 'fallback': A_PRIORI['fallback']}


def config_params(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Thresholds of a configuration in the shape of layout_eval.default_params().

    Input:  cfg — configuration
    Output: {'dedup', 'containment_ratio', 'cut_tol_frac', 'fallback', 'match_iou'}
            (C1 gets the 'off' thresholds; C1 never de-duplicates anyway)
    """
    return {'dedup': dict(DEDUP_SETTINGS[cfg['dedup'] or 'off']),
            'containment_ratio': float(cfg['containment_ratio']),
            'cut_tol_frac': float(cfg['cut_tol_frac']), 'fallback': cfg['fallback'],
            'match_iou': LE.MATCH_IOU}


def distance_to_a_priori(cfg: Mapping[str, Any]) -> Tuple[int, int]:
    """(parameters differing from A_PRIORI, grid steps from A_PRIORI) of a configuration (rule 4d).

    Input:  cfg — configuration (C4 or C1; C1 has no de-duplication parameter)
    Output: (n_changed, n_steps)
    """
    axes = [('cut_tol_frac', GRID_CUT_TOL_FRAC), ('containment_ratio', GRID_CONTAINMENT_RATIO),
            ('fallback', GRID_FALLBACK)]
    if cfg.get('dedup') is not None:
        axes.append(('dedup', DEDUP_ORDER))
    changed = steps = 0
    for name, grid in axes:
        a, b = list(grid).index(cfg[name]), list(grid).index(A_PRIORI[name])
        changed += a != b
        steps += abs(a - b)
    return changed, steps


# ---------------------------------------------------------------------------
# Per-screen sweep
# ---------------------------------------------------------------------------

def compact(rec: Mapping[str, Any]) -> Dict[str, Any]:
    """Keeps the fields of an evaluate_condition record that the sweep summary needs.

    Input:  rec — layout_eval.evaluate_condition record
    Output: dictionary with the KEEP_FIELDS present in rec
    """
    return {k: rec[k] for k in KEEP_FIELDS if k in rec}


def sweep_screen(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Builds one screen's reference once and evaluates every configuration on it.

    Input:  task — layout_eval.make_tasks task plus 'configs' (sweep_configs)
    Output: {'stem', 'reference' (record without the tree), 'configs':
             {config_key: compact record}, 'time_s'}
    """
    t0 = time.perf_counter()
    ref_rec, ref_tree, gt = LE.build_reference(task['stem'], task['xml'], task['label'],
                                               task['image'], task['names'])
    ref_rec.pop('tree', None)
    dets = LE.load_predictions_file(task.get('pred'))
    out: Dict[str, Any] = {'stem': task['stem'], 'reference': ref_rec, 'configs': {}}
    for cfg in task['configs']:
        key = config_key(cfg)
        if ref_tree is None:
            out['configs'][key] = {'status': ref_rec['status']}
            continue
        rec = LE.evaluate_condition(cfg['condition'], task['stem'], ref_tree, gt,
                                    ref_rec['image_size'], dets, config_params(cfg),
                                    zss_check=False, save_tree=False)
        out['configs'][key] = compact(rec)
    out['time_s'] = time.perf_counter() - t0
    return out


def run_sweep(data_dir: str | Path, split: str, out_dir: str | Path, pred_cache: str | Path,
              conditions: Sequence[str] = SWEPT_CONDITIONS, images: Optional[str] = None,
              limit: Optional[int] = None, workers: int = 1,
              progress_every: int = 100) -> Path:
    """Runs every configuration of the sweep on the screens of a split and writes sweep.jsonl.

    Input:  data_dir — package root; split — 'val' (the test split is
            refused); out_dir — output directory; pred_cache — directory of
            <stem>.json cached detections; conditions — subset of
            SWEPT_CONDITIONS; images — image list or directory (default: the
            whole split); limit — first N stems; workers — processes;
            progress_every — progress line interval (0: none)
    Output: path of <out_dir>/sweep.jsonl (plus <out_dir>/sweep_meta.json)
    """
    if split == 'test':
        raise ValueError("the threshold sweep never runs on the test split: thresholds are "
                         "selected on the validation split only")
    configs = sweep_configs(conditions)
    stems = LE.select_stems(data_dir, split, images, limit)
    tasks = LE.make_tasks(data_dir, split, stems, [], LE.default_params(), pred_cache, None,
                          zss_check=0, save_trees=False)
    for t in tasks:
        t['configs'] = configs
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / SWEEP_FILE
    t_start = time.perf_counter()
    with open(path, 'w', encoding='utf-8') as f:
        if workers > 1:
            import multiprocessing as mp

            with mp.get_context('spawn').Pool(workers) as pool:
                _write(pool.imap(sweep_screen, tasks, chunksize=2), f, len(tasks), progress_every)
        else:
            _write((sweep_screen(t) for t in tasks), f, len(tasks), progress_every)
    meta = {'tool': 'ui2compose.evaluation.layout_sweep run', 'ui2compose_version': __version__,
            'created_utc': _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'data_dir': str(data_dir), 'split': split, 'n_screens': len(stems),
            'pred_cache': str(pred_cache), 'prediction_meta': LE.prediction_meta(pred_cache),
            'conditions': list(conditions), 'n_configs': len(configs),
            'workers': workers, 'wall_seconds': time.perf_counter() - t_start}
    (out / 'sweep_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return path


def _write(results: Iterable[Dict[str, Any]], f, n: int, progress_every: int) -> None:
    """Writes per-screen sweep results as they arrive.

    Input:  results — sweep_screen outputs; f — open file; n — number of
            screens; progress_every — progress interval (0: none)
    Output: none
    """
    t0 = time.perf_counter()
    for k, res in enumerate(results, 1):
        f.write(json.dumps(res) + '\n')
        if progress_every and k % progress_every == 0:
            print(f"  {k}/{n} screens ({time.perf_counter() - t0:.0f} s)", flush=True)


def read_sweep(path: str | Path) -> Tuple[Dict[str, Dict[str, Any]],
                                          Dict[str, Dict[str, Dict[str, Any]]]]:
    """Reads sweep.jsonl.

    Input:  path — sweep.jsonl or its directory
    Output: (reference records {stem: record}, {config_key: {stem: record}})
    """
    p = Path(path)
    if p.is_dir():
        p = p / SWEEP_FILE
    refs: Dict[str, Dict[str, Any]] = {}
    cfgs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for line in p.read_text(encoding='utf-8').splitlines():
        if not line:
            continue
        r = json.loads(line)
        refs[r['stem']] = r['reference']
        for key, rec in r['configs'].items():
            cfgs.setdefault(key, {})[r['stem']] = {'stem': r['stem'], **rec}
    return refs, cfgs


# ---------------------------------------------------------------------------
# Detector check of the de-duplication settings
# ---------------------------------------------------------------------------

def _as_entry(dets: Sequence[Mapping[str, Any]], name2id: Mapping[str, int]) -> Dict[str, list]:
    """Converts detection dicts to an ap.py prediction entry.

    Input:  dets — detection dicts ('bbox', 'confidence', 'class_name');
            name2id — class name -> id
    Output: {'boxes', 'conf', 'cls'}
    """
    return {'boxes': [list(d['bbox']) for d in dets],
            'conf': [float(d['confidence']) for d in dets],
            'cls': [int(name2id[d['class_name']]) for d in dets]}


def _f1_counts(counts: np.ndarray) -> Tuple[float, float, float]:
    """Micro precision, recall and F1 from summed (TP, FP, FN).

    Input:  counts — array (3,) of TP, FP, FN
    Output: (precision, recall, f1), NaN where undefined
    """
    tp, fp, fn = (float(x) for x in counts)
    p = tp / (tp + fp) if tp + fp else math.nan
    r = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else math.nan
    return p, r, f1


def dedup_operating_points(data_dir: str | Path, split: str, pred_cache: str | Path,
                           stems: Sequence[str], groupings: Mapping[str, Mapping[str, Any]],
                           n_boot: int = apm.DEFAULT_BOOTSTRAP_N,
                           seed: int = apm.DEFAULT_BOOTSTRAP_SEED,
                           conf: float = CHECK_CONF, iou: float = CHECK_IOU) -> Dict[str, Any]:
    """Detector operating point (P / R / F1 at conf, IoU) after every de-duplication setting (item 3).

    Input:  data_dir — package root; split — 'val'; pred_cache — cache
            directory; stems — screens; groupings — {scheme: {stem: group}}
            (layout_eval.load_groupings); n_boot, seed — bootstrap; conf,
            iou — operating point
    Output: {'conf', 'iou', 'n_images', 'max_f1_drop', 'settings': {name:
             {'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'n_removed',
              'f1_minus_off' {scheme: interval}, 'eligible'}}}
    """
    from ui2compose.evaluation.detector_breakdown import operating_point_counts
    from ui2compose.evaluation.evaluate_detector import load_ground_truth

    if split == 'test':
        raise ValueError("the de-duplication check runs on the validation split only")
    names = LE.load_names(data_dir)
    name2id = {n: i for i, n in enumerate(names)}
    d = Path(data_dir)
    images = [d / 'dataset' / 'images' / split / f"{s}.png" for s in stems]
    gts, gt_report = load_ground_truth(images, d / 'dataset' / 'labels' / split, len(names))
    ids = sorted(gts)
    raw = {s: LE.load_predictions_file(Path(pred_cache) / f"{s}.json") for s in ids}
    missing = sorted(s for s, v in raw.items() if v is None)
    per_setting: Dict[str, np.ndarray] = {}
    out: Dict[str, Any] = {'conf': conf, 'iou': iou, 'n_images': len(ids), 'missing_predictions':
                           len(missing), 'label_report': {k: (len(v) if isinstance(v, list) else v)
                                                          for k, v in gt_report.items()},
                           'max_f1_drop': MAX_F1_DROP, 'settings': {}}
    for name in DEDUP_ORDER:
        preds: Dict[str, Dict[str, list]] = {}
        n_removed = 0
        for s in ids:
            dets = raw[s] or []
            kept, removed = deduplicate_detections(list(dets), **DEDUP_SETTINGS[name])
            n_removed += len(removed)
            preds[s] = _as_entry(kept, name2id)
        _, counts = operating_point_counts(preds, gts, len(names), conf, iou)
        per_setting[name] = counts.sum(1).astype(np.float64)       # (n_images, 3)
        op = apm.operating_point(preds, gts, names, conf, iou)
        out['settings'][name] = {'tp': op['tp'], 'fp': op['fp'], 'fn': op['fn'],
                                 'precision': op['precision'], 'recall': op['recall'],
                                 'f1': op['f1'], 'n_detections': op['tp'] + op['fp'],
                                 'n_removed': n_removed, 'thresholds': DEDUP_SETTINGS[name]}
    base = per_setting['off']
    for scheme, groups in groupings.items():
        boot = LE.Bootstrap(ids, groups, n_boot, seed)
        sums = {k: np.stack([boot.unit_sums(v[:, j])[0] for j in range(3)], 1)
                for k, v in per_setting.items()}                    # (n_units, 3)
        rep = {k: boot.counts @ v for k, v in sums.items()}           # (n_boot, 3)

        def f1_reps(r: np.ndarray) -> np.ndarray:
            """F1 of replicate counts.

            Input:  r — (n, 3) TP, FP, FN
            Output: (n,) F1 (NaN where undefined)
            """
            den = 2 * r[:, 0] + r[:, 1] + r[:, 2]
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.where(den > 0, 2 * r[:, 0] / np.where(den > 0, den, 1), np.nan)

        f_off = f1_reps(rep['off'])
        for name in DEDUP_ORDER:
            s = out['settings'][name]
            point = _f1_counts(per_setting[name].sum(0))[2] - _f1_counts(base.sum(0))[2]
            s.setdefault('f1_ci', {})[scheme] = boot.interval(f1_reps(rep[name]),
                                                              s['f1'])
            s.setdefault('f1_minus_off', {})[scheme] = boot.interval(
                f1_reps(rep[name]) - f_off, float(point))
    f1_off = out['settings']['off']['f1']
    for name in DEDUP_ORDER:
        s = out['settings'][name]
        s['f1_minus_off_point'] = s['f1'] - f1_off
        s['eligible'] = bool(s['f1'] >= f1_off - MAX_F1_DROP - 1e-12)
    return out


# ---------------------------------------------------------------------------
# Summary and selection
# ---------------------------------------------------------------------------

def _r(v: Any) -> Any:
    """Rounds the floats of a JSON-like object to ROUND decimals (NaN -> None).

    Input:  v — JSON-like object
    Output: rounded copy
    """
    if isinstance(v, float):
        return None if not math.isfinite(v) else round(v, ROUND)
    if isinstance(v, dict):
        return {k: _r(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_r(x) for x in v]
    if isinstance(v, np.floating):
        return _r(float(v))
    if isinstance(v, np.integer):
        return int(v)
    return v


def summarize_config(recs: Mapping[str, Mapping[str, Any]], boots: Mapping[str, LE.Bootstrap],
                     subsets: Mapping[str, Sequence[str]]) -> Dict[str, Any]:
    """Summary of one configuration: means with intervals on the whole split, point means on subsets.

    Input:  recs — {stem: record}; boots — {scheme: Bootstrap} over the whole
            split; subsets — {name: stems} besides the whole split
    Output: {'n_evaluated', 'status', metric: {'mean', 'median', 'ci'
             {scheme: interval}}, 'grouping_micro_f1', 'mean_fallback', ...,
             'subsets': {name: {metric: mean}}}
    """
    stems = next(iter(boots.values())).stems
    rs = [recs.get(s, {'status': 'missing'}) for s in stems]
    ok = [r for r in rs if r.get('status') == 'ok']
    out: Dict[str, Any] = {'n_screens': len(rs), 'n_evaluated': len(ok),
                           'status': dict(sorted(Counter(r.get('status') for r in rs).items()))}
    for m in LE.CI_METRICS:
        vals = [LE._num(r.get(m)) if r.get('status') == 'ok' else math.nan for r in rs]
        d = LE.describe(vals)
        out[m] = {'n': d['n'], 'mean': d.get('mean'), 'median': d.get('median'),
                  'ci': {k: b.mean(vals) for k, b in boots.items()}}
    tp = sum(r['grouping']['tp'] for r in ok)
    den = sum(r['grouping']['n_gen'] + r['grouping']['n_ref'] for r in ok)
    out['grouping_micro_f1'] = 2 * tp / den if den else None
    out['n_grouping_undefined'] = sum(1 for r in ok if r.get('grouping_f1') is None)
    for k in ('n_fallback', 'n_dedup_removed', 'gen_nodes', 'n_layout_input', 'n_matched',
              'n_placeholder'):
        out[f"mean_{k[2:] if k.startswith('n_') else k}"] = (
            float(np.mean([r[k] for r in ok])) if ok else None)
    out['median_ted_o'] = float(np.median([r['ted_o'] for r in ok])) if ok else None
    out['subsets'] = {}
    for name, sub in subsets.items():
        sub_ok = [recs[s] for s in sub if recs.get(s, {}).get('status') == 'ok']
        out['subsets'][name] = {'n_evaluated': len(sub_ok)}
        for m in LE.CI_METRICS:
            v = [r[m] for r in sub_ok if r.get(m) is not None]
            out['subsets'][name][m] = float(np.mean(v)) if v else None
    return out


def select_config(rows: Sequence[Mapping[str, Any]],
                  eligible: Optional[Mapping[str, bool]] = None,
                  tie_tol: float = TIE_TOL) -> Dict[str, Any]:
    """Applies the selection rule (item 4 of the module docstring) to the configurations of one condition.

    Input:  rows — [{'config', 'key', OBJECTIVE: mean, TIE_BREAK: mean}];
            eligible — {de-duplication setting: eligible} (None: all
            eligible; C1 has no setting); tie_tol — tie width
    Output: {'selected' (key), 'argmax' (key of the best objective among
             the eligible), 'best_objective', 'tied_objective' (keys),
             'tied_tie_break' (keys), 'ineligible' (keys), 'rule'}
    """
    def ok(r: Mapping[str, Any]) -> bool:
        """Tells whether a configuration is eligible.

        Input:  r — row
        Output: bool
        """
        d = r['config'].get('dedup')
        return eligible is None or d is None or bool(eligible.get(d, False))

    cand = [r for r in rows if ok(r) and r[OBJECTIVE] is not None]
    if not cand:
        raise ValueError("no eligible configuration with a defined objective")
    best = max(r[OBJECTIVE] for r in cand)
    argmax = max(cand, key=lambda r: (r[OBJECTIVE], -sum(distance_to_a_priori(r['config']))))
    tier1 = [r for r in cand if r[OBJECTIVE] >= best - tie_tol]
    best_tb = max((r[TIE_BREAK] for r in tier1 if r[TIE_BREAK] is not None), default=None)
    tier2 = [r for r in tier1 if best_tb is None or
             (r[TIE_BREAK] is not None and r[TIE_BREAK] >= best_tb - tie_tol)]
    chosen = min(tier2, key=lambda r: (*distance_to_a_priori(r['config']), -r[OBJECTIVE]))
    return {'selected': chosen['key'], 'argmax': argmax['key'], 'best_objective': best,
            'tied_objective': [r['key'] for r in tier1],
            'tied_tie_break': [r['key'] for r in tier2],
            'ineligible': [r['key'] for r in rows if not ok(r)],
            'rule': (f"max mean {OBJECTIVE} among eligible configurations; within {tie_tol} of the "
                     f"best: max mean {TIE_BREAK}; within {tie_tol} of that: fewest parameters "
                     "changed from the a-priori default, then fewest grid steps, then higher "
                     f"mean {OBJECTIVE}")}


def grid_edges(cfg: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    """Tells which values of a configuration lie on the edge of their grid.

    Input:  cfg — configuration (C4: with a de-duplication setting)
    Output: {parameter: 'first' / 'last' (position in its grid: DEDUP_ORDER
             from fewest to most removals, GRID_CONTAINMENT_RATIO from strict
             to loose, GRID_CUT_TOL_FRAC from 0 up) or None (interior)};
            the two-valued fallback is not listed
    """
    out: Dict[str, Optional[str]] = {}
    axes = [('containment_ratio', GRID_CONTAINMENT_RATIO), ('cut_tol_frac', GRID_CUT_TOL_FRAC)]
    if cfg.get('dedup') is not None:
        axes.insert(0, ('dedup', DEDUP_ORDER))
    for name, grid in axes:
        i = list(grid).index(cfg[name])
        out[name] = 'first' if i == 0 else 'last' if i == len(grid) - 1 else None
    return out


def _read_objective_arrays(sweep: str | Path, condition: str = 'C4'
                           ) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """Per-screen TEDS-o and grouping F1 of every configuration of one condition, as arrays.

    Input:  sweep — sweep.jsonl or its directory; condition — swept condition
    Output: (stems, config keys, teds_o (n_screens, n_configs), grouping_f1
             (same shape)); NaN where a screen is not 'ok' or F1 is undefined
    """
    p = Path(sweep)
    if p.is_dir():
        p = p / SWEEP_FILE
    keys = [config_key(c) for c in sweep_configs([condition])]
    stems: List[str] = []
    to_rows: List[List[float]] = []
    f1_rows: List[List[float]] = []
    with open(p, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            stems.append(r['stem'])
            cf = r['configs']
            to_rows.append([LE._num(cf.get(k, {}).get(OBJECTIVE))
                            if cf.get(k, {}).get('status') == 'ok' else math.nan for k in keys])
            f1_rows.append([LE._num(cf.get(k, {}).get(TIE_BREAK))
                            if cf.get(k, {}).get('status') == 'ok' else math.nan for k in keys])
    return stems, keys, np.asarray(to_rows, dtype=np.float64), np.asarray(f1_rows, dtype=np.float64)


def _column_means(x: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Means over the selected rows of every column, ignoring NaN.

    Input:  x — (n, k) array; rows — boolean (n,) mask
    Output: (k,) means (NaN for a column without a value)
    """
    sub = x[rows]
    ok = np.isfinite(sub)
    cnt = ok.sum(0)
    tot = np.where(ok, sub, 0.0).sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def selection_crossval(sweep: str | Path, data_dir: str | Path, split: str = 'val',
                       repeats: int = CROSSVAL_REPEATS, seed: int = 0,
                       scheme: str = CROSSVAL_SCHEME, condition: str = 'C4',
                       shipped: Optional[str] = None) -> Dict[str, Any]:
    """Post-hoc out-of-sample check of the selection rule on random halves of the val app groups (item 6).

    Input:  sweep — sweep.jsonl or its directory; data_dir — package root
            (groupings); split — 'val' (the test split is refused);
            repeats — random two-fold splits; seed — RNG seed; scheme —
            grouping kept whole (layout_eval.default_grouping_files name);
            condition — swept condition; shipped — key of the shipped
            configuration (default: the rule applied to the whole split)
    Output: {'scheme', 'n_groups', 'n_screens', 'repeats', 'seed', 'a_priori',
             'shipped', 'whole_split' {'selected', 'gain'}, 'in_fold_gain',
             'out_of_fold_gain' (summaries over 2 x repeats selections: mean,
             median, 2.5 / 97.5 % quantiles, share > 0; a half without an
             evaluated screen is skipped and counted in 'n_skipped_halves'),
             'optimism' (mean of
             in-fold minus out-of-fold gain), 'selected_counts' {key: n},
             'shipped_selected_share',
             'grid_edges' (of the shipped configuration)}
    """
    if split == 'test':
        raise ValueError("the selection check runs on the validation split only")
    stems, keys, teds_o, f1 = _read_objective_arrays(sweep, condition)
    files = LE.default_grouping_files(data_dir)
    if scheme not in files:
        raise ValueError(f"unknown grouping {scheme!r} (known: {sorted(files)})")
    groups = LE.load_groupings({scheme: files[scheme]}, stems)[scheme]
    labels = sorted({str(groups.get(s, f"__screen__{s}")) for s in stems})
    gidx = {g: i for i, g in enumerate(labels)}
    unit = np.asarray([gidx[str(groups.get(s, f"__screen__{s}"))] for s in stems])
    configs = {config_key(c): c for c in sweep_configs([condition])}
    apr = config_key(a_priori_config(condition))
    ia = keys.index(apr)

    def select(rows: np.ndarray) -> str:
        """Applies the selection rule to the means over some screens.

        Input:  rows — boolean mask of the screens
        Output: selected configuration key
        """
        mo, mf = _column_means(teds_o, rows), _column_means(f1, rows)
        cand = [{'config': configs[k], 'key': k,
                 OBJECTIVE: None if not math.isfinite(mo[j]) else float(mo[j]),
                 TIE_BREAK: None if not math.isfinite(mf[j]) else float(mf[j])}
                for j, k in enumerate(keys)]
        return select_config(cand)['selected']

    everything = np.ones(len(stems), dtype=bool)
    whole = select(everything)
    shipped = shipped or whole
    m_all = _column_means(teds_o, everything)
    rng = np.random.default_rng(seed)
    gains_in, gains_out = [], []
    counts: Counter = Counter()
    skipped = 0
    for _ in range(int(repeats)):
        perm = rng.permutation(len(labels))
        half = np.zeros(len(labels), dtype=bool)
        half[perm[: len(labels) // 2]] = True
        in_a = half[unit]
        for fit, score in ((in_a, ~in_a), (~in_a, in_a)):
            if not (np.isfinite(_column_means(teds_o, fit)).any()
                    and np.isfinite(_column_means(teds_o, score)).any()):
                skipped += 1                   # a half without an evaluated screen
                continue
            k = select(fit)
            j = keys.index(k)
            m_fit, m_sc = _column_means(teds_o, fit), _column_means(teds_o, score)
            counts[k] += 1
            gains_in.append(m_fit[j] - m_fit[ia])
            gains_out.append(m_sc[j] - m_sc[ia])

    def summary(v: Sequence[float]) -> Dict[str, Any]:
        """Mean, median, 2.5 / 97.5 % quantiles and share > 0.

        Input:  v — values
        Output: dictionary
        """
        a = np.asarray(v, dtype=np.float64)
        if not a.size:
            return {'n': 0}
        lo, med, hi = np.percentile(a, [2.5, 50, 97.5])
        return {'n': int(a.size), 'mean': float(a.mean()), 'median': float(med),
                'q025': float(lo), 'q975': float(hi), 'share_positive': float(np.mean(a > 0))}

    n_sel = len(gains_out)
    return _r({
        'note': 'post hoc (added after the review of the revision); nothing is re-selected',
        'condition': condition, 'objective': OBJECTIVE, 'scheme': scheme,
        'n_groups': len(labels), 'n_screens': len(stems), 'repeats': int(repeats),
        'seed': int(seed), 'a_priori': apr, 'shipped': shipped,
        'whole_split': {'selected': whole, 'gain': float(m_all[keys.index(whole)] - m_all[ia])},
        'in_fold_gain': summary(gains_in), 'out_of_fold_gain': summary(gains_out),
        'optimism': (float(np.mean(np.asarray(gains_in) - np.asarray(gains_out)))
                     if gains_out else None),
        'n_skipped_halves': skipped,
        'selected_counts': dict(counts.most_common()),
        'shipped_selected_share': counts[shipped] / n_sel if n_sel else None,
        'grid_edges': grid_edges(configs[shipped]),
        'eligibility': 'every de-duplication setting eligible (each raised the detector F1 on '
                       'the whole val split)'})


def summarize_sweep(data_dir: str | Path, split: str, sweep: str | Path, pred_cache: str | Path,
                    n_boot: int = apm.DEFAULT_BOOTSTRAP_N, seed: int = apm.DEFAULT_BOOTSTRAP_SEED,
                    groupings: Optional[Mapping[str, str | Path]] = None) -> Dict[str, Any]:
    """Summarises a finished sweep, runs the detector check and applies the selection rule.

    Input:  data_dir — package root; split — 'val'; sweep — sweep.jsonl or
            its directory; pred_cache — cache directory (detector check);
            n_boot, seed — bootstrap; groupings — {scheme: manifest.csv}
            (default layout_eval.default_grouping_files)
    Output: summary dictionary (written as val_sweep.json by write_outputs)
    """
    if split == 'test':
        raise ValueError("the threshold sweep is summarised on the validation split only")
    t0 = time.perf_counter()
    refs, cfg_recs = read_sweep(sweep)
    stems = sorted(refs)
    group_files = dict(groupings) if groupings is not None else LE.default_grouping_files(data_dir)
    groups = LE.load_groupings(group_files, stems)
    boots = {k: LE.Bootstrap(stems, g, n_boot, seed) for k, g in groups.items()}
    subsets = {name: [s for s in sub if s in refs]
               for name, sub in LE.load_subsets(data_dir, split).items()}
    configs = {config_key(c): c for c in sweep_configs()}
    present = [k for k in configs if k in cfg_recs]
    summary: Dict[str, Any] = {'configs': {}}
    for key in present:
        summary['configs'][key] = {'config': configs[key],
                                   **summarize_config(cfg_recs[key], boots, subsets)}
    # Paired differences against the a-priori configuration of the same condition.
    for key in present:
        cond = configs[key]['condition']
        base = config_key(a_priori_config(cond))
        if base in cfg_recs:
            summary['configs'][key]['minus_a_priori'] = {
                scheme: LE.paired_difference(cfg_recs[key], cfg_recs[base], b)
                for scheme, b in boots.items()}
    detector = dedup_operating_points(data_dir, split, pred_cache, stems, groups, n_boot, seed)
    eligible = {k: v['eligible'] for k, v in detector['settings'].items()}
    selection: Dict[str, Any] = {}
    for cond in ('C4', 'C1'):
        rows = [{'config': configs[k], 'key': k,
                 OBJECTIVE: summary['configs'][k][OBJECTIVE]['mean'],
                 TIE_BREAK: summary['configs'][k][TIE_BREAK]['mean']}
                for k in present if configs[k]['condition'] == cond]
        if rows:
            selection[cond] = select_config(rows, eligible if cond == 'C4' else None)
            selection[cond]['a_priori'] = config_key(a_priori_config(cond))
    if 'C4' in selection:
        sel = selection['C4']['selected']
        base = selection['C4']['a_priori']
        selection['C4']['selected_minus_a_priori'] = summary['configs'][sel].get('minus_a_priori')
        # C1 at the C4-selected layout thresholds (the GT-box effect of the shipped default).
        c4 = configs[sel]
        c1_at = config_key({**c4, 'condition': 'C1', 'dedup': None})
        selection['C1_at_C4_selection'] = c1_at if c1_at in cfg_recs else None
        if 'C1' in selection:
            c1 = configs[selection['C1']['selected']]
            same = {k: c1[k] == c4[k] for k in ('containment_ratio', 'cut_tol_frac', 'fallback')}
            selection['gt_vs_pred_optimal'] = {'C4_selected': sel, 'C1_selected':
                                               selection['C1']['selected'], 'same': same,
                                               'identical_layout_thresholds': all(same.values())}
    reference = LE.summarize_reference(refs, stems)
    summary.update({
        'selection': selection, 'detector_dedup_check': detector,
        'reference': {k: reference.get(k) for k in ('n_screens', 'n_ok', 'status', 'flat')},
        'bootstrap': {k: {'n': b.n, 'seed': b.seed, 'alpha': b.alpha, 'n_units': len(b.units),
                          'largest_unit_screens': max((len(u) for u in b.units), default=0)}
                      for k, b in boots.items()},
        'meta': {'tool': 'ui2compose.evaluation.layout_sweep report',
                 'ui2compose_version': __version__,
                 'created_utc': _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                 'split': split, 'n_screens': len(stems), 'sweep': str(sweep),
                 'pred_cache': str(pred_cache), 'prediction_meta': LE.prediction_meta(pred_cache),
                 'groupings': {**{k: str(v) for k, v in group_files.items()},
                               LE.SCREEN_SCHEME: 'every screen its own unit'},
                 'primary_grouping': next(iter(group_files), LE.SCREEN_SCHEME),
                 'subsets': {k: len(v) for k, v in subsets.items()},
                 'grid': {'cut_tol_frac': list(GRID_CUT_TOL_FRAC), 'fallback': list(GRID_FALLBACK),
                          'containment_ratio': list(GRID_CONTAINMENT_RATIO),
                          'dedup': DEDUP_SETTINGS},
                 'a_priori': A_PRIORI, 'tie_tol': TIE_TOL, 'max_f1_drop': MAX_F1_DROP,
                 'objective': OBJECTIVE, 'tie_break': TIE_BREAK,
                 'summary_seconds': time.perf_counter() - t0},
    })
    return _r(summary)


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------

def _ci_text(iv: Optional[Mapping[str, Any]], digits: int = 3, signed: bool = False) -> str:
    """Formats an interval as 'point [lo, hi]'.

    Input:  iv — interval dictionary or None; digits — decimals; signed — '+' sign
    Output: text ('n/a' when undefined)
    """
    if not iv or iv.get('point') is None:
        return 'n/a'
    f = f"{{:{'+' if signed else ''}.{digits}f}}"
    lo, hi = iv.get('lo'), iv.get('hi')
    rng = '' if lo is None else f" [{f.format(lo)}, {f.format(hi)}]"
    return f.format(iv['point']) + rng


def _fmt(v: Optional[float], digits: int = 3) -> str:
    """Formats a number ('n/a' for None).

    Input:  v — number or None; digits — decimals
    Output: text
    """
    return 'n/a' if v is None else f"{v:.{digits}f}"


def selected_thresholds(summary: Mapping[str, Any], date: Optional[str] = None,
                        crossval: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Content of selected_thresholds.json.

    Input:  summary — summarize_sweep output; date — ISO date (default today,
            UTC); crossval — optional selection_crossval output
    Output: dictionary
    """
    sel = summary['selection']['C4']
    cfg = summary['configs'][sel['selected']]
    apr = summary['configs'][sel['a_priori']]
    c1 = summary['selection'].get('C1')
    det = summary['detector_dedup_check']['settings'][cfg['config']['dedup']]
    out: Dict[str, Any] = {
        'note': 'selected on validation only (val split of the regenerated ReDraw package, '
                f"{summary['meta']['n_screens']} screens); the test split was not used",
        'date': date or _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%d'),
        'ui2compose_version': summary['meta']['ui2compose_version'],
        'objective': f"mean {OBJECTIVE} of C4 (cached predictions + v1.2.0 de-duplication, "
                     "containment and XY-cut) over the val screens",
        'tie_break': sel['rule'],
        'constraint': f"detector operating-point F1 (val, conf > {CHECK_CONF}, IoU {CHECK_IOU}) "
                      f"after de-duplication >= F1 without de-duplication - {MAX_F1_DROP}",
        'values': {
            'dedup': {'setting': cfg['config']['dedup'],
                      'enabled': cfg['config']['dedup'] != 'off',
                      **DEDUP_SETTINGS[cfg['config']['dedup']]},
            'containment_ratio': cfg['config']['containment_ratio'],
            'cut_tol_frac': cfg['config']['cut_tol_frac'],
            'fallback': cfg['config']['fallback'],
        },
        'a_priori': {'dedup': {'setting': A_PRIORI['dedup'], **DEDUP_SETTINGS[A_PRIORI['dedup']]},
                     'containment_ratio': A_PRIORI['containment_ratio'],
                     'cut_tol_frac': A_PRIORI['cut_tol_frac'], 'fallback': A_PRIORI['fallback']},
        'selected_config': sel['selected'],
        'objective_value': {'selected': cfg[OBJECTIVE]['mean'], 'a_priori': apr[OBJECTIVE]['mean'],
                            'best_eligible': sel['best_objective'], 'argmax_config': sel['argmax']},
        'selected_minus_a_priori': {
            scheme: {m: d[m] for m in (*LE.CI_METRICS, 'grouping_micro_f1')}
            for scheme, d in (sel.get('selected_minus_a_priori') or {}).items()},
        'selected_minus_a_priori_note': (
            'in-sample: measured on the val screens on which the configuration was selected as '
            'the best of the grid, so it overstates the gain on new screens; see '
            'selection_check'),
        'grid_edges': {k: v for k, v in grid_edges(cfg['config']).items() if v},
        'detector_f1_check': {'f1_selected_dedup': det['f1'],
                              'f1_without_dedup':
                                  summary['detector_dedup_check']['settings']['off']['f1'],
                              'difference': det['f1_minus_off_point'], 'eligible': det['eligible']},
        'gt_boxes_C1': None if c1 is None else {
            'selected_config': c1['selected'],
            'objective_value': summary['configs'][c1['selected']][OBJECTIVE]['mean'],
            'same_layout_thresholds_as_C4': summary['selection']['gt_vs_pred_optimal'],
            'note': 'reported only; the shipped default is the C4 (predicted-box) selection'},
        'files': {'sweep_summary': 'results/eval/val_sweep.json',
                  'sweep_table': 'results/eval/val_sweep.md'},
    }
    if crossval is not None:
        out['selection_check'] = {
            'note': crossval.get('note'), 'scheme': crossval['scheme'],
            'repeats': crossval['repeats'], 'out_of_fold_gain': crossval['out_of_fold_gain'],
            'in_fold_gain': crossval['in_fold_gain'], 'optimism': crossval['optimism'],
            'shipped_selected_share': crossval['shipped_selected_share']}
        out['files']['selection_check'] = f"results/eval/{CROSSVAL_FILE}"
    return _r(out)


def markdown(summary: Mapping[str, Any], crossval: Optional[Mapping[str, Any]] = None) -> str:
    """Markdown report of the sweep (val_sweep.md).

    Input:  summary — summarize_sweep output; crossval — optional
            selection_crossval output (item 6)
    Output: text
    """
    meta, sel = summary['meta'], summary['selection']
    prim = meta['primary_grouping']
    cfgs = summary['configs']
    L: List[str] = []
    L += ["# Threshold sweep on the validation split", "",
          f"Generated by `python -m ui2compose.evaluation.layout_sweep report` "
          f"({meta['created_utc']} UTC, ui2compose {meta['ui2compose_version']}). "
          f"Split: **{meta['split']}** ({meta['n_screens']} screens); the test split was not used. "
          "The method (grid, objective, tie-breaks, eligibility) is documented in "
          "`src/ui2compose/evaluation/layout_sweep.py` and was fixed before the sweep was run.", "",
          "- Objective: mean TEDS-o of C4 (cached predictions at "
          "conf 0.3 / NMS IoU 0.7 / imgsz 640 + v1.2.0 de-duplication, containment and XY-cut).",
          f"- Ties: within {meta['tie_tol']} of the best mean TEDS-o, then within "
          f"{meta['tie_tol']} of the best mean grouping F1; then the configuration closest "
          "to the a-priori default (fewest changed parameters, then fewest grid steps).",
          f"- Eligibility: a de-duplication setting may lower the detector's operating-point F1 "
          f"(conf > {CHECK_CONF}, IoU {CHECK_IOU}, ap.py matching) by at most "
          f"{meta['max_f1_drop'] * 100:.1f} F1 points relative to no de-duplication.",
          f"- Intervals: 95 % percentile bootstrap, {summary['bootstrap'][prim]['n']} replicates, "
          f"seed {summary['bootstrap'][prim]['seed']}, app groups of the `{prim}` grouping "
          f"({summary['bootstrap'][prim]['n_units']} units, largest "
          f"{summary['bootstrap'][prim]['largest_unit_screens']} screens) and of `package_md5` "
          f"({summary['bootstrap'].get('package_md5', {}).get('n_units')} units). "
          "They are descriptive; the selection uses point means only.",
          f"- A priori (the provisional v1.2.0 defaults): de-duplication "
          f"{'/'.join(str(v) for v in DEDUP_SETTINGS[A_PRIORI['dedup']].values())}, "
          f"containment_ratio {A_PRIORI['containment_ratio']}, cut_tol_frac "
          f"{A_PRIORI['cut_tol_frac']}, fallback {A_PRIORI['fallback']}.", ""]
    c4 = sel['C4']
    s, a = cfgs[c4['selected']], cfgs[c4['a_priori']]
    d = c4.get('selected_minus_a_priori') or {}
    L += ["## Selection", "",
          f"- **Selected (C4, shipped default): `{c4['selected']}`** — mean TEDS-o "
          f"{_ci_text(s['teds_o']['ci'][prim])} ({prim} groups).",
          f"- A priori: `{c4['a_priori']}` — mean TEDS-o {_ci_text(a['teds_o']['ci'][prim])}.",
          f"- Unconstrained argmax among eligible configurations: `{c4['argmax']}` "
          f"(mean TEDS-o {_fmt(c4['best_objective'], 4)}).",
          f"- Tied on TEDS-o (within {meta['tie_tol']}): {len(c4['tied_objective'])} "
          f"configurations; also tied on grouping F1: {len(c4['tied_tie_break'])}.",
          f"- Ineligible (detector F1 check): {len(c4['ineligible'])} configurations.", ""]
    if d:
        L += ["Selected minus a priori (paired, same screens):", "",
              "| Metric | " + " | ".join(d) + " |", "|---|" + "---|" * len(d)]
        for m, label in (('teds_o', 'TEDS-o'), ('teds_g', 'TEDS-g'),
                         ('grouping_f1', 'grouping F1 (mean)'),
                         ('grouping_micro_f1', 'grouping F1 (micro)')):
            L.append(f"| {label} | " + " | ".join(_ci_text(d[k][m], 4, True) for k in d) + " |")
        L.append("")
        L += ["These differences are **in-sample**: they are measured on the screens on which the "
              "selected configuration won among the grid, so they overstate the gain on new "
              "screens (see \"Selection check\" below when present).", ""]
    selected_cfg = cfgs[c4['selected']]['config']
    edges = {k: v for k, v in grid_edges(selected_cfg).items() if v}
    if edges:
        L += ["Selected values on the edge of their grid (no value beyond them was tried, so the "
              "grid does not show whether one would be better): "
              + ', '.join(f"`{k}` = {selected_cfg[k]} ({EDGE_TEXT[(k, v)]})"
                          for k, v in edges.items()) + ".", ""]
    if 'C1' in sel:
        c1 = sel['C1']
        g = sel.get('gt_vs_pred_optimal', {})
        L += [f"- GT boxes (C1): selected `{c1['selected']}` (mean TEDS-o "
              f"{_ci_text(cfgs[c1['selected']]['teds_o']['ci'][prim])}); a priori `{c1['a_priori']}` "
              f"({_ci_text(cfgs[c1['a_priori']]['teds_o']['ci'][prim])}).",
              "- GT-optimal and pred-optimal layout thresholds: " + (
                  "**identical**." if g.get('identical_layout_thresholds') else
                  "**differ** in " + ', '.join(f"`{k}`" for k, v in g.get('same', {}).items()
                                               if not v) + "."), ""]
        at = sel.get('C1_at_C4_selection')
        if at:
            L += [f"- C1 at the C4-selected thresholds (`{at}`): mean TEDS-o "
                  f"{_ci_text(cfgs[at]['teds_o']['ci'][prim])}.", ""]
    if crossval:
        o, i = crossval['out_of_fold_gain'], crossval['in_fold_gain']
        L += ["## Selection check (post hoc)", "",
              f"Added after the review; nothing is re-selected. The rule was applied to the means "
              f"on one random half of the {crossval['n_groups']} `{crossval['scheme']}` app "
              f"groups and scored on the other half ({crossval['repeats']} splits × 2 directions, "
              f"seed {crossval['seed']}; all de-duplication settings eligible).", ""]
    if crossval and not o.get('n'):
        L += ["No half split had evaluated screens on both sides.", ""]
    elif crossval:
        L += [
              "| quantity | mean | median | 2.5 % | 97.5 % | share > 0 |", "|---|---|---|---|---|---|",
              f"| TEDS-o gain over a priori, fold of selection (in-sample) | {i['mean']:+.4f} | "
              f"{i['median']:+.4f} | {i['q025']:+.4f} | {i['q975']:+.4f} | {i['share_positive']:.2f} |",
              f"| TEDS-o gain over a priori, other fold (out of sample) | {o['mean']:+.4f} | "
              f"{o['median']:+.4f} | {o['q025']:+.4f} | {o['q975']:+.4f} | {o['share_positive']:.2f} |",
              "",
              f"- Optimism (in-fold minus out-of-fold gain): {crossval['optimism']:+.4f}.",
              f"- The shipped configuration `{crossval['shipped']}` was selected in "
              f"{100 * crossval['shipped_selected_share']:.0f} % of the half-split selections; "
              "most frequent: " + ', '.join(f"`{k}` ({v})" for k, v in
                                           list(crossval['selected_counts'].items())[:3]) + ".",
              ""]
    det = summary['detector_dedup_check']
    L += ["## Detector check of the de-duplication settings", "",
          f"Val, {det['n_images']} images, conf > {det['conf']}, IoU {det['iou']}, micro over all "
          "boxes (the 'operating_point_pipeline' of ui2compose-evaluate-detector; 'off' equals it).",
          "", "| Setting | thresholds (IoU / IoMin / cross IoU) | removed | P | R | F1 | "
          f"F1 − off [{prim}] | F1 − off [package_md5] | eligible |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name in DEDUP_ORDER:
        x = det['settings'][name]
        th = '/'.join('none' if v is None else f"{v:g}" for v in x['thresholds'].values())
        L.append(f"| {name} | {th} | {x['n_removed']} | {_fmt(x['precision'], 4)} | "
                 f"{_fmt(x['recall'], 4)} | {_fmt(x['f1'], 4)} | "
                 f"{_ci_text(x['f1_minus_off'].get(prim), 4, True)} | "
                 f"{_ci_text(x['f1_minus_off'].get('package_md5'), 4, True)} | {x['eligible']} |")
    L.append("")

    def table(cond: str, title: str) -> None:
        """Appends the table of one condition, sorted by mean TEDS-o.

        Input:  cond — condition; title — section title
        Output: none
        """
        keys = [k for k, v in cfgs.items() if v['config']['condition'] == cond]
        keys.sort(key=lambda k: -(cfgs[k]['teds_o']['mean'] or -1))
        sel_key = sel.get(cond, {}).get('selected')
        apr_key = config_key(a_priori_config(cond))
        L.extend([f"## {title}", "",
                  "Sorted by mean TEDS-o. ★ selected, ◆ a priori. Δ = paired difference to the "
                  f"a-priori configuration ({prim} groups).", "",
                  "| # | dedup | containment | cut_tol_frac | fallback | TEDS-o [" + prim +
                  "] | TEDS-o [package_md5] | Δ TEDS-o | TEDS-g | grouping F1 | micro F1 | "
                  "fallback nodes / screen | dedup removed / screen | n |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"])
        for i, k in enumerate(keys, 1):
            v = cfgs[k]
            c = v['config']
            mark = (' ★' if k == sel_key else '') + (' ◆' if k == apr_key else '')
            dd = (v.get('minus_a_priori') or {}).get(prim, {}).get('teds_o')
            L.append(
                f"| {i}{mark} | {c['dedup'] or '–'} | {c['containment_ratio']:g} | "
                f"{c['cut_tol_frac']:g} | {c['fallback']} | {_ci_text(v['teds_o']['ci'][prim], 4)} | "
                f"{_ci_text(v['teds_o']['ci'].get('package_md5'), 4)} | "
                f"{'–' if k == apr_key else _ci_text(dd, 4, True)} | {_fmt(v['teds_g']['mean'], 4)} | "
                f"{_fmt(v['grouping_f1']['mean'], 4)} | {_fmt(v['grouping_micro_f1'], 4)} | "
                f"{_fmt(v['mean_fallback'], 3)} | {_fmt(v['mean_dedup_removed'], 3)} | "
                f"{v['n_evaluated']} |")
        L.append("")

    table('C4', 'C4: predicted boxes (120 configurations)')
    table('C1', 'C1: ground-truth boxes (30 configurations; no de-duplication)')
    keys5 = [k for k, v in cfgs.items() if v['config']['condition'] == 'C5']
    if keys5:
        L += ["## C5: predicted boxes, flat Column (context only)", "",
              f"| dedup | TEDS-o [{prim}] | TEDS-g | grouping F1 | dedup removed / screen |",
              "|---|---|---|---|---|"]
        for k in keys5:
            v = cfgs[k]
            L.append(f"| {v['config']['dedup']} | {_ci_text(v['teds_o']['ci'][prim], 4)} | "
                     f"{_fmt(v['teds_g']['mean'], 4)} | {_fmt(v['grouping_f1']['mean'], 4)} | "
                     f"{_fmt(v['mean_dedup_removed'], 3)} |")
        L.append("")
    L += ["## Marginal view (C4)", "",
          "Best and mean of the mean TEDS-o over the configurations with each parameter value.", "",
          "| Parameter | value | best TEDS-o | mean over its configurations |", "|---|---|---|---|"]
    c4keys = [k for k, v in cfgs.items() if v['config']['condition'] == 'C4']
    for name, grid in (('dedup', DEDUP_ORDER), ('containment_ratio', GRID_CONTAINMENT_RATIO),
                       ('cut_tol_frac', GRID_CUT_TOL_FRAC), ('fallback', GRID_FALLBACK)):
        for val in grid:
            vals = [cfgs[k]['teds_o']['mean'] for k in c4keys
                    if cfgs[k]['config'][name] == val and cfgs[k]['teds_o']['mean'] is not None]
            if vals:
                L.append(f"| {name} | {val} | {max(vals):.4f} | {np.mean(vals):.4f} |")
    L += ["", "Unseen subsets (point means of the selected and a-priori C4 configurations; not "
          "used in the selection):", "", "| Subset | n | selected TEDS-o | a priori TEDS-o |",
          "|---|---|---|---|"]
    for name, x in s['subsets'].items():
        L.append(f"| {name} | {x['n_evaluated']} | {_fmt(x['teds_o'], 4)} | "
                 f"{_fmt(a['subsets'][name]['teds_o'], 4)} |")
    L.append("")
    return '\n'.join(L)


def write_outputs(summary: Mapping[str, Any], results_dir: str | Path,
                  date: Optional[str] = None,
                  crossval: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    """Writes val_sweep.json, val_sweep.md and selected_thresholds.json (and the selection check).

    Input:  summary — summarize_sweep output; results_dir — directory
            (results/eval); date — ISO date of the selection; crossval —
            optional selection_crossval output (written as CROSSVAL_FILE)
    Output: {name: path}
    """
    d = Path(results_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = {'val_sweep.json': d / 'val_sweep.json', 'val_sweep.md': d / 'val_sweep.md',
             'selected_thresholds.json': d / 'selected_thresholds.json'}
    paths['val_sweep.json'].write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n',
                                       encoding='utf-8')
    paths['val_sweep.md'].write_text(markdown(summary, crossval), encoding='utf-8')
    paths['selected_thresholds.json'].write_text(
        json.dumps(selected_thresholds(summary, date, crossval), indent=2, ensure_ascii=False)
        + '\n', encoding='utf-8')
    if crossval is not None:
        paths[CROSSVAL_FILE] = d / CROSSVAL_FILE
        paths[CROSSVAL_FILE].write_text(json.dumps(crossval, indent=1, ensure_ascii=False) + '\n',
                                        encoding='utf-8')
    return {k: str(v) for k, v in paths.items()}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Defines and reads the command-line arguments.

    Input:  argv — argument list (default sys.argv[1:])
    Output: argparse namespace
    """
    ap = argparse.ArgumentParser(description="Grid sweep of the v1.2.0 thresholds on the "
                                             "validation split and selection of the defaults")
    sub = ap.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help="evaluate every configuration (writes sweep.jsonl)")
    rep = sub.add_parser('report', help="summarise a sweep, check the detector, select")
    cv = sub.add_parser('crossval', help="post-hoc check of the selection on halves of the val "
                                         "app groups; rewrites val_sweep.md and "
                                         "selected_thresholds.json from val_sweep.json")
    cv.add_argument('--data-dir', required=True, help="regenerated ReDraw val/test package")
    cv.add_argument('--split', default='val', choices=('val',),
                    help="only the validation split is accepted")
    cv.add_argument('--sweep', required=True, help="sweep.jsonl or its directory")
    cv.add_argument('--results-dir', required=True,
                    help="directory holding val_sweep.json (outputs go there too)")
    cv.add_argument('--repeats', type=int, default=CROSSVAL_REPEATS)
    cv.add_argument('--seed', type=int, default=0)
    cv.add_argument('--scheme', default=CROSSVAL_SCHEME)
    for p in (run, rep):
        p.add_argument('--data-dir', required=True, help="regenerated ReDraw val/test package")
        p.add_argument('--split', default='val', choices=('val',),
                       help="only the validation split is accepted")
        p.add_argument('--pred-cache', required=True, help="directory of <stem>.json detections")
    run.add_argument('--out', required=True, help="output directory (sweep.jsonl)")
    run.add_argument('--conditions', default=','.join(SWEPT_CONDITIONS))
    run.add_argument('--images', default=None)
    run.add_argument('--limit', type=int, default=None)
    run.add_argument('--workers', type=int, default=1)
    rep.add_argument('--sweep', required=True, help="sweep.jsonl or its directory")
    rep.add_argument('--results-dir', required=True, help="where the summary files go")
    rep.add_argument('--bootstrap', type=int, default=apm.DEFAULT_BOOTSTRAP_N)
    rep.add_argument('--seed', type=int, default=apm.DEFAULT_BOOTSTRAP_SEED)
    rep.add_argument('--date', default=None, help="ISO date written to selected_thresholds.json")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Runs the sweep or its report from the command line.

    Input:  argv — argument list (default sys.argv[1:])
    Output: exit code 0 (2 for invalid arguments)
    """
    args = parse_args(argv)
    try:
        if args.command == 'run':
            conds = [c.strip() for c in args.conditions.split(',') if c.strip()]
            bad = [c for c in conds if c not in SWEPT_CONDITIONS]
            if bad:
                print(f"[ERR] unknown conditions {bad} (swept: {SWEPT_CONDITIONS})")
                return 2
            path = run_sweep(args.data_dir, args.split, args.out, args.pred_cache, conds,
                             images=args.images, limit=args.limit, workers=args.workers)
            print(f"written: {path}")
            return 0
        if args.command == 'crossval':
            res = Path(args.results_dir)
            summary = json.loads((res / 'val_sweep.json').read_text(encoding='utf-8'))
            shipped = summary['selection']['C4']['selected']
            cvr = selection_crossval(args.sweep, args.data_dir, args.split, args.repeats,
                                     args.seed, args.scheme, shipped=shipped)
            old = res / 'selected_thresholds.json'
            date = (json.loads(old.read_text(encoding='utf-8')).get('date')
                    if old.is_file() else None)
            paths = write_outputs(summary, res, date, cvr)
            o = cvr['out_of_fold_gain']
            print(f"out-of-fold TEDS-o gain of the rule: mean {o['mean']:+.4f} "
                  f"[{o['q025']:+.4f}, {o['q975']:+.4f}], share > 0 {o['share_positive']:.2f}; "
                  f"optimism {cvr['optimism']:+.4f}")
            for name, p in paths.items():
                print(f"written: {p}")
            return 0
        summary = summarize_sweep(args.data_dir, args.split, args.sweep, args.pred_cache,
                                  args.bootstrap, args.seed)
        paths = write_outputs(summary, args.results_dir, args.date)
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERR] {e}")
        return 2
    c4 = summary['selection']['C4']
    print(f"selected (C4): {c4['selected']}  (a priori {c4['a_priori']})")
    for name, p in paths.items():
        print(f"written: {p}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
