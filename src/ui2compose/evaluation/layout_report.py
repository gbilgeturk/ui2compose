# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Table-1 report of a layout evaluation run, with the structure-only correction-effort proxy.

Typical use (after ui2compose-eval-layout has written <run>/aggregate.json
and the per-screen JSONL files):
    python -m ui2compose.evaluation.layout_report --run <outside>/layout_test \
        --data-dir <package> --out-json results/eval/layout_test.json \
        --out-md results/eval/layout_test.md

Nothing is recomputed except the correction-effort proxy: every number of
the tables is read from aggregate.json (layout_eval.aggregate), so the
report and the run cannot disagree.

1. Table 1 (one per subset: all screens of the split, the unseen-app subset
   of the full grouping, the package-only unseen subset). Rows = conditions
   (C1, C2, C3, C4, C5, C1p when run). Columns: mean TEDS-g, mean TEDS-o and
   mean grouping F1 with 95 % percentile bootstrap intervals over app groups
   (TABLE_SCHEMES: 'full' = group_id of splits/full/manifest.csv, the
   primary scheme of the method, and 'package_md5'), the median (IQR) of
   the edit operations of the TEDS-o and TEDS-g mappings, and the median
   (IQR) of the generated / reference canonical node counts. A second table
   gives micro grouping F1, matching recall / precision, class mismatches,
   fallback nodes, placeholders, duplicates, lost boxes and de-duplication
   removals.
2. Paired differences (layout_eval.paired_difference; the same draws for
   both conditions): C1-C2, C4-C5, C4-C3, C1-C4 (and C1-C1p), mean
   per-screen difference with intervals per scheme and the fraction of
   replicates <= 0.
3. Correction-effort proxy (correction_proxy), structure only: the number
   of edit operations (insert + delete + rename of the optimal unit-cost
   mapping, = TED) that turn the generated tree into the reference tree;
   texts, sizes, styles, modifiers and the code itself are not counted. Per
   condition (PROXY_CONDITIONS: C1 and C4) and variant ('o' and 'g'):
   median, IQR, mean, 90th percentile, share of screens with 0 edits and
   with at most PROXY_SMALL (5) edits, both shares with bootstrap
   intervals (same schemes, seed and replicates as the run), and the
   median of each edit kind. The edit kinds are those of APTED's own
   optimal mapping (layout_eval.tree_edit); other optimal mappings can
   split the same total differently.

4. Grouping F1 is defined only on screens where at least one tree groups
   a pair of matched boxes, so its means in Table 1 are over different
   screens per condition (with predicted boxes, over what the detector
   found). The report gives the number of screens behind every mean and
   paired difference, splits the undefined screens into those with fewer
   than two matched boxes and those where neither tree groups a pair
   (grouping_undefined), and repeats the means on the screens where every
   condition of the run defines F1 (grouping_common).

Added after the review of the revision, labelled post hoc in both outputs:
5. Stratified paired differences (stratified_differences): the paired
   differences on the screens of a stratum of the reference (STRATA: flat /
   not flat dump, one / at least two ground-truth boxes), same bootstrap
   draws restricted to the stratum.
6. Containment-Box sensitivity (box_sensitivity, --box-sensitivity-run): a
   second run of the conditions with containment Boxes (C1, C4) with
   ui2compose-eval-layout --box-repr container (see layout_eval, item 2) is
   merged with the primary records of the other conditions and aggregated
   with the same subsets, groupings, seed and replicates: means, changes
   against the primary scoring, and the paired differences.

Intervals are always labelled by their scheme. The full grouping chains
about half of a full split into one group (880 of the 1,870 test screens),
so its intervals on 'all' are wide and asymmetric; on the unseen subsets
both schemes have no such group.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ui2compose import __version__
from ui2compose.evaluation import layout_eval as LE

#: Order of the rows.
CONDITION_ORDER = ('C1', 'C2', 'C3', 'C4', 'C5', 'C1p')

#: Resampling schemes shown in the tables (the first is the primary one).
TABLE_SCHEMES = ('full', 'package_md5')

#: Conditions and threshold of the correction-effort proxy.
PROXY_CONDITIONS = ('C1', 'C4')
PROXY_SMALL = 5

#: Strata of the post-hoc stratified differences: name -> predicate on the reference record.
STRATA = {
    'reference_flat': lambda r: bool(r.get('flat')),
    'reference_not_flat': lambda r: not r.get('flat'),
    'one_gt_box': lambda r: r.get('n_gt') == 1,
    'two_or_more_gt_boxes': lambda r: (r.get('n_gt') or 0) >= 2,
}

#: Conditions whose trees contain containment Boxes (rescored by the sensitivity run).
BOX_CONDITIONS = ('C1', 'C4')

#: Subset names and their captions.
SUBSET_TITLES = {'all': 'all screens', 'unseen': 'unseen apps (full grouping)',
                 'unseen_package_only': 'unseen packages (package only)'}


def _r(v: Any, digits: int = 4) -> Any:
    """Rounds floats for the JSON output (other values unchanged).

    Input:  v — value; digits — decimals
    Output: rounded value
    """
    if isinstance(v, float):
        return None if math.isnan(v) else round(v, digits)
    if isinstance(v, dict):
        return {k: _r(x, digits) for k, x in v.items()}
    if isinstance(v, list):
        return [_r(x, digits) for x in v]
    return v


def _ci(iv: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Compact interval: point, lo, hi, P(<= 0).

    Input:  iv — interval dictionary of layout_eval.Bootstrap
    Output: {'point', 'lo', 'hi', 'p_le_0'} or None
    """
    if not iv:
        return None
    return {'point': iv.get('point'), 'lo': iv.get('lo'), 'hi': iv.get('hi'),
            'p_le_0': iv.get('p_le_0')}


def _iqr(d: Mapping[str, Any]) -> Dict[str, Any]:
    """Median and quartiles of a layout_eval.describe dictionary.

    Input:  d — describe() output
    Output: {'median', 'q1', 'q3', 'mean', 'n'}
    """
    return {k: d.get(k) for k in ('median', 'q1', 'q3', 'mean', 'n')}


def condition_row(s: Mapping[str, Any]) -> Dict[str, Any]:
    """Table-1 row of one condition on one subset.

    Input:  s — aggregate['conditions'][cond][subset]
    Output: dictionary of the row (means with intervals per scheme, medians, counts)
    """
    row: Dict[str, Any] = {'n_screens': s['n_screens'], 'n_evaluated': s['n_evaluated'],
                           'status': s['status']}
    for m in LE.CI_METRICS:
        row[m] = {'mean': s[m].get('mean'), 'median': s[m].get('median'),
                  'q1': s[m].get('q1'), 'q3': s[m].get('q3'), 'n': s[m].get('n'),
                  'ci_mean': {k: _ci(v) for k, v in s[m].get('ci_mean', {}).items()},
                  'ci_median': {k: _ci(v) for k, v in s[m].get('ci_median', {}).items()}}
    row['grouping_f1']['n_undefined'] = s['grouping_f1'].get('n_undefined')
    gm = s['grouping_micro']
    row['grouping_micro'] = {'precision': gm['precision'], 'recall': gm['recall'], 'f1': gm['f1'],
                             'ci_f1': {k: _ci(v) for k, v in gm['ci_f1'].items()}}
    for v in LE.VARIANTS:
        row[f"edit_ops_{v}"] = {k: _iqr(s[f"edit_ops_{v}"][k]) for k in LE.EDIT_OPS + ('total',)}
    row['nodes'] = {'generated': _iqr(s['nodes']['generated']),
                    'reference': _iqr(s['nodes']['reference'])}
    row['generated_depth'] = _iqr(s['depth']['generated'])
    for k in ('fallback', 'placeholder', 'duplicates', 'lost', 'dedup_removed'):
        row[k] = s[k]
    row['matching'] = s['matching']
    return row


def correction_proxy(records: Mapping[str, Mapping[str, Any]], stems: Sequence[str],
                     groupings: Mapping[str, Mapping[str, Any]], n_boot: int, seed: int,
                     small: int = PROXY_SMALL) -> Dict[str, Any]:
    """Structure-only correction-effort proxy of one condition on one subset.

    Input:  records — {stem: condition record}; stems — subset; groupings —
            {scheme: {stem: group}}; n_boot, seed — bootstrap; small — the
            'few edits' threshold
    Output: {variant: {'n', 'median', 'q1', 'q3', 'mean', 'p90', 'max',
             'share_0', 'share_le_small' (with 'ci' per scheme), 'kinds'
             {insert/delete/rename: median, mean}}}
    """
    stems = sorted(set(stems))
    ok = [s for s in stems if records.get(s, {}).get('status') == 'ok']
    boots = {k: LE.Bootstrap(stems, g, n_boot, seed) for k, g in groupings.items()}
    out: Dict[str, Any] = {'n_screens': len(stems), 'n_evaluated': len(ok), 'small': small}
    for v in LE.VARIANTS:
        ted = np.asarray([records[s][f"ted_{v}"] for s in ok], dtype=float)
        if not ted.size:
            out[v] = {'n': 0}
            continue
        q1, med, q3, p90 = np.percentile(ted, [25, 50, 75, 90])

        def indicator(pred) -> List[float]:
            """Per-screen 0/1 values aligned with the subset's stems (NaN if not evaluated).

            Input:  pred — function of the screen's TED
            Output: list of floats
            """
            return [float(pred(records[s][f"ted_{v}"])) if records.get(s, {}).get('status') == 'ok'
                    else math.nan for s in stems]
        zero, few = indicator(lambda t: t == 0), indicator(lambda t: t <= small)
        out[v] = {
            'n': int(ted.size), 'median': float(med), 'q1': float(q1), 'q3': float(q3),
            'mean': float(ted.mean()), 'p90': float(p90), 'max': float(ted.max()),
            'share_0': {'point': float(np.mean(ted == 0)),
                        'ci': {k: _ci(b.mean(zero)) for k, b in boots.items()}},
            'share_le_small': {'point': float(np.mean(ted <= small)),
                               'ci': {k: _ci(b.mean(few)) for k, b in boots.items()}},
            'kinds': {k: {'median': float(np.median([records[s][f"ops_{v}"][k] for s in ok])),
                          'mean': float(np.mean([records[s][f"ops_{v}"][k] for s in ok]))}
                      for k in LE.EDIT_OPS},
            'per_reference_node': {
                'median': float(np.median([records[s][f"ted_{v}"] / records[s]['ref_nodes']
                                           for s in ok]))},
        }
    return out


def grouping_undefined(records: Mapping[str, Mapping[str, Any]], stems: Sequence[str]
                       ) -> Dict[str, int]:
    """Screens of a subset without a defined grouping F1, by reason.

    Input:  records — {stem: condition record}; stems — subset
    Output: {'n_evaluated', 'n_defined', 'n_undefined',
             'n_undefined_fewer_than_2_matched' (no pair of matched boxes
             exists), 'n_undefined_no_grouped_pair' (pairs exist, neither
             tree groups one)}
    """
    ok = [records[s] for s in stems if records.get(s, {}).get('status') == 'ok']
    und = [r for r in ok if r.get('grouping_f1') is None]
    few = sum(1 for r in und if r.get('n_matched', 0) < 2)
    return {'n_evaluated': len(ok), 'n_defined': len(ok) - len(und), 'n_undefined': len(und),
            'n_undefined_fewer_than_2_matched': few, 'n_undefined_no_grouped_pair': len(und) - few}


def _boots(stems: Sequence[str], groupings: Mapping[str, Mapping[str, Any]], n_boot: int,
           seed: int, schemes: Sequence[str] = TABLE_SCHEMES) -> Dict[str, Any]:
    """Bootstrap objects of a subset for the table schemes (same draws as the run).

    Input:  stems — subset; groupings — {scheme: {stem: group}}; n_boot,
            seed — bootstrap; schemes — schemes to build
    Output: {scheme: layout_eval.Bootstrap}
    """
    stems = sorted(set(stems))
    return {k: LE.Bootstrap(stems, groupings[k], n_boot, seed) for k in schemes if k in groupings}


def grouping_common(cond_records: Mapping[str, Mapping[str, Mapping[str, Any]]],
                    boots: Mapping[str, Any]) -> Dict[str, Any]:
    """Mean grouping F1 of every condition on the screens where all conditions define it.

    Input:  cond_records — {cond: {stem: record}}; boots — {scheme:
            Bootstrap} of the subset
    Output: {'n_screens', 'conditions': {cond: {'mean', 'ci': {scheme: interval}}}}
    """
    stems = next(iter(boots.values())).stems
    common = {s for s in stems if all(rec.get(s, {}).get('status') == 'ok'
                                      and rec[s].get('grouping_f1') is not None
                                      for rec in cond_records.values())}
    out: Dict[str, Any] = {'n_screens': len(common), 'conditions': {}}
    for c, rec in cond_records.items():
        vals = [rec[s]['grouping_f1'] if s in common else math.nan for s in stems]
        mean = float(np.mean([rec[s]['grouping_f1'] for s in common])) if common else None
        out['conditions'][c] = {'mean': mean,
                                'ci': {k: _ci(b.mean(vals)) for k, b in boots.items()}}
    return out


def _diff_block(rec_a: Mapping[str, Any], rec_b: Mapping[str, Any], boots: Mapping[str, Any]
                ) -> Dict[str, Any]:
    """Paired differences of two record sets under every scheme, compacted.

    Input:  rec_a, rec_b — {stem: record}; boots — {scheme: Bootstrap}
    Output: {scheme: {metric: {'point', 'lo', 'hi', 'p_le_0', 'n_screens_both'}}}
    """
    return {scheme: {m: _ci(v) | {'n_screens_both': v.get('n_screens_both')}
                     for m, v in LE.paired_difference(rec_a, rec_b, boot).items()}
            for scheme, boot in boots.items()}


def stratified_differences(ref_records: Mapping[str, Mapping[str, Any]],
                           cond_records: Mapping[str, Mapping[str, Mapping[str, Any]]],
                           boots: Mapping[str, Any],
                           diffs: Sequence[Sequence[str]] = LE.DEFAULT_DIFFS) -> Dict[str, Any]:
    """Post-hoc paired differences on the strata of the reference (STRATA).

    The draws are those of the subset; screens outside the stratum are left
    out of each replicate's mean.

    Input:  ref_records — {stem: reference record}; cond_records — {cond:
            {stem: record}}; boots — {scheme: Bootstrap} of the subset;
            diffs — (a, b) pairs
    Output: {stratum: {'n_screens', 'differences': {'a-b': _diff_block}}}
    """
    stems = next(iter(boots.values())).stems
    out: Dict[str, Any] = {}
    for name, pred in STRATA.items():
        ss = {s for s in stems if ref_records.get(s, {}).get('status') == 'ok'
              and pred(ref_records[s])}
        entry: Dict[str, Any] = {'n_screens': len(ss), 'differences': {}}
        for a, b in diffs:
            if a in cond_records and b in cond_records:
                ra = {s: cond_records[a][s] for s in ss if s in cond_records[a]}
                rb = {s: cond_records[b][s] for s in ss if s in cond_records[b]}
                entry['differences'][f"{a}-{b}"] = _diff_block(ra, rb, boots)
        out[name] = entry
    return out


def box_sensitivity(box_run: str | Path, run_meta: Mapping[str, Any],
                    ref_records: Mapping[str, Mapping[str, Any]],
                    cond_records: Mapping[str, Mapping[str, Mapping[str, Any]]],
                    subsets: Mapping[str, Sequence[str]],
                    groupings: Mapping[str, Mapping[str, Any]], n_boot: int, seed: int,
                    diffs: Sequence[Sequence[str]] = LE.DEFAULT_DIFFS) -> Dict[str, Any]:
    """Post-hoc sensitivity of the layout metrics to the scoring of containment Boxes.

    The records of box_run (ui2compose-eval-layout --box-repr container, the
    conditions of BOX_CONDITIONS) replace the primary records of those
    conditions; everything else (references, other conditions, subsets,
    groupings, seed, replicates) is the primary run's.

    Input:  box_run — directory of the sensitivity run; run_meta — meta of
            the primary aggregate; ref_records, cond_records — records of
            the primary run; subsets — {name: stems}; groupings — {scheme:
            {stem: group}}; n_boot, seed — bootstrap; diffs — (a, b) pairs
    Output: {'run_dir', 'run_created_utc', 'box_repr', 'conditions',
             'checks', 'subsets': {sub: {'n_screens_changed', 'table'
             {cond: condition_row}, 'change' {cond: _diff_block (container
             minus composite)}}}, 'paired_differences' (as the report's)};
            raises ValueError if the runs do not belong together
    """
    run = Path(box_run)
    meta_b = json.loads((run / 'aggregate.json').read_text(encoding='utf-8'))['meta']
    ref_b, cond_b = LE.read_run(run)
    conds = [c for c in BOX_CONDITIONS if c in cond_b]
    if meta_b.get('box_repr') != 'container':
        raise ValueError(f"{run}: not a --box-repr container run ({meta_b.get('box_repr')!r})")
    for k in ('split', 'params', 'n_screens'):
        if meta_b.get(k) != run_meta.get(k):
            raise ValueError(f"{run}: {k} differs from the primary run")
    if not conds or set(cond_b) - set(BOX_CONDITIONS):
        raise ValueError(f"{run}: expected conditions among {BOX_CONDITIONS}, got {sorted(cond_b)}")
    same_ref = sorted(ref_b) == sorted(ref_records) and all(
        (ref_b[s].get('status'), ref_b[s].get('tree')) ==
        (ref_records[s].get('status'), ref_records[s].get('tree')) for s in ref_records)
    if not same_ref:
        raise ValueError(f"{run}: reference trees differ from the primary run")
    merged = {**cond_records, **{c: cond_b[c] for c in conds}}
    agg = LE.aggregate(ref_records, merged, subsets, groupings, n_boot, seed, diffs)
    out: Dict[str, Any] = {
        'run_dir': str(box_run), 'run_created_utc': meta_b.get('created_utc'),
        'box_repr': meta_b.get('box_repr'), 'conditions': conds,
        'checks': {'reference_identical': same_ref,
                   'zss_cross_check': {c: agg['zss_cross_check'].get(c) for c in conds},
                   'status': {c: dict(Counter(r.get('status') for r in cond_b[c].values()))
                              for c in conds}},
        'subsets': {}, 'paired_differences': {}}
    for sub, stems in subsets.items():
        boots = _boots(stems, groupings, n_boot, seed)
        out['subsets'][sub] = {
            'n_screens_changed': {c: sum(1 for s in set(stems)
                                         if cond_b[c].get(s, {}).get('tree')
                                         != cond_records[c].get(s, {}).get('tree'))
                                  for c in conds},
            'table': {c: condition_row(agg['conditions'][c][sub]) for c in conds},
            'change': {c: _diff_block(cond_b[c], cond_records[c], boots) for c in conds}}
    out['paired_differences'] = {
        pair: {sub: {scheme: {m: _ci(v) | {'n_screens_both': v.get('n_screens_both')}
                              for m, v in by_scheme.items()}
                     for scheme, by_scheme in per_sub.items() if scheme in TABLE_SCHEMES}
               for sub, per_sub in subs.items()}
        for pair, subs in agg['paired_differences'].items()
        if any(c in pair.split('-') for c in conds)}
    return out


def build_report(run_dir: str | Path, data_dir: str | Path,
                 proxy_conditions: Sequence[str] = PROXY_CONDITIONS,
                 small: int = PROXY_SMALL,
                 box_run: Optional[str | Path] = None) -> Dict[str, Any]:
    """Reads a finished run and assembles the report dictionary.

    Input:  run_dir — ui2compose-eval-layout output directory; data_dir —
            package root (subsets and groupings for the proxy intervals);
            proxy_conditions — conditions of the correction proxy; small —
            'few edits' threshold; box_run — optional --box-repr container
            run of the same split (post-hoc sensitivity, item 6)
    Output: report dictionary (see the module docstring)
    """
    run = Path(run_dir)
    agg = json.loads((run / 'aggregate.json').read_text(encoding='utf-8'))
    meta = agg['meta']
    ref_records, cond_records = LE.read_run(run)
    conds = [c for c in CONDITION_ORDER if c in agg['conditions']]
    subsets_meta = meta['subsets']
    stems_all = sorted(ref_records)
    subsets: Dict[str, List[str]] = {'all': stems_all}
    for name, sub in LE.load_subsets(data_dir, meta['split']).items():
        subsets[name] = [s for s in sub if s in ref_records]
    if {k: len(set(v)) for k, v in subsets.items()} != subsets_meta:
        raise ValueError(f"subsets {({k: len(set(v)) for k, v in subsets.items()})} differ from "
                         f"the run's {subsets_meta}")
    defaults = LE.default_grouping_files(data_dir)
    group_files = {k: defaults.get(k, v) for k, v in meta['groupings'].items()
                   if k != LE.SCREEN_SCHEME}
    groupings = LE.load_groupings(group_files, stems_all)
    boot = agg['bootstrap']['all'][meta['primary_grouping']]
    report: Dict[str, Any] = {
        'meta': {
            'tool': 'ui2compose.evaluation.layout_report', 'ui2compose_version': __version__,
            'created_utc': _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'run_dir': str(run_dir), 'run_created_utc': meta['created_utc'], 'split': meta['split'],
            'n_screens': meta['n_screens'], 'subsets': subsets_meta, 'conditions': meta['conditions'],
            'params': meta['params'], 'prediction_meta': meta['prediction_meta'],
            'v110_code': meta['v110_code'], 'groupings': meta['groupings'],
            'primary_grouping': meta['primary_grouping'], 'table_schemes': list(TABLE_SCHEMES),
            'bootstrap': {'n': boot['n'], 'seed': boot['seed'], 'alpha': boot['alpha'],
                          'method': 'percentile; app groups drawn with replacement'},
            'parse_settings': meta['parse_settings'], 'versions': meta['versions'],
            'wall_seconds': meta['wall_seconds'], 'workers': meta['workers'],
        },
        'bootstrap_units': agg['bootstrap'],
        'reference': agg['reference'],
        'table': {sub: {c: condition_row(agg['conditions'][c][sub]) for c in conds}
                  for sub in subsets},
        'paired_differences': {
            pair: {sub: {scheme: {m: _ci(v) | {'n_screens_both': v.get('n_screens_both')}
                                  for m, v in by_scheme.items()}
                         for scheme, by_scheme in per_sub.items()}
                   for sub, per_sub in subs.items()}
            for pair, subs in agg['paired_differences'].items()},
        'correction_proxy': {
            sub: {c: correction_proxy(cond_records[c], subsets[sub], groupings, boot['n'],
                                      boot['seed'], small)
                  for c in proxy_conditions if c in cond_records}
            for sub in subsets},
        'zss_cross_check': agg['zss_cross_check'],
        'timing_seconds_per_1000_screens': {
            k: v.get('seconds_per_1000_screens') for k, v in agg['timing'].items()},
    }
    ordered = {c: cond_records[c] for c in conds if c in cond_records}
    report['grouping_undefined'] = {sub: {c: grouping_undefined(rec, stems)
                                          for c, rec in ordered.items()}
                                    for sub, stems in subsets.items()}
    report['grouping_common'] = {}
    report['post_hoc'] = {'note': 'added after the review of the revision; not part of the '
                                  'pre-specified evaluation', 'stratified_differences': {}}
    for sub, stems in subsets.items():
        boots = _boots(stems, groupings, boot['n'], boot['seed'])
        report['grouping_common'][sub] = grouping_common(ordered, boots)
        report['post_hoc']['stratified_differences'][sub] = stratified_differences(
            ref_records, ordered, boots)
    if box_run is not None:
        report['post_hoc']['box_sensitivity'] = box_sensitivity(
            box_run, meta, ref_records, cond_records, subsets, groupings, boot['n'], boot['seed'])
    return _r(report)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _f(v: Optional[float], d: int = 3) -> str:
    """Formats a number (en dash for None).

    Input:  v — number or None; d — decimals
    Output: string
    """
    return '–' if v is None else f"{v:.{d}f}"


def _cell(m: Mapping[str, Any], scheme: str, d: int = 3) -> str:
    """'mean [lo, hi]' cell of a metric for one scheme.

    Input:  m — row metric with 'mean' and 'ci_mean'; scheme — resampling scheme
    Output: string
    """
    iv = (m.get('ci_mean') or {}).get(scheme) or {}
    return f"{_f(m.get('mean'), d)} [{_f(iv.get('lo'), d)}, {_f(iv.get('hi'), d)}]"


def _med(d: Mapping[str, Any], digits: int = 0) -> str:
    """'median (q1–q3)' text.

    Input:  d — {'median', 'q1', 'q3'}; digits — decimals
    Output: string
    """
    return f"{_f(d.get('median'), digits)} ({_f(d.get('q1'), digits)}–{_f(d.get('q3'), digits)})"


def _diff_cell(iv: Optional[Mapping[str, Any]], d: int = 3) -> str:
    """'+0.123 [+0.100, +0.140]' text of a paired difference.

    Input:  iv — interval or None
    Output: string
    """
    if not iv or iv.get('point') is None:
        return '–'
    return f"{iv['point']:+.{d}f} [{iv['lo']:+.{d}f}, {iv['hi']:+.{d}f}]"


def markdown(rep: Mapping[str, Any]) -> str:
    """Readable Table-1 report.

    Input:  rep — build_report output
    Output: Markdown text
    """
    m = rep['meta']
    L: List[str] = []
    L.append(f"# Layout evaluation, {m['split']} split ({m['n_screens']} screens)")
    L.append('')
    p = m['params']
    pm = m.get('prediction_meta') or {}
    L.append(f"Generated by `python -m ui2compose.evaluation.layout_report` from the run "
             f"`{Path(m['run_dir']).name}` (created {m['run_created_utc']} UTC, ui2compose "
             f"{m['ui2compose_version']}). Method: `results/eval/README.md` and the docstring of "
             f"`src/ui2compose/evaluation/layout_eval.py`.")
    L.append('')
    L.append(f"* Thresholds (selected on val): de-duplication {p['dedup']['same_class_iou']} / "
             f"{p['dedup']['same_class_iomin']} / {p['dedup']['cross_class_iou']}, containment "
             f"ratio {p['containment_ratio']}, cut_tol_frac {p['cut_tol_frac']}, fallback "
             f"{p['fallback']}; matching IoU {p['match_iou']}.")
    if pm:
        st = pm.get('settings') or {}
        L.append(f"* Predictions: `{pm.get('weights')}` (MD5 {pm.get('weights_md5')}), conf "
                 f"{st.get('conf_threshold')}, NMS IoU {st.get('iou_threshold')}, imgsz "
                 f"{st.get('imgsz_effective')}, max_det {st.get('max_det')}, ultralytics "
                 f"{pm.get('ultralytics_version')}.")
    b = m['bootstrap']
    L.append(f"* Intervals: 95 % percentile bootstrap, {b['n']} replicates, seed {b['seed']}, app "
             f"groups drawn with replacement. Cells show `mean [lo, hi]` with the **{TABLE_SCHEMES[0]}** "
             f"grouping (primary, splits/full/manifest.csv); the second table of each subset repeats "
             f"them with **{TABLE_SCHEMES[1]}** groups.")
    L.append('* Edit operations are structure only (insert + delete + rename of the optimal '
             'unit-cost mapping = TED); nodes = canonical tree sizes.')
    L.append('* Grouping F1 is the mean over the screens where it is defined (at least one tree '
             'groups a pair of matched boxes); their number is given in parentheses, and differs '
             'between conditions. With predicted boxes it is computed over the boxes the detector '
             'found. "Grouping F1 on common screens" repeats the means on the screens where every '
             'condition defines it.')
    L.append('')
    for sub, rows in rep['table'].items():
        n = m['subsets'].get(sub)
        units = rep['bootstrap_units'][sub]
        L.append(f"## {SUBSET_TITLES.get(sub, sub)} ({n} screens; app groups: "
                 + ', '.join(f"{k} {v['n_units']} (largest {v['largest_unit_screens']} screens)"
                             for k, v in units.items() if k in TABLE_SCHEMES) + ')')
        L.append('')
        ref = rep['reference'][sub]
        L.append(f"Reference trees: {ref['n_ok']} of {ref['n_screens']} built "
                 f"(status {ref['status']}); flat dumps (depth ≤ 2) {ref['flat']['n_flat']} "
                 f"({_f(100 * ref['flat']['rate'], 1) if ref['flat']['rate'] is not None else '–'} %); "
                 f"median {_f(ref['n_gt']['median'], 0)} GT boxes and "
                 f"{_f(ref['ref_nodes']['median'], 0)} nodes per screen.")
        L.append('')
        for scheme in TABLE_SCHEMES:
            L.append(f"Intervals: **{scheme}**.")
            L.append('')
            L.append('| condition | n | TEDS-g | TEDS-o | grouping F1 (n defined) | edit ops TEDS-o, median (IQR) '
                     '| edit ops TEDS-g, median (IQR) | nodes gen / ref, median (IQR) |')
            L.append('|---|---:|---|---|---|---|---|---|')
            for c, r in rows.items():
                L.append(f"| {c} | {r['n_evaluated']} | {_cell(r['teds_g'], scheme)} | "
                         f"{_cell(r['teds_o'], scheme)} | {_cell(r['grouping_f1'], scheme)} "
                         f"({r['grouping_f1'].get('n', 0)}) | "
                         f"{_med(r['edit_ops_o']['total'])} | {_med(r['edit_ops_g']['total'])} | "
                         f"{_med(r['nodes']['generated'])} / {_med(r['nodes']['reference'])} |")
            L.append('')
        L.append('| condition | median TEDS-g / TEDS-o | grouping F1 undefined (< 2 matched boxes / '
                 'no grouped pair) | micro grouping F1 '
                 f'[{TABLE_SCHEMES[0]}] | GT boxes matched | layout boxes matched | class mismatches '
                 '| fallback nodes (screens) | placeholders (screens) | duplicates | lost | '
                 'de-dup removals |')
        L.append('|---|---|---:|---|---:|---:|---:|---|---|---:|---:|---:|')
        for c, r in rows.items():
            gm = r['grouping_micro']
            iv = (gm.get('ci_f1') or {}).get(TABLE_SCHEMES[0]) or {}
            mt = r['matching']
            und = (rep.get('grouping_undefined') or {}).get(sub, {}).get(c)
            und_txt = (f"{r['grouping_f1']['n_undefined']}" if not und else
                       f"{und['n_undefined']} ({und['n_undefined_fewer_than_2_matched']} / "
                       f"{und['n_undefined_no_grouped_pair']})")
            L.append(f"| {c} | {_f(r['teds_g']['median'])} / {_f(r['teds_o']['median'])} | "
                     f"{und_txt} | {_f(gm['f1'])} [{_f(iv.get('lo'))}, "
                     f"{_f(iv.get('hi'))}] | {_f(mt['gt_recall'])} | "
                     f"{_f(mt['layout_input_precision'])} | {mt['n_class_mismatch']} | "
                     f"{r['fallback']['total']:.0f} ({r['fallback']['screens_with']}) | "
                     f"{r['placeholder']['total']:.0f} ({r['placeholder']['screens_with']}) | "
                     f"{r['duplicates']['total']:.0f} | {r['lost']['total']:.0f} | "
                     f"{r['dedup_removed']['total']:.0f} |")
        L.append('')
    L.append('## Paired differences (mean per-screen difference, same bootstrap draws)')
    L.append('')
    L.append(f"Cells: point [95 % interval {TABLE_SCHEMES[0]}] / [{TABLE_SCHEMES[1]}]; P = "
             f"fraction of {TABLE_SCHEMES[0]} replicates ≤ 0. TEDS differences are over every "
             f"screen of the subset; a grouping-F1 difference is over the screens where both "
             f"conditions define F1 (n).")
    L.append('')
    L.append('| difference | subset | TEDS-g | TEDS-o | grouping F1 | micro grouping F1 |')
    L.append('|---|---|---|---|---|---|')
    for pair, subs in rep['paired_differences'].items():
        for sub, by_scheme in subs.items():
            cells = []
            for metric in ('teds_g', 'teds_o', 'grouping_f1', 'grouping_micro_f1'):
                a = (by_scheme.get(TABLE_SCHEMES[0]) or {}).get(metric)
                bb = (by_scheme.get(TABLE_SCHEMES[1]) or {}).get(metric)
                tail = '' if not a or a.get('p_le_0') is None else f" P={a['p_le_0']:.3f}"
                second = '' if not bb or bb.get('lo') is None else f" / [{bb['lo']:+.3f}, {bb['hi']:+.3f}]"
                if metric == 'grouping_f1' and a and a.get('n_screens_both') is not None:
                    tail += f" n={a['n_screens_both']}"
                cells.append(_diff_cell(a) + second + tail)
            L.append(f"| {pair} | {sub} | " + ' | '.join(cells) + ' |')
    L.append('')
    if rep.get('grouping_common'):
        L.append('## Grouping F1 on common screens')
        L.append('')
        L.append('Mean grouping F1 on the screens where every condition of the run defines it, '
                 f"95 % interval [{TABLE_SCHEMES[0]}] / [{TABLE_SCHEMES[1]}].")
        L.append('')
        conds = list(next(iter(rep['grouping_common'].values()))['conditions'])
        L.append('| subset | n screens | ' + ' | '.join(conds) + ' |')
        L.append('|---|---:|' + '---|' * len(conds))
        for sub, gc in rep['grouping_common'].items():
            cells = []
            for c in conds:
                x = gc['conditions'][c]
                a = x['ci'].get(TABLE_SCHEMES[0]) or {}
                bb = x['ci'].get(TABLE_SCHEMES[1]) or {}
                cells.append(f"{_f(x['mean'])} [{_f(a.get('lo'))}, {_f(a.get('hi'))}] / "
                             f"[{_f(bb.get('lo'))}, {_f(bb.get('hi'))}]")
            L.append(f"| {sub} | {gc['n_screens']} | " + ' | '.join(cells) + ' |')
        L.append('')
    L.extend(_post_hoc_markdown(rep))
    L.append(f"## Correction-effort proxy (structure only, edits = TED; ≤ {PROXY_SMALL} edits)")
    L.append('')
    L.append('Edits that turn the generated layout tree into the reference tree. Texts, sizes, '
             'styles, modifiers and the Kotlin itself are not counted, so this is a lower bound '
             'of the manual work on the structure, not an estimate of developer time.')
    L.append('')
    L.append(f"| subset | condition | variant | median (IQR) | mean | 90th pct | share 0 edits | "
             f"share ≤ {PROXY_SMALL} edits [{TABLE_SCHEMES[0]}] / [{TABLE_SCHEMES[1]}] | "
             "median insert / delete / rename | median edits per reference node |")
    L.append('|---|---|---|---|---:|---:|---:|---|---|---:|')
    for sub, by_cond in rep['correction_proxy'].items():
        for c, pr in by_cond.items():
            for v in LE.VARIANTS:
                x = pr.get(v) or {}
                if not x.get('n'):
                    continue
                a = x['share_le_small']['ci'].get(TABLE_SCHEMES[0]) or {}
                bb = x['share_le_small']['ci'].get(TABLE_SCHEMES[1]) or {}
                k = x['kinds']
                L.append(f"| {sub} | {c} | TEDS-{v} | {_med(x)} | {_f(x['mean'], 1)} | "
                         f"{_f(x['p90'], 0)} | {_f(x['share_0']['point'])} | "
                         f"{_f(x['share_le_small']['point'])} [{_f(a.get('lo'))}, {_f(a.get('hi'))}] / "
                         f"[{_f(bb.get('lo'))}, {_f(bb.get('hi'))}] | "
                         f"{_f(k['insert']['median'], 0)} / {_f(k['delete']['median'], 0)} / "
                         f"{_f(k['rename']['median'], 0)} | "
                         f"{_f(x['per_reference_node']['median'], 2)} |")
    L.append('')
    z = rep['zss_cross_check']
    L.append('## Checks')
    L.append('')
    L.append('* zss (Zhang-Shasha) cross-check of APTED: '
             + ', '.join(f"{c} {v['n_agree']}/{v['n_checked']}" for c, v in z.items()) + '.')
    ex = {c: r['status'] for c, r in rep['table']['all'].items()}
    L.append('* Status per condition (all screens): '
             + '; '.join(f"{c} {s}" for c, s in ex.items()) + '.')
    t = rep['timing_seconds_per_1000_screens']
    L.append('* Single-process seconds per 1,000 screens (generation + metrics): '
             + ', '.join(f"{k} {_f(v, 1)}" for k, v in t.items()) + '.')
    L.append('')
    return '\n'.join(L)


def _post_hoc_markdown(rep: Mapping[str, Any]) -> List[str]:
    """Markdown of the post-hoc sections (strata of the reference, containment-Box sensitivity).

    Input:  rep — build_report output
    Output: list of lines (empty without post-hoc results)
    """
    ph = rep.get('post_hoc') or {}
    L: List[str] = []
    sd = ph.get('stratified_differences')
    s0 = TABLE_SCHEMES[0]
    if sd:
        L.append('## Post hoc: paired differences by stratum of the reference')
        L.append('')
        L.append('Added after the review; not part of the pre-specified evaluation. Strata: flat '
                 'dump (depth ≤ 2) or not; one ground-truth box (every tree is one root with one '
                 'box, so C1 = C2) or at least two. Cells: point [95 % interval '
                 f"{s0}]; grouping F1 with the number of screens where both conditions define it. "
                 'All differences of the run are in the JSON.')
        L.append('')
        L.append('| subset | stratum | n | difference | TEDS-g | TEDS-o | grouping F1 |')
        L.append('|---|---|---:|---|---|---|---|')
        for sub, strata in sd.items():
            for name, entry in strata.items():
                for pair in ('C1-C2', 'C4-C5'):
                    d = (entry['differences'].get(pair) or {}).get(s0)
                    if not d:
                        continue
                    g = d.get('grouping_f1') or {}
                    gtxt = _diff_cell(g) + (f" n={g['n_screens_both']}"
                                            if g.get('n_screens_both') is not None else '')
                    L.append(f"| {sub} | {name} | {entry['n_screens']} | {pair} | "
                             f"{_diff_cell(d.get('teds_g'))} | {_diff_cell(d.get('teds_o'))} | "
                             f"{gtxt} |")
        L.append('')
    bs = ph.get('box_sensitivity')
    if bs:
        L.append('## Post hoc: containment Boxes scored as containers (sensitivity)')
        L.append('')
        L.append('Added after the review; the primary metric is unchanged. The run '
                 f"`{Path(bs['run_dir']).name}` (created {bs['run_created_utc']} UTC, "
                 f"`--box-repr {bs['box_repr']}`) scores a containment Box as a container whose "
                 "children are the container's own widget and the Box's content, as the Kotlin "
                 "draws them, instead of a node of the container's class over the content. "
                 f"Rescored: {', '.join(bs['conditions'])} (the other conditions have no "
                 'containment Boxes and keep their records). Reference trees identical to the '
                 f"primary run: {bs['checks']['reference_identical']}.")
        L.append('')
        L.append('| subset | condition | screens changed | TEDS-g | TEDS-o | grouping F1 (n) | '
                 'change TEDS-g | change TEDS-o | change grouping F1 |')
        L.append('|---|---|---:|---|---|---|---|---|---|')
        for sub, x in bs['subsets'].items():
            for c in bs['conditions']:
                r = x['table'][c]
                ch = x['change'][c].get(s0) or {}
                L.append(f"| {sub} | {c} | {x['n_screens_changed'][c]} | {_cell(r['teds_g'], s0)} | "
                         f"{_cell(r['teds_o'], s0)} | {_cell(r['grouping_f1'], s0)} "
                         f"({r['grouping_f1'].get('n', 0)}) | {_diff_cell(ch.get('teds_g'))} | "
                         f"{_diff_cell(ch.get('teds_o'))} | {_diff_cell(ch.get('grouping_f1'))} |")
        L.append('')
        L.append(f"Paired differences with the rescored conditions (cells: point [{s0}] / "
                 f"[{TABLE_SCHEMES[1]}]):")
        L.append('')
        L.append('| difference | subset | TEDS-g | TEDS-o | grouping F1 |')
        L.append('|---|---|---|---|---|')
        for pair, subs in bs['paired_differences'].items():
            for sub, by_scheme in subs.items():
                cells = []
                for metric in ('teds_g', 'teds_o', 'grouping_f1'):
                    a = (by_scheme.get(s0) or {}).get(metric)
                    bb = (by_scheme.get(TABLE_SCHEMES[1]) or {}).get(metric)
                    second = ('' if not bb or bb.get('lo') is None
                              else f" / [{bb['lo']:+.3f}, {bb['hi']:+.3f}]")
                    if metric == 'grouping_f1' and a and a.get('n_screens_both') is not None:
                        second += f" n={a['n_screens_both']}"
                    cells.append(_diff_cell(a) + second)
                L.append(f"| {pair} | {sub} | " + ' | '.join(cells) + ' |')
        L.append('')
    return L


def write_report(rep: Mapping[str, Any], out_json: str | Path, out_md: str | Path) -> None:
    """Writes the report JSON and Markdown.

    Input:  rep — build_report output; out_json, out_md — target files
    Output: none
    """
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    Path(out_md).write_text(markdown(rep), encoding='utf-8')


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Defines and reads the command-line arguments.

    Input:  argv — argument list (default sys.argv[1:])
    Output: argparse namespace
    """
    ap = argparse.ArgumentParser(description="Table-1 report of a layout evaluation run")
    ap.add_argument('--run', required=True, help="ui2compose-eval-layout output directory")
    ap.add_argument('--data-dir', required=True, help="the val/test package of the run")
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-md', required=True)
    ap.add_argument('--small', type=int, default=PROXY_SMALL,
                    help="'few edits' threshold of the correction proxy")
    ap.add_argument('--box-sensitivity-run', default=None,
                    help="post-hoc sensitivity: ui2compose-eval-layout --box-repr container run "
                         "(C1, C4) of the same split")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Writes the report of a finished run.

    Input:  argv — argument list
    Output: exit code
    """
    a = parse_args(argv)
    rep = build_report(a.run, a.data_dir, small=a.small, box_run=a.box_sensitivity_run)
    write_report(rep, a.out_json, a.out_md)
    for c, r in rep['table']['all'].items():
        print(f"  {c}: TEDS-g {r['teds_g']['mean']} TEDS-o {r['teds_o']['mean']} "
              f"grouping F1 {r['grouping_f1']['mean']}")
    print(f"written: {a.out_json}, {a.out_md}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
