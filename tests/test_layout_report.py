# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Tests of the Table-1 report (ui2compose.evaluation.layout_report).

Sections:
  1. Correction-effort proxy on hand-made records: quantiles, shares,
     intervals (constant values give point intervals), edit kinds.
  2. End to end on the synthetic package of test_layout_eval: the table is
     read from aggregate.json unchanged, the paired differences and the
     proxy are present, the Markdown renders, the CLI writes both files
     (skipped without apted / scipy).
  3. Grouping-F1 bookkeeping (undefined screens by reason, means on the
     common screens), the post-hoc strata and the containment-Box
     sensitivity merged from a second run (skipped without apted / scipy).
"""

import json
import math

import pytest

from test_layout_eval import make_package

from ui2compose.evaluation import layout_eval as LE
from ui2compose.evaluation import layout_report as R


def _rec(stem, ted_o, ted_g=None, ok=True):
    """Condition record with the fields the proxy reads.

    Input:  stem — stem; ted_o, ted_g — distances; ok — status ok
    Output: dict
    """
    if not ok:
        return {'stem': stem, 'status': 'excluded:no_predictions'}
    ted_g = ted_o if ted_g is None else ted_g
    return {'stem': stem, 'status': 'ok', 'ted_o': ted_o, 'ted_g': ted_g, 'ref_nodes': 10,
            'ops_o': {'insert': ted_o, 'delete': 0, 'rename': 0},
            'ops_g': {'insert': 0, 'delete': ted_g, 'rename': 0}}


# ── 1. proxy ───────────────────────────────────────────────

def test_correction_proxy_on_hand_made_records():
    """Median, IQR, shares of 0 and <= 5 edits, edit kinds and per-node edits; excluded screens are left out."""
    recs = {f"s{i}": _rec(f"s{i}", t) for i, t in enumerate([0, 0, 2, 5, 6, 12])}
    recs['x'] = _rec('x', 0, ok=False)
    stems = sorted(recs)
    groups = {'full': {s: s[:2] for s in stems}}
    out = R.correction_proxy(recs, stems, groups, n_boot=50, seed=0, small=5)
    o = out['o']
    assert (out['n_screens'], out['n_evaluated']) == (7, 6)
    assert (o['median'], o['q1'], o['q3'], o['max']) == (3.5, 0.5, 5.75, 12.0)
    assert o['share_0']['point'] == pytest.approx(2 / 6)
    assert o['share_le_small']['point'] == pytest.approx(4 / 6)
    assert o['kinds']['insert']['median'] == 3.5 and o['kinds']['delete']['median'] == 0.0
    assert o['per_reference_node']['median'] == pytest.approx(0.35)
    iv = o['share_le_small']['ci']['full']
    assert iv['lo'] <= iv['point'] <= iv['hi']
    const = R.correction_proxy({s: _rec(s, 3) for s in stems}, stems, groups, 50, 0)
    c = const['g']['share_le_small']['ci']['full']
    assert (c['point'], c['lo'], c['hi']) == (1.0, 1.0, 1.0)


def test_formatting_helpers():
    """Cells, medians and differences are formatted as in the tables."""
    m = {'mean': 0.5, 'ci_mean': {'full': {'lo': 0.4, 'hi': 0.61}}}
    assert R._cell(m, 'full') == '0.500 [0.400, 0.610]'
    assert R._cell(m, 'package_md5') == '0.500 [–, –]'
    assert R._med({'median': 3.0, 'q1': 1.0, 'q3': 7.0}) == '3 (1–7)'
    assert R._diff_cell({'point': 0.1, 'lo': -0.02, 'hi': 0.2}) == '+0.100 [-0.020, +0.200]'
    assert R._r({'a': float('nan'), 'b': [1.234567, 2]}) == {'a': None, 'b': [1.2346, 2]}


# ── 2. end to end ─────────────────────────────────────────

def test_report_end_to_end_on_the_synthetic_package(tmp_path, capsys):
    """The report reproduces aggregate.json, holds the proxy and the differences, and the CLI writes both files."""
    pytest.importorskip('apted')
    pytest.importorskip('scipy')
    pk = make_package(tmp_path)
    agg = LE.run_evaluation(pk['data'], 'val', tmp_path / 'run', conditions=['C1', 'C2', 'C4', 'C5'],
                            pred_cache=pk['pred'], n_boot=40, zss_check=2, progress_every=0)
    rep = R.build_report(tmp_path / 'run', pk['data'])
    assert list(rep['table']) == ['all', 'unseen', 'unseen_package_only']
    assert list(rep['table']['all']) == ['C1', 'C2', 'C4', 'C5']
    for c in ('C1', 'C4'):
        for m in LE.CI_METRICS:
            a = agg['conditions'][c]['all'][m]
            r = rep['table']['all'][c][m]
            if a.get('mean') is None:
                assert r['mean'] is None
            else:
                assert r['mean'] == pytest.approx(a['mean'], abs=1e-4)
                iv, iv_a = r['ci_mean']['full'], a['ci_mean']['full']
                assert (iv['lo'], iv['hi']) == (pytest.approx(iv_a['lo'], abs=1e-4),
                                                pytest.approx(iv_a['hi'], abs=1e-4))
    assert set(rep['paired_differences']) == {'C1-C2', 'C4-C5', 'C1-C4'}
    assert set(rep['correction_proxy']['all']) == {'C1', 'C4'}
    c1 = rep['correction_proxy']['all']['C1']
    assert c1['n_evaluated'] == 2 and math.isfinite(c1['o']['median'])
    assert rep['meta']['primary_grouping'] == 'full'
    md = R.markdown(rep)
    assert '| C4 |' in md and 'Correction-effort proxy' in md and 'Paired differences' in md
    code = R.main(['--run', str(tmp_path / 'run'), '--data-dir', pk['data'],
                   '--out-json', str(tmp_path / 'r.json'), '--out-md', str(tmp_path / 'r.md')])
    assert code == 0
    assert json.loads((tmp_path / 'r.json').read_text())['meta']['split'] == 'val'
    assert (tmp_path / 'r.md').read_text().startswith('# Layout evaluation, val split')
    assert 'written:' in capsys.readouterr().out


# ── 3. grouping bookkeeping and post-hoc sections ─────────

def _grec(stem, f1, n_matched):
    """Condition record with a grouping F1 and a number of matched boxes.

    Input:  stem — stem; f1 — grouping F1 or None; n_matched — matched boxes
    Output: dict
    """
    return {'stem': stem, 'status': 'ok', 'grouping_f1': f1, 'n_matched': n_matched}


def test_grouping_undefined_and_common_screens():
    """Undefined F1 is split by reason; the common-screen means use the screens every condition defines."""
    a = {'s1': _grec('s1', 1.0, 3), 's2': _grec('s2', None, 1), 's3': _grec('s3', None, 4),
         's4': _grec('s4', 0.5, 2), 's5': {'stem': 's5', 'status': 'excluded:no_predictions'}}
    b = {'s1': _grec('s1', 0.0, 3), 's2': _grec('s2', 0.2, 2), 's3': _grec('s3', 0.4, 4),
         's4': _grec('s4', None, 1), 's5': _grec('s5', 1.0, 5)}
    stems = sorted(a)
    assert R.grouping_undefined(a, stems) == {
        'n_evaluated': 4, 'n_defined': 2, 'n_undefined': 2,
        'n_undefined_fewer_than_2_matched': 1, 'n_undefined_no_grouped_pair': 1}
    boots = {'full': LE.Bootstrap(stems, {s: s for s in stems}, 30, 0)}
    gc = R.grouping_common({'A': a, 'B': b}, boots)
    assert gc['n_screens'] == 1                                    # only s1
    assert gc['conditions']['A']['mean'] == 1.0 and gc['conditions']['B']['mean'] == 0.0
    iv = gc['conditions']['A']['ci']['full']
    assert (iv['lo'], iv['hi']) == (1.0, 1.0)


def test_post_hoc_strata_and_box_sensitivity_on_the_synthetic_package(tmp_path):
    """Strata partition the screens; the Box-sensitivity run is merged, checked and rendered."""
    pytest.importorskip('apted')
    pytest.importorskip('scipy')
    pk = make_package(tmp_path)
    kw = dict(pred_cache=pk['pred'], n_boot=30, zss_check=0, progress_every=0)
    LE.run_evaluation(pk['data'], 'val', tmp_path / 'run', conditions=['C1', 'C2', 'C4', 'C5'], **kw)
    LE.run_evaluation(pk['data'], 'val', tmp_path / 'box', conditions=['C1', 'C4'],
                      box_repr='container', **kw)
    rep = R.build_report(tmp_path / 'run', pk['data'], box_run=tmp_path / 'box')
    und = rep['grouping_undefined']['all']['C1']
    assert und['n_defined'] + und['n_undefined'] == und['n_evaluated'] == 2
    assert set(rep['grouping_common']['all']['conditions']) == {'C1', 'C2', 'C4', 'C5'}
    sd = rep['post_hoc']['stratified_differences']['all']
    assert set(sd) == set(R.STRATA)
    assert sd['reference_flat']['n_screens'] + sd['reference_not_flat']['n_screens'] == 2
    assert sd['one_gt_box']['n_screens'] + sd['two_or_more_gt_boxes']['n_screens'] == 2
    bs = rep['post_hoc']['box_sensitivity']
    assert bs['box_repr'] == 'container' and bs['conditions'] == ['C1', 'C4']
    assert bs['checks']['reference_identical'] is True
    assert set(bs['paired_differences']) == {'C1-C2', 'C4-C5', 'C1-C4'}
    assert set(bs['subsets']['all']['change']) == {'C1', 'C4'}
    md = R.markdown(rep)
    assert 'Grouping F1 on common screens' in md
    assert 'Post hoc: paired differences by stratum' in md
    assert 'Post hoc: containment Boxes scored as containers' in md
    with pytest.raises(ValueError, match='box-repr container'):
        R.build_report(tmp_path / 'run', pk['data'], box_run=tmp_path / 'run')
    code = R.main(['--run', str(tmp_path / 'run'), '--data-dir', pk['data'], '--out-json',
                   str(tmp_path / 'r.json'), '--out-md', str(tmp_path / 'r.md'),
                   '--box-sensitivity-run', str(tmp_path / 'box')])
    assert code == 0 and 'box_sensitivity' in json.loads((tmp_path / 'r.json').read_text())['post_hoc']
