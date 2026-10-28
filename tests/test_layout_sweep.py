# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Tests of the validation threshold sweep (ui2compose.evaluation.layout_sweep).

Sections:
  1. Grid: sizes, unique keys, the a-priori literals, 'off' disables every
     de-duplication rule, distances to the a-priori configuration.
  2. Selection rule on hand-made rows: objective, ties, tie-break,
     closeness to the a-priori default, eligibility.
  3. End to end on the synthetic package of test_layout_eval: the sweep's
     a-priori records equal ui2compose-eval-layout's, the detector check
     without de-duplication equals ap.operating_point, the output files,
     the test-split guard (skipped without apted / scipy).
  4. Post-hoc selection check on a hand-made sweep: grid edges, the rule
     applied on halves of the app groups, the rewritten summary files.
"""

import json

import pytest

from test_layout_eval import make_package

from ui2compose.evaluation import ap as apm
from ui2compose.evaluation import layout_eval as LE
from ui2compose.evaluation import layout_sweep as S
from ui2compose.infer.postprocess import deduplicate_detections


# ── 1. grid ────────────────────────────────────────────────

def test_grid_sizes_keys_and_a_priori_literals():
    """120 C4, 30 C1 and 4 C5 configurations with unique keys; the a-priori values are the provisional v1.2.0 literals."""
    cfgs = S.sweep_configs()
    by = {c: [x for x in cfgs if x['condition'] == c] for c in S.SWEPT_CONDITIONS}
    assert (len(by['C4']), len(by['C1']), len(by['C5'])) == (120, 30, 4)
    keys = [S.config_key(c) for c in cfgs]
    assert len(set(keys)) == len(keys)
    assert S.A_PRIORI == {'dedup': 'a_priori', 'containment_ratio': 0.95,
                          'cut_tol_frac': 0.005, 'fallback': 'y_sorted'}
    assert S.DEDUP_SETTINGS['a_priori'] == {'same_class_iou': 0.5, 'same_class_iomin': 0.9,
                                            'cross_class_iou': 0.6}
    assert S.DEDUP_SETTINGS['stricter'] == {'same_class_iou': 0.6, 'same_class_iomin': 0.95,
                                            'cross_class_iou': 0.7}
    assert S.DEDUP_SETTINGS['looser'] == {'same_class_iou': 0.4, 'same_class_iomin': 0.85,
                                          'cross_class_iou': 0.5}
    assert set(S.DEDUP_ORDER) == set(S.DEDUP_SETTINGS)
    for cond in S.SWEPT_CONDITIONS:
        assert S.config_key(S.a_priori_config(cond)) in keys
    assert S.config_key(S.a_priori_config('C4')) == 'C4|dedup=a_priori|cr=0.95|tol=0.005|fb=y_sorted'
    assert S.config_key(S.a_priori_config('C1')) == 'C1|cr=0.95|tol=0.005|fb=y_sorted'


def test_dedup_off_removes_nothing_and_params_have_the_eval_shape():
    """'off' disables every rule (as run_pipeline(dedup=False)); config_params has layout_eval's keys."""
    dets = [{'id': 0, 'class_name': 'TextView', 'confidence': 0.9, 'bbox': [0, 0, 10, 10]},
            {'id': 1, 'class_name': 'TextView', 'confidence': 0.8, 'bbox': [0, 0, 10, 10]},
            {'id': 2, 'class_name': 'Button', 'confidence': 0.7, 'bbox': [0, 0, 10, 10]}]
    kept, removed = deduplicate_detections(dets, **S.DEDUP_SETTINGS['off'])
    assert len(kept) == 3 and removed == []
    kept, removed = deduplicate_detections(dets, **S.DEDUP_SETTINGS['a_priori'])
    assert [d['id'] for d in kept] == [0]
    p = S.config_params(S.a_priori_config('C4'))
    assert set(p) == set(LE.default_params())
    assert p['dedup'] == S.DEDUP_SETTINGS['a_priori'] and p['match_iou'] == LE.MATCH_IOU


def test_distance_to_a_priori():
    """Changed parameters and grid steps from the a-priori configuration."""
    a = S.a_priori_config('C4')
    assert S.distance_to_a_priori(a) == (0, 0)
    assert S.distance_to_a_priori({**a, 'cut_tol_frac': 0.02}) == (1, 2)
    assert S.distance_to_a_priori({**a, 'fallback': 'least_overlap', 'dedup': 'off'}) == (2, 3)
    assert S.distance_to_a_priori(S.a_priori_config('C1')) == (0, 0)
    assert S.distance_to_a_priori({**S.a_priori_config('C1'), 'containment_ratio': 1.0}) == (1, 1)


# ── 2. selection rule ──────────────────────────────────────

def _row(obj, tb, **changes):
    """A hand-made summary row of a C4 configuration.

    Input:  obj — mean TEDS-o; tb — mean grouping F1; changes — parameters
            that differ from the a-priori configuration
    Output: row for select_config
    """
    cfg = {**S.a_priori_config('C4'), **changes}
    return {'config': cfg, 'key': S.config_key(cfg), S.OBJECTIVE: obj, S.TIE_BREAK: tb}


def test_select_prefers_the_objective_then_the_tie_break_then_the_a_priori():
    """Ties within TIE_TOL go to the higher grouping F1, then to the configuration closest to the a priori."""
    rows = [_row(0.500, 0.60),                                   # a priori
            _row(0.520, 0.50, cut_tol_frac=0.02),                # clear winner on the objective
            _row(0.5195, 0.70, cut_tol_frac=0.01),               # tied, better tie-break
            _row(0.5199, 0.7005, cut_tol_frac=0.01, fallback='least_overlap')]   # tied twice, farther
    r = S.select_config(rows)
    assert r['argmax'] == rows[1]['key']
    assert set(r['tied_objective']) == {rows[1]['key'], rows[2]['key'], rows[3]['key']}
    assert set(r['tied_tie_break']) == {rows[2]['key'], rows[3]['key']}
    assert r['selected'] == rows[2]['key']
    # Without a near-tie the best objective wins outright.
    r = S.select_config([rows[0], _row(0.53, 0.1, containment_ratio=1.0)])
    assert r['selected'] == r['argmax'] == S.config_key({**S.a_priori_config('C4'),
                                                         'containment_ratio': 1.0})


def test_select_respects_the_detector_eligibility():
    """An ineligible de-duplication setting cannot be selected, even with the best objective."""
    rows = [_row(0.50, 0.6), _row(0.60, 0.6, dedup='looser')]
    r = S.select_config(rows, eligible={'off': True, 'stricter': True, 'a_priori': True,
                                        'looser': False})
    assert r['selected'] == rows[0]['key'] and r['ineligible'] == [rows[1]['key']]
    with pytest.raises(ValueError, match='no eligible'):
        S.select_config(rows[1:], eligible={'looser': False})


# ── 3. end to end ──────────────────────────────────────────

def test_sweep_end_to_end_on_a_synthetic_package(tmp_path):
    """The sweep reproduces ui2compose-eval-layout at the a-priori thresholds and writes the selection files."""
    pytest.importorskip('apted')
    pytest.importorskip('scipy')
    pk = make_package(tmp_path)
    path = S.run_sweep(pk['data'], 'val', tmp_path / 'sweep', pk['pred'], progress_every=0)
    refs, cfgs = S.read_sweep(path)
    assert len(refs) == 4 and len(cfgs) == 154
    _ = LE.run_evaluation(pk['data'], 'val', tmp_path / 'eval', conditions=['C1', 'C4', 'C5'],
                          params=S.config_params(S.a_priori_config('C4')), pred_cache=pk['pred'],
                          n_boot=0, zss_check=0, progress_every=0)
    _, ev = LE.read_run(tmp_path / 'eval')
    for cond in ('C1', 'C4', 'C5'):
        mine = cfgs[S.config_key(S.a_priori_config(cond))]
        for stem, rec in ev[cond].items():
            assert mine[stem]['status'] == rec['status']
            for k in ('teds_g', 'teds_o', 'grouping_f1', 'n_fallback', 'gen_nodes'):
                assert mine[stem].get(k) == rec.get(k), (cond, stem, k)

    summary = S.summarize_sweep(pk['data'], 'val', path, pk['pred'], n_boot=20)
    sel = summary['selection']
    assert sel['C4']['selected'] in summary['configs'] and sel['C1']['selected'] in summary['configs']
    assert sel['C4']['a_priori'] == S.config_key(S.a_priori_config('C4'))
    assert set(sel['gt_vs_pred_optimal']['same']) == {'containment_ratio', 'cut_tol_frac', 'fallback'}
    a = summary['configs'][sel['C4']['a_priori']]
    assert a['minus_a_priori']['full']['teds_o']['point'] == 0.0
    assert summary['meta']['n_screens'] == 4 and summary['meta']['primary_grouping'] == 'full'

    # Detector check: 'off' equals ap.operating_point on the raw cached detections.
    det = summary['detector_dedup_check']['settings']
    gts, _ = __import__('ui2compose.evaluation.evaluate_detector', fromlist=['x']).load_ground_truth(
        [tmp_path / 'pkg' / 'dataset' / 'images' / 'val' / f'{s}.png' for s in sorted(refs)],
        tmp_path / 'pkg' / 'dataset' / 'labels' / 'val', 13)
    names = LE.load_names(pk['data'])
    raw = {}
    for s in sorted(refs):
        d = LE.load_predictions_file(tmp_path / 'pred' / f'{s}.json') or []
        raw[s] = S._as_entry(d, {n: i for i, n in enumerate(names)})
    op = apm.operating_point(raw, gts, names, S.CHECK_CONF, S.CHECK_IOU)
    assert (det['off']['tp'], det['off']['fp'], det['off']['fn']) == (op['tp'], op['fp'], op['fn'])
    assert det['off']['eligible'] and det['off']['f1_minus_off_point'] == 0.0

    paths = S.write_outputs(summary, tmp_path / 'results', date='2026-10-20')
    sel_json = json.loads(open(paths['selected_thresholds.json'], encoding='utf-8').read())
    assert 'selected on validation only' in sel_json['note']
    assert sel_json['date'] == '2026-10-20'
    assert set(sel_json['values']) == {'dedup', 'containment_ratio', 'cut_tol_frac', 'fallback'}
    md = open(paths['val_sweep.md'], encoding='utf-8').read()
    assert '## Selection' in md and '★' in md and '◆' in md and 'in-sample' in md
    # Post-hoc selection check: written next to the summary, referenced from both files.
    cv = S.selection_crossval(path, pk['data'], repeats=3, scheme='full')
    assert cv['out_of_fold_gain']['n'] + cv['n_skipped_halves'] == 6
    paths = S.write_outputs(summary, tmp_path / 'results2', date='2026-10-20', crossval=cv)
    assert S.CROSSVAL_FILE in paths
    sel2 = json.loads(open(paths['selected_thresholds.json'], encoding='utf-8').read())
    assert sel2['selection_check']['repeats'] == 3
    assert sel2['files']['selection_check'] == f"results/eval/{S.CROSSVAL_FILE}"
    assert 'Selection check (post hoc)' in open(paths['val_sweep.md'], encoding='utf-8').read()


def test_sweep_refuses_the_test_split(tmp_path):
    """The sweep, its summary, the detector check and the CLI never accept the test split."""
    pk = make_package(tmp_path)
    with pytest.raises(ValueError, match='validation'):
        S.run_sweep(pk['data'], 'test', tmp_path / 'o', pk['pred'])
    with pytest.raises(ValueError, match='validation'):
        S.summarize_sweep(pk['data'], 'test', tmp_path / 'o', pk['pred'])
    with pytest.raises(ValueError, match='validation'):
        S.dedup_operating_points(pk['data'], 'test', pk['pred'], [], {})
    with pytest.raises(SystemExit):
        S.parse_args(['run', '--data-dir', pk['data'], '--split', 'test', '--pred-cache',
                      pk['pred'], '--out', str(tmp_path / 'o')])


# ── 4. post-hoc selection check ───────────────────────────

def _hand_made_sweep(root, best_key, n_screens=8):
    """A sweep.jsonl and a package with a package_md5 manifest: one configuration is best on every screen.

    Input:  root — directory; best_key — configuration with TEDS-o 0.6
            (all others 0.5); n_screens — screens (one app group each)
    Output: (sweep path, package dir)
    """
    stems = [f"app{i}-screens_screenshot_1" for i in range(n_screens)]
    keys = [S.config_key(c) for c in S.sweep_configs(['C4'])]
    path = root / 'sweep.jsonl'
    with open(path, 'w', encoding='utf-8') as f:
        for s in stems:
            cfgs = {k: {'status': 'ok', 'teds_o': 0.6 if k == best_key else 0.5,
                        'grouping_f1': 0.5} for k in keys}
            f.write(json.dumps({'stem': s, 'reference': {'status': 'ok'}, 'configs': cfgs}) + '\n')
    man = root / 'pkg' / 'splits' / 'package_md5' / 'manifest.csv'
    man.parent.mkdir(parents=True)
    man.write_text('stem,group_id\n' + ''.join(f"{s},{i}\n" for i, s in enumerate(stems)),
                   encoding='utf-8')
    return path, root / 'pkg'


def test_grid_edges():
    """Edges are named by their position in the grid; interior values are None."""
    shipped = {'condition': 'C4', 'dedup': 'looser', 'containment_ratio': 0.9,
               'cut_tol_frac': 0.0025, 'fallback': 'y_sorted'}
    assert S.grid_edges(shipped) == {'dedup': 'last', 'containment_ratio': 'last',
                                     'cut_tol_frac': None}
    assert S.grid_edges(S.a_priori_config('C4')) == {'dedup': None, 'containment_ratio': None,
                                                     'cut_tol_frac': None}
    off = {**shipped, 'dedup': 'off', 'containment_ratio': 1.0, 'cut_tol_frac': 0.02}
    assert S.grid_edges(off) == {'dedup': 'first', 'containment_ratio': 'first',
                                 'cut_tol_frac': 'last'}
    assert 'dedup' not in S.grid_edges(S.a_priori_config('C1'))


def test_selection_crossval_on_a_hand_made_sweep(tmp_path):
    """A configuration best on every screen is selected on every half, with no optimism; files are rewritten."""
    best = 'C4|dedup=looser|cr=0.9|tol=0.0025|fb=y_sorted'
    path, pkg = _hand_made_sweep(tmp_path, best)
    cv = S.selection_crossval(path, pkg, repeats=5, seed=0)
    assert cv['whole_split']['selected'] == best and cv['shipped'] == best
    assert cv['n_groups'] == 8 and cv['out_of_fold_gain']['n'] == 10
    assert cv['out_of_fold_gain']['mean'] == pytest.approx(0.1)
    assert cv['in_fold_gain']['mean'] == pytest.approx(0.1) and cv['optimism'] == pytest.approx(0.0)
    assert cv['shipped_selected_share'] == 1.0 and cv['selected_counts'] == {best: 10}
    assert cv['grid_edges'] == {'dedup': 'last', 'containment_ratio': 'last', 'cut_tol_frac': None}
    with pytest.raises(ValueError, match='validation'):
        S.selection_crossval(path, pkg, split='test')
    with pytest.raises(ValueError, match='unknown grouping'):
        S.selection_crossval(path, pkg, scheme='full')
    assert S.parse_args(['crossval', '--data-dir', str(pkg), '--sweep', str(path),
                         '--results-dir', str(tmp_path)]).repeats == S.CROSSVAL_REPEATS
    with pytest.raises(SystemExit):
        S.parse_args(['crossval', '--data-dir', str(pkg), '--split', 'test', '--sweep', str(path),
                      '--results-dir', str(tmp_path)])
