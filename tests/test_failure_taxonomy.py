# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Tests of the failure taxonomy (ui2compose.evaluation.failure_taxonomy).

Sections:
  1. Geometry helpers: exact union area, stacked vs flat-2D reference
     containers, tolerance splits.
  2. Reference evidence from XML and pixels: covered by a later sibling of
     comparable size (a large background does not hide), open drawer,
     shared pixels (a small icon on a large button is not stacked), boxes
     beyond the screen, uniform pixels, dumped window.
  3. Edit mapping: the edits add up to layout_eval's TED; the tie-break
     pairs matched detections (skipped without apted).
  4. Classification of single edits on hand-made trees: wrong class, FP,
     FN, de-duplication removal, fallback, tolerance cut, containment,
     stacked-equivalent composite, window artifact (skipped without scipy).
  5. Sample, summary, case selection, reports and the test-split guard on
     the synthetic package of test_layout_eval (skipped without apted /
     scipy).
"""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from test_layout_eval import H, W, hierarchy, make_package, xml_node

from ui2compose.evaluation import failure_taxonomy as FT
from ui2compose.evaluation import layout_eval as LE


def det(cls, i, box, src='leaf', children=None):
    """Shorthand for a detection node.

    Input:  cls — class; i — id; box — [x1, y1, x2, y2]; src — provenance; children — nodes
    Output: TreeNode
    """
    return LE.detection(cls, i, box, src, children)


def grp(src, *children):
    """Shorthand for a container node.

    Input:  src — provenance (e.g. 'Column:gap' or 'xml:LinearLayout@3'); children — nodes
    Output: TreeNode
    """
    return LE.container(src, list(children))


def gt_box(i, cls, box):
    """Ground-truth box dictionary as parse_label_lines returns it.

    Input:  i — id; cls — class; box — [x1, y1, x2, y2]
    Output: dict
    """
    return {'id': i, 'class_name': cls, 'bbox': [float(v) for v in box], 'confidence': 1.0}


def det_box(i, cls, box, conf=0.9):
    """Detection dictionary as the detector returns it.

    Input:  i — id; cls — class; box — [x1, y1, x2, y2]; conf — confidence
    Output: dict
    """
    return {'id': i, 'class_name': cls, 'bbox': [float(v) for v in box], 'confidence': conf}


# ── 1. geometry ────────────────────────────────────────────

def test_union_area_is_exact():
    """Overlapping, nested and degenerate boxes are counted once."""
    assert FT.union_area([]) == 0.0
    assert FT.union_area([[0, 0, 10, 10], [5, 5, 15, 15]]) == 175.0
    assert FT.union_area([[0, 0, 10, 10], [2, 2, 3, 3], [4, 4, 4, 9]]) == 100.0
    assert FT.union_area([[0, 0, 1, 1], [2, 0, 3, 1]]) == 2.0


def test_stacked_flat_and_regular_reference_containers():
    """'Box' containers are stacked when two children intersect, flat 2D otherwise; Column / Row are neither."""
    flat = LE.canonicalize(grp('xml:RelativeLayout@1', det('ImageView', 0, [0, 0, 10, 30]),
                               det('TextView', 1, [12, 0, 50, 10]), det('TextView', 2, [12, 12, 50, 22])))
    stacked = LE.canonicalize(grp('xml:FrameLayout@1', det('ImageView', 0, [0, 0, 50, 30]),
                                  det('TextView', 1, [5, 20, 45, 28]), det('Button', 2, [60, 0, 70, 10])))
    column = LE.canonicalize(grp('xml:LinearLayout@1', det('TextView', 0, [0, 0, 50, 10]),
                                 det('TextView', 1, [0, 10, 50, 20])))
    assert (LE.label_o(flat), FT.stacked_or_flat(flat)) == ('Box', 'reference_flat2d')
    assert (LE.label_o(stacked), FT.stacked_or_flat(stacked)) == ('Box', 'reference_overlap')
    assert FT.stacked_or_flat(column) is None
    assert FT.stacked_or_flat(det('TextView', 0, [0, 0, 1, 1])) is None


def test_tolerance_split_needs_a_positive_overlap_along_the_cut_axis():
    """A gap_tolerance Column split two children apart only if their y-intervals overlap."""
    a, b, c = det('TextView', 0, [0, 0, 50, 12]), det('TextView', 1, [0, 10, 50, 20]), \
        det('TextView', 2, [0, 25, 50, 30])
    col = LE.canonicalize(grp('Column:gap_tolerance', a, b, c))
    k = col.children
    assert FT.tolerance_split(col, k[:2]) and not FT.tolerance_split(col, k[1:])
    plain = LE.canonicalize(grp('Column:gap', a, b, c))
    assert not FT.tolerance_split(plain, plain.children)
    row = LE.canonicalize(grp('Row:gap_tolerance', det('Button', 0, [0, 0, 22, 10]),
                              det('Button', 1, [20, 0, 40, 10])))
    assert FT.tolerance_split(row, row.children)
    assert FT.is_fallback(LE.canonicalize(grp('Column:fallback_y_sorted', a, b)))
    assert FT.is_fallback(det('ListView', 9, [0, 0, 50, 50], 'LazyColumn:least_overlap', [a, b]))
    assert not FT.is_fallback(det('ListView', 9, [0, 0, 50, 50], 'LazyColumn:gap', [a, b]))


# ── 2. reference evidence ─────────────────────────────────

def _evidence(xml_text, boxes, gray=None, size=(W, H)):
    """XmlEvidence of a hierarchy whose label elements are the nodes with a 'label' attribute.

    Input:  xml_text — hierarchy text; boxes — {label id: [x1, y1, x2, y2]};
            gray — grey-level image or None; size — (w, h)
    Output: XmlEvidence
    """
    root = ET.fromstring(xml_text)
    elems = sorted((e for e in root.iter() if 'label' in e.attrib), key=lambda e: int(e.attrib['label']))
    gt = [gt_box(int(e.attrib['label']), 'TextView', boxes[int(e.attrib['label'])]) for e in elems]
    return FT.XmlEvidence(root, elems, gt, size, gray)


def test_covered_by_a_later_sibling_of_comparable_size_but_not_by_a_background():
    """A later sibling of comparable size hides a label; a much larger later surface does not."""
    xml = hierarchy(xml_node('android.widget.FrameLayout', (0, 0, 120, 192), [
        xml_node('android.widget.TextView', (10, 10, 50, 30), label=0),
        xml_node('android.widget.FrameLayout', (5, 5, 55, 35), [
            xml_node('android.widget.Button', (5, 5, 55, 35), label=1)]),
        xml_node('android.widget.TextView', (10, 100, 50, 120), label=2),
        xml_node('android.widget.ImageView', (0, 50, 120, 192), label=3),
    ]))
    boxes = {0: [10, 10, 50, 30], 1: [5, 5, 55, 35], 2: [10, 100, 50, 120], 3: [0, 50, 120, 192]}
    ev = _evidence(xml, boxes)
    assert 0 in ev.hidden and 'covered by later sibling' in ev.hidden[0]
    assert 2 not in ev.hidden            # the later image is 17x its area: a background
    assert ev.window == [0.0, 0.0, 120.0, 192.0]


def test_open_drawer_beyond_screen_and_uniform_pixels():
    """Content under an open drawer, boxes beyond the screen and uniform regions are flagged."""
    xml = hierarchy(xml_node('android.support.v4.widget.DrawerLayout', (0, 0, 120, 192), [
        xml_node('android.widget.LinearLayout', (0, 0, 120, 192), [
            xml_node('android.widget.TextView', (60, 10, 110, 30), label=0),
            xml_node('android.widget.ImageView', (100, 150, 140, 190), label=2),
        ]),
        xml_node('android.widget.ListView', (0, 0, 50, 192), [
            xml_node('android.widget.TextView', (0, 40, 50, 60), label=1)]),
    ]))
    boxes = {0: [60, 10, 110, 30], 1: [0, 40, 50, 60], 2: [100, 150, 140, 190]}
    gray = np.zeros((H, W), dtype=np.float32)
    gray[::2, :] = 255.0                  # texture everywhere ...
    gray[0:40, 55:120] = 128.0            # ... except a flat patch under label 0
    ev = _evidence(xml, boxes, gray)
    assert 'open drawer' in ev.hidden[0]
    assert 1 not in ev.hidden             # the drawer itself is on top
    assert 'beyond' in ev.artifact[2]
    assert 'uniform' in ev.artifact[0] and 1 not in ev.artifact


def test_labels_sharing_their_pixels_are_stacked():
    """Two sibling labels on the same pixels are stacked; a small icon on a large button is not."""
    xml = hierarchy(xml_node('android.widget.FrameLayout', (0, 0, 120, 192), [
        xml_node('android.widget.EditText', (0, 100, 40, 120), label=0),
        xml_node('android.widget.TextView', (0, 100, 40, 118), label=1),
        xml_node('android.widget.Button', (0, 10, 120, 50), label=2),
        xml_node('android.widget.ImageView', (100, 20, 110, 30), label=3),
    ]))
    boxes = {0: [0, 100, 40, 120], 1: [0, 100, 40, 118], 2: [0, 10, 120, 50], 3: [100, 20, 110, 30]}
    ev = _evidence(xml, boxes)
    assert 'under other labels' in ev.hidden[0] and 'under other labels' in ev.hidden[1]
    assert 3 not in ev.hidden and 2 not in ev.hidden


def test_dumped_window_of_a_dialog():
    """The window is the union of the top-level nodes' bounds."""
    xml = hierarchy(xml_node('android.widget.FrameLayout', (20, 60, 100, 120), [
        xml_node('android.widget.TextView', (30, 70, 90, 90), label=0)]))
    ev = _evidence(xml, {0: [30, 70, 90, 90]})
    assert ev.window == [20.0, 60.0, 100.0, 120.0]
    assert ev.element_of('xml:FrameLayout@1') is not None and ev.element_of('Column:gap') is None


# ── 3. edit mapping ───────────────────────────────────────

def test_edit_mapping_adds_up_and_prefers_matched_pairs():
    """The edits add up to tree_edit's TED; with the tie-break, matched detections are paired."""
    pytest.importorskip('apted')
    ref = LE.canonicalize(grp('xml:TableRow@1', det('ImageView', 10, [0, 0, 10, 10]),
                              det('ImageView', 11, [20, 0, 30, 10]), det('ImageView', 12, [40, 0, 50, 10])))
    gen = LE.canonicalize(grp('Row:gap', det('Button', 1, [20, 0, 30, 10]),
                              det('Button', 2, [40, 0, 50, 10])))
    ted, edits = FT.edit_mapping(ref, gen, 'o', {(1, 11), (2, 12)})
    assert ted == LE.tree_edit(ref, gen, 'o')['ted'] == 3
    renames = {(g.det_id, r.det_id) for g, r in edits if g is not None and r is not None}
    inserts = [r.det_id for g, r in edits if g is None]
    assert renames == {(1, 11), (2, 12)} and inserts == [10]
    ted0, edits0 = FT.edit_mapping(ref, gen, 'o')
    assert ted0 == ted and len(edits0) == ted


# ── 4. classification ─────────────────────────────────────

def _classifier(ref, gen, gt, kept, raw=None, removed=(), ev=None):
    """ScreenClassifier on canonical copies.

    Input:  ref, gen — raw trees; gt, kept, raw, removed — box lists; ev — XmlEvidence or None
    Output: (classifier, canonical ref, canonical gen)
    """
    r, g = LE.canonicalize(ref), LE.canonicalize(gen)
    return FT.ScreenClassifier(r, g, gt, kept, raw if raw is not None else kept, list(removed), ev), r, g


def _by_det(tree, det_id):
    """Detection node of a tree by id.

    Input:  tree — TreeNode; det_id — id
    Output: node
    """
    return next(n for n in FT._detections(tree) if n.det_id == det_id)


def test_box_level_categories():
    """wrong_class, detector_fp, detector_fn and dedup_removed on one screen."""
    pytest.importorskip('scipy')
    gt = [gt_box(0, 'TextView', [0, 0, 50, 10]), gt_box(1, 'Button', [0, 20, 50, 30]),
          gt_box(2, 'TextView', [0, 40, 50, 50]), gt_box(3, 'TextView', [0, 60, 50, 70])]
    kept = [det_box(5, 'TextView', [0, 0, 50, 10]), det_box(6, 'TextView', [0, 20, 50, 30]),
            det_box(7, 'ImageView', [0, 80, 50, 100])]
    raw = kept + [det_box(8, 'TextView', [0, 40, 50, 50], 0.4)]
    removed = [dict(raw[-1], dedup_rule='cross_class_iou', removed_by=7)]
    ref = grp('xml:LinearLayout@1', *[det(b['class_name'], b['id'], b['bbox']) for b in gt])
    gen = grp('Column:gap', *[det(d['class_name'], d['id'], d['bbox']) for d in kept])
    clf, r, g = _classifier(ref, gen, gt, kept, raw, removed)
    assert clf.classify(_by_det(g, 6), _by_det(r, 1))[1] == 'wrong_class'
    assert clf.classify(_by_det(g, 7), None)[1] == 'detector_fp'
    kind, cat, _, eid = clf.classify(None, _by_det(r, 2))
    assert (kind, cat, eid) == ('insert', 'dedup_removed', 8)
    assert clf.classify(None, _by_det(r, 3))[1] == 'detector_fn'
    assert (clf.n_fp, clf.n_fn, clf.n_class_mismatch) == (1, 2, 1)


def test_layout_rule_categories():
    """Fallback containers, tolerance cuts and containment differences."""
    pytest.importorskip('scipy')
    gt = [gt_box(0, 'TextView', [0, 0, 50, 10]), gt_box(1, 'TextView', [0, 8, 50, 20]),
          gt_box(2, 'ImageView', [0, 30, 50, 60]), gt_box(3, 'TextView', [5, 40, 45, 50])]
    kept = [det_box(i, b['class_name'], b['bbox']) for i, b in enumerate(gt)]
    ref = grp('xml:LinearLayout@1', grp('xml:LinearLayout@2', det('TextView', 0, gt[0]['bbox']),
                                        det('TextView', 1, gt[1]['bbox'])),
              grp('xml:LinearLayout@3', det('ImageView', 2, gt[2]['bbox']),
                  det('TextView', 3, gt[3]['bbox'])))
    gen = grp('Column:gap', grp('Column:fallback_y_sorted', det('TextView', 0, gt[0]['bbox']),
                                det('TextView', 1, gt[1]['bbox'])),
              det('ImageView', 2, gt[2]['bbox'], 'Box', [det('TextView', 3, gt[3]['bbox'])]))
    clf, r, g = _classifier(ref, gen, gt, kept)
    fb = g.children[0]
    assert clf.classify(fb, None)[1] == 'xycut_fallback'
    assert clf.classify(None, _by_det(r, 3))[1] == 'containment'
    tol = LE.canonicalize(grp('Column:gap_tolerance', det('TextView', 0, gt[0]['bbox']),
                              det('TextView', 1, gt[1]['bbox'])))
    clf2 = FT.ScreenClassifier(LE.canonicalize(ref), tol, gt, kept, kept, [], None)
    assert clf2.classify(tol, None)[1] == 'tolerance_cut'


def test_stacked_reference_and_window_artifact():
    """A composite Box over a FrameLayout of siblings is the same picture; a detection outside the dump is an artifact."""
    pytest.importorskip('scipy')
    gt = [gt_box(0, 'ImageView', [0, 0, 100, 60]), gt_box(1, 'TextView', [10, 40, 90, 55])]
    kept = [det_box(4, 'ImageView', [0, 0, 100, 60]), det_box(5, 'TextView', [10, 40, 90, 55]),
            det_box(6, 'TextView', [0, 150, 100, 170])]
    ref = grp('xml:FrameLayout@1', det('ImageView', 0, gt[0]['bbox']), det('TextView', 1, gt[1]['bbox']))
    gen = grp('Column:gap', det('ImageView', 4, kept[0]['bbox'], 'Box', [det('TextView', 5, kept[1]['bbox'])]),
              det('TextView', 6, kept[2]['bbox']))
    xml = hierarchy(xml_node('android.widget.FrameLayout', (0, 0, 120, 100), [
        xml_node('android.widget.ImageView', (0, 0, 100, 60), label=0),
        xml_node('android.widget.TextView', (10, 40, 90, 55), label=1)]))
    ev = _evidence(xml, {0: gt[0]['bbox'], 1: gt[1]['bbox']})
    clf, r, g = _classifier(ref, gen, gt, kept, ev=ev)
    comp = _by_det(g, 4)
    stacked = next(n for n, _, _ in LE.iter_preorder(r) if n.cls is None and LE.label_o(n) == 'Box')
    assert clf.classify(comp, stacked)[1] == 'reference_overlap'
    assert clf.classify(_by_det(g, 6), None)[1] == 'reference_artifact'


# ── 5. end to end ─────────────────────────────────────────

def test_sample_is_reproducible_and_without_replacement():
    """sample_stems = numpy default_rng(seed).choice over the sorted stems."""
    stems = [f"s{i:03d}" for i in range(100)][::-1]
    a = FT.sample_stems(stems, 10, 42)
    assert a == FT.sample_stems(list(reversed(stems)), 10, 42) and len(set(a)) == 10
    idx = np.random.default_rng(42).choice(100, size=10, replace=False)
    assert a == [sorted(stems)[int(i)] for i in idx]
    assert len(FT.sample_stems(stems[:3], 10, 42)) == 3


def test_end_to_end_on_the_synthetic_package(tmp_path):
    """Statuses, perfect boxes without edits, the run cross-check, reports and the test-split guard."""
    pytest.importorskip('apted')
    pytest.importorskip('scipy')
    pk = make_package(tmp_path)
    LE.run_evaluation(pk['data'], 'val', tmp_path / 'run', conditions=['C4'], pred_cache=pk['pred'],
                      n_boot=0, zss_check=0, progress_every=0)
    tx = FT.run_taxonomy(pk['data'], 'val', pk['pred'], tmp_path / 'tx', n=4, seed=42,
                         run_dir=tmp_path / 'run')
    st = {r['stem']: r['status'] for r in tx['screens']}
    assert st == {'a-screens_screenshot_1': 'ok', 'a-screens_screenshot_2': 'excluded:no_predictions',
                  'b-screens_screenshot_1': 'excluded:xml_parse_error',
                  'c-screens_screenshot_1': 'excluded:no_mapping'}
    ok = next(r for r in tx['screens'] if r['status'] == 'ok')
    assert ok['ted'] == 0 and ok['edits'] == [] and ok['dominant'] == 'none'
    assert tx['meta']['run_check'] == {'run_dir': str(tmp_path / 'run'), 'n_checked': 1,
                                       'n_mismatch': 0, 'mismatches': []}
    s = tx['summary']
    assert s['n_ok'] == 1 and s['n_edits'] == 0 and s['dominant']['none'] == 1
    assert tx['cases'] == []
    pj, pm = FT.write_results(tx, tmp_path / 'res', 'tx')
    compact = json.loads(pj.read_text())
    assert 'edits' not in compact['screens'][0] and compact['summary'] == s
    assert '# Failure taxonomy of C4' in pm.read_text()
    with pytest.raises(ValueError, match='allow_test'):
        FT.run_taxonomy(pk['data'], 'test', pk['pred'], tmp_path / 'tx2')
    assert FT.main(['run', '--data-dir', pk['data'], '--split', 'test', '--pred-cache', pk['pred'],
                    '--out-dir', str(tmp_path / 'tx3')]) == 2


def test_summary_and_case_selection_on_hand_made_records():
    """Counts per category, dominant categories, families and the case rule (screens without a trace skipped)."""
    def rec(stem, cats, eid=1):
        edits = [{'kind': 'insert', 'category': c, 'reason': c, 'gen': None, 'ref': 'x',
                  'explain_id': None if c == 'detector_fn' else eid}
                 for c, n in cats.items() for _ in range(n)]
        order = list(FT.CATEGORIES)
        dom = sorted(cats, key=lambda k: (-cats[k], order.index(k)))[0] if cats else 'none'
        base = {k: 0 for k in ('n_gt', 'n_raw', 'n_kept', 'n_matched', 'n_fp', 'n_fn',
                               'n_class_mismatch', 'n_dedup_removed', 'n_dedup_removed_matching_gt',
                               'n_fallback')}
        return {'stem': stem, 'status': 'ok', 'edits': edits, 'categories': cats, 'dominant': dom,
                'dominant_share': (cats[dom] / sum(cats.values())) if cats else None,
                'teds': 0.5, **base}
    recs = [rec('a', {'detector_fn': 5}), rec('b', {'detector_fp': 2, 'grouping': 1}),
            rec('c', {'xycut_fallback': 3}), rec('d', {'reference_artifact': 1}), rec('e', {}),
            rec('f', {'reference_artifact': 2}, eid=None)]
    s = FT.summarize(recs)
    assert s['n_edits'] == 14 and s['dominant']['detector_fn'] == 1 and s['dominant']['none'] == 1
    assert s['families']['detector']['n'] == 7 and s['dominant_by_family']['layout_rules'] == 1
    cases = FT.select_cases(recs)
    assert [(c['family'], c['category'], c['stem']) for c in cases] == [
        ('detector', 'detector_fp', 'b'), ('layout_rules', 'xycut_fallback', 'c'),
        ('reference', 'reference_artifact', 'd')]


def test_key_trace_line_picks_the_rule_line():
    """The quoted trace line is the one naming the category's rule, else the detection line."""
    lines = ['Explain trace for detection id 2', '[detection] ImageView conf=0.40 bbox=[1, 2, 3, 4]',
             '[post-processing] kept', '  Column (rule=gap_tolerance, axis=y, 2 cuts, 1 within tolerance 4.8 px)',
             '  Column (rule=fallback_y_sorted: no cut along y) -> position 1 of 2']
    assert FT.key_trace_line(lines, 'xycut_fallback').startswith('Column (rule=fallback_y_sorted')
    assert 'within tolerance' in FT.key_trace_line(lines, 'tolerance_cut')
    assert FT.key_trace_line(lines, 'dedup_removed').startswith('[detection]')
    assert FT.key_trace_line(None, 'grouping') is None
    edit = {'gen': 'Box(Column:gap_tolerance) [0,0,10,10]', 'reason': 'container Column:gap_tolerance'}
    assert FT.key_trace_line(lines, 'grouping', edit).startswith('Column (rule=gap_tolerance')
    fp = {'gen': 'Row(Row:gap) [0,0,1,1]', 'reason': '2 of 2 unmatched: ImageView#2 matches no box'}
    assert FT.key_trace_line(lines, 'detector_fp', fp).startswith('[detection]')
