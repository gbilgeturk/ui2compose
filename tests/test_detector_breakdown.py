# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Tests for ui2compose.evaluation.detector_breakdown (summary and seen/unseen tables).

Sections:
  1. operating_point_bootstrap and subset_difference_bootstrap on synthetic
     data (points equal ap.py's, determinism, grouping, overlap check).
  2. Manifest helpers (seen / unseen split, --groups parsing).
  3. Both commands end to end on an evaluate_detector result made from saved
     synthetic predictions (no model).
  4. The interval check (percentile / basic / BCa, bootstrap bias): it
     reproduces the percentile intervals of bootstrap_ci and of the
     seen-unseen difference, and the 'intervals' command end to end.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from test_ap import NAMES, _cli_dataset, synthetic_dataset
from ui2compose.evaluation import ap as A
from ui2compose.evaluation import detector_breakdown as D
from ui2compose.evaluation import evaluate_detector as E


def test_operating_point_bootstrap_point_equals_ap_operating_point():
    preds, gts = synthetic_dataset(3, n_images=20)
    ref = A.operating_point(preds, gts, NAMES, 0.3)
    res = D.operating_point_bootstrap(preds, gts, NAMES, 0.3, n=50, seed=1)
    assert (res["tp"], res["fp"], res["fn"]) == (ref["tp"], ref["fp"], ref["fn"])
    for k in ("precision", "recall", "f1"):
        assert res[k]["point"] == pytest.approx(ref[k])
        assert res[k]["lo"] <= res[k]["point"] <= res[k]["hi"]
    for c in NAMES:
        assert res["per_class"][c]["tp"] == ref["per_class"][c]["tp"]
        assert res["per_class"][c]["recall"] == ref["per_class"][c]["recall"]
    assert res == D.operating_point_bootstrap(preds, gts, NAMES, 0.3, n=50, seed=1)
    groups = {k: str(i // 4) for i, k in enumerate(sorted(gts))}
    g = D.operating_point_bootstrap(preds, gts, NAMES, 0.3, n=50, seed=1, groups=groups)
    assert g["resample_unit"] == "group" and g["n_units"] == 5
    assert g["precision"]["point"] == res["precision"]["point"]


def test_subset_difference_bootstrap():
    preds, gts = synthetic_dataset(4, n_images=24)
    ids = sorted(gts)
    a, b = ids[:12], ids[12:]
    res = D.subset_difference_bootstrap(preds, gts, NAMES, a, b, n=40, seed=2)
    ma = A.evaluate({k: preds[k] for k in a if k in preds}, {k: gts[k] for k in a}, NAMES)
    mb = A.evaluate({k: preds[k] for k in b if k in preds}, {k: gts[k] for k in b}, NAMES)
    assert res["mAP50"]["point"] == pytest.approx(mb["mAP50"] - ma["mAP50"])
    assert res["mAP50_95"]["point"] == pytest.approx(mb["mAP50_95"] - ma["mAP50_95"])
    assert res["mAP50"]["lo"] <= res["mAP50"]["hi"] and 0 <= res["mAP50"]["p_diff_le_0"] <= 1
    groups = {k: str(i // 3) for i, k in enumerate(ids)}
    g = D.subset_difference_bootstrap(preds, gts, NAMES, a, b, n=40, seed=2, groups=groups)
    assert (g["n_units_a"], g["n_units_b"]) == (4, 4) and g["resample_unit"] == "group"
    with pytest.raises(ValueError):
        D.subset_difference_bootstrap(preds, gts, NAMES, a, a[:3] + b, n=5)
    # Fixed-class mean: the mean AP50 over the given classes in each subset, and its difference.
    fc = D.subset_difference_bootstrap(preds, gts, NAMES, a, b, n=40, seed=2,
                                       classes=NAMES[:2])["mAP50_classes"]
    ea = np.mean([ma["per_class"][c]["ap50"] for c in NAMES[:2]])
    eb = np.mean([mb["per_class"][c]["ap50"] for c in NAMES[:2]])
    assert fc["classes"] == NAMES[:2]
    assert fc["a"]["point"] == pytest.approx(ea) and fc["b"]["point"] == pytest.approx(eb)
    assert fc["diff"]["point"] == pytest.approx(eb - ea) and fc["diff"]["lo"] <= fc["diff"]["hi"]
    with pytest.raises(ValueError):
        D.subset_difference_bootstrap(preds, gts, NAMES, a, b, n=5, classes=["Nope"])


def test_manifest_helpers(tmp_path):
    m = tmp_path / "m.csv"
    m.write_text("stem,package,group_id,seen_in_train\na,p,0,1\nb,q,1,0\nc,r,2,1\n")
    man = D.load_manifest(m)
    assert D.split_seen_unseen(["c", "b", "a"], man) == (["a", "c"], ["b"])
    with pytest.raises(ValueError):
        D.split_seen_unseen(["a", "zzz"], man)
    assert D.parse_groups_arg(["x=a.csv", "y=b.csv"]) == {"x": Path("a.csv"), "y": Path("b.csv")}
    for bad in (["x"], ["x=a", "x=b"], ["image=a"]):
        with pytest.raises(ValueError):
            D.parse_groups_arg(bad)


def _evaluation(tmp_path: Path):
    """An evaluate_detector result made from saved synthetic predictions.

    Input:  tmp_path — scratch directory
    Output: (result path, manifest path, gts, preds, pipeline preds)
    """
    preds, img_dir, lab_dir, yaml_path = _cli_dataset(tmp_path, seed=7, n_images=16)
    stems = sorted(p.stem for p in img_dir.iterdir())
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("stem,package,group_id,seen_in_train\n" + "".join(
        f"{s},app{k // 2},{k // 2},{int(k >= 6)}\n" for k, s in enumerate(stems)))
    pdir = tmp_path / "preds"
    pdir.mkdir()
    ap_file = pdir / "detector_test_full_preds_m.json"
    E.save_predictions(ap_file, preds, {"source": "val", "settings": {"conf": 0.001}})
    pipe = {k: {f: [x for x, c in zip(v[f], v["conf"]) if c > 0.3] for f in ("boxes", "conf", "cls")}
            for k, v in preds.items()}
    E.save_predictions(pdir / "detector_test_full_preds_m_pipeline.json", pipe,
                       {"source": "predict", "settings": {"conf": 0.3, "multi_label": False}})
    out = tmp_path / "res" / "detector_test_full.json"
    assert E.main(["--preds", str(ap_file), "--images", str(img_dir), "--labels", str(lab_dir),
                   "--dataset-yaml", str(yaml_path), "--out", str(out), "--bootstrap", "30",
                   "--groups", str(manifest)]) == 0
    gts, _ = E.load_ground_truth(E.read_image_list(img_dir), lab_dir, len(NAMES))
    return out, manifest, gts, preds, pipe


def test_summary_command(tmp_path):
    out, manifest, gts, preds, pipe = _evaluation(tmp_path)
    pk = tmp_path / "pk.csv"
    stems = sorted(gts)
    pk.write_text("stem,group_id\n" + "".join(f"{s},{k}\n" for k, s in enumerate(stems)))
    js, cs = tmp_path / "s.json", tmp_path / "s.csv"
    assert D.main(["summary", str(out), "--groups", f"full={manifest}", "--groups", f"pk={pk}",
                   "--bootstrap", "30", "--min-support", "5", "--out-json", str(js),
                   "--out-csv", str(cs)]) == 0
    ev = json.loads(js.read_text())["evaluations"][0]
    assert (ev["split"], ev["subset"]) == ("test", "full")
    assert ev["reused_from_evaluation_file"] == ["full", "image"]
    res = json.loads(out.read_text())["models"][0]
    assert ev["mAP50"] == res["metrics"]["mAP50"]
    assert ev["ci"]["full"]["mAP50"] == [res["bootstrap_by_group"]["mAP50"]["lo"],
                                         res["bootstrap_by_group"]["mAP50"]["hi"]]
    direct = A.bootstrap_ci(preds, gts, NAMES, 30, 0, groups={s: str(k) for k, s in enumerate(stems)})
    assert ev["ci"]["pk"]["mAP50"] == [direct["mAP50"]["lo"], direct["mAP50"]["hi"]]
    op = A.operating_point(pipe, gts, NAMES, 0.3)
    assert ev["operating_point_pipeline"]["recall"] == pytest.approx(op["recall"])
    assert set(ev["operating_point_pipeline"]["ci"]) == {"image", "full", "pk"}
    for c, pc in ev["per_class"].items():
        assert pc["reliable"] == (pc["support"] >= 5)
    rc = ev["mAP50_reliable_classes"]
    assert rc["classes"] == [c for c in NAMES if ev["per_class"][c]["support"] >= 5]
    assert rc["point"] == pytest.approx(np.mean([ev["per_class"][c]["ap50"] for c in rc["classes"]]))
    rows = list(csv.DictReader(cs.open()))
    assert {r["metric"] for r in rows} == {"mAP50", "mAP50_95", "mAP50_reliable_classes", "AP50",
                                           "op0.3_precision", "op0.3_recall", "op0.3_f1"}
    assert sum(r["metric"] == "AP50" for r in rows) == len(NAMES)


def test_seen_unseen_command(tmp_path):
    out, manifest, gts, preds, pipe = _evaluation(tmp_path)
    js, cs = tmp_path / "su.json", tmp_path / "su.csv"
    assert D.main(["seen-unseen", str(out), "--manifest", str(manifest),
                   "--groups", f"full={manifest}", "--bootstrap", "25",
                   "--out-json", str(js), "--out-csv", str(cs)]) == 0
    res = json.loads(js.read_text())
    stems = sorted(gts)
    seen, unseen = stems[6:], stems[:6]
    assert res["counts"] == {"all": 16, "seen": 10, "unseen": 6}
    m_un = A.evaluate({k: preds[k] for k in unseen if k in preds}, {k: gts[k] for k in unseen}, NAMES)
    assert res["subsets"]["unseen"]["mAP50"] == pytest.approx(m_un["mAP50"])
    m_se = A.evaluate({k: preds[k] for k in seen if k in preds}, {k: gts[k] for k in seen}, NAMES)
    d = res["difference_unseen_minus_seen"]
    assert d["scheme"] == "full" and d["mAP50"]["point"] == pytest.approx(m_un["mAP50"] - m_se["mAP50"])
    rows = list(csv.DictReader(cs.open()))
    assert [r["class"] for r in rows][:len(NAMES) + 1] == NAMES + ["ALL"]
    allrow = rows[len(NAMES)]
    assert int(allrow["images_seen"]) == 10 and int(allrow["images_unseen"]) == 6
    for r in rows[:len(NAMES)]:
        assert int(r["support_all"]) == int(r["support_seen"]) + int(r["support_unseen"])
    # The fixed-class row uses the classes with >= min_support instances in both subsets.
    rel = [c for c in NAMES if res["subsets"]["seen"]["per_class"][c]["support"] >= 20
           and res["subsets"]["unseen"]["per_class"][c]["support"] >= 20]
    if rel:
        assert rows[-1]["class"] == "MEAN_RELIABLE_CLASSES"
        assert d["mAP50_classes"]["classes"] == rel
    else:
        assert "mAP50_classes" not in d


# ── 4. interval check ──────────────────────────────────────

def test_interval_check_set_reproduces_bootstrap_ci():
    """Same replicates as bootstrap_ci; the fixed classes are those with enough support."""
    preds, gts = synthetic_dataset(5, n_images=20)
    groups = {k: str(i // 2) for i, k in enumerate(sorted(gts))}
    res = D.interval_check_set(preds, gts, NAMES, "g", groups, n=30, seed=0, min_support=5)
    b = A.bootstrap_ci(preds, gts, NAMES, 30, 0, groups=groups)
    assert res["matches_bootstrap_ci"] and res["n_units"] == 10
    assert res["mAP50"]["percentile"] == [b["mAP50"]["lo"], b["mAP50"]["hi"]]
    m = A.evaluate(preds, gts, NAMES)
    fixed = [c for c in NAMES if m["per_class"][c]["support"] >= 5]
    assert res["classes"] == fixed
    assert res["mAP50_classes"]["point"] == pytest.approx(
        np.mean([m["per_class"][c]["ap50"] for c in fixed]))
    for k in ("mAP50", "mAP50_95", "mAP50_classes"):
        iv = res[k]
        assert iv["basic"][0] == pytest.approx(2 * iv["point"] - iv["percentile"][1])
        assert iv["bias"] == pytest.approx(iv["replicate_mean"] - iv["point"])
    plain = D.interval_check_set(preds, gts, NAMES, "image", None, n=30, seed=0, bca=False)
    assert plain["mAP50"]["bca"] is None and plain["n_units"] == 20


def test_intervals_command(tmp_path):
    """The command reproduces the evaluation file's and the seen-unseen command's percentile intervals."""
    out, manifest, gts, preds, pipe = _evaluation(tmp_path)
    js, cs = tmp_path / "i.json", tmp_path / "i.csv"
    assert D.main(["intervals", str(out), "--groups", f"full={manifest}", "--bootstrap", "30",
                   "--min-support", "5", "--seen-unseen", str(out), "--manifest", str(manifest),
                   "--out-json", str(js), "--out-csv", str(cs)]) == 0
    res = json.loads(js.read_text())
    model = json.loads(out.read_text())["models"][0]
    st = res["sets"][0]
    assert (st["split"], st["subset"], st["n_images"]) == ("test", "full", 16)
    assert set(st["schemes"]) == {"image", "full"}
    assert st["schemes"]["image"]["mAP50"]["percentile"] == [model["bootstrap"]["mAP50"]["lo"],
                                                              model["bootstrap"]["mAP50"]["hi"]]
    assert st["schemes"]["full"]["mAP50"]["percentile"] == [
        model["bootstrap_by_group"]["mAP50"]["lo"], model["bootstrap_by_group"]["mAP50"]["hi"]]
    d = res["differences"][0]
    assert d["scheme"] == "full" and d["counts"] == {"seen": 10, "unseen": 6}
    assert d["matches_seen_unseen"] and d["mAP50"]["bca"] is None
    su = tmp_path / "su.json"
    assert D.main(["seen-unseen", str(out), "--manifest", str(manifest), "--groups",
                   f"full={manifest}", "--bootstrap", "30", "--out-json", str(su),
                   "--out-csv", str(tmp_path / "su.csv")]) == 0
    ref = json.loads(su.read_text())["difference_unseen_minus_seen"]["mAP50"]
    assert d["mAP50"]["percentile"] == pytest.approx([ref["lo"], ref["hi"]])
    assert d["mAP50"]["point"] == pytest.approx(ref["point"])
    rows = list(csv.DictReader(cs.open()))
    assert {r["quantity"] for r in rows} == {"set", "difference"}
    assert {r["metric"] for r in rows if r["quantity"] == "set"} == {"mAP50", "mAP50_95",
                                                                     "mAP50_classes"}
    with pytest.raises(ValueError, match="manifest"):
        D.interval_check([out], {}, seen_unseen_results=[out])
