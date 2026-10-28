# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Tests for ui2compose.evaluation.detector_ablation (checkpoints x sets, paired differences).

Two synthetic "checkpoints" (the noisy predictions of test_ap and a
perturbed copy) are written as evaluate_detector results on the same
images; the ablation must reproduce the evaluation files' percentile
intervals, the paired differences of ap.bootstrap_ci (other_preds), the
interval check of detector_breakdown, and its cache must return the same
values.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from test_ap import NAMES, _cli_dataset
from ui2compose.evaluation import ap as A
from ui2compose.evaluation import detector_ablation as X
from ui2compose.evaluation import detector_breakdown as D
from ui2compose.evaluation import evaluate_detector as E


def _perturbed(preds, seed=3):
    """A second 'checkpoint': the same predictions with jittered confidences and some dropped.

    Input:  preds — predictions by image; seed — RNG seed
    Output: predictions by image
    """
    rng = np.random.default_rng(seed)
    out = {}
    for k, v in preds.items():
        keep = [i for i in range(len(v["cls"])) if rng.random() < 0.85]
        out[k] = {"boxes": [v["boxes"][i] for i in keep], "cls": [v["cls"][i] for i in keep],
                  "conf": [float(np.clip(v["conf"][i] + rng.normal(0, 0.1), 0.001, 1.0))
                           for i in keep]}
    return out


def _write_eval(tmp_path: Path, name: str, preds, img_dir, lab_dir, yaml_path, manifest):
    """evaluate_detector result (--preds) of one checkpoint, with its pipeline predictions.

    Input:  tmp_path — scratch directory; name — checkpoint name; preds — AP
            predictions; img_dir, lab_dir, yaml_path — dataset; manifest — groups CSV
    Output: directory holding detector_test_full.json
    """
    pdir = tmp_path / "preds"
    pdir.mkdir(exist_ok=True)
    ap_file = pdir / f"detector_test_full_preds_{name}.json"
    E.save_predictions(ap_file, preds, {"source": "val", "settings": {"conf": 0.001, "imgsz": 640}})
    pipe = {k: {f: [x for x, c in zip(v[f], v["conf"]) if c > 0.3] for f in ("boxes", "conf", "cls")}
            for k, v in preds.items()}
    E.save_predictions(pdir / f"detector_test_full_preds_{name}_pipeline.json", pipe,
                       {"source": "predict", "settings": {"conf": 0.3, "multi_label": False}})
    out = tmp_path / name / "detector_test_full.json"
    assert E.main(["--preds", str(ap_file), "--images", str(img_dir), "--labels", str(lab_dir),
                   "--dataset-yaml", str(yaml_path), "--out", str(out), "--bootstrap", "40",
                   "--groups", str(manifest)]) == 0
    return out.parent


@pytest.fixture
def two_checkpoints(tmp_path):
    preds, img_dir, lab_dir, yaml_path = _cli_dataset(tmp_path, seed=11, n_images=18)
    stems = sorted(p.stem for p in img_dir.iterdir())
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("stem,group_id\n" + "".join(f"{s},{k // 2}\n" for k, s in enumerate(stems)))
    other = _perturbed(preds)
    d_a = _write_eval(tmp_path, "a", preds, img_dir, lab_dir, yaml_path, manifest)
    d_b = _write_eval(tmp_path, "b", other, img_dir, lab_dir, yaml_path, manifest)
    gts, _ = E.load_ground_truth(E.read_image_list(img_dir), lab_dir, len(NAMES))
    return {"a": d_a, "b": d_b, "manifest": manifest, "preds": preds, "other": other, "gts": gts,
            "tmp": tmp_path}


def test_parse_helpers():
    assert X.parse_named(["x=1", "y=a=b"], "--m") == {"x": "1", "y": "a=b"}
    for bad in (["x"], ["=1"], ["x=1", "x=2"]):
        with pytest.raises(ValueError):
            X.parse_named(bad, "--m")
    models = ["baseline", "oversample_2k", "oversample_5k"]
    assert X.parse_pairs(["oversample_5k-baseline", "oversample_5k-oversample_2k"], models) == [
        ("oversample_5k", "baseline"), ("oversample_5k", "oversample_2k")]
    with pytest.raises(ValueError):
        X.parse_pairs(["oversample_5k-nope"], models)
    assert X.oversampled_classes() == ["CheckBox", "ProgressBar", "RadioButton", "RecyclerView",
                                       "SeekBar", "Spinner", "Switch", "WebView"]


def test_ablation_reproduces_bootstrap_and_interval_check(two_checkpoints):
    t = two_checkpoints
    groupings = {"g": t["manifest"]}
    res = X.run_ablation({"a": str(t["a"]), "b": str(t["b"])}, ["test_full"], groupings,
                         [("b", "a")], n=40, seed=0, min_support=5, focus_classes=NAMES[:2])
    ch = res["checks"]
    assert ch["settings_identical"] and ch["same_images_labels_units"]
    assert ch["op_matches_operating_point_bootstrap"]
    assert ch["percentile_matches_evaluation_files"]["ok"]
    assert ch["percentile_matches_evaluation_files"]["n"] == 2
    sc = res["sets"]["test_full"]["schemes"]["g"]
    assert sc["n_units"] == 9
    groups = {s: str(k // 2) for k, s in enumerate(sorted(t["gts"]))}
    # Paired differences: the same replicates as bootstrap_ci with other_preds (other - preds).
    b = A.bootstrap_ci(t["preds"], t["gts"], NAMES, 40, 0, other_preds=t["other"], groups=groups)
    for k in ("mAP50", "mAP50_95"):
        d = sc["pairs"]["b - a"][k]
        assert d["point"] == pytest.approx(b["diff"][k]["point"])
        assert d["percentile"] == pytest.approx([b["diff"][k]["lo"], b["diff"][k]["hi"]], abs=1e-12)
        assert d["p_le_0"] == pytest.approx(b["diff"][k]["p_diff_le_0"])
    # Per checkpoint: exactly the interval check of the 'intervals' command.
    ic = D.interval_check_set(t["other"], t["gts"], NAMES, "g", groups, n=40, seed=0, min_support=5)
    assert sc["classes_fixed"] == ic["classes"]
    for k in ("mAP50", "mAP50_95", "mAP50_classes"):
        for f in ("point", "percentile", "basic", "bca", "bias"):
            assert sc["models"]["b"][k][f] == pytest.approx(ic[k][f], abs=1e-12)
    # Operating point: the micro P / R / F1 of the pipeline predictions.
    pipe = {k: {f: [x for x, c in zip(v[f], v["conf"]) if c > 0.3] for f in ("boxes", "conf", "cls")}
            for k, v in t["preds"].items()}
    op = A.operating_point(pipe, t["gts"], NAMES, 0.3)
    assert sc["models"]["a"]["f1"]["point"] == pytest.approx(op["f1"])
    # The difference of a checkpoint with itself is zero in every replicate.
    same = X.run_ablation({"a": str(t["a"]), "a2": str(t["a"])}, ["test_full"], groupings,
                          [("a2", "a")], n=20, min_support=5, focus_classes=NAMES[:2])
    z = same["sets"]["test_full"]["schemes"]["g"]["pairs"]["a2 - a"]
    assert z["mAP50"]["point"] == 0 and z["mAP50"]["percentile"] == [0, 0]
    assert z["mAP50"]["p_le_0"] == 1.0


def test_ablation_cli_cache_and_reference(two_checkpoints, tmp_path):
    t = two_checkpoints
    # A reference made by the 'intervals' command for checkpoint b.
    ref = tmp_path / "ref.json"
    assert D.main(["intervals", str(t["b"] / "detector_test_full.json"), "--groups",
                   f"g={t['manifest']}", "--bootstrap", "30", "--min-support", "5",
                   "--out-json", str(ref), "--out-csv", str(tmp_path / "ref.csv")]) == 0
    outs = [tmp_path / f"o{k}.{e}" for k in (1, 2) for e in ("json", "csv", "md")]
    common = ["--model", f"a={t['a']}", "--model", f"b={t['b']}", "--pairs", "b-a",
              "--sets", "test_full", "--groups", f"g={t['manifest']}", "--bootstrap", "30",
              "--min-support", "5", "--classes", ",".join(NAMES[:2]),
              "--intervals-reference", str(ref), "--cache-dir", str(tmp_path / "cache")]
    assert X.main(common + ["--out-json", str(outs[0]), "--out-csv", str(outs[1]),
                            "--out-md", str(outs[2])]) == 0
    assert len(list((tmp_path / "cache").glob("*.pkl"))) == 2
    assert X.main(common + ["--out-json", str(outs[3]), "--out-csv", str(outs[4]),
                            "--out-md", str(outs[5])]) == 0
    r1, r2 = json.loads(outs[0].read_text()), json.loads(outs[3].read_text())
    assert r1["sets"] == r2["sets"]
    mr = r1["checks"]["matches_intervals_reference"]
    assert mr["ok"] and mr["n"] == 3
    rows = list(csv.DictReader(outs[1].open()))
    assert {r["quantity"] for r in rows} == {"checkpoint", "difference"}
    assert {r["name"] for r in rows if r["quantity"] == "difference"} == {"b - a"}
    ap_rows = [r for r in rows if r["metric"] == "AP50"]
    assert {r["class"] for r in ap_rows} == set(NAMES[:2])
    for r in ap_rows:
        assert (r["reliable"] == "True") == (int(r["support"]) >= 5)
    md = outs[2].read_text()
    assert "## Paired differences" in md and "b - a" in md
    # A changed evaluation file invalidates its cache entry.
    f = t["b"] / "detector_test_full.json"
    f.write_text(f.read_text() + "\n")
    key_task = {"set": "test_full", "model": "b", "result": str(f), "base_dir": ".",
                "scheme": "g", "groups_csv": str(t["manifest"]), "n": 30, "seed": 0,
                "conf_op": 0.3, "cache_dir": str(tmp_path / "cache")}
    assert "cached" not in X.cached_model_task(key_task)
    assert X.cached_model_task(key_task)["cached"] is True


def test_checkpoint_provenance(tmp_path):
    torch = pytest.importorskip("torch")
    cfg = tmp_path / "train.yaml"
    cfg.write_text("model: yolov8s.pt\ntrain:\n  epochs: 60\n  seed: 42\n")
    for name, data, ep in (("a", "/c/ablation_640/d.yaml", 60), ("b", "/c/ablation_os2k/d.yaml", 59)):
        torch.save({"train_args": {"data": data, "model": "yolov8s.pt", "resume": False,
                                   "epochs": ep, "seed": 42},
                    "train_metrics": {"metrics/mAP50(B)": 0.5}, "version": "8.4.24",
                    "date": "2026-03-20T00:00:00"}, tmp_path / f"{name}.pt")
    prov = X.checkpoint_provenance({"a": tmp_path / "a.pt", "b": tmp_path / "b.pt"},
                                   {"a": cfg, "b": cfg})
    assert set(prov["train_args_differing_keys"]) == {"data", "epochs"}
    assert prov["checkpoints"]["a"]["config"]["equal"] is True
    assert prov["checkpoints"]["b"]["config"]["mismatches"] == {"epochs": [60, 59]}
    assert prov["checkpoints"]["a"]["stored_val_metrics"]["mAP50"] == 0.5
    assert prov["checkpoints"]["a"]["md5"] == E.md5_of(tmp_path / "a.pt")
