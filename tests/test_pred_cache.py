# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Tests for ui2compose.evaluation.pred_cache (the pipeline-detection cache of the layout evaluation).

Sections:
  1. Settings: the cache uses run_pipeline's own detection defaults.
  2. build_cache / compare_detections / verify_cache with a fake detector (no model).
  3. Smoke test with the default weights (skips without ultralytics or the
     weights): the cache of two bundled examples reproduces the golden raw
     detections of tests/golden within the tolerances of tests/test_golden.py.
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from _reference_data import require_reference_data
from ui2compose.evaluation import pred_cache as P

REPO_ROOT = Path(__file__).resolve().parents[1]


def _det(i: int, y1: float, cls: str = "TextView", conf: float = 0.5) -> Dict[str, Any]:
    """One detection dict in ComponentDetector.detect() format (image 1200 x 1920).

    Input:  i — id; y1 — top edge; cls — class name; conf — confidence
    Output: detection dict
    """
    x1, x2, y2 = 10.0, 210.0, y1 + 40.0
    return {"id": i, "class_name": cls, "class_id": 11, "confidence": conf,
            "bbox": [x1, y1, x2, y2],
            "bbox_norm": [(x1 + x2) / 2 / 1200, (y1 + y2) / 2 / 1920, 200 / 1200, 40 / 1920],
            "center": [(x1 + x2) / 2, (y1 + y2) / 2], "width": 200.0, "height": 40.0,
            "img_w": 1200, "img_h": 1920}


class FakeDetector:
    """Returns fixed detections per image name and records the detect() calls."""

    def __init__(self, by_name: Dict[str, List[Dict[str, Any]]]):
        """Stores the detections to return.

        Input:  by_name — {image file name: detections}
        Output: none
        """
        self.by_name = by_name
        self.calls: List[Dict[str, Any]] = []

    def detect(self, image_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Returns a copy of the detections of the image.

        Input:  image_path — path; kwargs — detection settings (recorded)
        Output: list of detection dicts
        """
        self.calls.append({"image": Path(image_path).name, **kwargs})
        return copy.deepcopy(self.by_name[Path(image_path).name])


def test_settings_are_the_pipeline_defaults():
    """The cache calls detect() with exactly run_pipeline's default detection settings."""
    assert P.pipeline_detect_settings() == {"conf_threshold": 0.3, "iou_threshold": 0.7,
                                            "imgsz": None, "agnostic_nms": False, "max_det": 300}


def test_build_cache_writes_pipeline_format_and_resumes(tmp_path):
    images = [tmp_path / "a.png", tmp_path / "b.png"]
    fake = FakeDetector({"a.png": [_det(0, 5.0), _det(1, 50.0, "Button", 0.9)], "b.png": []})
    out = tmp_path / "cache"
    stats = P.build_cache(fake, images, out)
    assert stats["computed"] == 2 and stats["kept_existing"] == 0
    assert stats["n_detections"] == 2 and stats["empty_screens"] == 1
    assert set(stats["seconds_per_image"]) == {"a", "b"} and stats["first_call"]["stem"] == "a"
    assert fake.calls[0] == {"image": "a.png", **P.pipeline_detect_settings()}
    # The file is what run_pipeline writes as 1_detections_raw.json (json.dump, indent 2).
    expected = json.dumps(fake.by_name["a.png"], indent=2)
    assert (out / "a.json").read_text(encoding="utf-8") == expected
    assert P.load_cached(out, "b") == []
    # A second run keeps the existing files; --overwrite recomputes them.
    again = P.build_cache(fake, images, out)
    assert again["computed"] == 0 and again["kept_existing"] == 2 and len(fake.calls) == 2
    assert P.build_cache(fake, images, out, overwrite=True)["computed"] == 2
    with pytest.raises(ValueError):
        P.build_cache(fake, [tmp_path / "x" / "a.png", tmp_path / "y" / "a.png"], out)


def test_compare_detections_exact_tolerant_and_mismatch():
    ref = [_det(0, 5.0), _det(1, 50.0)]
    v110 = [{k: v for k, v in d.items() if k not in ("img_w", "img_h")} for d in ref]
    r = P.compare_detections(ref, v110)
    assert r["exact"] and r["match"] and r["keys_only_cached"] == ["img_h", "img_w"]
    assert r["max_abs_diff"]["bbox"] == 0.0
    near = copy.deepcopy(ref)
    near[1]["bbox"][0] += 0.004
    near[1]["confidence"] += 5e-5
    r = P.compare_detections(near, ref)
    assert not r["exact"] and r["match"]
    assert r["max_abs_diff"]["bbox"] == pytest.approx(0.004)
    far = copy.deepcopy(ref)
    far[0]["class_name"] = "Button"
    far[1]["bbox"][3] += 1.0
    r = P.compare_detections(far, ref)
    assert not r["match"] and len(r["mismatches"]) == 2
    r = P.compare_detections(ref[:1], ref)
    assert not r["match"] and "1 cached vs 2 reference" in r["mismatches"][0]


def test_verify_cache_uses_only_screens_with_a_reference(tmp_path):
    out = tmp_path / "cache"
    P.write_detections(out / "a.json", [_det(0, 5.0)])
    P.write_detections(out / "b.json", [_det(0, 7.0)])
    ref_dir = tmp_path / "ref"
    P.write_detections(ref_dir / "a" / "1_detections.json", [_det(0, 5.0)])
    rep = P.verify_cache(out, ["a", "b"], ref_dir, "1_detections.json")
    assert rep["n_compared"] == 1 and rep["n_exact"] == 1 and rep["n_match"] == 1
    assert set(rep["screens"]) == {"a"}


def test_main_with_existing_cache_needs_no_model(tmp_path):
    """With every screen cached, main() only verifies and writes the metadata (no model load)."""
    pytest.importorskip("ultralytics")  # main() records the Ultralytics version
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    from PIL import Image

    Image.new("RGB", (12, 20)).save(img_dir / "a.png")
    out = tmp_path / "cache" / "test"
    P.write_detections(out / "a.json", [_det(0, 5.0)])
    P.write_detections(tmp_path / "ref" / "a" / "f.json", [_det(0, 5.0, conf=0.9)])
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"x")
    rc = P.main(["--model", str(weights), "--images", str(img_dir), "--out-dir", str(out),
                 "--verify", str(tmp_path / "ref"), "f.json"])
    assert rc == 3                                  # confidence differs by 0.4
    meta = json.loads((tmp_path / "cache" / "meta_test.json").read_text())
    assert meta["computed"] == 0 and meta["kept_existing"] == 1 and meta["model_load_seconds"] is None
    assert meta["verification"][0]["n_compared"] == 1 and meta["verification"][0]["n_match"] == 0


def test_cache_reproduces_golden_raw_detections(tmp_path):
    """Default weights at the pipeline defaults: the cache equals tests/golden/<stem>/1_detections_raw.json."""
    pytest.importorskip("ultralytics")
    weights = REPO_ROOT / "runs" / "oversample_5k_appsplit" / "weights" / "best.pt"
    golden = REPO_ROOT / "tests" / "golden"
    require_reference_data(weights, golden)
    from ui2compose.infer.detect_components import ComponentDetector

    images = sorted((REPO_ROOT / "examples").glob("*.png"))[:2]
    det = ComponentDetector(str(weights), str(REPO_ROOT / "configs" / "dataset.yaml"))
    P.build_cache(det, images, tmp_path)
    rep = P.verify_cache(tmp_path, [p.stem for p in images], golden, "1_detections_raw.json")
    assert rep["n_compared"] == 2 and rep["n_match"] == 2, rep["screens"]
