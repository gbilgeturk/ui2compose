# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Evaluates detector weights on an image list with Ultralytics-equivalent AP and bootstrap CIs.

Typical use (unseen-app test subset written by app_split.py):
    ui2compose-evaluate-detector --model runs/oversample_5k/weights/best.pt \
        --images results/splits/unseen_test.txt --labels data/yolo/labels/test \
        --dataset-yaml configs/dataset.yaml --groups results/splits/manifest.csv \
        --out results/eval_unseen_test.json

Steps:
  1. The AP predictions use the Ultralytics *validation* defaults (conf
     0.001, NMS IoU 0.7, the checkpoint's imgsz (640 for the released
     weights), max_det 300, multi-label NMS as in DetectionValidator). By
     default (--pred-source val) they are captured from YOLO.val(), so the
     numbers are comparable with Ultralytics' own validation (and with the
     published 48.70 % test mAP@50); --pred-source predict makes them image
     by image with YOLO.predict() (--single-label: predict()'s own
     best-class-only NMS).
  2. The raw predictions (original-image pixels) are saved per model to
     <out stem>_preds_<model>.json next to --out (or in --preds-dir, e.g. a
     folder outside the repository for large prediction sets), so that
     metrics and bootstrap intervals can be recomputed later without the
     model (--preds).
  3. Metrics are computed with ui2compose.evaluation.ap (an exact port of
     Ultralytics' matching and ap_per_class) with percentile bootstrap
     intervals (--bootstrap N, paired between consecutive models): over
     images, and with --groups also over app groups (cluster bootstrap; the
     appropriate interval when an evaluation set holds several screens per
     app).
  4. Pipeline operating point ('operating_point_pipeline'): precision /
     recall / F1 at IoU 0.5 of the detections the pipeline itself produces,
     i.e. YOLO.predict() with best-class-only NMS at conf --conf-op (0.3),
     NMS IoU --iou and the same imgsz and max_det as ui2compose.infer.pipeline
     (a separate prediction pass, saved to <out stem>_preds_<model>_pipeline.json).
     The 'operating_point' inside 'metrics' is instead computed from the AP
     predictions (multi-label, and with --pred-source val letterboxed as
     YOLO.val() does), so it includes boxes the pipeline never outputs.
  5. --expect-metrics FILE checks that the result reproduces published
     metrics (e.g. results/perclass_ap_results.json --expect-key
     oversample_5k for the test split, results/oversample_5k_metrics.json for
     the val split) within --expect-tol / --expect-class-tol, and
     --expect-images N the number of listed images; the command exits with
     status 3 if a check fails. This is the necessary condition for the unseen-app
     subset to be valid: the regenerated dataset must have the split the
     released weights were trained on (see app_split.py).
  6. --cross-check-val additionally runs YOLO.val() on a temporary dataset
     whose test split is exactly the image list (symlinks) with the same
     labels, and reports (a) Ultralytics' own numbers, (b) ap.py applied to
     the predictions YOLO.val() made (captured from the validator; this
     isolates the metric port and should agree to rounding) and (c) the
     difference of this run's metrics to (a). (c) is expected to be non-zero
     with --pred-source predict, because the predictions themselves differ:
     YOLO.val() resizes the long side to imgsz and pads to a per-batch
     rectangular shape (rect batching with 0.5 stride padding, e.g. 672x416
     for 1200x1920 screens), uses FP16 on CUDA and does not clip boxes to the
     image, whereas predict() letterboxes each image on its own with minimal
     padding (640x416) and clips boxes to the image. A different padding
     shifts the image relative to the stride-32 grid and changes the boxes
     slightly, which matters most at the strict IoU thresholds (mAP50-95).
     --pred-source val evaluates the predictions YOLO.val() makes instead,
     so that numbers are comparable with YOLO.val()-based results.

Ground truth: labels/<stem>.txt in YOLO format, read with the same checks as
ultralytics.data.utils.verify_image_label (5 columns, coordinates <= 1.01,
values >= -0.01, class < nc; exact duplicate rows removed; segment-format
rows, which Ultralytics converts to boxes, are treated as corrupt). A missing or
empty label file makes the image a background image; a corrupt label file
excludes the image (as Ultralytics does), and the exclusion is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ui2compose.evaluation import ap as apm

IMG_EXTS = (".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp")

# Ultralytics validation defaults (ultralytics/cfg/default.yaml, DetectionValidator).
DEFAULT_CONF = 0.001
DEFAULT_IOU = 0.7
# None = not passed to Ultralytics, which then uses the imgsz stored in the
# checkpoint (640 for the released weights); the value used is recorded.
DEFAULT_IMGSZ: Optional[int] = None
DEFAULT_MAX_DET = 300
DEFAULT_VAL_BATCH = 16
# Source of the AP predictions: 'val' (YOLO.val(), comparable with the
# published Ultralytics validation numbers) or 'predict'.
PRED_SOURCES = ("val", "predict")
DEFAULT_PRED_SOURCE = "val"
# Absolute tolerances of --expect-metrics. PROVISIONAL, chosen a priori (not
# tuned) as allowances for GPU FP16 vs CPU FP32 inference and Ultralytics
# version differences between the training run and this evaluation: 0.005
# for mAP50 / mAP50-95 and 0.02 for per-class AP50 (classes with few test
# instances move more). Passing the check is necessary, not sufficient, for
# the regenerated split to be the training split.
DEFAULT_EXPECT_TOL = 0.005
DEFAULT_EXPECT_CLASS_TOL = 0.02


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def read_image_list(spec: str | Path) -> List[Path]:
    """Reads the evaluation images from a directory or an image-list file.

    A list file has one path per line; as in Ultralytics, lines starting with
    "./" are relative to the list file's directory, other relative paths to
    the current directory. Blank lines and lines starting with '#' are skipped.

    Input:  spec — directory of images, or .txt list file
    Output: sorted list of image paths (only known image extensions);
            raises FileNotFoundError if spec does not exist
    """
    p = Path(spec)
    if p.is_dir():
        files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS]
    elif p.is_file():
        parent = str(p.parent) + os.sep
        files = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            files.append(Path(parent + line[2:] if line.startswith("./") else line))
        files = [f for f in files if f.suffix.lower() in IMG_EXTS]
    else:
        raise FileNotFoundError(f"image list or directory not found: {spec}")
    return sorted(files, key=lambda f: str(f))


def load_names(yaml_path: str | Path) -> List[str]:
    """Reads the class names (list or {id: name}) from a dataset yaml.

    Input:  yaml_path — dataset yaml
    Output: names indexed by class id; raises ValueError if absent
    """
    import yaml

    data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    if not names:
        raise ValueError(f"no 'names' in {yaml_path}")
    return [str(n) for n in names]


def exif_size(path: str | Path) -> Tuple[int, int]:
    """Image size corrected for EXIF rotation (as ultralytics.data.utils.exif_size).

    Input:  path — image file
    Output: (width, height)
    """
    with Image.open(path) as im:
        w, h = im.size
        if im.format == "JPEG":
            try:
                rotation = im.getexif().get(274, None)
                if rotation in {6, 8}:
                    w, h = h, w
            except Exception:
                pass
    return w, h


def load_yolo_label(label_path: str | Path, img_w: int, img_h: int, nc: int
                    ) -> Tuple[Optional[Dict[str, list]], str]:
    """Reads one YOLO label file into pixel xyxy boxes with Ultralytics' checks.

    Input:  label_path — .txt path; img_w, img_h — image size; nc — class count
    Output: (entry {'boxes': [[x1, y1, x2, y2], ...], 'cls': [...]} or None
            if corrupt, status) — status is 'ok', 'missing', 'empty',
            'duplicates_removed:<n>' or 'corrupt:<reason>'
    """
    p = Path(label_path)
    if not p.is_file():
        return {"boxes": [], "cls": []}, "missing"
    rows = [x.split() for x in p.read_text(encoding="utf-8").strip().splitlines() if len(x)]
    if not rows:
        return {"boxes": [], "cls": []}, "empty"
    try:
        lb = np.array(rows, dtype=np.float32)
    except ValueError as e:
        return None, f"corrupt:{e}"
    if lb.ndim != 2 or lb.shape[1] != 5:
        return None, f"corrupt:labels require 5 columns, got {lb.shape}"
    if lb[:, 1:].max() > 1.01:
        return None, "corrupt:non-normalized or out of bounds coordinates"
    if lb.min() < -0.01:
        return None, "corrupt:negative class labels or coordinates"
    if lb[:, 0].max() >= nc:
        return None, f"corrupt:class {int(lb[:, 0].max())} >= nc {nc}"
    status = "ok"
    _, keep = np.unique(lb, axis=0, return_index=True)
    if len(keep) < len(lb):
        status = f"duplicates_removed:{len(lb) - len(keep)}"
        lb = lb[keep]
    xc, yc, w, h = lb[:, 1], lb[:, 2], lb[:, 3], lb[:, 4]
    boxes = np.stack([(xc - w / 2) * img_w, (yc - h / 2) * img_h,
                      (xc + w / 2) * img_w, (yc + h / 2) * img_h], 1)
    return {"boxes": boxes.tolist(), "cls": lb[:, 0].astype(int).tolist()}, status


def load_ground_truth(images: Sequence[Path], labels_dir: str | Path, nc: int
                      ) -> Tuple[Dict[str, Dict[str, list]], Dict[str, Any]]:
    """Loads the ground truth of all images (labels_dir/<stem>.txt).

    Input:  images — image paths; labels_dir — label directory; nc — class count
    Output: (gts_by_image keyed by stem without corrupt images, report with
            counts of missing / empty / duplicate-row / corrupt label files);
            raises ValueError for duplicate image stems
    """
    stems = [p.stem for p in images]
    if len(set(stems)) != len(stems):
        dup = sorted({s for s in stems if stems.count(s) > 1})
        raise ValueError(f"duplicate image stems in the list: {dup[:5]}")
    gts: Dict[str, Dict[str, list]] = {}
    report: Dict[str, Any] = {"missing": 0, "empty": 0, "duplicate_rows_removed": 0,
                              "corrupt": []}
    for img in images:
        w, h = exif_size(img)
        entry, status = load_yolo_label(Path(labels_dir) / f"{img.stem}.txt", w, h, nc)
        if entry is None:
            report["corrupt"].append({"image": img.stem, "reason": status[len("corrupt:"):]})
            continue
        if status in ("missing", "empty"):
            report[status] += 1
        elif status.startswith("duplicates_removed:"):
            report["duplicate_rows_removed"] += int(status.split(":")[1])
        gts[img.stem] = entry
    return gts, report


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def _multi_label_predictor():
    """Builds a DetectionPredictor subclass whose NMS is multi-label, as in validation.

    Input:  none (imports ultralytics)
    Output: predictor class for YOLO.predict(predictor=...)
    """
    from ultralytics.models.yolo.detect import DetectionPredictor
    from ultralytics.utils import nms, ops

    class MultiLabelDetectionPredictor(DetectionPredictor):
        """DetectionPredictor with DetectionValidator's NMS settings (multi_label=True)."""

        def postprocess(self, preds, img, orig_imgs, **kwargs):
            """Applies multi-label NMS and builds Results scaled to the original image.

            Input:  preds — raw model output; img — network input batch;
                    orig_imgs — original images
            Output: list of ultralytics Results
            """
            preds = nms.non_max_suppression(
                preds, self.args.conf, self.args.iou, self.args.classes,
                self.args.agnostic_nms, multi_label=True, max_det=self.args.max_det,
                nc=0 if self.args.task == "detect" else len(self.model.names),
                end2end=getattr(self.model, "end2end", False),
                rotated=self.args.task == "obb")
            if not isinstance(orig_imgs, list):
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
            return self.construct_results(preds, img, orig_imgs)

    return MultiLabelDetectionPredictor


def _ultralytics_version() -> Optional[str]:
    """Installed Ultralytics version, read from the package metadata (no import of the package).

    Input:  none
    Output: version string, or None if Ultralytics is not installed
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("ultralytics")
    except PackageNotFoundError:
        return None


def md5_of(path: str | Path) -> str:
    """MD5 of a file (to identify weights).

    Input:  path — file
    Output: hex digest
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _checkpoint_imgsz(model: Any, imgsz: Optional[int]) -> Optional[int]:
    """Returns the imgsz Ultralytics uses: the argument, else the checkpoint's stored value.

    Input:  model — ultralytics YOLO model; imgsz — requested size or None
    Output: effective size (None if neither is known)
    """
    if imgsz is not None:
        return imgsz
    return (getattr(model, "overrides", None) or {}).get("imgsz")


def predict_images(model_path: str | Path, images: Sequence[Path],
                   conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU,
                   imgsz: Optional[int] = DEFAULT_IMGSZ, max_det: int = DEFAULT_MAX_DET,
                   device: Optional[str] = None, multi_label: bool = True
                   ) -> Tuple[Dict[str, Dict[str, list]], Dict[str, Any]]:
    """Runs the detector on every image (one image per call, batch 1).

    With multi_label False and conf = the pipeline threshold this is the
    prediction call of ui2compose.infer.pipeline (ComponentDetector.detect).

    Input:  model_path — weights; images — image paths; conf, iou, imgsz,
            max_det — predict() settings (imgsz None = the checkpoint's);
            device — e.g. 'cpu', 'mps', '0'; multi_label — multi-label NMS
            (validation behaviour) instead of best-class-only NMS
    Output: (preds_by_image keyed by stem: {'boxes' xyxy pixels, 'conf',
            'cls'}, metadata dict with the effective settings)
    """
    import ultralytics
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    eff_imgsz = _checkpoint_imgsz(model, imgsz)
    predictor = _multi_label_predictor() if multi_label else None
    preds: Dict[str, Dict[str, list]] = {}
    for img in images:
        kwargs = dict(source=str(img), conf=conf, iou=iou, max_det=max_det, verbose=False)
        if imgsz is not None:
            kwargs["imgsz"] = imgsz
        if device is not None:
            kwargs["device"] = device
        res = model.predict(predictor=predictor, **kwargs)[0]
        b = res.boxes
        preds[img.stem] = {"boxes": b.xyxy.cpu().numpy().astype(np.float32).tolist(),
                           "conf": b.conf.cpu().numpy().astype(np.float32).tolist(),
                           "cls": b.cls.cpu().numpy().astype(int).tolist()}
    meta = {"weights": str(model_path), "weights_md5": md5_of(model_path),
            "model_names": [model.names[k] for k in sorted(model.names)],
            "ultralytics_version": ultralytics.__version__,
            "settings": {"conf": conf, "iou": iou, "imgsz": eff_imgsz, "max_det": max_det,
                         "multi_label": multi_label, "batch": 1,
                         "device": str(model.predictor.device) if model.predictor else device},
            "n_images": len(images)}
    return preds, meta


def _json_default(o: Any) -> Any:
    """Converts NumPy scalars/arrays for json.dumps.

    Input:  o — object json cannot serialise
    Output: Python equivalent; raises TypeError for other types
    """
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def save_predictions(path: str | Path, preds: Dict[str, Any], meta: Dict[str, Any]) -> None:
    """Writes raw predictions and their metadata to JSON.

    Input:  path — output .json; preds — preds_by_image; meta — metadata
    Output: none (file written)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"meta": meta, "predictions": preds}, default=_json_default),
                          encoding="utf-8")


def load_predictions(path: str | Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Reads a predictions JSON written by save_predictions.

    Input:  path — .json file
    Output: (preds_by_image, metadata)
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["predictions"], data.get("meta", {})


def model_tag(weights: str | Path) -> str:
    """Short name of a weights file: 'oversample_5k' for runs/oversample_5k/weights/best.pt.

    The run folder names the weights only for Ultralytics' own file names
    (best.pt / last.pt); any other file is named by its stem, e.g.
    'imgsz_640' for data/weights/imgsz_640.pt (the ablation checkpoints of
    the v1.0.2 release).

    Input:  weights — path to a .pt (or a predictions .json)
    Output: tag string
    """
    p = Path(weights)
    if p.parent.name == "weights" and p.parent.parent.name and p.stem in ("best", "last"):
        return p.parent.parent.name
    return p.stem


# ---------------------------------------------------------------------------
# Cross-check with YOLO.val
# ---------------------------------------------------------------------------

def _capturing_validator(store: Dict[str, Dict[str, list]]):
    """Builds a DetectionValidator subclass that records every image's predictions.

    The predictions are mapped from letterboxed network coordinates back to
    original-image pixels with the image's ratio_pad (as
    DetectionValidator.scale_preds does) but without clipping, i.e. exactly
    the boxes that YOLO.val() matched, in the coordinates of the labels.

    Input:  store — dict filled with {stem: {'boxes', 'conf', 'cls'}}
    Output: validator class for YOLO.val(validator=...)
    """
    from ultralytics.models.yolo.detect import DetectionValidator

    class CapturingDetectionValidator(DetectionValidator):
        """DetectionValidator that also stores the predictions per image."""

        def update_metrics(self, preds, batch):
            """Updates the metrics (unchanged) and records the predictions.

            Input:  preds — post-processed predictions per image; batch — data batch
            Output: none
            """
            super().update_metrics(preds, batch)
            for si, pred in enumerate(preds):
                gain = float(batch["ratio_pad"][si][0][0])
                pad_x, pad_y = (float(v) for v in batch["ratio_pad"][si][1])
                b = pred["bboxes"].detach().float().cpu().numpy().astype(np.float32).copy()
                b[:, [0, 2]] -= pad_x
                b[:, [1, 3]] -= pad_y
                b /= gain
                store[Path(batch["im_file"][si]).stem] = {
                    "boxes": b.tolist(),
                    "conf": pred["conf"].detach().float().cpu().numpy().astype(np.float32).tolist(),
                    "cls": pred["cls"].detach().cpu().numpy().astype(int).tolist()}

    return CapturingDetectionValidator


def run_val(model_path: str | Path, images: Sequence[Path], labels_dir: str | Path,
            names: Sequence[str], imgsz: Optional[int] = DEFAULT_IMGSZ, conf: float = DEFAULT_CONF,
            iou: float = DEFAULT_IOU, max_det: int = DEFAULT_MAX_DET,
            batch: int = DEFAULT_VAL_BATCH, device: Optional[str] = None
            ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, list]]]:
    """Runs YOLO.val() on exactly the given images and labels (temporary dataset).

    The temporary dataset has images/test (symlinks to the images) and
    labels/test (copies of labels_dir/<stem>.txt), so Ultralytics evaluates
    the same images with the same labels.

    Input:  model_path — weights; images — image paths; labels_dir — label
            directory (<stem>.txt); names — class names; imgsz, conf, iou,
            max_det, batch, device — val() settings (imgsz None = the checkpoint's)
    Output: (metrics {'mAP50', 'mAP50_95', 'precision', 'recall',
             'per_class': {name: {'ap50', 'ap50_95'}}, 'settings'},
             predictions_by_image captured from the validator)
    """
    from ultralytics import YOLO

    store: Dict[str, Dict[str, list]] = {}
    with tempfile.TemporaryDirectory(prefix="ui2c_val_") as tmp:
        root = Path(tmp).resolve()
        (root / "images" / "test").mkdir(parents=True)
        (root / "labels" / "test").mkdir(parents=True)
        for img in images:
            (root / "images" / "test" / img.name).symlink_to(Path(img).resolve())
            lab = Path(labels_dir) / f"{img.stem}.txt"
            if lab.is_file():
                shutil.copyfile(lab, root / "labels" / "test" / f"{img.stem}.txt")
        yaml_text = (f"path: {root}\ntrain: images/test\nval: images/test\ntest: images/test\n"
                     "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
        (root / "data.yaml").write_text(yaml_text, encoding="utf-8")
        kwargs = dict(data=str(root / "data.yaml"), split="test", conf=conf,
                      iou=iou, max_det=max_det, batch=batch, plots=False, verbose=False,
                      project=str(root / "val"), name="run", save_json=False)
        if imgsz is not None:
            kwargs["imgsz"] = imgsz
        if device is not None:
            kwargs["device"] = device
        model = YOLO(str(model_path))
        eff_imgsz = _checkpoint_imgsz(model, imgsz)
        m = model.val(validator=_capturing_validator(store), **kwargs)
        box = m.box
        per_class = {n: {"ap50": None, "ap50_95": None} for n in names}
        for k, c in enumerate(box.ap_class_index):
            per_class[names[int(c)]] = {"ap50": float(box.ap50[k]), "ap50_95": float(box.ap[k])}
        metrics = {"mAP50": float(box.map50), "mAP50_95": float(box.map),
                   "precision": float(box.mp), "recall": float(box.mr), "per_class": per_class,
                   "settings": {**{k: v for k, v in kwargs.items()
                                   if k not in ("data", "project", "name")}, "imgsz": eff_imgsz}}
    return metrics, store


def cross_check_val(model_path: str | Path, images: Sequence[Path], labels_dir: str | Path,
                    names: Sequence[str], imgsz: Optional[int] = DEFAULT_IMGSZ,
                    conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU,
                    max_det: int = DEFAULT_MAX_DET, batch: int = DEFAULT_VAL_BATCH,
                    device: Optional[str] = None) -> Dict[str, Any]:
    """YOLO.val() metrics on the given images (see run_val), without the predictions.

    Input:  as run_val
    Output: metrics dict of run_val
    """
    return run_val(model_path, images, labels_dir, names, imgsz, conf, iou, max_det,
                   batch, device)[0]


def compare_with_val(metrics: Dict[str, Any], val: Dict[str, Any]) -> Dict[str, Any]:
    """Absolute differences between ap.py metrics and YOLO.val() metrics.

    Input:  metrics — evaluate() result; val — cross_check_val() result
    Output: {'mAP50', 'mAP50_95', 'precision', 'recall', 'max_abs_class_ap50'}
    """
    diff = {k: abs(metrics[k] - val[k]) for k in ("mAP50", "mAP50_95", "precision", "recall")}
    per = [abs(metrics["per_class"][n]["ap50"] - v["ap50"])
           for n, v in val["per_class"].items()
           if v["ap50"] is not None and metrics["per_class"].get(n, {}).get("ap50") is not None]
    diff["max_abs_class_ap50"] = max(per) if per else None
    return diff


# ---------------------------------------------------------------------------
# App groups, pipeline operating point, reproduction check
# ---------------------------------------------------------------------------

def load_groups(manifest_path: str | Path) -> Dict[str, str]:
    """Reads the app group of every screen from app_split's manifest.csv.

    Input:  manifest_path — manifest.csv with columns 'stem' and 'group_id'
    Output: {stem: group_id}; raises ValueError if a column is missing
    """
    import csv

    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"stem", "group_id"} <= set(reader.fieldnames):
            raise ValueError(f"{manifest_path}: needs the columns 'stem' and 'group_id'")
        return {row["stem"]: row["group_id"] for row in reader}


def pipeline_predictions_usable(meta: Dict[str, Any], conf_op: float) -> Tuple[bool, str]:
    """Tells whether saved predictions reproduce the pipeline's detections at conf_op.

    That is the case for predict() output with best-class-only NMS made at a
    confidence threshold <= conf_op (boxes above conf_op are then the same).

    Input:  meta — metadata of a predictions file; conf_op — operating threshold
    Output: (usable, reason when not usable)
    """
    settings = meta.get("settings", {})
    if meta.get("source") != "predict":
        return False, f"predictions come from {meta.get('source')!r}, not predict()"
    if settings.get("multi_label", True):
        return False, "predictions use multi-label NMS (the pipeline uses best-class-only NMS)"
    if settings.get("conf") is None or float(settings["conf"]) > conf_op:
        return False, f"predictions were made at conf {settings.get('conf')} > {conf_op}"
    return True, ""


def pipeline_operating_point(preds: Dict[str, Any], gts: Dict[str, Any], names: Sequence[str],
                             conf_op: float, source: str) -> Dict[str, Any]:
    """Operating point of the pipeline's own detections (best-class NMS, conf > conf_op).

    Input:  preds — single-label predict() predictions (see predict_images);
            gts — ground truth; names — class names; conf_op — threshold;
            source — description of the predictions (recorded)
    Output: ap.operating_point() result plus 'predictions' (source)
    """
    op = apm.operating_point(preds, gts, names, conf_op)
    op["predictions"] = source
    return op


def load_expected_metrics(path: str | Path, key: Optional[str] = None) -> Dict[str, Any]:
    """Reads published metrics to reproduce: overall mAP50 / mAP50-95 and optional per-class AP50.

    Accepted layouts: {'mAP50', 'mAP50_95', ...} (e.g.
    results/oversample_5k_metrics.json) or {'overall': {'mAP50', 'mAP50_95'},
    'per_class_ap50': {name: value}} (an entry of
    results/perclass_ap_results.json); key selects an entry of a file that
    holds several (e.g. 'oversample_5k').

    Input:  path — JSON file; key — optional top-level key
    Output: {'mAP50', 'mAP50_95', 'per_class_ap50' (dict, possibly empty), 'source'};
            raises ValueError if the values are not found
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if key is not None:
        if key not in data:
            raise ValueError(f"{path}: no entry {key!r} (entries: {sorted(data)})")
        data = data[key]
    overall = data.get("overall", data)
    if "mAP50" not in overall or "mAP50_95" not in overall:
        raise ValueError(f"{path}: no mAP50 / mAP50_95 values" + (f" in {key!r}" if key else ""))
    return {"mAP50": float(overall["mAP50"]), "mAP50_95": float(overall["mAP50_95"]),
            "per_class_ap50": {k: float(v) for k, v in data.get("per_class_ap50", {}).items()},
            "source": f"{path}" + (f"[{key}]" if key else "")}


def check_reproduction(metrics: Dict[str, Any], expected: Dict[str, Any],
                       tol: float = DEFAULT_EXPECT_TOL,
                       class_tol: float = DEFAULT_EXPECT_CLASS_TOL) -> Dict[str, Any]:
    """Compares computed metrics with published ones (absolute differences).

    Input:  metrics — evaluate() result; expected — load_expected_metrics()
            result; tol — tolerance of mAP50 / mAP50-95; class_tol — tolerance
            of each per-class AP50 (classes without ground truth here fail)
    Output: {'ok', 'tolerance', 'class_tolerance', 'source', 'overall': {key:
             {'expected', 'got', 'abs_diff', 'ok'}}, 'per_class_ap50': {...}}
    """
    def row(exp: float, got: Optional[float], t: float) -> Dict[str, Any]:
        """One comparison.

        Input:  exp — expected value; got — computed value (None = missing); t — tolerance
        Output: {'expected', 'got', 'abs_diff', 'ok'}
        """
        diff = None if got is None else abs(float(got) - exp)
        return {"expected": exp, "got": got, "abs_diff": diff, "ok": diff is not None and diff <= t}

    overall = {k: row(expected[k], metrics.get(k), tol) for k in ("mAP50", "mAP50_95")}
    per_class = {n: row(v, metrics.get("per_class", {}).get(n, {}).get("ap50"), class_tol)
                 for n, v in expected.get("per_class_ap50", {}).items()}
    ok = all(r["ok"] for r in overall.values()) and all(r["ok"] for r in per_class.values())
    return {"ok": ok, "tolerance": tol, "class_tolerance": class_tol,
            "source": expected.get("source"), "overall": overall, "per_class_ap50": per_class}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Defines and reads the command-line arguments.

    Input:  argv — argument list (default sys.argv[1:])
    Output: argparse namespace
    """
    ap = argparse.ArgumentParser(description="Evaluate detector weights on an image list "
                                             "(Ultralytics-equivalent AP, bootstrap CIs)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", action="append", help="weights (repeatable)")
    src.add_argument("--preds", action="append",
                     help="predictions JSON saved by an earlier run (repeatable)")
    ap.add_argument("--images", required=True, help="image directory or image-list .txt")
    ap.add_argument("--labels", required=True, help="label directory (<stem>.txt)")
    ap.add_argument("--dataset-yaml", required=True, help="dataset yaml with class names")
    ap.add_argument("--out", required=True, help="output metrics JSON")
    ap.add_argument("--preds-dir", default=None,
                    help="directory of the raw prediction files written with --model "
                         "(default: the directory of --out)")
    ap.add_argument("--bootstrap", type=int, default=apm.DEFAULT_BOOTSTRAP_N,
                    help="bootstrap replicates (0 disables)")
    ap.add_argument("--seed", type=int, default=apm.DEFAULT_BOOTSTRAP_SEED, help="bootstrap seed")
    ap.add_argument("--groups", default=None, metavar="MANIFEST_CSV",
                    help="app_split manifest.csv: also report a bootstrap over app groups "
                         "(resampling whole groups of screens)")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--iou", type=float, default=DEFAULT_IOU)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                    help="inference size (default: the checkpoint's training size)")
    ap.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET)
    ap.add_argument("--device", default=None)
    ap.add_argument("--single-label", action="store_true",
                    help="with --pred-source predict: best-class-only NMS instead of multi-label")
    ap.add_argument("--conf-op", type=float, default=apm.DEFAULT_CONF_OP,
                    help="confidence threshold of the operating points (pipeline threshold 0.3)")
    ap.add_argument("--pred-source", choices=PRED_SOURCES, default=DEFAULT_PRED_SOURCE,
                    help="val (default): the predictions YOLO.val() makes (comparable with "
                         "Ultralytics' validation); predict: YOLO.predict() per image")
    ap.add_argument("--cross-check-val", action="store_true",
                    help="also run YOLO.val() on the same images/labels and report both")
    ap.add_argument("--val-batch", type=int, default=DEFAULT_VAL_BATCH)
    ap.add_argument("--expect-metrics", default=None, metavar="JSON",
                    help="published metrics to reproduce (exit status 3 if they are not)")
    ap.add_argument("--expect-key", default=None,
                    help="entry of --expect-metrics, e.g. oversample_5k")
    ap.add_argument("--expect-tol", type=float, default=DEFAULT_EXPECT_TOL,
                    help="absolute tolerance for mAP50 / mAP50-95 (provisional 0.005)")
    ap.add_argument("--expect-class-tol", type=float, default=DEFAULT_EXPECT_CLASS_TOL,
                    help="absolute tolerance for per-class AP50 (provisional 0.02)")
    ap.add_argument("--expect-images", type=int, default=None, metavar="N",
                    help="expected number of listed images (exit status 3 if different)")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Evaluates one or more models (or saved predictions) and writes a metrics JSON.

    Input:  argv — argument list (default sys.argv[1:])
    Output: exit code 0 (1 without images or with --expect-metrics for several
            models, 3 if --expect-metrics is not reproduced or the number of
            listed images differs from --expect-images); files: --out
            metrics JSON and, with --model, <out stem>_preds_<tag>.json (AP
            predictions) and <out stem>_preds_<tag>_pipeline.json (pipeline
            predictions) per model, in --preds-dir (default: next to --out)
    """
    args = parse_args(argv)
    names = load_names(args.dataset_yaml)
    images = read_image_list(args.images)
    if not images:
        print(f"[ERR] no images in {args.images}")
        return 1
    specs = args.model or args.preds
    if args.expect_metrics and len(specs) != 1:
        print("[ERR] --expect-metrics needs exactly one --model / --preds")
        return 1
    expected = load_expected_metrics(args.expect_metrics, args.expect_key) if args.expect_metrics else None
    gts, gt_report = load_ground_truth(images, args.labels, len(names))
    eval_images = [p for p in images if p.stem in gts]
    groups = None
    if args.groups:
        all_groups = load_groups(args.groups)
        groups = {stem: all_groups[stem] for stem in gts if stem in all_groups}
        missing = sorted(set(gts) - set(groups))
        if missing:
            print(f"[ERR] {len(missing)} evaluated image(s) are not in {args.groups}, e.g. {missing[:3]}")
            return 1
    out = Path(args.out)
    preds_dir = Path(args.preds_dir) if args.preds_dir else out.parent

    sources: List[Tuple[str, Dict[str, Any], Dict[str, Any], str, Optional[tuple],
                        Dict[str, Any]]] = []
    tags: List[str] = []
    if args.cross_check_val and not args.model:
        print("[WARN] --cross-check-val needs --model; skipped for --preds")
    for spec in specs:
        tag = model_tag(spec)
        while tag in tags:
            tag += "_"
        tags.append(tag)
        val_info = None
        if args.model and args.pred_source == "val":
            if args.single_label:
                print("[WARN] --single-label has no effect with --pred-source val")
            val_metrics, preds = run_val(spec, eval_images, args.labels, names, args.imgsz,
                                         args.conf, args.iou, args.max_det, args.val_batch,
                                         args.device)
            val_info = (val_metrics, preds)
            meta = {"weights": str(spec), "weights_md5": md5_of(spec),
                    "ultralytics_version": _ultralytics_version(), "source": "val",
                    "settings": val_metrics["settings"], "n_images": len(eval_images)}
        elif args.model:
            preds, meta = predict_images(spec, eval_images, args.conf, args.iou, args.imgsz,
                                         args.max_det, args.device, not args.single_label)
            meta["source"] = "predict"
        if args.model:
            pred_file = preds_dir / f"{out.stem}_preds_{tag}.json"
            save_predictions(pred_file, preds, meta)
            if meta.get("model_names") not in (None, list(names)):
                print(f"[WARN] model class names differ from {args.dataset_yaml}")
            # The pipeline's own detections: best-class-only NMS at the pipeline threshold.
            pipe_preds, pipe_meta = predict_images(spec, eval_images, args.conf_op, args.iou,
                                                   args.imgsz, args.max_det, args.device,
                                                   multi_label=False)
            pipe_meta["source"] = "predict"
            pipe_file = preds_dir / f"{out.stem}_preds_{tag}_pipeline.json"
            save_predictions(pipe_file, pipe_preds, pipe_meta)
            pipe = {k: v for k, v in pipe_preds.items() if k in gts}
            op_pipe = pipeline_operating_point(
                pipe, gts, names, args.conf_op,
                f"YOLO.predict(), best-class-only NMS, conf > {args.conf_op} "
                f"(as ui2compose.infer.pipeline); {pipe_file}")
        else:
            preds, meta = load_predictions(spec)
            pred_file = str(spec)
            usable, reason = pipeline_predictions_usable(meta, args.conf_op)
            op_pipe = (pipeline_operating_point({k: v for k, v in preds.items() if k in gts},
                                                gts, names, args.conf_op,
                                                f"{spec} (single-label predict())")
                       if usable else {"predictions": None, "reason": reason})
        preds = {k: v for k, v in preds.items() if k in gts}
        sources.append((tag, preds, meta, str(pred_file), val_info, op_pipe))

    result: Dict[str, Any] = {
        "images": {"spec": str(args.images), "listed": len(images), "evaluated": len(gts),
                   "expected_listed": args.expect_images,
                   "labels_dir": str(args.labels), "labels": gt_report},
        "settings": {"conf": args.conf, "iou": args.iou, "imgsz": args.imgsz,
                     "max_det": args.max_det, "pred_source": args.pred_source,
                     "multi_label": args.pred_source == "val" or not args.single_label,
                     "conf_op": args.conf_op, "bootstrap": args.bootstrap, "seed": args.seed,
                     "groups": args.groups},
        "numpy_version": np.__version__,
        "models": [], "paired": [],
    }
    exit_code = 0
    if args.expect_images is not None:
        if len(images) == args.expect_images:
            print(f"[OK] {len(images)} images listed, as expected")
        else:
            print(f"[FAIL] {len(images)} images listed, expected {args.expect_images}")
            exit_code = 3
    for tag, preds, meta, pred_file, val_info, op_pipe in sources:
        entry: Dict[str, Any] = {"name": tag, "meta": meta, "predictions_file": pred_file,
                                 "metrics": apm.evaluate(preds, gts, names, args.conf_op),
                                 "operating_point_pipeline": op_pipe}
        if args.bootstrap > 0:
            entry["bootstrap"] = apm.bootstrap_ci(preds, gts, names, args.bootstrap, args.seed)
            if groups is not None:
                entry["bootstrap_by_group"] = apm.bootstrap_ci(preds, gts, names, args.bootstrap,
                                                               args.seed, groups=groups)
        if args.cross_check_val and args.model:
            if val_info is None:
                val_info = run_val(meta["weights"], eval_images, args.labels, names, args.imgsz,
                                   args.conf, args.iou, args.max_det, args.val_batch,
                                   args.device)
            val, val_preds = val_info
            on_val = apm.evaluate({k: v for k, v in val_preds.items() if k in gts}, gts, names,
                                  args.conf_op)
            entry["val_cross_check"] = {
                "ultralytics_val": val,
                # ap.py on the predictions YOLO.val() made: isolates the metric port
                "ap_py_on_val_predictions": {k: on_val[k] for k in
                                             ("mAP50", "mAP50_95", "precision", "recall")},
                "abs_diff_metric_port": compare_with_val(on_val, val),
                # this run's metrics (from --pred-source predictions) vs YOLO.val()
                "abs_diff": compare_with_val(entry["metrics"], val)}
        if expected is not None:
            entry["reproduction"] = check_reproduction(entry["metrics"], expected,
                                                       args.expect_tol, args.expect_class_tol)
        result["models"].append(entry)
        m = entry["metrics"]
        print(f"{tag}: mAP50 {m['mAP50']:.4f}  mAP50-95 {m['mAP50_95']:.4f}  "
              f"P {m['precision']:.4f}  R {m['recall']:.4f}  ({m['n_images']} images)")
        if op_pipe.get("predictions") is not None:
            p, r = op_pipe["precision"], op_pipe["recall"]
            print(f"{tag}: pipeline operating point (conf > {args.conf_op}): "
                  f"P {'n/a' if p is None else f'{p:.4f}'}  R {'n/a' if r is None else f'{r:.4f}'}")
        if "reproduction" in entry:
            rep = entry["reproduction"]
            bad = [k for k, v in {**rep["overall"], **rep["per_class_ap50"]}.items() if not v["ok"]]
            if rep["ok"]:
                print(f"[OK] reproduces {rep['source']} within {args.expect_tol} "
                      f"(per class {args.expect_class_tol})")
            else:
                print(f"[FAIL] does not reproduce {rep['source']}: {', '.join(bad)} outside tolerance")
                exit_code = 3
    if args.bootstrap > 0:
        for (t0, p0, *_), (t1, p1, *_) in zip(sources, sources[1:]):
            bs = apm.bootstrap_ci(p0, gts, names, args.bootstrap, args.seed, other_preds=p1)
            pair = {"base": t0, "other": t1, "diff": bs["diff"]}
            if groups is not None:
                pair["diff_by_group"] = apm.bootstrap_ci(p0, gts, names, args.bootstrap, args.seed,
                                                         other_preds=p1, groups=groups)["diff"]
            result["paired"].append(pair)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
    print(f"Saved: {out}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
