# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Caches the pipeline's raw detections of many screens, so layout evaluations can run without the model.

Typical use (test split of the regenerated ReDraw dataset):
    python -m ui2compose.evaluation.pred_cache --model runs/oversample_5k/weights/best.pt \
        --images <dataset>/images/test --out-dir <outside the repo>/pred_cache/test \
        --verify results/v1.1.0_examples 1_detections.json \
        --verify tests/golden 1_detections_raw.json \
        --meta-out <outside the repo>/pred_cache/meta_test.json

Steps:
  1. One ComponentDetector (the class run_pipeline uses) is loaded, and
     detect() is called on every image with run_pipeline's own default
     detection settings (PIPELINE_DETECT_SETTINGS, read from run_pipeline's
     signature: conf 0.3, NMS IoU 0.7, the checkpoint's imgsz, class-aware
     NMS, max_det 300) and Ultralytics' default device, i.e. exactly the call
     that writes 1_detections_raw.json in a default pipeline run.
  2. The detections of each screen are written to <out-dir>/<stem>.json in
     the format of 1_detections_raw.json (the list returned by detect(),
     including img_w / img_h, indented as run_pipeline writes it), so the
     later pipeline stages can be replayed from the file with a detector that
     returns it (as tests/test_golden.py does), or with
     pipeline.layout_from_boxes after de-duplication.
  3. Existing cache files are kept (--overwrite recomputes them), so an
     interrupted run resumes; the metadata records how many were computed.
  4. The wall-clock time of every detect() call (image read + predict) and of
     the model load are recorded (time.perf_counter). The first detect() call
     includes Ultralytics' predictor set-up and is reported separately.
  5. --verify REF_DIR FILE compares, for every cached screen that has
     REF_DIR/<stem>/FILE, the cached detections with that reference file
     (compare_detections): v1.1.0 wrote its raw detections (there was no
     de-duplication yet) to results/v1.1.0_examples/<stem>/1_detections.json,
     v1.2.0 to tests/golden/<stem>/1_detections_raw.json. Keys that only one
     side has (v1.1.0 has no img_w / img_h) are listed, not compared.

The metadata (--meta-out, JSON) holds the settings, weights MD5, versions,
host, counts, timings (per image and summarised) and the verification.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Tolerances of a "match" in compare_detections: those tests/test_golden.py uses
# for the released weights on another machine (confidence 1e-4, box 1e-2 px).
# Exact equality is reported separately.
DEFAULT_CONF_TOL = 1e-4
DEFAULT_BOX_TOL = 1e-2
# Numeric keys of a detection dict compared by compare_detections.
NUMERIC_KEYS = ("confidence", "bbox", "bbox_norm", "center", "width", "height")
# Keys compared for equality.
EXACT_KEYS = ("id", "class_name", "class_id", "img_w", "img_h")


def pipeline_detect_settings() -> Dict[str, Any]:
    """Reads the detection defaults of run_pipeline from its signature.

    Input:  none (imports ui2compose.infer.pipeline, which does not import torch)
    Output: {'conf_threshold', 'iou_threshold', 'imgsz', 'agnostic_nms',
             'max_det'} — the keyword arguments run_pipeline passes to
            ComponentDetector.detect() when called with its defaults
    """
    from ui2compose.infer.pipeline import run_pipeline

    params = inspect.signature(run_pipeline).parameters
    return {k: params[k].default for k in
            ("conf_threshold", "iou_threshold", "imgsz", "agnostic_nms", "max_det")}


def cache_file(out_dir: str | Path, stem: str) -> Path:
    """Path of one screen's cached detections.

    Input:  out_dir — cache directory of a split; stem — image stem
    Output: <out_dir>/<stem>.json
    """
    return Path(out_dir) / f"{stem}.json"


def load_cached(out_dir: str | Path, stem: str) -> List[Dict[str, Any]]:
    """Reads one screen's cached detections.

    Input:  out_dir — cache directory; stem — image stem
    Output: list of detection dicts (format of 1_detections_raw.json);
            raises FileNotFoundError if the screen is not cached
    """
    return json.loads(cache_file(out_dir, stem).read_text(encoding="utf-8"))


def write_detections(path: str | Path, detections: Sequence[Mapping[str, Any]]) -> None:
    """Writes detections as run_pipeline writes 1_detections_raw.json (json.dump, indent 2).

    Input:  path — target file; detections — detection dicts
    Output: none (file written; the parent directory is created)
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(list(detections), f, indent=2)


def _timing_summary(values: Sequence[float]) -> Dict[str, Any]:
    """Median / quartiles / IQR / min / max / mean / total of timings (benchmark.summarize plus total).

    Input:  values — seconds
    Output: dict of statistics ({'n': 0} for no values)
    """
    from ui2compose.evaluation.benchmark import summarize

    s = summarize(values)
    if s["n"]:
        s["total"] = float(sum(values))
    return s


def build_cache(detector: Any, images: Sequence[Path], out_dir: str | Path,
                settings: Optional[Mapping[str, Any]] = None, overwrite: bool = False,
                progress_every: int = 200) -> Dict[str, Any]:
    """Runs detector.detect() on every image and writes <out_dir>/<stem>.json.

    Input:  detector — object with detect(image_path, conf_threshold=...,
                iou_threshold=..., imgsz=..., agnostic_nms=..., max_det=...)
                (a ComponentDetector, or a fake one in tests);
            images — image paths (unique stems); out_dir — cache directory;
            settings — detect() keyword arguments (default:
                pipeline_detect_settings()); overwrite — recompute existing
                files; progress_every — print a line every N images (0: never)
    Output: {'n_images', 'computed', 'kept_existing', 'n_detections' (of the
             computed screens), 'empty_screens' (computed screens without a
             detection), 'seconds_per_image' {stem: s} (computed only),
             'first_call' {'stem', 'seconds'} or None, 'warm' (summary of the
             other calls), 'wall_seconds'}; raises ValueError for duplicate stems
    """
    stems = [Path(p).stem for p in images]
    if len(set(stems)) != len(stems):
        raise ValueError("duplicate image stems: "
                         f"{sorted({s for s in stems if stems.count(s) > 1})[:5]}")
    settings = dict(pipeline_detect_settings() if settings is None else settings)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_image: Dict[str, float] = {}
    n_det, empty, kept = 0, 0, 0
    t_all = time.perf_counter()
    for k, img in enumerate(images):
        target = cache_file(out, Path(img).stem)
        if target.is_file() and not overwrite:
            kept += 1
            continue
        t = time.perf_counter()
        dets = detector.detect(str(img), **settings)
        per_image[Path(img).stem] = time.perf_counter() - t
        write_detections(target, dets)
        n_det += len(dets)
        empty += int(not dets)
        if progress_every and (k + 1) % progress_every == 0:
            print(f"  {k + 1}/{len(images)} images")
    order = list(per_image)
    first = {"stem": order[0], "seconds": per_image[order[0]]} if order else None
    return {"n_images": len(images), "computed": len(per_image), "kept_existing": kept,
            "n_detections": n_det, "empty_screens": empty,
            "seconds_per_image": per_image, "first_call": first,
            "warm": _timing_summary([per_image[s] for s in order[1:]]),
            "wall_seconds": time.perf_counter() - t_all}


def _flatten(v: Any) -> List[float]:
    """Flattens a number or a (nested) list of numbers.

    Input:  v — number or list
    Output: list of floats
    """
    if isinstance(v, (list, tuple)):
        return [x for e in v for x in _flatten(e)]
    return [float(v)]


def compare_detections(cached: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]],
                       conf_tol: float = DEFAULT_CONF_TOL, box_tol: float = DEFAULT_BOX_TOL
                       ) -> Dict[str, Any]:
    """Compares two detection lists of one screen, detection by detection in list order.

    Input:  cached, reference — lists of detection dicts (detect() format);
            conf_tol — tolerance of 'confidence'; box_tol — tolerance (px) of
            'bbox', 'center', 'width', 'height' ('bbox_norm' is compared with
            box_tol / 1000, i.e. well below a pixel of a 1200 x 1920 screen)
    Output: {'n_cached', 'n_reference', 'exact' (every shared key equal),
             'match' (same length, equal EXACT_KEYS, numeric keys within
             tolerance), 'max_abs_diff' {key: value or None},
             'keys_only_cached', 'keys_only_reference', 'mismatches' (first
             five descriptions)}
    """
    keys_c = set().union(*(d.keys() for d in cached)) if cached else set()
    keys_r = set().union(*(d.keys() for d in reference)) if reference else set()
    shared = keys_c & keys_r
    tol = {"confidence": conf_tol, "bbox": box_tol, "center": box_tol, "width": box_tol,
           "height": box_tol, "bbox_norm": box_tol / 1000.0}
    max_diff: Dict[str, Optional[float]] = {k: None for k in NUMERIC_KEYS if k in shared}
    mismatches: List[str] = []
    exact = len(cached) == len(reference)
    if not exact:
        mismatches.append(f"{len(cached)} cached vs {len(reference)} reference detections")
    for i, (a, b) in enumerate(zip(cached, reference)):
        for k in sorted(shared):
            if a.get(k) != b.get(k):
                exact = False
            if (k in a) != (k in b):
                mismatches.append(f"#{i} {k}: present on one side only")
            elif k in max_diff and k in a:
                va, vb = _flatten(a[k]), _flatten(b[k])
                if len(va) != len(vb):
                    mismatches.append(f"#{i} {k}: lengths {len(va)} vs {len(vb)}")
                    continue
                d = max((abs(x - y) for x, y in zip(va, vb)), default=0.0)
                max_diff[k] = d if max_diff[k] is None else max(max_diff[k], d)
                if d > tol[k]:
                    mismatches.append(f"#{i} {k}: |diff| {d:.3g} > {tol[k]:g}")
            elif k in EXACT_KEYS and a.get(k) != b.get(k):
                mismatches.append(f"#{i} {k}: {a.get(k)!r} vs {b.get(k)!r}")
    return {"n_cached": len(cached), "n_reference": len(reference), "exact": exact,
            "match": not mismatches, "max_abs_diff": max_diff,
            "keys_only_cached": sorted(keys_c - keys_r),
            "keys_only_reference": sorted(keys_r - keys_c), "mismatches": mismatches[:5]}


def verify_cache(out_dir: str | Path, stems: Sequence[str], ref_dir: str | Path, ref_file: str,
                 conf_tol: float = DEFAULT_CONF_TOL, box_tol: float = DEFAULT_BOX_TOL
                 ) -> Dict[str, Any]:
    """Compares the cache with reference files REF_DIR/<stem>/FILE for the cached stems that have one.

    Input:  out_dir — cache directory; stems — cached image stems; ref_dir,
            ref_file — reference location; conf_tol, box_tol — see compare_detections
    Output: {'reference', 'n_compared', 'n_exact', 'n_match', 'max_abs_diff'
             {key: max over screens}, 'keys_only_cached', 'keys_only_reference',
             'screens': {stem: compare_detections result}}
    """
    screens: Dict[str, Any] = {}
    for stem in stems:
        ref = Path(ref_dir) / stem / ref_file
        if ref.is_file():
            screens[stem] = compare_detections(load_cached(out_dir, stem),
                                               json.loads(ref.read_text(encoding="utf-8")),
                                               conf_tol, box_tol)
    max_diff: Dict[str, Optional[float]] = {}
    for r in screens.values():
        for k, v in r["max_abs_diff"].items():
            if v is not None:
                max_diff[k] = v if max_diff.get(k) is None else max(max_diff[k], v)
    return {"reference": f"{ref_dir}/<stem>/{ref_file}", "n_compared": len(screens),
            "n_exact": sum(r["exact"] for r in screens.values()),
            "n_match": sum(r["match"] for r in screens.values()),
            "max_abs_diff": max_diff,
            "keys_only_cached": sorted({k for r in screens.values() for k in r["keys_only_cached"]}),
            "keys_only_reference": sorted({k for r in screens.values()
                                           for k in r["keys_only_reference"]}),
            "conf_tol": conf_tol, "box_tol": box_tol, "screens": screens}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Defines and reads the command-line arguments.

    Input:  argv — argument list (default sys.argv[1:])
    Output: argparse namespace
    """
    ap = argparse.ArgumentParser(description="Cache the pipeline's raw detections of many screens")
    ap.add_argument("--model", required=True, help="weights (e.g. runs/oversample_5k/weights/best.pt)")
    ap.add_argument("--dataset-yaml", default="configs/dataset.yaml",
                    help="passed to ComponentDetector (unused there; class names come from the model)")
    ap.add_argument("--images", required=True, help="image directory or image-list .txt")
    ap.add_argument("--out-dir", required=True, help="cache directory (<stem>.json per screen)")
    ap.add_argument("--meta-out", default=None, help="metadata JSON (default: <out-dir>/../meta_<out-dir name>.json)")
    ap.add_argument("--overwrite", action="store_true", help="recompute existing cache files")
    ap.add_argument("--verify", nargs=2, action="append", default=[], metavar=("REF_DIR", "FILE"),
                    help="compare with REF_DIR/<stem>/FILE where it exists (repeatable)")
    ap.add_argument("--conf-tol", type=float, default=DEFAULT_CONF_TOL)
    ap.add_argument("--box-tol", type=float, default=DEFAULT_BOX_TOL)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Builds (or completes) the cache of one image set, verifies it and writes the metadata.

    Input:  argv — argument list (default sys.argv[1:])
    Output: exit code 0, 1 without images, 3 if a --verify comparison does
            not match; files: the cache and the metadata JSON
    """
    from ui2compose.evaluation.benchmark import host_info
    from ui2compose.evaluation.evaluate_detector import md5_of, read_image_list

    args = parse_args(argv)
    images = read_image_list(args.images)
    if not images:
        print(f"[ERR] no images in {args.images}")
        return 1
    out_dir = Path(args.out_dir)
    meta_out = Path(args.meta_out) if args.meta_out else out_dir.parent / f"meta_{out_dir.name}.json"
    settings = pipeline_detect_settings()
    pending = [p for p in images if args.overwrite or not cache_file(out_dir, p.stem).is_file()]
    load_seconds = None
    detector = None
    if pending:
        from ui2compose.infer.detect_components import ComponentDetector

        t = time.perf_counter()
        detector = ComponentDetector(args.model, args.dataset_yaml)
        load_seconds = time.perf_counter() - t
    stats = build_cache(detector, images, out_dir, settings, overwrite=args.overwrite)
    import ultralytics

    meta: Dict[str, Any] = {
        "images": {"spec": str(args.images), "n": len(images)},
        "out_dir": str(out_dir), "format": "list of ComponentDetector.detect() dicts "
                                           "(= 1_detections_raw.json of run_pipeline)",
        "weights": str(args.model), "weights_md5": md5_of(args.model),
        "settings": {**settings,
                     "imgsz_effective": detector.effective_imgsz(settings["imgsz"]) if detector else None,
                     "device": str(detector.model.predictor.device)
                     if detector is not None and detector.model.predictor is not None else None},
        "ultralytics_version": ultralytics.__version__, "host": host_info(),
        "model_load_seconds": load_seconds, **stats, "verification": [],
    }
    exit_code = 0
    stems = [p.stem for p in images]
    for ref_dir, ref_file in args.verify:
        rep = verify_cache(out_dir, stems, ref_dir, ref_file, args.conf_tol, args.box_tol)
        meta["verification"].append(rep)
        print(f"verify {rep['reference']}: {rep['n_compared']} screens, {rep['n_exact']} exact, "
              f"{rep['n_match']} within tolerance; max |diff| {rep['max_abs_diff']}")
        if rep["n_match"] != rep["n_compared"]:
            exit_code = 3
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    w = stats["warm"]
    print(f"{stats['computed']} computed, {stats['kept_existing']} kept; "
          f"warm detect median {w.get('median', float('nan')):.4f} s; meta: {meta_out}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
