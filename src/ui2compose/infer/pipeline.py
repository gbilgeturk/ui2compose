# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""End-to-end pipeline: detection -> de-duplication -> UI graph -> layout and Compose code.

Stages (numbered as in the README) and the files they write into the output
directory:

1. Detection (ComponentDetector.detect): ``1_detections_raw.json``.
2. De-duplication (postprocess.py), preceded by the optional per-class
   confidence filter: ``1_detections.json`` (kept), ``1_dedup_removed.json``
   (removed, each with the rule that removed it) and, if requested,
   ``1_detections_viz.png``.
3. UI graph (UIGraphBuilder.build_graph): ``3_ui_graph.json``. Code generation
   consumes only its containment hierarchy (see build_ui_graph.py).
4. Layout and code generation (ComposeCodeGenerator.generate):
   ``2_layout_tree.json`` (the layout tree the code was generated from) and
   ``4_GeneratedScreen.kt``; with ``explain_id`` also ``2_explain_<id>.txt``,
   a trace of that detection through every stage.

The file-name prefixes are not stage numbers: ``1_``, ``3_`` and ``4_`` are
the names v1.1.0 used (kept for compatibility), and the files added in v1.2.0
by stage 4 take the free prefix ``2_``.

``0_run_config.json`` records every setting of the run, with defaults resolved
to the values actually used. Detection ids are the detector's ids;
post-processing never renumbers them, so an id can be followed through all
files.

The model is loaded lazily inside run_pipeline, so this module (and the
config/CLI helpers) can be imported without torch or ultralytics.
"""

import argparse
import inspect
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from ui2compose import __version__
from ui2compose.infer.build_ui_graph import (DEFAULT_ALIGN_TOL_FRAC,
                                             DEFAULT_CONTAINMENT_RATIO,
                                             UIGraphBuilder, explain_containment)
from ui2compose.infer.generate_compose_code import (DEFAULT_CUT_TOL_FRAC, DEFAULT_FALLBACK,
                                                    LAYOUT_MODES, ComposeCodeGenerator,
                                                    build_layout_tree, explain_path)
from ui2compose.infer.postprocess import (DETECTOR_CLASSES, apply_class_conf,
                                          deduplicate_detections, validate_class_conf)

# Defaults of the path arguments of the `ui2compose` command.
CLI_DEFAULTS: Dict[str, str] = {
    'model_path': 'runs/oversample_5k_appsplit/weights/best.pt',
    'dataset_yaml': 'configs/dataset.yaml',
    'output_dir': 'output',
}

# Mapping of the YAML config (configs/pipeline_config.yaml) to run_pipeline
# keywords: section -> {key -> keyword}. Keys of the 'dedup' section other than
# 'enabled' are passed to deduplicate_detections() via dedup_params.
CONFIG_KEYS: Dict[str, Dict[str, str]] = {
    'input': {'image': 'image_path', 'model': 'model_path', 'dataset_yaml': 'dataset_yaml'},
    'output': {'directory': 'output_dir'},
    'detection': {'confidence_threshold': 'conf_threshold', 'iou_threshold': 'iou_threshold',
                  'imgsz': 'imgsz', 'agnostic_nms': 'agnostic_nms', 'max_det': 'max_det',
                  'class_conf': 'class_conf'},
    'dedup': {'enabled': 'dedup'},
    'graph': {'containment_ratio': 'containment_ratio', 'align_tol_frac': 'align_tol_frac'},
    'layout': {'mode': 'layout', 'size_from_boxes': 'size_from_boxes',
               'cut_tol_frac': 'cut_tol_frac', 'fallback': 'fallback'},
    'features': {'visualization': 'visualize'},
}


def print_detection_statistics(detections: list, title: str = "Detection Statistics"):
    """Prints detection statistics (class distribution, confidence range) to the console.

    Input:  detections — list of detection dictionaries
            title — heading text to show in the output
    Output: none (side effect of printing to the console)
    """
    print(f"\n{'=' * 60}")
    print(f"📊 {title}")
    print(f"{'=' * 60}")

    if not detections:
        print("  No detections found!")
        return

    # Count by class
    class_counts = Counter(d['class_name'] for d in detections)

    print(f"\n  Total components: {len(detections)}")
    print(f"\n  Class distribution:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        percentage = (count / len(detections)) * 100
        print(f"    {class_name:15s}: {count:3d} ({percentage:5.1f}%)")

    # Confidence stats
    confidences = [d['confidence'] for d in detections]
    print(f"\n  Confidence range:")
    print(f"    Min:  {min(confidences):.3f}")
    print(f"    Max:  {max(confidences):.3f}")
    print(f"    Avg:  {sum(confidences) / len(confidences):.3f}")


def _write_json(path: Path, obj: Any):
    """Writes an object as indented JSON and prints a console notice.

    Input:  path — target file; obj — JSON-serialisable object
    Output: none (side effect of writing the file)
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
    print(f"      ✓ Saved: {path}")


def resolve_pipeline_image_size(image_path: str, detections: List[Dict]) -> Tuple[int, int]:
    """Determines the size of the input image.

    Uses the 'img_w'/'img_h' fields the detector writes into every detection; if
    there are no detections (or they lack the fields) the image file is opened
    with PIL (header only).

    Input:  image_path — screenshot path; detections — raw detector output
    Output: (width, height) in pixels
    """
    sizes = {(d['img_w'], d['img_h']) for d in detections
             if d.get('img_w') is not None and d.get('img_h') is not None}
    if len(sizes) == 1:
        return sizes.pop()
    from PIL import Image  # lazy: only needed without detector-provided sizes
    with Image.open(image_path) as img:
        return img.size


def explain_detection(det_id: int, raw: List[Dict], kept: List[Dict], removed: List[Dict],
                      graph: Dict, layout_tree: Optional[Dict]) -> List[str]:
    """Traces one detection through post-processing, graph construction and layout.

    Input:  det_id — detection id (the detector's id); raw — raw detections;
            kept — detections after post-processing; removed — de-duplication removals;
            graph — UI graph; layout_tree — generator.last_layout_tree
    Output: list of explanation lines (containment lines from
            build_ui_graph.explain_containment, layout lines from
            generate_compose_code.explain_path)
    """
    lines = [f"Explain trace for detection id {det_id}"]
    raw_by_id = {d['id']: d for d in raw}
    if det_id not in raw_by_id:
        lines.append(f"[detection] no raw detection has id {det_id} "
                     f"(valid ids: {sorted(raw_by_id)})")
        return lines
    det = raw_by_id[det_id]
    bbox = ", ".join(f"{v:.0f}" for v in det['bbox'])
    lines.append(f"[detection] {det['class_name']} conf={det['confidence']:.2f} bbox=[{bbox}]")

    removed_by_id = {d['id']: d for d in removed}
    if det_id in removed_by_id:
        rec = removed_by_id[det_id]
        lines.append(f"[post-processing] removed by de-duplication: rule "
                     f"'{rec.get('dedup_rule')}', overlap {rec.get('overlap', float('nan')):.3f} "
                     f"with kept detection {rec.get('removed_by')}")
        return lines
    if det_id not in {d['id'] for d in kept}:
        lines.append("[post-processing] removed by the per-class confidence threshold (class_conf)")
        return lines
    lines.append("[post-processing] kept")

    lines.append("[graph] containment:")
    lines.extend("  " + line for line in explain_containment(graph, det_id))

    lines.append("[layout] path in the layout tree:")
    if layout_tree is None:
        lines.append("  no layout tree available")
    else:
        lines.extend("  " + line for line in explain_path(layout_tree, det_id))
    return lines


def effective_dedup_params(dedup_params: Optional[Dict[str, Optional[float]]] = None
                           ) -> Dict[str, Optional[float]]:
    """Resolves de-duplication parameters to the values deduplicate_detections will use.

    Input:  dedup_params — keyword arguments for deduplicate_detections, or None
    Output: dictionary of every threshold parameter of deduplicate_detections
            (its signature defaults, overridden by dedup_params); raises
            ValueError for a key that is not a parameter of deduplicate_detections
    """
    params = {name: p.default for name, p in inspect.signature(deduplicate_detections).parameters.items()
              if name != 'detections'}
    for key, value in (dedup_params or {}).items():
        if key not in params:
            raise ValueError(f"unknown de-duplication parameter {key!r} (known: {sorted(params)})")
        params[key] = value
    return params


def layout_from_boxes(boxes: List[Dict], image_size: Optional[Sequence[float]] = None, *,
                      containment_ratio: float = DEFAULT_CONTAINMENT_RATIO,
                      align_tol_frac: float = DEFAULT_ALIGN_TOL_FRAC,
                      layout: str = 'xycut', cut_tol_frac: Optional[float] = None,
                      fallback: str = DEFAULT_FALLBACK) -> Dict:
    """Lays out a plain list of boxes exactly as run_pipeline lays out its kept detections.

    Runs UIGraphBuilder.build_graph (containment) and then
    generate_compose_code.build_layout_tree with the same parameters and in the
    same way as run_pipeline, without detection, class filter or
    de-duplication; for ground-truth boxes this gives the layout the pipeline
    would produce from perfect detections.  (build_layout_tree on the plain
    list alone would skip containment.)

    Input:  boxes — list of dictionaries with 'bbox' [x1, y1, x2, y2] and
                'class_name' (optional 'id', default list position, and
                'confidence', default 1.0); not modified
            image_size — (width, height) in px (default: resolved by
                build_ui_graph.resolve_image_size)
            containment_ratio, align_tol_frac — UIGraphBuilder thresholds
            layout, cut_tol_frac, fallback — build_layout_tree parameters
                (cut_tol_frac None = the generator's default)
    Output: root layout node (as ComposeCodeGenerator.last_layout_tree)
    """
    builder = UIGraphBuilder(containment_ratio=containment_ratio, align_tol_frac=align_tol_frac)
    graph = builder.build_graph(boxes, image_size=image_size)
    kwargs: Dict[str, Any] = {'layout': layout, 'fallback': fallback}
    if cut_tol_frac is not None:
        kwargs['cut_tol_frac'] = cut_tol_frac
    return build_layout_tree(graph, image_size=graph['image_size'], **kwargs)


def run_pipeline(
        image_path: str,
        model_path: str,
        dataset_yaml: str,
        output_dir: str = "output",
        conf_threshold: float = 0.3,
        visualize: bool = True,
        *,
        iou_threshold: float = 0.7,
        imgsz: Optional[int] = None,
        agnostic_nms: bool = False,
        max_det: int = 300,
        class_conf: Optional[Dict[str, float]] = None,
        dedup: bool = True,
        dedup_params: Optional[Dict[str, Optional[float]]] = None,
        containment_ratio: float = DEFAULT_CONTAINMENT_RATIO,
        align_tol_frac: float = DEFAULT_ALIGN_TOL_FRAC,
        layout: str = 'xycut',
        size_from_boxes: bool = False,
        cut_tol_frac: Optional[float] = None,
        fallback: str = DEFAULT_FALLBACK,
        explain_id: Optional[int] = None,
        detector: Optional[Any] = None,
) -> Dict[str, Any]:
    """Runs the full pipeline: detection -> class_conf filter -> de-duplication -> graph -> Compose code.

    The detection defaults (conf 0.3, NMS IoU 0.7, the checkpoint's own imgsz,
    class-aware NMS, max_det 300) are the settings v1.1.0 actually ran with, so
    the raw detections are identical to v1.1.0's.

    Input:  image_path — input screenshot path
            model_path — YOLO model weights file path
            dataset_yaml — dataset configuration file path
            output_dir — output folder (default "output")
            conf_threshold — detection confidence threshold (default 0.3)
            visualize — if True, the detection visualization is also saved
            iou_threshold, imgsz, agnostic_nms, max_det — passed to ComponentDetector.detect
                (imgsz None = the checkpoint's training size; the value used is
                recorded as 'imgsz_effective' in 0_run_config.json)
            class_conf — optional per-class minimum confidences (postprocess.apply_class_conf);
                its class names are checked against the detector's classes
                (detector.class_names, else postprocess.DETECTOR_CLASSES) before
                detection, and an unknown name raises ValueError
            dedup — if True, postprocess.deduplicate_detections is applied
            dedup_params — keyword arguments for deduplicate_detections (None = its defaults)
            containment_ratio, align_tol_frac — UIGraphBuilder thresholds
            layout, size_from_boxes, cut_tol_frac, fallback — passed to
                ComposeCodeGenerator.generate (cut_tol_frac only if not None, so that
                None means the generator's default)
            explain_id — optional detection id to trace (printed and written to
                2_explain_<id>.txt)
            detector — optional object with detect()/visualize() used instead of loading
                ComponentDetector (e.g. a fake detector in tests)
    Output: result dictionary with keys {'raw_detections', 'detections' (kept), 'removed',
            'graph', 'layout_tree', 'code', 'explain' (lines or None), 'output_dir'}
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("🚀 UI-to-Compose Pipeline")
    print("=" * 60)
    print(f"\n📷 Input: {image_path}")
    print(f"🤖 Model: {Path(model_path).name}")
    # Fail before loading the model if a dedup parameter is misspelt.
    effective_dedup = effective_dedup_params(dedup_params)

    # ========== STAGE 1: DETECTION ==========
    print("\n" + "─" * 60)
    print("[1/4] 🔍 Detecting UI Components...")
    print("─" * 60)

    if detector is None:
        # Lazy import: loads ultralytics/torch only when a real model is needed
        from ui2compose.infer.detect_components import ComponentDetector
        detector = ComponentDetector(model_path, dataset_yaml)
    known_classes = list(getattr(detector, 'class_names', None) or DETECTOR_CLASSES)
    validate_class_conf(class_conf, known_classes)
    effective_imgsz = detector.effective_imgsz(imgsz) if hasattr(detector, 'effective_imgsz') else imgsz
    print(f"🎯 Confidence threshold: {conf_threshold}   NMS IoU: {iou_threshold}   "
          f"imgsz: {effective_imgsz if effective_imgsz is not None else 'checkpoint default'}")

    # Every setting, with the defaults resolved to the values actually used.
    run_config = {
        'version': __version__,
        'image_path': str(image_path), 'model_path': str(model_path),
        'dataset_yaml': str(dataset_yaml), 'output_dir': str(output_dir),
        'conf_threshold': conf_threshold, 'visualize': visualize,
        'iou_threshold': iou_threshold, 'imgsz': imgsz, 'imgsz_effective': effective_imgsz,
        'agnostic_nms': agnostic_nms, 'max_det': max_det, 'class_conf': class_conf,
        'dedup': dedup, 'dedup_params': effective_dedup if dedup else None,
        'containment_ratio': containment_ratio, 'align_tol_frac': align_tol_frac,
        'layout': layout, 'size_from_boxes': size_from_boxes,
        'cut_tol_frac': DEFAULT_CUT_TOL_FRAC if cut_tol_frac is None else cut_tol_frac,
        'fallback': fallback, 'explain_id': explain_id,
    }
    _write_json(output_path / "0_run_config.json", run_config)

    raw_detections = detector.detect(image_path, conf_threshold=conf_threshold,
                                     iou_threshold=iou_threshold, imgsz=imgsz,
                                     agnostic_nms=agnostic_nms, max_det=max_det)
    print(f"      ✓ Raw detections: {len(raw_detections)} components")
    _write_json(output_path / "1_detections_raw.json", raw_detections)

    # ========== STAGE 2: DE-DUPLICATION ==========
    print("\n" + "─" * 60)
    print("[2/4] 🧹 Post-processing (class filter, de-duplication)...")
    print("─" * 60)

    filtered = apply_class_conf(raw_detections, class_conf, known_classes=known_classes)
    if class_conf:
        print(f"      ✓ Per-class confidence filter: {len(raw_detections) - len(filtered)} removed")
    if dedup:
        detections, removed = deduplicate_detections(filtered, **effective_dedup)
        print(f"      ✓ De-duplication: {len(removed)} removed, {len(detections)} kept")
        for rec in removed:
            print(f"        - id {rec['id']} {rec['class_name']} ({rec['confidence']:.2f}): "
                  f"{rec['dedup_rule']} {rec['overlap']:.2f} with id {rec['removed_by']}")
    else:
        detections, removed = list(filtered), []
        print(f"      ✓ De-duplication: off")
    _write_json(output_path / "1_detections.json", detections)
    _write_json(output_path / "1_dedup_removed.json", removed)

    # Print statistics
    print_detection_statistics(detections, "Detection Results (after post-processing)")

    # Visualize detections
    if visualize:
        vis_path = output_path / "1_detections_viz.png"
        detector.visualize(image_path, detections, str(vis_path), removed=removed)
        print(f"      ✓ Visualization: {vis_path}")

    image_size = resolve_pipeline_image_size(image_path, raw_detections)

    # ========== STAGE 3: UI GRAPH ==========
    print("\n" + "─" * 60)
    print("[3/4] 🌳 Building UI Hierarchy Graph...")
    print("─" * 60)

    builder = UIGraphBuilder(containment_ratio=containment_ratio, align_tol_frac=align_tol_frac)
    graph = builder.build_graph(detections, image_size=image_size)

    print(f"      ✓ Nodes: {len(graph['nodes'])}")
    print(f"      ✓ Edges: {len(graph['edges'])} (inspection only, not used by code generation)")
    print(f"      ✓ Root components: {len(graph['hierarchy']['roots'])}")
    print(f"      ✓ Image size: {graph['image_size'][0]}x{graph['image_size'][1]}")

    # Save graph JSON
    _write_json(output_path / "3_ui_graph.json", graph)

    # Print hierarchy preview
    print(f"\n      Hierarchy preview:")

    def print_tree(node, indent=0, max_depth=2):
        """Prints the hierarchy tree to the console in indented form.

        Input:  node — hierarchy node dictionary (contains 'class' and 'children')
                indent — current indentation level
                max_depth — maximum depth to print
        Output: none (side effect of printing to the console)
        """
        if indent > max_depth:
            return
        prefix = "      " + "  " * indent
        print(f"{prefix}└─ {node['class']}")
        for child in node.get('children', [])[:2]:
            print_tree(child, indent + 1, max_depth)
        if len(node.get('children', [])) > 2:
            print(f"{prefix}   └─ ... ({len(node.get('children', [])) - 2} more)")

    for i, root in enumerate(graph['hierarchy']['roots'][:2]):
        print(f"\n      Root {i + 1}:")
        print_tree(root)

    if len(graph['hierarchy']['roots']) > 2:
        print(f"\n      ... and {len(graph['hierarchy']['roots']) - 2} more roots")

    # ========== STAGE 4: LAYOUT AND CODE GENERATION ==========
    print("\n" + "─" * 60)
    print("[4/4] 📝 Layout and Jetpack Compose Code...")
    print("─" * 60)

    generator = ComposeCodeGenerator()
    gen_kwargs: Dict[str, Any] = {'image_size': image_size, 'layout': layout,
                                  'size_from_boxes': size_from_boxes, 'fallback': fallback}
    if cut_tol_frac is not None:
        gen_kwargs['cut_tol_frac'] = cut_tol_frac
    code = generator.generate(graph, **gen_kwargs)
    layout_tree = getattr(generator, 'last_layout_tree', None)
    _write_json(output_path / "2_layout_tree.json", layout_tree)

    # Count lines
    num_lines = len(code.split('\n'))
    num_composables = code.count('@Composable')

    print(f"      ✓ Generated {num_lines} lines of code")
    print(f"      ✓ Composable functions: {num_composables}")

    # Save code
    code_path = output_path / "4_GeneratedScreen.kt"
    with open(code_path, 'w', encoding='utf-8') as f:
        f.write(code)
    print(f"      ✓ Compose code saved: {code_path}")

    # Print code preview
    print(f"\n      Code preview (first 15 lines):")
    code_lines = code.split('\n')
    for i, line in enumerate(code_lines[:15], 1):
        print(f"      {i:3d} | {line}")
    if len(code_lines) > 15:
        print(f"      ... | ({len(code_lines) - 15} more lines)")

    explain_lines = None
    if explain_id is not None:
        explain_lines = explain_detection(explain_id, raw_detections, detections, removed,
                                          graph, layout_tree)
        print(f"\n      Explain id {explain_id}:")
        for line in explain_lines:
            print(f"      {line}")
        explain_file = output_path / f"2_explain_{explain_id}.txt"
        with open(explain_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(explain_lines) + "\n")
        print(f"      ✓ Saved: {explain_file}")

    # ========== SUMMARY ==========
    print("\n" + "─" * 60)
    print("📊 Pipeline Summary")
    print("─" * 60)

    print(f"\n  📷 Input:  {Path(image_path).name}")
    print(f"  📁 Output: {output_dir}/")
    print(f"\n  📦 Results:")
    print(f"     • Components detected:  {len(raw_detections)} raw, {len(detections)} kept")
    print(f"     • Relationships found: {len(graph['edges'])}")
    print(f"     • Root components:     {len(graph['hierarchy']['roots'])}")
    print(f"     • Generated code:      {num_lines} lines")

    print(f"\n  📄 Output files:")
    output_files = sorted(output_path.glob("*"))
    for f in output_files:
        size = f.stat().st_size / 1024
        print(f"     • {f.name:<30s} ({size:>6.1f} KB)")

    print("\n" + "=" * 60)
    print("✅ Pipeline Completed Successfully!")
    print("=" * 60)

    # Final class distribution
    print(f"\n📊 Final Component Distribution:")
    final_classes = Counter(d['class_name'] for d in detections)
    for cls, count in sorted(final_classes.items()):
        print(f"   {cls:15s}: {count:2d}")

    return {
        'raw_detections': raw_detections,
        'detections': detections,
        'removed': removed,
        'graph': graph,
        'layout_tree': layout_tree,
        'code': code,
        'explain': explain_lines,
        'output_dir': str(output_path),
    }


# ── Configuration file and command line ────────────────────


def load_config(path: str) -> Dict:
    """Reads a pipeline YAML config file (configs/pipeline_config.yaml format).

    Input:  path — YAML file path
    Output: config dictionary ({} for an empty file); raises ValueError if the
            top level is not a mapping
    """
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"{path}: the top level of a pipeline config must be a mapping")
    return config


def config_to_kwargs(config: Dict) -> Dict[str, Any]:
    """Converts a pipeline config dictionary into run_pipeline keyword arguments.

    Only the keys present in the config are returned, so missing keys fall back
    to the run_pipeline defaults. Unknown sections or keys raise ValueError, so a
    misspelt threshold cannot be silently ignored. The 'dedup' keys other than
    'enabled' are checked against the parameters of deduplicate_detections.

    Input:  config — dictionary as read from the YAML file (see CONFIG_KEYS)
    Output: dictionary of run_pipeline keyword arguments
    """
    dedup_names = [p for p in inspect.signature(deduplicate_detections).parameters
                   if p != 'detections']
    kwargs: Dict[str, Any] = {}
    dedup_params: Dict[str, Any] = {}
    for section, values in config.items():
        if section not in CONFIG_KEYS:
            raise ValueError(f"Unknown config section '{section}' "
                             f"(known: {sorted(CONFIG_KEYS)})")
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ValueError(f"Config section '{section}' must be a mapping")
        for key, value in values.items():
            if key in CONFIG_KEYS[section]:
                kwargs[CONFIG_KEYS[section][key]] = value
            elif section == 'dedup' and key in dedup_names:
                dedup_params[key] = value
            else:
                known = sorted(CONFIG_KEYS[section])
                if section == 'dedup':
                    known += dedup_names
                raise ValueError(f"Unknown config key '{section}.{key}' (known: {known})")
    if dedup_params:
        kwargs['dedup_params'] = dedup_params
    return kwargs


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the parser of the `ui2compose` command.

    Every option defaults to "not given" (argparse.SUPPRESS), so that
    resolve_cli_settings can tell explicit flags from config-file values.

    Input:  none
    Output: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="UI Screenshot to Jetpack Compose Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        argument_default=argparse.SUPPRESS,
        epilog="Precedence: explicit flags > --config file > built-in defaults.",
    )
    parser.add_argument("image", nargs="?",
                        help="Input screenshot path (default: input.image of --config)")
    parser.add_argument("--config", metavar="YAML",
                        help="Pipeline config (configs/pipeline_config.yaml format)")
    parser.add_argument("--model", dest="model_path",
                        help=f"YOLO model path (default: {CLI_DEFAULTS['model_path']})")
    parser.add_argument("--dataset-yaml", dest="dataset_yaml",
                        help=f"Dataset config YAML (default: {CLI_DEFAULTS['dataset_yaml']})")
    parser.add_argument("--output", dest="output_dir",
                        help=f"Output directory (default: {CLI_DEFAULTS['output_dir']})")
    parser.add_argument("--conf", dest="conf_threshold", type=float,
                        help="Detection confidence threshold (default: 0.3)")
    parser.add_argument("--iou", dest="iou_threshold", type=float,
                        help="NMS IoU threshold (default: 0.7)")
    parser.add_argument("--imgsz", type=int,
                        help="Detector input size (default: the checkpoint's training size, "
                             "640 for both bundled weights)")
    parser.add_argument("--agnostic-nms", dest="agnostic_nms",
                        action=argparse.BooleanOptionalAction,
                        help="Class-agnostic NMS (default: off)")
    parser.add_argument("--max-det", dest="max_det", type=int,
                        help="Maximum detections per image (default: 300)")
    parser.add_argument("--no-dedup", dest="dedup", action="store_false",
                        help="Skip the de-duplication step")
    parser.add_argument("--layout", choices=list(LAYOUT_MODES),
                        help="Layout mode of the code generator (default: xycut)")
    parser.add_argument("--size-from-boxes", dest="size_from_boxes",
                        action=argparse.BooleanOptionalAction,
                        help="Size components from their boxes (default: off)")
    parser.add_argument("--explain", dest="explain_id", type=int, metavar="ID",
                        help="Trace detection ID through all stages")
    parser.add_argument("--no-viz", dest="visualize", action="store_false",
                        help="Skip visualization")
    return parser


def resolve_cli_settings(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Parses the command line and merges it with the optional --config file.

    Precedence: explicit command-line flags > values in the --config file >
    CLI_DEFAULTS (paths) / run_pipeline defaults (everything else).

    Input:  argv — argument list (None = sys.argv[1:])
    Output: dictionary of run_pipeline keyword arguments (always with
            image_path, model_path, dataset_yaml, output_dir)
    """
    parser = build_arg_parser()
    args = vars(parser.parse_args(argv))
    settings: Dict[str, Any] = dict(CLI_DEFAULTS)
    config_path = args.pop('config', None)
    if config_path is not None:
        try:
            settings.update(config_to_kwargs(load_config(config_path)))
        except (OSError, ValueError, yaml.YAMLError) as e:
            parser.error(f"--config {config_path}: {e}")
    if 'image' in args:
        args['image_path'] = args.pop('image')
    settings.update(args)
    if not settings.get('image_path'):
        parser.error("an input image is required (positional argument or input.image in --config)")
    return settings


def main(argv: Optional[Sequence[str]] = None):
    """Reads command-line arguments, validates the inputs and runs the pipeline.

    Input:  argv — argument list (None = sys.argv[1:])
    Output: none (pipeline outputs are written to the output folder; sys.exit on error)
    """
    settings = resolve_cli_settings(argv)
    image_path = settings['image_path']
    model_path = settings['model_path']

    # Validate inputs
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)

    if not Path(model_path).exists():
        print(f"❌ Error: Model not found: {model_path}")
        print(f"\n🔍 Searching for available models...")
        runs_dir = Path("runs")
        if runs_dir.exists():
            found_models = list(runs_dir.rglob("best.pt"))
            if found_models:
                print(f"✓ Found {len(found_models)} model(s):")
                for model in found_models[:5]:
                    print(f"  - {model}")
                print(f"\n💡 Usage: python {sys.argv[0]} {image_path} --model {found_models[0]}")
            else:
                print("  No models found")
        sys.exit(1)

    # Run pipeline
    try:
        results = run_pipeline(**settings)

        print(f"\n💾 All results saved to: {results['output_dir']}")

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
