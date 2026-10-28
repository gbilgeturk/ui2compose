# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Per-stage wall-clock benchmark of the pipeline on a set of screenshots.

Usage:
    ui2compose-benchmark --images examples --model runs/oversample_5k_appsplit/weights/best.pt \
        --repeats 5 [--out results/benchmark_<host>.json]

Stages (time.perf_counter, seconds):
  import      — `import torch, ultralytics` plus the pipeline modules, measured
                (a) in a fresh Python subprocess per repeat (the first
                subprocess may still benefit from the OS file cache) and
                (b) once in this process;
  model_load  — ComponentDetector(...) construction, once per repeat
                (first construction vs. later ones);
  detect      — ComponentDetector.detect() at the pipeline threshold; the
                first call in the process includes predictor set-up and
                warm-up and is reported separately from the warm calls;
  dedup       — postprocess.deduplicate_detections();
  graph       — UIGraphBuilder().build_graph();
  codegen     — ComposeCodeGenerator().generate().
For every stage the first call and the median / quartiles / IQR of the warm
calls (all images x repeats, first call excluded) are reported, together with
the host (OS, CPU model, cores, RAM, Python, torch, ultralytics, device) and
the system load average (1 / 5 / 15 min) at the start and end of the run,
since other processes on the machine slow every stage.

The default output name uses a host tag made of OS, architecture and CPU
model (not the network host name).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg")
DEFAULT_REPEATS = 5
DEFAULT_CONF = 0.3  # pipeline/demo detection threshold

_IMPORT_SNIPPET = (
    "import time; t = time.perf_counter(); import torch, ultralytics; "
    "import ui2compose.infer.detect_components, ui2compose.infer.build_ui_graph, "
    "ui2compose.infer.generate_compose_code\n"
    "try:\n    import ui2compose.infer.postprocess\nexcept ImportError:\n    pass\n"
    "print(time.perf_counter() - t)")


def summarize(values: Sequence[float]) -> Dict[str, Any]:
    """Median, quartiles (linear interpolation), IQR, min, max and mean of timings.

    Input:  values — timings in seconds
    Output: {'n', 'median', 'q1', 'q3', 'iqr', 'min', 'max', 'mean'} (only
            'n' = 0 for an empty input)
    """
    v = np.asarray(list(values), dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"n": int(v.size), "median": float(med), "q1": float(q1), "q3": float(q3),
            "iqr": float(q3 - q1), "min": float(v.min()), "max": float(v.max()),
            "mean": float(v.mean())}


def _sysctl(name: str) -> Optional[str]:
    """Reads a macOS sysctl value.

    Input:  name — sysctl key
    Output: value string or None
    """
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _version(dist: str) -> Optional[str]:
    """Installed version of a distribution without importing it.

    Input:  dist — distribution name
    Output: version string or None
    """
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def host_info() -> Dict[str, Any]:
    """Describes the host (does not import torch or ultralytics).

    Input:  none
    Output: {'platform', 'system', 'machine', 'cpu_model', 'logical_cores',
             'physical_cores', 'ram_bytes', 'python', 'torch', 'ultralytics',
             'numpy', 'opencv'}; unknown values are None
    """
    system = platform.system()
    cpu, phys, ram = None, None, None
    if system == "Darwin":
        cpu = _sysctl("machdep.cpu.brand_string")
        phys = _sysctl("hw.physicalcpu")
        ram = _sysctl("hw.memsize")
    elif system == "Linux":
        try:
            text = Path("/proc/cpuinfo").read_text()
            m = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
            cpu = m.group(1).strip() if m else None
            cores = {(a, b) for a, b in re.findall(
                r"physical id\s*:\s*(\d+).*?core id\s*:\s*(\d+)", text, re.S)}
            phys = len(cores) or None
            m = re.search(r"^MemTotal:\s*(\d+) kB", Path("/proc/meminfo").read_text(), re.M)
            ram = int(m.group(1)) * 1024 if m else None
        except OSError:
            pass
    if cpu is None:
        cpu = platform.processor() or None
    if ram is None and hasattr(os, "sysconf"):
        try:
            ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            ram = None
    return {
        "platform": platform.platform(), "system": system, "machine": platform.machine(),
        "cpu_model": cpu, "logical_cores": os.cpu_count(),
        "physical_cores": int(phys) if phys else None,
        "ram_bytes": int(ram) if ram else None,
        "python": platform.python_version(), "torch": _version("torch"),
        "ultralytics": _version("ultralytics"), "numpy": _version("numpy"),
        "opencv": _version("opencv-python") or _version("opencv-python-headless"),
    }


def host_tag(info: Dict[str, Any]) -> str:
    """Builds a file-name tag from OS, architecture and CPU model.

    Input:  info — host_info() result
    Output: lower-case tag such as 'darwin-arm64-apple-m2'
    """
    parts = [info.get("system") or "", info.get("machine") or "", info.get("cpu_model") or ""]
    tag = re.sub(r"[^a-z0-9]+", "-", " ".join(parts).lower()).strip("-")
    return tag[:60] or "host"


def _timed(fn: Callable[[], Any]) -> tuple:
    """Calls fn and measures its wall time.

    Input:  fn — zero-argument callable
    Output: (result, seconds)
    """
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def _list_images(spec: str) -> List[Path]:
    """Lists the benchmark images (a directory or a single file).

    Input:  spec — directory or image path
    Output: sorted image paths
    """
    p = Path(spec)
    if p.is_file():
        return [p]
    return sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS)


def load_average() -> Optional[List[float]]:
    """System load average over 1, 5 and 15 minutes (None where unavailable).

    Input:  none
    Output: [1 min, 5 min, 15 min] rounded to 2 decimals, or None
    """
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def run_benchmark(images: Sequence[Path], model_path: str, dataset_yaml: str,
                  repeats: int = DEFAULT_REPEATS, conf: float = DEFAULT_CONF) -> Dict[str, Any]:
    """Times every pipeline stage (see module docstring).

    Input:  images — screenshots; model_path — weights; dataset_yaml —
            passed to ComponentDetector; repeats — passes over the images
            (>= 1); conf — detection threshold. The device is the one
            ultralytics selects by default (reported in host.device)
    Output: result dict {'host', 'settings', 'stages', 'per_image',
            'api_notes', 'created'}
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    load_start = load_average()
    notes: List[str] = []
    sub = []
    for _ in range(repeats):
        r = subprocess.run([sys.executable, "-c", _IMPORT_SNIPPET], capture_output=True,
                           text=True, timeout=600)
        try:
            sub.append(float(r.stdout.strip().splitlines()[-1]))
        except (IndexError, ValueError):
            notes.append(f"import subprocess failed: {r.stderr.strip()[-300:]}")

    t0 = time.perf_counter()
    import torch  # noqa: F401  (timed import)
    import ultralytics  # noqa: F401
    from ui2compose.infer.build_ui_graph import UIGraphBuilder
    from ui2compose.infer.detect_components import ComponentDetector
    from ui2compose.infer.generate_compose_code import ComposeCodeGenerator
    try:
        from ui2compose.infer.postprocess import deduplicate_detections
    except ImportError:  # module not available in this version
        deduplicate_detections = None
        notes.append("postprocess.deduplicate_detections not importable: dedup stage skipped")
    import_in_process = time.perf_counter() - t0

    loads = []
    detector = None
    for _ in range(repeats):
        detector, dt = _timed(lambda: ComponentDetector(model_path, dataset_yaml))
        loads.append(dt)

    stages = {k: [] for k in ("detect", "dedup", "graph", "codegen", "end_to_end")}
    first: Dict[str, Optional[float]] = {k: None for k in stages}
    per_image: Dict[str, Dict[str, List[float]]] = {p.stem: {k: [] for k in stages} for p in images}
    for _ in range(repeats):
        for img in images:
            dets, t_det = _timed(lambda: detector.detect(str(img), conf_threshold=conf))
            if deduplicate_detections is not None:
                (kept, _removed), t_dd = _timed(lambda: deduplicate_detections(dets))
            else:
                kept, t_dd = dets, None
            graph, t_gr = _timed(lambda: UIGraphBuilder().build_graph(kept))
            _code, t_cg = _timed(lambda: ComposeCodeGenerator().generate(graph))
            vals = {"detect": t_det, "dedup": t_dd, "graph": t_gr, "codegen": t_cg}
            vals["end_to_end"] = sum(v for v in vals.values() if v is not None)
            for k, v in vals.items():
                if v is None:
                    continue
                per_image[img.stem][k].append(v)
                if first[k] is None:
                    first[k] = v
                else:
                    stages[k].append(v)

    device_used = None
    pred = getattr(detector.model, "predictor", None)
    if pred is not None and getattr(pred, "device", None) is not None:
        device_used = str(pred.device)
    return {
        "host": {**host_info(), "device": device_used,
                 "load_average_1_5_15min": {"start": load_start, "end": load_average()}},
        "settings": {"images": [p.name for p in images], "n_images": len(images),
                     "model": str(model_path), "repeats": repeats, "conf": conf,
                     "timer": "time.perf_counter", "unit": "seconds"},
        "stages": {
            "import_subprocess": {"first": sub[0] if sub else None,
                                  "all": summarize(sub)},
            "import_in_process": import_in_process,
            "model_load": {"first": loads[0], "warm": summarize(loads[1:])},
            **{k: {"first": first[k], "warm": summarize(stages[k])} for k in stages},
        },
        "per_image": {s: {k: summarize(v)["median"] if v else None for k, v in d.items()}
                      for s, d in per_image.items()},
        "api_notes": notes,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Defines and reads the command-line arguments.

    Input:  argv — argument list (default sys.argv[1:])
    Output: argparse namespace
    """
    ap = argparse.ArgumentParser(description="Per-stage pipeline benchmark")
    ap.add_argument("--images", default="examples", help="image directory or file")
    ap.add_argument("--model", default="runs/oversample_5k_appsplit/weights/best.pt")
    ap.add_argument("--dataset-yaml", default="configs/dataset.yaml")
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--out", default=None,
                    help="output JSON (default results/benchmark_<host tag>.json)")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Runs the benchmark and writes the JSON report.

    Input:  argv — argument list (default sys.argv[1:])
    Output: exit code 0; file written to --out
    """
    args = parse_args(argv)
    images = _list_images(args.images)
    if not images:
        print(f"[ERR] no images in {args.images}")
        return 1
    res = run_benchmark(images, args.model, args.dataset_yaml, args.repeats, args.conf)
    out = Path(args.out) if args.out else Path("results") / f"benchmark_{host_tag(res['host'])}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    imp = res["stages"]["import_subprocess"]["all"]
    if imp.get("n"):
        print(f"{'import':12s} subprocess median {imp['median'] * 1000:.1f} ms "
              f"(IQR {imp['iqr'] * 1000:.1f}); in-process "
              f"{res['stages']['import_in_process'] * 1000:.1f} ms")
    for k, v in res["stages"].items():
        if isinstance(v, dict) and "warm" in v:
            w = v["warm"]
            med = f"{w['median'] * 1000:.1f} ms (IQR {w['iqr'] * 1000:.1f})" if w.get("n") else "-"
            fst = f"{v['first'] * 1000:.1f} ms" if v.get("first") is not None else "-"
            print(f"{k:12s} first {fst:>12s}   warm median {med}")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
