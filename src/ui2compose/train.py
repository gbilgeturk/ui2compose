# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Trains the YOLO detector from a configuration file (configs/train_*.yaml).

A configuration holds the initial weights (`model`), the dataset yaml
(`data`), the output location (`project`, `name`), the Ultralytics version
the published results were produced with (`ultralytics_version`) and a
`train:` mapping that is passed verbatim to ultralytics.YOLO.train().
configs/train_oversample_5k.yaml reproduces the train_args stored in the
released checkpoint; the baseline and 2k configurations differ only in data.

Usage:
    ui2compose-train --config configs/train_oversample_5k.yaml [--data D]
                     [--project P] [--name N] [--device DEV] [--dry-run]
    ui2compose-train --resume runs/oversample_5k/weights/last.pt

--dry-run prints the resolved arguments as JSON and exits without importing
ultralytics. --resume continues an interrupted run from its last.pt; in that
case Ultralytics restores all training arguments from the checkpoint and the
configuration file is not used (only --device is forwarded).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml

REQUIRED_KEYS = ("model", "data", "train")
# Keys of the `train:` mapping that are set from the top level / CLI instead.
RESERVED_TRAIN_KEYS = ("model", "data", "project", "name", "resume", "device")


def load_train_config(path: str | Path) -> Dict[str, Any]:
    """Reads and validates a training configuration file.

    Input:  path — YAML file with model, data, train (mapping) and optional
            project, name, ultralytics_version
    Output: configuration dict; raises ValueError if a required key is
            missing, `train` is not a mapping or contains a reserved key
    """
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing key(s) {missing}")
    if not isinstance(cfg["train"], dict):
        raise ValueError(f"{path}: 'train' must be a mapping")
    reserved = [k for k in cfg["train"] if k in RESERVED_TRAIN_KEYS]
    if reserved:
        raise ValueError(f"{path}: set {reserved} at the top level, not under 'train'")
    return cfg


def build_train_kwargs(cfg: Dict[str, Any], data: Optional[str] = None,
                       project: Optional[str] = None, name: Optional[str] = None,
                       device: Optional[str] = None) -> Dict[str, Any]:
    """Resolves the keyword arguments for YOLO.train() (CLI overrides win).

    Input:  cfg — configuration from load_train_config; data, project, name,
            device — optional overrides
    Output: dict of YOLO.train() keyword arguments (data, project, name,
            device if given, and every entry of cfg['train'])
    """
    kwargs: Dict[str, Any] = dict(cfg["train"])
    kwargs["data"] = data or cfg["data"]
    kwargs["project"] = project or cfg.get("project", "runs")
    kwargs["name"] = name or cfg.get("name", "train")
    if device is not None:
        kwargs["device"] = device
    return kwargs


def check_ultralytics_version(expected: Optional[str]) -> Optional[str]:
    """Compares the installed ultralytics version with the configured one.

    Input:  expected — version string from the configuration (None: no check)
    Output: warning text if the versions differ or ultralytics is missing, else None
    """
    if not expected:
        return None
    try:
        import ultralytics
    except ImportError:
        return "ultralytics is not installed"
    if ultralytics.__version__ != str(expected):
        return (f"installed ultralytics {ultralytics.__version__} differs from {expected}, "
                f"the version the published results were produced with")
    return None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Defines and reads the command-line arguments.

    Input:  argv — argument list (default sys.argv[1:])
    Output: argparse namespace
    """
    ap = argparse.ArgumentParser(description="Train the ui2compose YOLO detector")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="training configuration (configs/train_*.yaml)")
    src.add_argument("--resume", help="last.pt of an interrupted run to resume")
    ap.add_argument("--data", default=None, help="override the dataset yaml")
    ap.add_argument("--project", default=None, help="override the output project directory")
    ap.add_argument("--name", default=None, help="override the run name")
    ap.add_argument("--device", default=None, help="device, e.g. 0, cpu, mps")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved arguments and exit (no training)")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Runs (or, with --dry-run, only resolves) a training run.

    Input:  argv — argument list (default sys.argv[1:])
    Output: exit code 0; side effects: the Ultralytics run directory
            <project>/<name> (not with --dry-run)
    """
    args = parse_args(argv)
    if args.resume:
        kwargs: Dict[str, Any] = {"resume": True}
        if args.device is not None:
            kwargs["device"] = args.device
        model_path = args.resume
        expected = None
    else:
        cfg = load_train_config(args.config)
        kwargs = build_train_kwargs(cfg, args.data, args.project, args.name, args.device)
        model_path = cfg["model"]
        expected = cfg.get("ultralytics_version")

    if args.dry_run:
        print(json.dumps({"model": model_path, "train_kwargs": kwargs,
                          "ultralytics_version": expected}, indent=2))
        return 0

    warning = check_ultralytics_version(expected)
    if warning:
        print(f"[WARN] {warning}")
    from ultralytics import YOLO  # heavy import only when actually training

    YOLO(model_path).train(**kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
