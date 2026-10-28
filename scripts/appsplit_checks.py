# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Checks of the app-level split that results/eval_appsplit/README.md cites besides the evaluation.

Usage:
  python scripts/appsplit_checks.py package  PKG [--out results/eval_appsplit/package_verification.json]
  python scripts/appsplit_checks.py colab    PKG [--out results/eval_appsplit/detector_colab_check.json]
  python scripts/appsplit_checks.py previous VIEW SWEEP [--out results/eval_appsplit/val_selected_vs_previous_default.json]

PKG is the data package ui2compose_redraw_appsplit_v1.2.0; VIEW and SWEEP are the package view
and the val sweep that `scripts/run_evaluation.sh app_sweep` writes under $OUT ($OUT/pkg_view,
$OUT/layout_sweep_val). Every check is recomputed from the files; nothing is copied from the
package's own records except where a field says so ('package_audit').

  package   counts per split against the manifest, the summary and the fingerprints; the
            manifest's SHA-256; no package, app group, MD5 or near-duplicate (dHash Hamming
            distance <= 4) in two splits; the MD5 and dHash of every val and test image and the
            SHA-256 of every val and test XML dump against the manifests; one label file per screen.
  colab     the local CPU FP32 detector evaluation (results/eval_appsplit/detector/) against the
            Colab evaluation in the package (verification/metrics/evaluate_detector_*.json), with
            the a-priori tolerances of evaluate_detector --expect-metrics (mAP 0.005, class AP50
            0.02) and 0.01 for an interval bound.
  previous  the selected C4 configuration against the previous default (cut_tol_frac 0.0025,
            the other values selected unchanged) on the val screens, paired, per resampling scheme.
Exit status 0 when every check passes, 3 otherwise (previous: always 0).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SPLITS = ("train", "val", "test")
NEAR_DUP = 4
TOL = {"map": 0.005, "class_ap50": 0.02, "ci": 0.01}
SELECTED = "C4|dedup=looser|cr=0.9|tol=0|fb=y_sorted"
PREVIOUS = "C4|dedup=looser|cr=0.9|tol=0.0025|fb=y_sorted"


def _rows(path: Path) -> list:
    """Reads a CSV file.

    Input:  path — CSV file with a header
    Output: list of row dicts
    """
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _sha256(path: Path) -> str:
    """SHA-256 of a file.

    Input:  path — file
    Output: hex digest
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_package(pkg: Path) -> dict:
    """Independent checks of the app-level split package.

    Input:  pkg — package root
    Output: report dict with 'ok' and one entry per check
    """
    from ui2compose.preprocess.app_split import dhash, md5_file, _popcount_u64

    man_path = pkg / "splits" / "app_split_manifest.csv"
    man = _rows(man_path)
    summary = json.loads((pkg / "splits" / "app_split_summary.json").read_text())
    fp = json.loads((pkg / "splits" / "split_fingerprints.json").read_text())
    checks = {}

    counts = Counter(r["split"] for r in man)
    checks["counts"] = {
        "manifest": dict(counts),
        "summary": {s: summary["splits"][s]["screens"] for s in SPLITS},
        "fingerprints": {s: int(fp["sizes"][s]) for s in SPLITS},
        "stem_lists": {s: len((pkg / "splits" / f"split_{s}_stems.txt").read_text().split())
                       for s in SPLITS},
    }
    c = checks["counts"]
    c["ok"] = all(c["manifest"][s] == c["summary"][s] == c["fingerprints"][s] == c["stem_lists"][s]
                  for s in SPLITS) and len({r["stem"] for r in man}) == len(man)
    # Fingerprints: SHA-256 of the sorted "stem,split" lines of the whole split, and of the
    # sorted stems of each split (one per line, final newline).
    def lines_sha(lines) -> str:
        return hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode("utf-8")).hexdigest()
    fps = {"assignment": (lines_sha(f"{r['stem']},{r['split']}" for r in man), fp["manifest_sha256"])}
    for s in SPLITS:
        fps[s] = (lines_sha(r["stem"] for r in man if r["split"] == s), fp["splits"][s])
    checks["fingerprints"] = {k: {"computed": a, "recorded": b} for k, (a, b) in fps.items()}
    checks["fingerprints"]["ok"] = all(a == b for a, b in fps.values())
    checks["manifest_file_sha256"] = _sha256(man_path)
    stems_ok = all(set((pkg / "splits" / f"split_{s}_stems.txt").read_text().split())
                   == {r["stem"] for r in man if r["split"] == s} for s in SPLITS)
    checks["stem_lists_equal_manifest"] = {"ok": stems_ok}

    leak = {}
    for key, norm in (("package", str.lower), ("group_id", str), ("md5", str), ("dhash", str)):
        where = defaultdict(set)
        for r in man:
            where[norm(r[key])].add(r["split"])
        crossing = sorted(k for k, v in where.items() if len(v) > 1)
        leak[key] = {"values": len(where), "in_two_splits": len(crossing), "examples": crossing[:5]}
    leak["package"]["case_insensitive"] = True
    hashes = np.array([int(r["dhash"], 16) for r in man], dtype=np.uint64)
    split_of = np.array([SPLITS.index(r["split"]) for r in man])
    # Brute force over all pairs: near-duplicates across splits, and the edges of the grouping.
    parent = list(range(len(man)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    pairs = within = 0
    for i in range(len(man)):
        d = _popcount_u64(hashes[i + 1:] ^ hashes[i])
        close = np.nonzero(d <= NEAR_DUP)[0] + i + 1
        for j in close:
            union(i, int(j))
        cross = int(np.sum(split_of[close] != split_of[i]))
        pairs += cross
        within += len(close) - cross
    first = {}
    for key, norm in (("package", str.lower), ("md5", str)):
        for i, r in enumerate(man):
            union(i, first.setdefault((key, norm(r[key])), i))
    leak["near_dup"] = {"max_hamming": NEAR_DUP, "pairs_across_splits": pairs,
                        "pairs_within_splits": within}
    leak["ok"] = pairs == 0 and all(leak[k]["in_two_splits"] == 0
                                    for k in ("package", "group_id", "md5", "dhash"))
    checks["leakage"] = leak
    comp = defaultdict(set)
    gid = defaultdict(set)
    for i, r in enumerate(man):
        comp[find(i)].add(i)
        gid[r["group_id"]].add(i)
    same = {frozenset(v) for v in comp.values()} == {frozenset(v) for v in gid.values()}
    checks["grouping_recomputed"] = {
        "rule": f"connected components of same package (case-insensitive) OR same MD5 OR dHash <= {NEAR_DUP}",
        "groups_recomputed": len(comp), "groups_in_manifest": len(gid), "ok": same}

    by_stem = {r["stem"]: r for r in man}
    mapping = {r["stem"]: r for r in _rows(pkg / "raw_xml" / "mapping.csv")}
    files = {"images": 0, "md5_mismatch": [], "dhash_mismatch": [], "missing": [],
             "xml": 0, "xml_sha256_mismatch": [], "image_md5_vs_mapping_mismatch": []}
    for s in ("val", "test"):
        for r in (r for r in man if r["split"] == s):
            img = pkg / "dataset" / "images" / s / f"{r['stem']}.png"
            if not img.is_file():
                files["missing"].append(r["stem"])
                continue
            files["images"] += 1
            md5 = md5_file(img)
            if md5 != r["md5"]:
                files["md5_mismatch"].append(r["stem"])
            if f"{dhash(img):016x}" != r["dhash"]:
                files["dhash_mismatch"].append(r["stem"])
            m = mapping.get(r["stem"])
            if m is None or m["split"] != s:
                files["missing"].append(f"{r['stem']} (mapping)")
                continue
            if m["image_md5"] != md5:
                files["image_md5_vs_mapping_mismatch"].append(r["stem"])
            xml = pkg / m["xml_in_zip"]
            files["xml"] += 1
            if not xml.is_file() or _sha256(xml) != m["xml_sha256"]:
                files["xml_sha256_mismatch"].append(r["stem"])
    files["ok"] = (files["images"] == counts["val"] + counts["test"] == files["xml"]
                   and not any(files[k] for k in ("md5_mismatch", "dhash_mismatch", "missing",
                                                  "xml_sha256_mismatch",
                                                  "image_md5_vs_mapping_mismatch")))
    checks["val_test_files"] = files
    want = {r["stem"] for r in man if r["split"] in ("val", "test")}
    checks["xml_mapping"] = {
        "rows": len(mapping), "covers_exactly_val_test": set(mapping) == want,
        "split_and_group_equal_manifest": all(
            m["split"] == by_stem[s]["split"] and m["group_id"] == by_stem[s]["group_id"]
            for s, m in mapping.items() if s in by_stem)}
    checks["xml_mapping"]["ok"] = (checks["xml_mapping"]["covers_exactly_val_test"]
                                   and checks["xml_mapping"]["split_and_group_equal_manifest"])

    labels = {}
    for s in SPLITS:
        have = {p.stem for p in (pkg / "dataset" / "labels" / s).glob("*.txt")}
        want = {r["stem"] for r in man if r["split"] == s}
        labels[s] = {"files": len(have), "missing": len(want - have), "extra": len(have - want)}
    labels["ok"] = all(labels[s]["missing"] == 0 == labels[s]["extra"] for s in SPLITS)
    checks["labels"] = labels
    checks["package_audit"] = summary.get("audit")

    ok = all(v.get("ok") for k, v in checks.items()
             if isinstance(v, dict) and k != "package_audit")
    return {"ok": ok, "package": pkg.name, "manifest_rows": len(man),
            "by_stem_unique": len(by_stem) == len(man), "checks": checks}


def check_colab(pkg: Path, local_dir: Path) -> dict:
    """Local detector evaluation against the Colab evaluation of the same checkpoints.

    Input:  pkg — package root; local_dir — results/eval_appsplit/detector
    Output: report dict with 'ok', the largest differences and one entry per model and split
    """
    names = {"baseline": "baseline_app", "oversample_5k": "oversample_5k_appsplit"}
    colab_names = {"baseline": "baseline_app", "oversample_5k": "oversample_5k_app"}
    out = {"tolerances": TOL, "entries": [], "max_abs_diff": {"mAP50": 0.0, "mAP50_95": 0.0,
                                                              "class_ap50": 0.0, "ci_bound": 0.0}}
    ok = True
    for split in ("val", "test"):
        colab = json.loads((pkg / "verification" / "metrics" / f"evaluate_detector_{split}.json").read_text())
        cmods = {m["name"]: m for m in colab["models"]}
        for key in ("baseline", "oversample_5k"):
            loc = json.loads((local_dir / key / f"detector_{split}_full.json").read_text())["models"][0]
            col = cmods[colab_names[key]]
            e = {"model": key, "split": split,
                 "weights_md5_equal": loc["meta"]["weights_md5"] == col["meta"]["weights_md5"]}
            lm, cm = loc["metrics"], col["metrics"]
            e["mAP50"] = abs(lm["mAP50"] - cm["mAP50"])
            e["mAP50_95"] = abs(lm["mAP50_95"] - cm["mAP50_95"])
            e["class_ap50"] = max(abs(lm["per_class"][c]["ap50"] - cm["per_class"][c]["ap50"])
                                  for c in lm["per_class"])
            bounds = []
            for b in ("bootstrap", "bootstrap_by_group"):
                for m in ("mAP50", "mAP50_95"):
                    bounds += [abs(loc[b][m][k] - col[b][m][k]) for k in ("lo", "hi")]
            e["ci_bound"] = max(bounds)
            e["ok"] = (e["weights_md5_equal"] and e["mAP50"] <= TOL["map"]
                       and e["mAP50_95"] <= TOL["map"] and e["class_ap50"] <= TOL["class_ap50"]
                       and e["ci_bound"] <= TOL["ci"])
            ok &= e["ok"]
            for k in out["max_abs_diff"]:
                out["max_abs_diff"][k] = max(out["max_abs_diff"][k], e[k])
            out["entries"].append(e)
    out["ok"] = ok
    return out


def check_previous(view: Path, sweep: Path) -> dict:
    """Selected configuration minus the previous default on the val screens, paired.

    Input:  view — package view of the sweep (groupings); sweep — sweep directory or sweep.jsonl
    Output: report dict: per scheme the paired intervals, and screens better / worse in TEDS-o
    """
    from ui2compose.evaluation import layout_eval as LE
    from ui2compose.evaluation.layout_sweep import read_sweep

    refs, cfgs = read_sweep(sweep)
    stems = sorted(refs)
    groups = LE.load_groupings(LE.default_grouping_files(view), stems)
    a, b = cfgs[SELECTED], cfgs[PREVIOUS]
    out = {"selected": SELECTED, "previous_default": PREVIOUS, "n_screens": len(stems),
           "difference": "selected minus previous default", "schemes": {}}
    for scheme, g in groups.items():
        out["schemes"][scheme] = LE.paired_difference(a, b, LE.Bootstrap(stems, g, 1000, 0))
    both = [s for s in stems if a[s].get("status") == "ok" and b[s].get("status") == "ok"]
    d = [a[s]["teds_o"] - b[s]["teds_o"] for s in both]
    out["teds_o_screens"] = {"better": sum(x > 0 for x in d), "worse": sum(x < 0 for x in d),
                             "equal": sum(x == 0 for x in d)}
    return out


def main(argv=None) -> int:
    """Runs one check and writes its JSON report.

    Input:  argv — see the module docstring
    Output: exit status (0 ok, 3 a check failed)
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("package")
    p.add_argument("pkg")
    p.add_argument("--out", default="results/eval_appsplit/package_verification.json")
    c = sub.add_parser("colab")
    c.add_argument("pkg")
    c.add_argument("--local", default="results/eval_appsplit/detector")
    c.add_argument("--out", default="results/eval_appsplit/detector_colab_check.json")
    v = sub.add_parser("previous")
    v.add_argument("view")
    v.add_argument("sweep")
    v.add_argument("--out", default="results/eval_appsplit/val_selected_vs_previous_default.json")
    a = ap.parse_args(argv)
    if a.cmd == "package":
        rep = check_package(Path(a.pkg))
    elif a.cmd == "colab":
        rep = check_colab(Path(a.pkg), Path(a.local))
    else:
        rep = check_previous(Path(a.view), Path(a.sweep))
    rep = {"tool": f"scripts/appsplit_checks.py {a.cmd}", **rep}
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"{'[OK]' if rep.get('ok', True) else '[FAIL]'} {a.out}")
    return 0 if rep.get("ok", True) else 3


if __name__ == "__main__":
    sys.exit(main())
