# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Tests for the split / training tooling.

Sections:
  1. app_split: file-name parsing, hashing, multi-index near-duplicate search,
     grouping, unseen subset, group-level split, leakage audit, CLI on the
     synthetic fixture tests/fixtures/tooling/yolo_tiny (see make_yolo_tiny.py).
  2. build_dataset: --split-by screenshot reproduces the v1.1.0 algorithm,
     --split-by app keeps groups together and writes the manifest.
  3. oversample_dataset: the documented procedure (counts estimated once,
     sorted round-robin copies, before/after statistics).
  4. train.py and configs/train_*.yaml: the 5k config reproduces the
     checkpoint's train_args; ablation configs differ only in data/name;
     --dry-run does not import ultralytics.

Only standard library, NumPy, PIL and PyYAML are imported.
"""

import csv
import hashlib
import itertools
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import yaml
from PIL import Image

from _reference_data import require_reference_data
from ui2compose.preprocess import app_split as A
from ui2compose.preprocess import build_dataset as BD
from ui2compose.preprocess import oversample_dataset as OS
from ui2compose import train as T

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "tooling" / "yolo_tiny"


def grid_image(seed: int, noise: int = 0, size=(90, 160)) -> Image.Image:
    """Seeded random block image (same construction as make_yolo_tiny.py).

    Input:  seed — RNG seed; noise — added pixel noise amplitude; size — (w, h)
    Output: PIL grayscale image
    """
    rng = np.random.default_rng(seed)
    grid = rng.integers(0, 256, size=(16, 9)).astype(np.int16)
    img = np.kron(grid, np.ones((size[1] // 16, size[0] // 9), dtype=np.int16))
    if noise:
        img = img + np.random.default_rng(seed + 1000).integers(-noise, noise + 1, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="L")


# ---------------------------------------------------------------------------
# 1. app_split
# ---------------------------------------------------------------------------

EXPECTED_EXAMPLE_PACKAGES = {
    "com.andromo.dev231870.app482676-screens_screenshot_5": "com.andromo.dev231870.app482676",
    "com.foxfi-screens_screenshot_1": "com.foxfi",
    "com.luigiedeveloper.battlesimulator-screens_screenshot_3": "com.luigiedeveloper.battlesimulator",
    "com.mcdeliveryonline.app-screens_screenshot_1": "com.mcdeliveryonline.app",
    "com.mineworld.swordsaddon-screens_screenshot_2": "com.mineworld.swordsaddon",
    "com.mynamecubeapps.ball-screens_screenshot_3": "com.mynamecubeapps.ball",
    "com.netbeast.yeti-screens_screenshot_1": "com.netbeast.yeti",
    "com.shazam.android-screens_screenshot_3": "com.shazam.android",
    "com.sign.language.learn1234-screens_screenshot_2": "com.sign.language.learn1234",
    "westport.andrewirwin.com.drivesafe-screens_screenshot_3": "westport.andrewirwin.com.drivesafe",
}


def test_parse_package_on_all_examples():
    stems = sorted(p.stem for p in (REPO / "examples").glob("*.png"))
    assert stems == sorted(EXPECTED_EXAMPLE_PACKAGES)
    for stem in stems:
        assert A.parse_package(stem) == EXPECTED_EXAMPLE_PACKAGES[stem]
        parsed = A.parse_stem(stem)
        assert parsed.copy_indices == () and parsed.base_stem == stem
        # the oversampled copy names written by oversample_dataset.py
        for suffix, copies in (("_os0", (0,)), ("_os17", (17,)), ("_os3_os0", (3, 0))):
            p = A.parse_stem(stem + suffix)
            assert p.package == EXPECTED_EXAMPLE_PACKAGES[stem]
            assert p.copy_indices == copies and p.base_stem == stem
            assert p.screen_index == parsed.screen_index


def test_parse_stem_edge_cases():
    p = A.parse_stem("com.foo_bar.os1-screens_screenshot_12_os2")
    assert (p.package, p.screen_index, p.copy_indices) == ("com.foo_bar.os1", 12, (2,))
    for bad in ("screenshot_1", "com.foo-screens_screenshot_", "com.foo-screens_screenshot_1_osx",
                "com.foo-screens_screenshot_1.png", "com.foo_screenshot_1"):
        with pytest.raises(ValueError):
            A.parse_stem(bad)


def test_builder_family():
    assert A.builder_family("com.andromo.dev231870.app482676") == "com.andromo.dev"
    assert A.builder_family("com.andromo.dev999999.app1") == "com.andromo.dev"
    assert A.builder_family("com.shazam.android") is None
    assert A.builder_family("app123.com") is None  # first segment is not a family


def test_md5_and_dhash(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    grid_image(1).save(a)
    shutil.copyfile(a, b)
    assert A.md5_file(a) == A.md5_file(b)
    grid_image(1).save(tmp_path / "a.jpg", quality=60)
    assert A.md5_file(a) != A.md5_file(tmp_path / "a.jpg")
    h = A.dhash(a)
    assert 0 <= h < 2 ** 64 and A.dhash(grid_image(1)) == h
    # re-encoding, noise and rescaling keep the hash (near-)identical
    assert A.hamming(h, A.dhash(tmp_path / "a.jpg")) <= 2
    assert A.hamming(h, A.dhash(grid_image(1, noise=6))) <= 2
    assert A.hamming(h, A.dhash(grid_image(1).resize((45, 80)))) <= 2
    # different screens are far apart
    far = [A.hamming(A.dhash(grid_image(i)), A.dhash(grid_image(j)))
           for i, j in itertools.combinations(range(2, 12), 2)]
    assert min(far) > A.DEFAULT_NEAR_DUP_MAX_DIST
    assert A.dhash(grid_image(1), size=4) < 2 ** 16


def test_dhash_bit_definition():
    # a horizontal ramp: every right neighbour is brighter -> all bits set
    ramp = Image.fromarray(np.tile(np.arange(0, 252, 28, dtype=np.uint8), (8, 1)), mode="L")
    assert A.dhash(ramp) == 2 ** 64 - 1
    assert A.dhash(ramp.transpose(Image.Transpose.FLIP_LEFT_RIGHT)) == 0
    assert A.hamming(0b1011, 0b0001) == 2


@pytest.mark.parametrize("k", [0, 1, 3, 4, 6])
def test_hamming_index_equals_brute_force(k):
    rng = random.Random(k)
    base = [rng.getrandbits(64) for _ in range(150)]
    hashes = []
    for b in base:
        hashes.append(b)
        for _ in range(2):
            x = b
            for bit in rng.sample(range(64), rng.randint(0, 8)):
                x ^= 1 << bit
            hashes.append(x)
    hashes = sorted(set(hashes))
    brute = [(i, j, A.hamming(hashes[i], hashes[j]))
             for i, j in itertools.combinations(range(len(hashes)), 2)
             if A.hamming(hashes[i], hashes[j]) <= k]
    assert A.near_duplicate_pairs(hashes, k) == brute
    idx = A.HammingIndex(hashes, k)
    q = hashes[5] ^ 0b101
    assert idx.query(q) == sorted((i, A.hamming(q, h)) for i, h in enumerate(hashes)
                                  if A.hamming(q, h) <= k)

    def brute_query(q, radius):
        return sorted((i, A.hamming(q, h)) for i, h in enumerate(hashes) if A.hamming(q, h) <= radius)

    # Every indexed hash as a query, and random queries with flips anywhere in the 64 bits
    # (a neighbour can agree with the query on a later band only).
    queries = list(hashes)
    for _ in range(200):
        x = rng.choice(hashes)
        for bit in rng.sample(range(64), rng.randint(0, k + 2)):
            x ^= 1 << bit
        queries.append(x)
    for q in queries:
        assert idx.query(q) == brute_query(q, k)
        if k:
            assert idx.query(q, max_dist=k - 1) == brute_query(q, k - 1)


def test_hamming_index_rejects_bad_radius():
    with pytest.raises(ValueError):
        A.HammingIndex([1, 2], 64)
    with pytest.raises(ValueError):
        A.HammingIndex([1, 2], -1)


def _rec(stem: str, md5=None, dh=None, split=None, counts=None) -> A.ScreenRecord:
    """Builds an in-memory record with the package parsed from the stem.

    Input:  stem — dataset stem; md5, dh — hashes; split — subset; counts — class counts
    Output: ScreenRecord
    """
    return A.ScreenRecord(stem=stem, package=A.parse_package(stem), md5=md5, dhash=dh,
                          split=split, class_counts=counts or {})


def test_group_screens_links_and_transitivity():
    recs = [
        _rec("com.a-screens_screenshot_1", "m1", 0x0F0F),
        _rec("com.a-screens_screenshot_2", "m2", 0xFFFF0000),
        _rec("com.b-screens_screenshot_1", "m1", 0x123456789),       # md5 = a1
        _rec("com.c-screens_screenshot_1", "m3", 0xFFFF0000 ^ 0b111),  # near-dup of a2 (d=3)
        _rec("com.d-screens_screenshot_1", "m4", 0xAAAAAAAAAAAAAAAA),
        _rec("com.D-screens_screenshot_2", "m5", 0x5555555555555555),  # same package, other case
        _rec("com.e-screens_screenshot_1", "m6", 0x00FF00FF00FF00FF),
    ]
    g = A.group_screens(recs)
    assert g[0] == g[1] == g[2] == g[3]
    assert g[4] == g[5] and g[4] != g[0]
    assert len(set(g)) == 3
    # canonical ids: independent of the record order
    perm = [6, 3, 0, 5, 1, 4, 2]
    g2 = A.group_screens([recs[i] for i in perm])
    assert [g2[perm.index(i)] for i in range(len(recs))] == g
    # near-duplicate radius below the distance -> c is separate
    g3 = A.group_screens(recs, near_dup_max_dist=2)
    assert g3[3] != g3[0]
    assert len(set(A.group_screens(recs, near_dup_max_dist=None))) == 4


def test_group_screens_builder_prefix_flag():
    recs = [_rec("com.andromo.dev1111.app1-screens_screenshot_1", "x", 1),
            _rec("com.andromo.dev2222.app9-screens_screenshot_1", "y", 2 ** 63 - 1)]
    assert len(set(A.group_screens(recs))) == 2
    assert len(set(A.group_screens(recs, builder_prefix=True))) == 1


def test_unseen_subset_reasons():
    train = [_rec("com.a-screens_screenshot_1", "m1", 0xF0F0F0F0),
             _rec("com.a-screens_screenshot_1_os0", "m1", 0xF0F0F0F0),
             _rec("com.b-screens_screenshot_1", "m2", 0x0F0F0F0F0F0F)]
    evals = [_rec("com.a-screens_screenshot_9", "m9", 0x1234),            # package
             _rec("com.x-screens_screenshot_1", "m2", 0x0F0F0F0F0F0F),    # md5
             _rec("com.y-screens_screenshot_1", "m7", 0xF0F0F0F0 ^ 0b11),  # near_dup d=2
             _rec("com.y-screens_screenshot_2", "m8", 0xFFFF000000000000),  # transitive
             _rec("com.z-screens_screenshot_1", "m10", 0xAAAA5555AAAA5555)]  # unseen
    sub, stats = A.unseen_subset(train, evals)
    assert [r.stem for r in sub] == ["com.z-screens_screenshot_1"]
    assert stats["n_eval"] == 5 and stats["n_unseen"] == 1 and stats["n_excluded"] == 4
    assert stats["excluded_by_reason"] == {"package": 1, "builder_prefix": 0, "md5": 1,
                                           "near_dup": 1, "transitive": 1}
    ex, _ = A.classify_exclusions(train, evals)
    assert ex[1].matched_stem == "com.b-screens_screenshot_1" and ex[1].distance == 0
    assert ex[2].reason == "near_dup" and ex[2].distance == 2
    assert ex[2].matched_stem == "com.a-screens_screenshot_1"  # original sorts before its copy
    assert ex[4] is None


def _synthetic_records(n_apps: int = 400, seed: int = 0) -> List[A.ScreenRecord]:
    """Many apps with 1..n screens and imbalanced class counts (no images).

    Input:  n_apps — number of packages; seed — RNG seed
    Output: list of ScreenRecord with md5/dhash set to unique values
    """
    rng = np.random.default_rng(seed)
    freq = [0.5, 0.03, 0.1, 0.8, 0.04, 0.03, 0.02, 0.06, 0.02, 0.03, 0.03, 0.95, 0.02]
    recs = []
    for a in range(n_apps):
        for s in range(int(rng.geometric(0.3))):
            cc = {c: int(rng.integers(1, 5)) for c, f in enumerate(freq) if rng.random() < f}
            recs.append(A.ScreenRecord(stem=f"com.app{a}-screens_screenshot_{s}",
                                       package=f"com.app{a}", class_counts=cc,
                                       md5=f"{a}-{s}", dhash=int(rng.integers(0, 2 ** 63))))
    return recs


def _class_fraction_deviation(recs: List[A.ScreenRecord], splits: List[str],
                              min_instances: int = 0) -> float:
    """Largest |instance fraction of a class in a subset - subset ratio|.

    Input:  recs — records; splits — subset per record; min_instances —
            only classes with at least this many instances
    Output: maximum absolute deviation
    """
    tot, per = {}, {s: {} for s in A.SPLITS}
    for rec, s in zip(recs, splits):
        for c, k in rec.class_counts.items():
            tot[c] = tot.get(c, 0) + k
            per[s][c] = per[s].get(c, 0) + k
    return max(abs(per[s].get(c, 0) / tot[c] - r)
               for s, r in zip(A.SPLITS, A.DEFAULT_RATIOS) for c in tot if tot[c] >= min_instances)


def test_split_by_group_ratios_stratification_and_no_leak():
    recs = _synthetic_records(1500)
    splits = A.split_by_group(recs)
    n = len(recs)
    frac = {s: splits.count(s) / n for s in A.SPLITS}
    for s, r in zip(A.SPLITS, A.DEFAULT_RATIOS):
        assert abs(frac[s] - r) < 0.03, frac
    # per-class instance fractions follow the ratios (stratification), and much
    # more closely than a group split that ignores the classes
    strat = _class_fraction_deviation(recs, splits)
    plain = _class_fraction_deviation(recs, A.split_by_group(recs, stratify=False))
    assert strat < 0.03 and strat < plain / 2, (strat, plain)
    # every package in exactly one subset
    where = {}
    for rec, s in zip(recs, splits):
        where.setdefault(rec.package, set()).add(s)
    assert all(len(v) == 1 for v in where.values())
    report = A.audit_split(recs, splits, group_ids=A.group_screens(recs))
    assert report["ok"] is True


def test_split_by_group_is_deterministic_and_order_independent():
    recs = _synthetic_records(200, seed=3)
    s1 = A.split_by_group(recs, seed=42)
    assert A.split_by_group(recs, seed=42) == s1
    perm = list(range(len(recs)))
    random.Random(0).shuffle(perm)
    s2 = A.split_by_group([recs[i] for i in perm], seed=42)
    assert [s2[perm.index(i)] for i in range(len(recs))] == s1
    assert A.split_by_group(recs, seed=7) != s1
    s3 = A.split_by_group(recs, stratify=False)
    assert A.audit_split(recs, s3)["ok"]


def test_split_by_group_validates_input():
    recs = _synthetic_records(10)
    with pytest.raises(ValueError):
        A.split_by_group(recs, ratios=(0.5, 0.2, 0.2))
    with pytest.raises(ValueError):
        A.split_by_group(recs, group_ids=[0])


def test_audit_split_catches_leakage():
    base = [_rec("com.a-screens_screenshot_1", "m1", 0xFF00),
            _rec("com.b-screens_screenshot_1", "m2", 0xFF00FF00FF00),
            _rec("com.c-screens_screenshot_1", "m3", 0x0F0F0F0F0F0F0F0F)]
    assert A.audit_split(base, ["train", "val", "test"])["ok"]
    cases = {
        "package": base + [_rec("com.a-screens_screenshot_2", "m4", 0xAAAA0000AAAA)],
        "md5": base + [_rec("com.z-screens_screenshot_1", "m1", 0x5555)],
        "near_dup": base + [_rec("com.y-screens_screenshot_1", "m5", 0xFF00FF00FF00 ^ 0b1)],
    }
    for name, recs in cases.items():
        with pytest.raises(A.SplitLeakageError, match=name):
            A.audit_split(recs, ["train", "val", "test", "test"])
    with pytest.raises(A.SplitLeakageError, match="group"):
        A.audit_split(base, ["train", "val", "test"], near_dup_max_dist=None, group_ids=[0, 0, 1])
    # identical dHash in two subsets is a near-duplicate leak too
    recs = base + [_rec("com.w-screens_screenshot_1", "m6", 0xFF00)]
    with pytest.raises(A.SplitLeakageError, match="near_dup"):
        A.audit_split(recs, ["train", "val", "test", "test"])
    # builder families only when enabled
    fam = [_rec("com.andromo.dev1111.app1-screens_screenshot_1", "x", 0x0F0F0F0F0F0F0F0F),
           _rec("com.andromo.dev2222.app2-screens_screenshot_1", "y", 0xF0F0F0F0F0F0F0F0)]
    assert A.audit_split(fam, ["train", "test"])["ok"]
    with pytest.raises(A.SplitLeakageError):
        A.audit_split(fam, ["train", "test"], builder_prefix=True)


def test_scan_fixture_dataset():
    recs = A.scan_yolo_dataset(FIXTURE)
    assert len(recs) == 13
    by = {r.stem: r for r in recs}
    copy = by["com.alpha.app-screens_screenshot_2_os0"]
    orig = by["com.alpha.app-screens_screenshot_2"]
    assert copy.is_copy and not orig.is_copy and copy.md5 == orig.md5 and copy.dhash == orig.dhash
    assert copy.class_counts == orig.class_counts
    assert by["com.clone.of.beta-screens_screenshot_1"].md5 == by["com.beta.app-screens_screenshot_1"].md5
    d = A.hamming(by["com.delta.app-screens_screenshot_1"].dhash, by["com.beta.app-screens_screenshot_1"].dhash)
    assert 0 < d <= A.DEFAULT_NEAR_DUP_MAX_DIST
    assert A.scan_yolo_dataset(FIXTURE, workers=2)[3].dhash == recs[3].dhash


def _read_csv(path: Path) -> List[Dict[str, str]]:
    """Reads a CSV file into dicts.

    Input:  path — CSV path
    Output: list of row dicts
    """
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_cli_on_fixture(tmp_path):
    out = tmp_path / "splits"
    assert A.main(["--dataset-dir", str(FIXTURE), "--out", str(out), "--workers", "1",
                   "--resplit"]) == 0
    rows = {r["stem"]: r for r in _read_csv(out / "manifest.csv")}
    assert list(_read_csv(out / "manifest.csv")[0])[:7] == [
        "stem", "package", "group_id", "split", "md5", "dhash", "seen_in_train"]
    expected = {
        "com.alpha.app-screens_screenshot_3": "package",
        "com.gamma.app-screens_screenshot_1": "",
        "com.clone.of.beta-screens_screenshot_1": "md5",
        "com.delta.app-screens_screenshot_1": "near_dup",
        "com.epsilon.app-screens_screenshot_1": "",
        "com.eta.app-screens_screenshot_1": "transitive",
        "com.eta.app-screens_screenshot_2": "md5",
        "com.andromo.dev111111.app333333-screens_screenshot_1": "",
    }
    for stem, reason in expected.items():
        assert rows[stem]["exclusion_reason"] == reason, stem
        assert rows[stem]["seen_in_train"] == ("1" if reason else "0")
    assert rows["com.alpha.app-screens_screenshot_2_os0"]["is_oversampled_copy"] == "1"
    test_list = [Path(p).name for p in (out / "unseen_test.txt").read_text().splitlines()]
    assert test_list == ["com.andromo.dev111111.app333333-screens_screenshot_1.png",
                         "com.epsilon.app-screens_screenshot_1.png"]
    assert all(Path(p).is_absolute() and Path(p).exists()
               for p in (out / "unseen_val.txt").read_text().splitlines())
    summary = json.loads((out / "summary.json").read_text())
    assert summary["unseen"]["test"]["excluded_by_reason"]["near_dup"] == 1
    assert summary["byte_identical_to_train"] == {"val": 1, "test": 1}
    # clone audit independent of packages: val clone.of.beta (md5), test eta_2 (md5) and
    # delta (near-dup of beta) are matches to a training image of another package
    assert summary["direct_matches_to_train"]["val"] == {
        "md5": 1, "dhash_0": 1, "dhash_le_k": 1, "md5_other_package": 1,
        "dhash_0_other_package": 1, "dhash_le_k_other_package": 1}
    assert summary["direct_matches_to_train"]["test"] == {
        "md5": 1, "dhash_0": 1, "dhash_le_k": 2, "md5_other_package": 1,
        "dhash_0_other_package": 1, "dhash_le_k_other_package": 2}
    full = summary["class_support"]["test_full"]
    unseen = summary["class_support"]["test_unseen"]
    assert set(full) == set(yaml.safe_load((FIXTURE / "dataset.yaml").read_text())["names"])
    assert all(unseen[c]["instances"] <= full[c]["instances"] for c in full)
    assert summary["splits"]["train"]["oversampled_copies"] == 1
    # --resplit: originals only, audited, groups intact
    rs = _read_csv(out / "app_split_manifest.csv")
    assert len(rs) == 12 and json.loads((out / "app_split_summary.json").read_text())["audit"]["ok"]
    split_of_group = {}
    for r in rs:
        split_of_group.setdefault(r["group_id"], set()).add(r["split"])
    assert all(len(v) == 1 for v in split_of_group.values())
    # builder-prefix grouping excludes the second andromo app as well
    out2 = tmp_path / "splits2"
    A.main(["--dataset-dir", str(FIXTURE), "--out", str(out2), "--workers", "1",
            "--builder-prefix"])
    rows2 = {r["stem"]: r for r in _read_csv(out2 / "manifest.csv")}
    assert rows2["com.andromo.dev111111.app333333-screens_screenshot_1"]["exclusion_reason"] == "builder_prefix"


def test_split_fingerprints_and_expect_check(tmp_path):
    """summary.json records per-split SHA-256 of the sorted stems; --expect-fingerprints compares them."""
    out = tmp_path / "splits"
    assert A.main(["--dataset-dir", str(FIXTURE), "--out", str(out), "--workers", "1"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    fp = summary["split_fingerprints"]
    for split in A.SPLITS:
        stems = sorted(p.stem for p in (FIXTURE / "images" / split).iterdir()
                       if p.suffix.lower() in A.IMG_EXTS)
        originals = [x for x in stems if not A.parse_stem(x).copy_indices]
        assert fp[split]["screens"] == len(stems)
        assert fp[split]["stems_sha256"] == hashlib.sha256("".join(f"{x}\n" for x in stems).encode()).hexdigest()
        assert fp[split]["originals"] == len(originals)
        assert fp[split]["originals_sha256"] == hashlib.sha256(
            "".join(f"{x}\n" for x in originals).encode()).hexdigest()
    assert fp["train"]["screens"] == fp["train"]["originals"] + 1          # one oversampled copy
    # The same dataset passes the check; a reference with another test split fails it (status 3).
    assert A.main(["--dataset-dir", str(FIXTURE), "--out", str(tmp_path / "again"), "--workers", "1",
                   "--expect-fingerprints", str(out / "summary.json")]) == 0
    other = dict(summary, split_fingerprints={**fp, "test": {**fp["test"], "originals_sha256": "0" * 64}})
    (tmp_path / "other.json").write_text(json.dumps(other))
    assert A.main(["--dataset-dir", str(FIXTURE), "--out", str(tmp_path / "x"), "--workers", "1",
                   "--expect-fingerprints", str(tmp_path / "other.json")]) == 3
    assert A.compare_fingerprints(fp, other["split_fingerprints"]) == [
        f"test.originals_sha256: got {fp['test']['originals_sha256']!r}, expected {'0' * 64!r}"]


def test_fixture_is_reproducible(tmp_path):
    """The committed fixture equals what make_yolo_tiny.py generates (pixels and labels)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_yolo_tiny", FIXTURE.parent / "make_yolo_tiny.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.OUT = tmp_path / "yolo_tiny"
    mod.main()
    for f in sorted(FIXTURE.rglob("*")):
        if f.is_file() and f.suffix in (".txt", ".yaml"):
            assert (mod.OUT / f.relative_to(FIXTURE)).read_text() == f.read_text(), f
        elif f.suffix == ".png":
            a = np.asarray(Image.open(f))
            b = np.asarray(Image.open(mod.OUT / f.relative_to(FIXTURE)))
            assert np.array_equal(a, b), f


# ---------------------------------------------------------------------------
# 2. build_dataset
# ---------------------------------------------------------------------------

def _make_raw(tmp_path: Path, n_apps: int = 24) -> Dict[str, Path]:
    """Creates a parse_redraw-style tree: labels/<app>/screenshot_N.txt + raw/<app>/screenshot_N.png.

    App 'com.clone0' contains byte copies of 'com.app0' screens.

    Input:  tmp_path — base directory; n_apps — number of apps
    Output: {'labels': ..., 'raw': ...} directories
    """
    labels, raw = tmp_path / "labels", tmp_path / "raw"
    rng = random.Random(1)
    seed = 100
    for a in range(n_apps):
        app = f"com.app{a}-screens"
        for s in range(1, rng.randint(1, 4) + 1):
            (labels / app).mkdir(parents=True, exist_ok=True)
            (raw / app).mkdir(parents=True, exist_ok=True)
            grid_image(seed).save(raw / app / f"screenshot_{s}.png")
            seed += 1
            cls = [rng.randrange(13) for _ in range(rng.randint(1, 3))]
            (labels / app / f"screenshot_{s}.txt").write_text(
                "".join(f"{c} 0.5 0.5 0.2 0.1\n" for c in cls))
    clone = "com.clone0-screens"
    shutil.copytree(raw / "com.app0-screens", raw / clone)
    shutil.copytree(labels / "com.app0-screens", labels / clone)
    (labels / "com.app1-screens" / "screenshot_9.txt").write_text("")  # empty label: skipped
    return {"labels": labels, "raw": raw}


def _built(yolo: Path) -> Dict[str, List[str]]:
    """Lists the image stems per subset of a built dataset.

    Input:  yolo — dataset root
    Output: {split: sorted stems}
    """
    return {s: sorted(p.stem for p in (yolo / "images" / s).iterdir()) for s in A.SPLITS}


def test_build_dataset_screenshot_split_reproduces_v110(tmp_path):
    d = _make_raw(tmp_path)
    yolo = tmp_path / "yolo"
    BD.main(["--labels-root", str(d["labels"]), "--images-root", str(d["raw"]),
             "--yolo-root", str(yolo), "--dataset-yaml", str(tmp_path / "none.yaml")])
    # the v1.1.0 algorithm, written out independently
    pairs = []
    for app_dir in sorted(d["labels"].iterdir()):
        for lf in sorted(app_dir.glob("*.txt")):
            if lf.read_text().strip():
                pairs.append(f"{app_dir.name}_{lf.stem}")
    random.seed(42)
    random.shuffle(pairs)
    t, v = int(len(pairs) * 0.8), int(len(pairs) * 0.1)
    expected = {"train": sorted(pairs[:t]), "val": sorted(pairs[t:t + v]), "test": sorted(pairs[t + v:])}
    assert _built(yolo) == expected
    assert not (yolo / "split_manifest.csv").exists()


def test_build_dataset_app_split_keeps_groups_together(tmp_path):
    d = _make_raw(tmp_path)
    yolo = tmp_path / "yolo_app"
    BD.main(["--labels-root", str(d["labels"]), "--images-root", str(d["raw"]),
             "--yolo-root", str(yolo), "--dataset-yaml", str(tmp_path / "none.yaml"),
             "--split-by", "app"])
    built = _built(yolo)
    where = {}
    for s, stems in built.items():
        for stem in stems:
            where.setdefault(A.parse_package(stem), set()).add(s)
            assert (yolo / "labels" / s / f"{stem}.txt").exists()
    assert all(len(v) == 1 for v in where.values())
    assert where["com.clone0"] == where["com.app0"]  # byte-identical clone stays with its source
    rows = _read_csv(yolo / "split_manifest.csv")
    assert sum(len(v) for v in built.values()) == len(rows)
    assert {r["split"] for r in rows} <= set(A.SPLITS)
    # deterministic
    yolo2 = tmp_path / "yolo_app2"
    BD.main(["--labels-root", str(d["labels"]), "--images-root", str(d["raw"]),
             "--yolo-root", str(yolo2), "--dataset-yaml", str(tmp_path / "none.yaml"),
             "--split-by", "app"])
    assert _built(yolo2) == built


# ---------------------------------------------------------------------------
# 3. oversample_dataset
# ---------------------------------------------------------------------------

def _make_train(root: Path, labels: Dict[str, str]) -> None:
    """Writes images/train + labels/train for the given stems.

    Input:  root — dataset root; labels — stem -> label text
    Output: none
    """
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    for i, (stem, text) in enumerate(labels.items()):
        grid_image(i).save(root / "images" / "train" / f"{stem}.png")
        (root / "labels" / "train" / f"{stem}.txt").write_text(text)


def test_oversample_documented_procedure(tmp_path, capsys):
    # CheckBox (1) and Switch (10) are rare; TextView (11) is frequent
    labels = {
        "a-screens_screenshot_1": "1 0.5 0.5 0.1 0.1\n11 0.5 0.2 0.3 0.1\n",
        "b-screens_screenshot_1": "1 0.5 0.5 0.1 0.1\n1 0.5 0.7 0.1 0.1\n10 0.2 0.2 0.1 0.1\n",
        "c-screens_screenshot_1": "11 0.5 0.5 0.3 0.1\n",
    }
    _make_train(tmp_path, labels)
    stats = OS.oversample(tmp_path, target_min=7)
    # CheckBox: n=3 in m=2 images -> needed = int((7-3)/(3/2)) + 1 = 3 copies,
    # round-robin in sorted order: a_os0, b_os0, a_os1
    # Switch: n=1 in 1 image -> needed = int(6/1) + 1 = 7 copies of b; in the first
    # pass b_os0 already exists (made for CheckBox): b is skipped, the suffix index
    # is incremented for the rest of the pass and again after it -> b_os2 .. b_os8
    assert stats["copies_per_class"] == {1: 3, 10: 7}
    names = sorted((p.stem for p in (tmp_path / "labels" / "train").glob("*_os*.txt")),
                   key=lambda n: (n[0], int(n.rsplit("_os", 1)[1])))
    assert names == ["a-screens_screenshot_1_os0", "a-screens_screenshot_1_os1",
                     "b-screens_screenshot_1_os0"] + [f"b-screens_screenshot_1_os{i}" for i in range(2, 9)]
    assert stats["instances_before"] == {1: 3, 11: 2, 10: 1}
    assert stats["images_before"] == {1: 2, 11: 2, 10: 1}
    # co-occurrence: the Switch copies also add CheckBoxes; TextView grows as a side effect
    assert stats["instances_after"][1] == 3 + 2 + 2 + 2 * 7
    assert stats["instances_after"][10] == 1 + 1 + 7
    assert stats["instances_after"][11] == 2 + 2
    assert stats["images_after"][1] == 2 + 10
    out = capsys.readouterr().out
    assert "instances" in out and "images" in out
    # the copies are byte-identical, so app_split groups them with the original
    recs = A.scan_yolo_dataset(tmp_path, splits=("train",))
    assert all(r.is_copy == ("_os" in r.stem) for r in recs)
    assert len(set(A.group_screens(recs))) == 3


def test_oversample_is_order_deterministic(tmp_path):
    labels = {f"x{i}-screens_screenshot_1": "8 0.5 0.5 0.1 0.1\n" for i in (3, 1, 2)}
    _make_train(tmp_path / "a", labels)
    _make_train(tmp_path / "b", dict(reversed(list(labels.items()))))
    OS.oversample(tmp_path / "a", target_min=4)
    OS.oversample(tmp_path / "b", target_min=4)
    la = sorted(p.name for p in (tmp_path / "a" / "labels" / "train").iterdir())
    lb = sorted(p.name for p in (tmp_path / "b" / "labels" / "train").iterdir())
    assert la == lb
    # needed = int((4-3)/1) + 1 = 2 copies: the first two images in sorted order
    assert [n for n in la if "_os" in n] == [f"x{i}-screens_screenshot_1_os0.txt" for i in (1, 2)]


def test_oversample_missing_dir_returns_empty(tmp_path):
    assert OS.oversample(tmp_path, 10) == {}


# ---------------------------------------------------------------------------
# 4. train.py and configs
# ---------------------------------------------------------------------------

CONFIGS = {name: REPO / "configs" / f"train_{name}.yaml"
           for name in ("baseline", "oversample_2k", "oversample_5k")}


def test_5k_config_reproduces_checkpoint_train_args():
    args_file = REPO / "results" / "train_args_oversample_5k.json"
    require_reference_data(args_file)
    ckpt = json.loads(args_file.read_text())
    cfg = T.load_train_config(CONFIGS["oversample_5k"])
    for key, value in cfg["train"].items():
        assert key in ckpt, key
        assert ckpt[key] == value, (key, ckpt[key], value)
    # the hyperparameters named in the manuscript
    expected = dict(optimizer="AdamW", lr0=0.008, lrf=0.01, momentum=0.937, weight_decay=5e-4,
                    warmup_epochs=3.0, epochs=60, patience=20, batch=8, nbs=64, imgsz=640,
                    rect=True, mosaic=0.0, mixup=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                    translate=0.1, scale=0.5, fliplr=0.5, seed=42, deterministic=True,
                    amp=True, cache=True)
    for k, v in expected.items():
        assert cfg["train"][k] == v, k
    assert cfg["model"] == "yolov8s.pt" and cfg["ultralytics_version"] == "8.4.24"


def test_ablation_configs_differ_only_in_data_and_name():
    cfgs = {k: T.load_train_config(p) for k, p in CONFIGS.items()}
    ref = cfgs["oversample_5k"]
    for name, cfg in cfgs.items():
        assert cfg["train"] == ref["train"], name
        assert cfg["model"] == ref["model"] and cfg["ultralytics_version"] == ref["ultralytics_version"]
        assert cfg["name"] == name
    assert len({c["data"] for c in cfgs.values()}) == 3


def test_build_train_kwargs_overrides():
    cfg = T.load_train_config(CONFIGS["baseline"])
    kw = T.build_train_kwargs(cfg, data="d.yaml", project="p", name="n", device="cpu")
    assert (kw["data"], kw["project"], kw["name"], kw["device"]) == ("d.yaml", "p", "n", "cpu")
    assert kw["lr0"] == 0.008
    kw = T.build_train_kwargs(cfg)
    assert kw["data"] == "configs/dataset.yaml" and "device" not in kw


def test_load_train_config_validation(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model: m.pt\ntrain: {epochs: 1}\n")
    with pytest.raises(ValueError, match="data"):
        T.load_train_config(p)
    p.write_text("model: m.pt\ndata: d.yaml\ntrain: {epochs: 1, data: x}\n")
    with pytest.raises(ValueError, match="top level"):
        T.load_train_config(p)


def test_train_dry_run_does_not_import_ultralytics():
    code = ("import sys, json; from ui2compose import train; "
            "rc = train.main(['--config', 'configs/train_oversample_5k.yaml', '--data', 'x.yaml', "
            "'--dry-run']); assert rc == 0; assert 'ultralytics' not in sys.modules; "
            "assert 'torch' not in sys.modules")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["model"] == "yolov8s.pt" and out["train_kwargs"]["data"] == "x.yaml"
    assert out["train_kwargs"]["name"] == "oversample_5k"
    r = subprocess.run([sys.executable, "-m", "ui2compose.train", "--resume", "last.pt",
                        "--dry-run"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0 and json.loads(r.stdout)["train_kwargs"] == {"resume": True}
