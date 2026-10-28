# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""App-level grouping of screens: leak audit, unseen-app subset, app-level split.

The v1.1.0 dataset was split at screenshot level (build_dataset.py shuffles
individual screens), so screens of the same app, and pixel-identical copies of
a screen published under another package name, can occur in train and in
val/test. This module makes that measurable and avoidable without retraining:

  * Every screen gets a *group*: the connected component (union-find) of the
    relation "same package OR identical file content (MD5) OR near-identical
    image (dHash Hamming distance <= k)", optionally also "same app-builder
    family" (see builder_family()). The relation is applied transitively.
  * unseen_subset() keeps the evaluation screens whose group contains no
    training screen (the "unseen-app subset") and reports, for every excluded
    screen, the first reason that applies (package, builder family, MD5,
    near-duplicate, or only transitively through other screens of its group).
  * split_by_group() makes a new group-level split (for future retraining)
    with greedy multi-label stratification over per-group class counts, and
    audit_split() checks a split for leakage independently of the grouping.
  * split_fingerprints() records a SHA-256 of every split's sorted stem list
    (summary.json), so that a regenerated dataset can be compared with the one
    the released weights were evaluated on (--expect-fingerprints).

The unseen-app subset is only meaningful if the dataset given here has
exactly the split the released weights were trained on. This module cannot
check that by itself (the original split lists were not archived with
v1.1.0); evaluate_detector.py --expect-metrics checks the necessary condition
that the released weights reproduce the published val and test metrics on
the regenerated split.

File naming. build_dataset.py names every dataset image
"<package>-screens_screenshot_<n>.<ext>" (ReDraw app folder "<package>-screens"
+ "_" + "screenshot_<n>"); oversample_dataset.py appends "_os<k>" to the stem
of every copy it writes (repeated runs would append it again). parse_stem()
is anchored on exactly this pattern and never splits on '_' or '.' blindly,
because package names contain both.

Near-duplicate search scales to the full ReDraw set (~19k screens): the 64-bit
dHash is split into k+1 disjoint bands; by the pigeonhole principle two hashes
within Hamming distance k agree exactly on at least one band, so only pairs
sharing a band value are candidates (multi-index hashing), and their exact
distance is then verified with vectorised NumPy popcounts.

Dependencies: standard library, NumPy and Pillow only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Dict, Iterable, List, Mapping, NamedTuple, Optional,
                    Sequence, Tuple, Union)

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants and parameters
# ---------------------------------------------------------------------------

SPLITS: Tuple[str, str, str] = ("train", "val", "test")
IMG_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg")

# Maximum dHash Hamming distance (out of 64 bits) at which two screens are
# treated as near-identical. PROVISIONAL default: chosen a priori (a common
# choice for 64-bit difference hashes), not tuned. It should be checked on the
# data by inspecting the near-duplicate matches written to the manifest
# (columns exclusion_reason / matched_train_stem / match_distance); it must
# not be tuned on test-set metrics.
DEFAULT_NEAR_DUP_MAX_DIST = 4

# dHash grid size: a (size+1) x size grayscale thumbnail gives size*size bits.
DEFAULT_DHASH_SIZE = 8

# Default split ratios and seed (identical to build_dataset.py: SPLIT, SEED).
DEFAULT_RATIOS: Tuple[float, float, float] = (0.80, 0.10, 0.10)
DEFAULT_SEED = 42

# Exclusion reasons reported by unseen_subset(), in the order they are tested.
REASON_PACKAGE = "package"
REASON_BUILDER = "builder_prefix"
REASON_MD5 = "md5"
REASON_NEAR_DUP = "near_dup"
REASON_TRANSITIVE = "transitive"
EXCLUSION_REASONS: Tuple[str, ...] = (REASON_PACKAGE, REASON_BUILDER, REASON_MD5,
                                      REASON_NEAR_DUP, REASON_TRANSITIVE)

# "<package>-screens_screenshot_<n>" followed by zero or more "_os<k>" copy
# suffixes. The package part is greedy so that the anchor is the LAST
# "-screens_screenshot_<n>" before the copy suffixes.
_STEM_RE = re.compile(
    r"^(?P<package>.+)-screens_screenshot_(?P<index>\d+)(?P<copies>(?:_os\d+)*)$")
_COPY_RE = re.compile(r"_os(\d+)")

# A package segment that looks like an app-builder generated id: letters
# followed by at least three digits (e.g. 'dev231870' in
# 'com.andromo.dev231870.app482676').
_BUILDER_SEGMENT_RE = re.compile(r"^(?P<prefix>[A-Za-z]+)(?P<digits>\d{3,})$")


class ParsedStem(NamedTuple):
    """Components of a dataset image stem.

    package      — Android package name (as written in the file name)
    screen_index — the <n> of "screenshot_<n>"
    copy_indices — the <k> of every "_os<k>" oversampling suffix, in order
                   (empty tuple for an original screen)
    base_stem    — the stem without oversampling suffixes
    """
    package: str
    screen_index: int
    copy_indices: Tuple[int, ...]
    base_stem: str


@dataclass
class ScreenRecord:
    """One dataset image and the attributes used for grouping.

    stem         — file stem, e.g. "com.shazam.android-screens_screenshot_3"
    path         — image path (None for records built in memory)
    split        — "train" / "val" / "test" (or None)
    package      — parsed package name (see parse_package)
    md5          — hex MD5 of the image file (None: not computed)
    dhash        — dHash as an int (None: not computed)
    class_counts — class id -> number of label instances in this image
    is_copy      — True for an oversampled copy ("_os<k>" suffix)
    """
    stem: str
    path: Optional[str] = None
    split: Optional[str] = None
    package: Optional[str] = None
    md5: Optional[str] = None
    dhash: Optional[int] = None
    class_counts: Dict[int, int] = field(default_factory=dict)
    is_copy: bool = False


class SplitLeakageError(AssertionError):
    """Raised by audit_split() when a split leaks screens across subsets."""


# ---------------------------------------------------------------------------
# File names
# ---------------------------------------------------------------------------

def parse_stem(stem: str) -> ParsedStem:
    """Parses a dataset image stem into package, screen index and copy suffixes.

    Input:  stem — file name without extension, e.g.
            "com.shazam.android-screens_screenshot_3" or
            "com.shazam.android-screens_screenshot_3_os0"
    Output: ParsedStem(package, screen_index, copy_indices, base_stem);
            raises ValueError if the stem does not follow the
            "<package>-screens_screenshot_<n>[_os<k>...]" pattern
    """
    m = _STEM_RE.match(stem)
    if not m:
        raise ValueError(
            f"stem {stem!r} does not match '<package>-screens_screenshot_<n>"
            f"[_os<k>...]' (the naming produced by build_dataset.py / "
            f"oversample_dataset.py)")
    copies = tuple(int(k) for k in _COPY_RE.findall(m.group("copies")))
    base = stem[: len(stem) - len(m.group("copies"))]
    return ParsedStem(m.group("package"), int(m.group("index")), copies, base)


def parse_package(stem: str) -> str:
    """Returns the Android package name encoded in a dataset image stem.

    Input:  stem — file name without extension (see parse_stem)
    Output: package name, e.g. "com.shazam.android"; raises ValueError for a
            stem that does not follow the dataset naming pattern
    """
    return parse_stem(stem).package


def builder_family(package: str) -> Optional[str]:
    """Returns the app-builder family key of a package, if it has one.

    The key is the package up to and including the letters of the first
    segment that consists of letters followed by >= 3 digits, e.g.
    'com.andromo.dev231870.app482676' -> 'com.andromo.dev'. Apps generated by
    the same builder account share it. This is a heuristic and is only used
    when builder-prefix grouping is switched on.

    Input:  package — Android package name
    Output: family key (lower case), or None if no segment matches
    """
    parts = package.lower().split(".")
    for i, seg in enumerate(parts):
        m = _BUILDER_SEGMENT_RE.match(seg)
        if m and i > 0:
            return ".".join(parts[:i] + [m.group("prefix")])
    return None


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def md5_file(path: Union[str, Path], chunk_size: int = 1 << 20) -> str:
    """Computes the MD5 digest of a file's bytes.

    Input:  path — file path; chunk_size — read size in bytes
    Output: lower-case hexadecimal MD5 string
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(image: Union[str, Path, Image.Image], size: int = DEFAULT_DHASH_SIZE) -> int:
    """Computes the difference hash (dHash) of an image.

    The image is converted to 8-bit grayscale, resized to (size+1) x size
    pixels (width x height, Lanczos filter), and bit (r, c) is 1 when pixel
    (r, c+1) is brighter than its left neighbour (r, c). Bits are packed row
    by row, most significant bit first.

    Input:  image — image path or PIL image; size — grid size (bits = size*size)
    Output: hash as a non-negative int with size*size bits
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    if isinstance(image, Image.Image):
        img = image
        gray = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    else:
        with Image.open(image) as img:
            gray = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = np.asarray(gray, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).ravel()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hamming(a: int, b: int) -> int:
    """Counts the differing bits of two integer hashes.

    Input:  a, b — non-negative ints
    Output: Hamming distance (number of set bits of a XOR b)
    """
    return bin(a ^ b).count("1")


_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount_u64(x: np.ndarray) -> np.ndarray:
    """Counts set bits of every element of a uint64 array.

    Input:  x — np.ndarray of dtype uint64
    Output: np.ndarray of the same shape with the bit counts (int64)
    """
    x = np.ascontiguousarray(x, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):  # NumPy >= 2.0
        return np.bitwise_count(x).astype(np.int64)
    as_bytes = x.view(np.uint8).reshape(x.shape + (8,))
    return _POPCOUNT_TABLE[as_bytes].sum(axis=-1).astype(np.int64)


def _bands(nbits: int, max_dist: int) -> List[Tuple[int, int]]:
    """Splits nbits into max_dist+1 disjoint contiguous bands.

    Input:  nbits — hash length in bits; max_dist — Hamming radius k
    Output: list of (shift, width) per band; widths differ by at most one
    """
    n_bands = max_dist + 1
    base, extra = divmod(nbits, n_bands)
    bands, shift = [], 0
    for i in range(n_bands):
        width = base + (1 if i < extra else 0)
        bands.append((shift, width))
        shift += width
    return bands


class HammingIndex:
    """Multi-index hashing over a set of <= 64-bit hashes for radius-k search."""

    def __init__(self, hashes: Sequence[int], max_dist: int, nbits: int = 64):
        """Builds one exact-match table per band.

        Input:  hashes — hash values (ints < 2**nbits); max_dist — radius k
                (0 <= k < nbits); nbits — hash length (<= 64)
        Output: none (index structures stored on the instance)
        """
        if not 1 <= nbits <= 64:
            raise ValueError("nbits must be in [1, 64]")
        if not 0 <= max_dist < nbits:
            raise ValueError(f"max_dist must be in [0, {nbits - 1}]")
        self.max_dist = max_dist
        self.nbits = nbits
        self.hashes = np.array([int(h) for h in hashes], dtype=np.uint64)
        self.bands = _bands(nbits, max_dist)
        self.tables: List[Dict[int, np.ndarray]] = []
        for shift, width in self.bands:
            vals = (self.hashes >> np.uint64(shift)) & np.uint64((1 << width) - 1)
            order = np.argsort(vals, kind="stable")
            sorted_vals = vals[order]
            cuts = np.flatnonzero(np.diff(sorted_vals)) + 1
            table: Dict[int, np.ndarray] = {}
            for run in np.split(order, cuts):
                if run.size:
                    table[int(vals[run[0]])] = run
            self.tables.append(table)

    def query(self, h: int, max_dist: Optional[int] = None) -> List[Tuple[int, int]]:
        """Finds all indexed hashes within Hamming distance of h.

        Input:  h — query hash; max_dist — radius (<= the index radius;
                default the index radius)
        Output: sorted list of (index, distance) pairs
        """
        k = self.max_dist if max_dist is None else max_dist
        if k > self.max_dist:
            raise ValueError("query radius exceeds the index radius")
        cand: List[np.ndarray] = []
        for (shift, width), table in zip(self.bands, self.tables):
            run = table.get((int(h) >> shift) & ((1 << width) - 1))
            if run is not None:
                cand.append(run)
        if not cand:
            return []
        idx = np.unique(np.concatenate(cand))
        d = _popcount_u64(self.hashes[idx] ^ np.uint64(int(h)))
        keep = d <= k
        return [(int(i), int(di)) for i, di in zip(idx[keep], d[keep])]

    def pairs(self) -> List[Tuple[int, int, int]]:
        """Finds all index pairs within the index radius.

        Input:  none
        Output: sorted list of (i, j, distance) with i < j
        """
        found: Dict[Tuple[int, int], int] = {}
        for table in self.tables:
            for run in table.values():
                if run.size < 2:
                    continue
                run = np.sort(run)
                hs = self.hashes[run]
                for p in range(run.size - 1):
                    d = _popcount_u64(hs[p + 1:] ^ hs[p])
                    for q in np.flatnonzero(d <= self.max_dist):
                        found[(int(run[p]), int(run[p + 1 + q]))] = int(d[q])
        return sorted((i, j, d) for (i, j), d in found.items())


def near_duplicate_pairs(hashes: Sequence[int], max_dist: int,
                         nbits: int = 64) -> List[Tuple[int, int, int]]:
    """Finds all pairs of hashes within a Hamming radius (multi-index hashing).

    Input:  hashes — hash values; max_dist — radius k; nbits — hash length
    Output: sorted list of (i, j, distance) with i < j (indices into hashes)
    """
    return HammingIndex(hashes, max_dist, nbits).pairs()


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

class UnionFind:
    """Disjoint-set forest with path halving and union by size."""

    def __init__(self, n: int):
        """Creates n singleton sets.

        Input:  n — number of elements
        Output: none
        """
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, a: int) -> int:
        """Returns the representative of a's set.

        Input:  a — element index
        Output: representative index
        """
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> bool:
        """Merges the sets of a and b.

        Input:  a, b — element indices
        Output: True if two different sets were merged
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def _package_key(rec: ScreenRecord) -> Optional[str]:
    """Returns the package grouping key of a record (lower case).

    Input:  rec — screen record
    Output: lower-cased package, or None if unknown
    """
    return rec.package.lower() if rec.package else None


def _hash_bits(records: Sequence[ScreenRecord]) -> int:
    """Returns the bit length used for near-duplicate search (always 64).

    Shorter hashes (dhash(size < 8)) are handled as 64-bit values whose high
    bits are zero, which keeps the multi-index search exact.

    Input:  records — screen records
    Output: 64; raises ValueError if a hash does not fit in 64 bits
    """
    max_h = max((r.dhash for r in records if r.dhash is not None), default=0)
    if max_h >= 1 << 64:
        raise ValueError("near-duplicate grouping supports hashes of at most 64 bits")
    return 64


def find_links(records: Sequence[ScreenRecord],
               near_dup_max_dist: Optional[int] = DEFAULT_NEAR_DUP_MAX_DIST,
               builder_prefix: bool = False) -> Dict[str, List[Tuple[int, int, int]]]:
    """Lists the pairwise links that define the screen groups.

    Package, builder-family and MD5 links are emitted as chains (each record is
    linked to the first record with the same key), which is sufficient for
    union-find. Near-duplicate links are computed between distinct dHash
    values with multi-index hashing; records with an identical dHash are
    chained with distance 0.

    Input:  records — screen records (package / md5 / dhash may be None);
            near_dup_max_dist — dHash radius k (None disables near-dup links);
            builder_prefix — also link records of the same builder family
    Output: {'package': [...], 'builder_prefix': [...], 'md5': [...],
             'near_dup': [...]} — lists of (i, j, distance) record-index
            triples (distance is 0 except for near-dup links)
    """
    links: Dict[str, List[Tuple[int, int, int]]] = {
        REASON_PACKAGE: [], REASON_BUILDER: [], REASON_MD5: [], REASON_NEAR_DUP: []}

    def chain(key_fn, name: str) -> None:
        """Links every record to the first record sharing its key.

        Input:  key_fn — record -> key (None: no link); name — link type
        Output: none (appends to links[name])
        """
        first: Dict[object, int] = {}
        for i, rec in enumerate(records):
            key = key_fn(rec)
            if key is None:
                continue
            if key in first:
                links[name].append((first[key], i, 0))
            else:
                first[key] = i

    chain(_package_key, REASON_PACKAGE)
    if builder_prefix:
        chain(lambda r: builder_family(r.package) if r.package else None, REASON_BUILDER)
    chain(lambda r: r.md5, REASON_MD5)

    if near_dup_max_dist is not None:
        by_hash: Dict[int, List[int]] = defaultdict(list)
        for i, rec in enumerate(records):
            if rec.dhash is not None:
                by_hash[int(rec.dhash)].append(i)
        if by_hash:
            nbits = _hash_bits(records)
            uniq = sorted(by_hash)
            for h in uniq:
                members = by_hash[h]
                links[REASON_NEAR_DUP].extend((members[0], m, 0) for m in members[1:])
            for a, b, d in near_duplicate_pairs(uniq, near_dup_max_dist, nbits):
                links[REASON_NEAR_DUP].append((by_hash[uniq[a]][0], by_hash[uniq[b]][0], d))
    return links


def group_screens(records: Sequence[ScreenRecord],
                  near_dup_max_dist: Optional[int] = DEFAULT_NEAR_DUP_MAX_DIST,
                  builder_prefix: bool = False) -> List[int]:
    """Assigns every screen to a leak group (connected component of the links).

    Two screens are in the same group if they are connected by a chain of
    links "same package (case-insensitive)", "identical MD5", "dHash distance
    <= near_dup_max_dist" and, if builder_prefix, "same builder family".
    Group ids are canonical: groups are numbered 0, 1, ... in the order of
    their lexicographically smallest stem, so they do not depend on the order
    of the input records.

    Input:  records — screen records; near_dup_max_dist — dHash radius k
            (None disables near-dup links); builder_prefix — enable
            builder-family links (see builder_family)
    Output: list of group ids aligned with records
    """
    uf = UnionFind(len(records))
    for pairs in find_links(records, near_dup_max_dist, builder_prefix).values():
        for i, j, _ in pairs:
            uf.union(i, j)
    root_min_stem: Dict[int, str] = {}
    for i, rec in enumerate(records):
        r = uf.find(i)
        if r not in root_min_stem or rec.stem < root_min_stem[r]:
            root_min_stem[r] = rec.stem
    order = sorted(root_min_stem, key=lambda r: root_min_stem[r])
    gid = {r: g for g, r in enumerate(order)}
    return [gid[uf.find(i)] for i in range(len(records))]


# ---------------------------------------------------------------------------
# Unseen-app subset
# ---------------------------------------------------------------------------

@dataclass
class Exclusion:
    """Why an evaluation screen is not in the unseen subset.

    reason        — one of EXCLUSION_REASONS
    matched_stem  — a training stem that triggered the reason
    distance      — dHash distance for 'near_dup' (0 for 'md5'), else None
    """
    reason: str
    matched_stem: Optional[str] = None
    distance: Optional[int] = None


def classify_exclusions(train_records: Sequence[ScreenRecord],
                        eval_records: Sequence[ScreenRecord],
                        near_dup_max_dist: Optional[int] = DEFAULT_NEAR_DUP_MAX_DIST,
                        builder_prefix: bool = False
                        ) -> Tuple[List[Optional[Exclusion]], List[int]]:
    """Decides, for every evaluation screen, whether its group touches training.

    Reasons are tested in this order and the first that applies is reported:
    'package' (same package as a training screen), 'builder_prefix' (same
    builder family; only if enabled), 'md5' (byte-identical to a training
    image), 'near_dup' (dHash distance <= k to a training image; the closest
    one is reported), 'transitive' (none of the above, but the group reaches a
    training screen through other screens, e.g. a near-duplicate of a test
    screen whose package is in training).

    Input:  train_records, eval_records — screen records;
            near_dup_max_dist — dHash radius k (None disables);
            builder_prefix — enable builder-family grouping
    Output: (exclusions, group_ids) — exclusions aligned with eval_records
            (None = unseen); group_ids aligned with train_records + eval_records
    """
    all_records = list(train_records) + list(eval_records)
    n_train = len(train_records)
    gids = group_screens(all_records, near_dup_max_dist, builder_prefix)
    train_groups = set(gids[:n_train])

    first_by_pkg: Dict[str, str] = {}
    first_by_family: Dict[str, str] = {}
    first_by_md5: Dict[str, str] = {}
    first_by_group: Dict[int, str] = {}
    for rec, g in sorted(zip(train_records, gids[:n_train]), key=lambda t: t[0].stem):
        key = _package_key(rec)
        if key is not None:
            first_by_pkg.setdefault(key, rec.stem)
            fam = builder_family(rec.package) if builder_prefix else None
            if fam is not None:
                first_by_family.setdefault(fam, rec.stem)
        if rec.md5 is not None:
            first_by_md5.setdefault(rec.md5, rec.stem)
        first_by_group.setdefault(g, rec.stem)

    # Near-duplicate index over the distinct training hashes; each hash maps to
    # its lexicographically first training stem.
    index: Optional[HammingIndex] = None
    stem_of_hash: Dict[int, str] = {}
    for rec in train_records:
        if rec.dhash is not None:
            h = int(rec.dhash)
            stem_of_hash[h] = min(stem_of_hash.get(h, rec.stem), rec.stem)
    train_hashes = sorted(stem_of_hash)
    if near_dup_max_dist is not None and train_hashes:
        index = HammingIndex(train_hashes, near_dup_max_dist, _hash_bits(all_records))

    exclusions: List[Optional[Exclusion]] = []
    for rec, g in zip(eval_records, gids[n_train:]):
        if g not in train_groups:
            exclusions.append(None)
            continue
        key = _package_key(rec)
        fam = builder_family(rec.package) if (builder_prefix and rec.package) else None
        if key is not None and key in first_by_pkg:
            exclusions.append(Exclusion(REASON_PACKAGE, first_by_pkg[key]))
        elif fam is not None and fam in first_by_family:
            exclusions.append(Exclusion(REASON_BUILDER, first_by_family[fam]))
        elif rec.md5 is not None and rec.md5 in first_by_md5:
            exclusions.append(Exclusion(REASON_MD5, first_by_md5[rec.md5], 0))
        else:
            hits = index.query(rec.dhash) if (index is not None and rec.dhash is not None) else []
            if hits:
                best = min(hits, key=lambda t: (t[1], stem_of_hash[train_hashes[t[0]]]))
                exclusions.append(Exclusion(REASON_NEAR_DUP,
                                            stem_of_hash[train_hashes[best[0]]], best[1]))
            else:
                exclusions.append(Exclusion(REASON_TRANSITIVE, first_by_group[g]))
    return exclusions, gids


def unseen_subset(train_records: Sequence[ScreenRecord],
                  eval_records: Sequence[ScreenRecord],
                  near_dup_max_dist: Optional[int] = DEFAULT_NEAR_DUP_MAX_DIST,
                  builder_prefix: bool = False
                  ) -> Tuple[List[ScreenRecord], Dict[str, object]]:
    """Selects the evaluation screens whose leak group contains no training screen.

    Input:  train_records — training screens (oversampled copies may be
            included; they share package and MD5 with their originals);
            eval_records — validation or test screens;
            near_dup_max_dist — dHash radius k (None disables near-dup links);
            builder_prefix — enable builder-family grouping
    Output: (subset, stats) — subset in input order; stats = {'n_eval',
            'n_unseen', 'n_excluded', 'excluded_by_reason': {reason: count}}
    """
    exclusions, _ = classify_exclusions(train_records, eval_records,
                                        near_dup_max_dist, builder_prefix)
    subset = [rec for rec, ex in zip(eval_records, exclusions) if ex is None]
    by_reason = Counter(ex.reason for ex in exclusions if ex is not None)
    stats: Dict[str, object] = {
        "n_eval": len(eval_records),
        "n_unseen": len(subset),
        "n_excluded": len(eval_records) - len(subset),
        "excluded_by_reason": {r: by_reason.get(r, 0) for r in EXCLUSION_REASONS},
    }
    return subset, stats


# ---------------------------------------------------------------------------
# Group-level split
# ---------------------------------------------------------------------------

def _check_ratios(ratios: Sequence[float]) -> np.ndarray:
    """Validates split ratios.

    Input:  ratios — non-negative fractions summing to 1
    Output: ratios as a float64 array; raises ValueError otherwise
    """
    r = np.asarray(ratios, dtype=np.float64)
    if r.ndim != 1 or r.size < 2 or (r < 0).any() or abs(r.sum() - 1.0) > 1e-6:
        raise ValueError(f"ratios must be >= 0 and sum to 1, got {list(ratios)}")
    return r


def split_by_group(records: Sequence[ScreenRecord],
                   ratios: Sequence[float] = DEFAULT_RATIOS,
                   seed: int = DEFAULT_SEED,
                   stratify: bool = True,
                   group_ids: Optional[Sequence[int]] = None,
                   near_dup_max_dist: Optional[int] = DEFAULT_NEAR_DUP_MAX_DIST,
                   builder_prefix: bool = False,
                   split_names: Sequence[str] = SPLITS) -> List[str]:
    """Splits screens so that every leak group lands in exactly one subset.

    Stratified mode (greedy multi-label iterative stratification, after
    Sechidis et al. 2011, applied to groups instead of single examples):
    the desired number of instances of every class in every subset is
    ratio * total. Repeatedly, the class with the fewest instances left in
    unassigned groups is taken; its groups are visited in descending order of
    their instance count of that class (ties in a seeded random order), and
    each group goes to the subset with the largest remaining demand for that
    class (ties: largest remaining demand in screens, then seeded random).
    Groups without labels are finally assigned by remaining screen demand.
    Unstratified mode visits all groups in a seeded random order and assigns
    by remaining screen demand only.

    The result depends only on the records' content and the seed, not on the
    order of the records.

    Input:  records — screen records with class_counts; ratios — subset
            fractions (sum 1); seed — RNG seed for tie-breaking/order;
            stratify — use the class-aware greedy; group_ids — precomputed
            group ids (default: group_screens(records, near_dup_max_dist,
            builder_prefix)); split_names — subset names aligned with ratios
    Output: list of subset names aligned with records
    """
    r = _check_ratios(ratios)
    if len(split_names) != r.size:
        raise ValueError("split_names and ratios must have the same length")
    if group_ids is None:
        group_ids = group_screens(records, near_dup_max_dist, builder_prefix)
    if len(group_ids) != len(records):
        raise ValueError("group_ids must align with records")

    # Canonical group order (by smallest stem) so the input order does not matter.
    members: Dict[int, List[int]] = defaultdict(list)
    for i, g in enumerate(group_ids):
        members[g].append(i)
    groups = sorted(members, key=lambda g: min(records[i].stem for i in members[g]))
    classes = sorted({c for rec in records for c in rec.class_counts})
    cidx = {c: k for k, c in enumerate(classes)}
    G = np.zeros((len(groups), len(classes)), dtype=np.float64)
    size = np.zeros(len(groups), dtype=np.float64)
    for gi, g in enumerate(groups):
        for i in members[g]:
            size[gi] += 1
            for c, n in records[i].class_counts.items():
                G[gi, cidx[c]] += n

    rng = np.random.default_rng(seed)
    rand_key = rng.permutation(len(groups))  # seeded tie-breaking key per group
    n_splits = r.size
    demand_size = r * size.sum()
    demand_cls = r[:, None] * G.sum(axis=0)[None, :]
    assign = np.full(len(groups), -1, dtype=np.int64)

    def pick(primary: Optional[np.ndarray]) -> int:
        """Chooses a subset by remaining demand with seeded tie-breaking.

        Input:  primary — per-subset primary demand (None: screens only)
        Output: index of the chosen subset
        """
        cand = np.arange(n_splits)
        if primary is not None:
            cand = cand[np.isclose(primary[cand], primary[cand].max())]
        cand = cand[np.isclose(demand_size[cand], demand_size[cand].max())]
        return int(cand[0]) if cand.size == 1 else int(rng.choice(cand))

    def place(gi: int, s: int) -> None:
        """Assigns group gi to subset s and updates the demands.

        Input:  gi — group index; s — subset index
        Output: none
        """
        assign[gi] = s
        demand_size[s] -= size[gi]
        demand_cls[s] -= G[gi]

    if stratify and classes:
        while True:
            left = assign < 0
            remaining = G[left].sum(axis=0)
            if not (remaining > 0).any():
                break
            c = int(np.flatnonzero(remaining > 0)[np.argmin(remaining[remaining > 0])])
            todo = np.flatnonzero(left & (G[:, c] > 0))
            todo = todo[np.lexsort((rand_key[todo], -G[todo, c]))]
            for gi in todo:
                place(int(gi), pick(demand_cls[:, c]))
    for gi in np.flatnonzero(assign < 0)[np.argsort(rand_key[assign < 0], kind="stable")]:
        place(int(gi), pick(None))

    split_of_group = {g: split_names[int(assign[gi])] for gi, g in enumerate(groups)}
    return [split_of_group[g] for g in group_ids]


def audit_split(records: Sequence[ScreenRecord],
                splits: Sequence[str],
                near_dup_max_dist: Optional[int] = DEFAULT_NEAR_DUP_MAX_DIST,
                builder_prefix: bool = False,
                group_ids: Optional[Sequence[int]] = None) -> Dict[str, object]:
    """Checks a split for leakage between its subsets.

    The checks are computed directly from the records (not from group ids):
    no package (case-insensitive), builder family (if enabled), MD5 or
    near-duplicate pair (dHash distance <= k) may occur in two different
    subsets; if group_ids are given, no group may span two subsets either.

    Input:  records — screen records; splits — subset name per record;
            near_dup_max_dist — dHash radius k (None disables the check);
            builder_prefix — also check builder families;
            group_ids — optional group ids to check
    Output: report dict {'ok': True, 'n_records', 'checks': {...: 0}};
            raises SplitLeakageError listing the violations otherwise
    """
    if len(splits) != len(records):
        raise ValueError("splits must align with records")
    violations: Dict[str, List[str]] = defaultdict(list)

    def check_key(name: str, key_fn) -> None:
        """Records keys that occur in more than one subset.

        Input:  name — check name; key_fn — record -> key (None: skip)
        Output: none (appends to violations[name])
        """
        where: Dict[object, set] = defaultdict(set)
        for rec, s in zip(records, splits):
            key = key_fn(rec)
            if key is not None:
                where[key].add(s)
        for key, ss in sorted(where.items(), key=lambda t: str(t[0])):
            if len(ss) > 1:
                violations[name].append(f"{key}: {sorted(ss)}")

    check_key(REASON_PACKAGE, _package_key)
    if builder_prefix:
        check_key(REASON_BUILDER, lambda r: builder_family(r.package) if r.package else None)
    check_key(REASON_MD5, lambda r: r.md5)
    if group_ids is not None:
        if len(group_ids) != len(records):
            raise ValueError("group_ids must align with records")
        where_g: Dict[int, set] = defaultdict(set)
        for g, s in zip(group_ids, splits):
            where_g[g].add(s)
        for g, ss in sorted(where_g.items()):
            if len(ss) > 1:
                violations["group"].append(f"group {g}: {sorted(ss)}")
    if near_dup_max_dist is not None:
        # Identical hashes first (distance 0), then pairs of distinct hashes.
        splits_of: Dict[int, set] = defaultdict(set)
        stem_of: Dict[int, str] = {}
        for rec, s in zip(records, splits):
            if rec.dhash is not None:
                h = int(rec.dhash)
                splits_of[h].add(s)
                stem_of[h] = min(stem_of.get(h, rec.stem), rec.stem)
        uniq = sorted(splits_of)
        for h in uniq:
            if len(splits_of[h]) > 1:
                violations[REASON_NEAR_DUP].append(
                    f"dhash {h:016x} ({stem_of[h]}...): {sorted(splits_of[h])}")
        if len(uniq) > 1:
            for a, b, d in near_duplicate_pairs(uniq, near_dup_max_dist, _hash_bits(records)):
                ss = splits_of[uniq[a]] | splits_of[uniq[b]]
                if len(ss) > 1:
                    violations[REASON_NEAR_DUP].append(
                        f"{stem_of[uniq[a]]} ~ {stem_of[uniq[b]]}, distance {d}: {sorted(ss)}")
    checks = ["package", "md5"] + (["near_dup"] if near_dup_max_dist is not None else []) \
        + (["builder_prefix"] if builder_prefix else []) \
        + (["group"] if group_ids is not None else [])
    if any(violations.values()):
        lines = [f"{name}: {len(v)} violation(s); first: {v[:5]}"
                 for name, v in violations.items() if v]
        raise SplitLeakageError("split leaks across subsets:\n  " + "\n  ".join(lines))
    return {"ok": True, "n_records": len(records), "checks": {c: 0 for c in checks}}


# ---------------------------------------------------------------------------
# Dataset scanning
# ---------------------------------------------------------------------------

def read_class_counts(label_path: Union[str, Path]) -> Dict[int, int]:
    """Counts the instances per class in a YOLO label file.

    Input:  label_path — .txt file with "cls xc yc w h" lines (missing file = empty)
    Output: dict class id -> instance count (lines with < 5 fields are ignored)
    """
    p = Path(label_path)
    counts: Counter = Counter()
    if not p.exists():
        return {}
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            counts[int(float(parts[0]))] += 1
    return dict(counts)


def _hash_one(path: str, size: int) -> int:
    """Computes the dHash of one file (picklable helper for worker processes).

    Input:  path — image path; size — dHash grid size
    Output: dHash int
    """
    return dhash(path, size)


def scan_yolo_dataset(dataset_dir: Union[str, Path],
                      splits: Sequence[str] = SPLITS,
                      compute_hashes: bool = True,
                      dhash_size: int = DEFAULT_DHASH_SIZE,
                      workers: int = 1) -> List[ScreenRecord]:
    """Reads a YOLO dataset directory into screen records.

    Expects images/<split>/<stem>.<ext> and labels/<split>/<stem>.txt.
    The dHash is computed once per distinct MD5 (oversampled copies are
    byte-identical to their originals and reuse the original's hash).

    Input:  dataset_dir — dataset root; splits — subset names to read;
            compute_hashes — compute MD5 and dHash; dhash_size — dHash grid;
            workers — processes for dHash computation (1 = in-process)
    Output: list of ScreenRecord sorted by (split order, stem); raises
            ValueError for a file name that parse_stem() cannot parse
    """
    root = Path(dataset_dir)
    records: List[ScreenRecord] = []
    for split in splits:
        img_dir = root / "images" / split
        if not img_dir.is_dir():
            continue
        files = sorted(p for p in img_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in IMG_EXTS)
        for p in files:
            parsed = parse_stem(p.stem)
            records.append(ScreenRecord(
                stem=p.stem, path=str(p), split=split, package=parsed.package,
                class_counts=read_class_counts(root / "labels" / split / f"{p.stem}.txt"),
                is_copy=bool(parsed.copy_indices)))
    if compute_hashes:
        hash_records(records, dhash_size, workers)
    return records


def hash_records(records: Sequence[ScreenRecord], dhash_size: int = DEFAULT_DHASH_SIZE,
                 workers: int = 1) -> None:
    """Fills in md5 and dhash of records that have a path.

    The dHash is computed once per distinct MD5 (byte-identical files, such as
    oversampled copies, share it).

    Input:  records — screen records with path set; dhash_size — dHash grid;
            workers — processes for dHash computation (1 = in-process)
    Output: none (records updated in place)
    """
    for rec in records:
        rec.md5 = md5_file(rec.path)
    first_path: Dict[str, str] = {}
    for rec in records:
        first_path.setdefault(rec.md5, rec.path)
    todo = sorted(first_path.items())
    if workers > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            values = list(ex.map(_hash_one, [p for _, p in todo],
                                 [dhash_size] * len(todo), chunksize=32))
    else:
        values = [_hash_one(p, dhash_size) for _, p in todo]
    by_md5 = {m: v for (m, _), v in zip(todo, values)}
    for rec in records:
        rec.dhash = by_md5[rec.md5]


def load_class_names(yaml_path: Optional[Union[str, Path]]) -> Dict[int, str]:
    """Reads class names from a YOLO dataset.yaml ('names' list or dict).

    Input:  yaml_path — path or None
    Output: dict class id -> name (empty if the file is absent)
    """
    if yaml_path is None or not Path(yaml_path).exists():
        return {}
    import yaml  # PyYAML is a package dependency; imported lazily

    data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _class_support(records: Iterable[ScreenRecord],
                   names: Mapping[int, str]) -> Dict[str, Dict[str, int]]:
    """Sums instance and image counts per class.

    Input:  records — screen records; names — class id -> name
    Output: {class name: {'instances': n, 'images': m}} for every class in
            names plus any class id found in the labels
    """
    inst: Counter = Counter()
    imgs: Counter = Counter()
    for rec in records:
        for c, n in rec.class_counts.items():
            inst[c] += n
            imgs[c] += 1
    ids = sorted(set(names) | set(inst))
    return {names.get(c, str(c)): {"instances": int(inst.get(c, 0)),
                                   "images": int(imgs.get(c, 0))} for c in ids}


def _split_stats(records: Sequence[ScreenRecord], gids: Sequence[int]) -> Dict[str, int]:
    """Counts screens, original screens, copies, apps and groups.

    Input:  records — screen records of one subset; gids — their group ids
    Output: dict of counts
    """
    return {
        "screens": len(records),
        "original_screens": sum(1 for r in records if not r.is_copy),
        "oversampled_copies": sum(1 for r in records if r.is_copy),
        "apps": len({_package_key(r) for r in records}),
        "groups": len(set(gids)),
    }


def write_manifest(path: Union[str, Path], rows: Sequence[Dict[str, object]],
                   columns: Sequence[str]) -> None:
    """Writes a CSV manifest.

    Input:  path — output .csv; rows — dicts; columns — column order
    Output: none (file written)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


MANIFEST_COLUMNS: Tuple[str, ...] = (
    "stem", "package", "group_id", "split", "md5", "dhash", "seen_in_train",
    "exclusion_reason", "matched_train_stem", "match_distance",
    "is_oversampled_copy", "n_instances")


def direct_matches_to_train(train: Sequence[ScreenRecord], evals: Sequence[ScreenRecord],
                            near_dup_max_dist: Optional[int] = DEFAULT_NEAR_DUP_MAX_DIST
                            ) -> Dict[str, Dict[str, int]]:
    """Counts evaluation screens that directly match a training image, by match type.

    Independent of packages and of the exclusion order, so that clone counts
    can be compared with other audits: 'md5' (byte-identical file),
    'dhash_0' (identical dHash, e.g. the same pixels re-encoded),
    'dhash_le_k' (dHash distance <= near_dup_max_dist), and the same three
    restricted to matches whose training image has a DIFFERENT package
    ('*_other_package', i.e. clones published under another name).

    Input:  train, evals — screen records (evals carry split 'val'/'test');
            near_dup_max_dist — dHash radius k (None: only md5 / dhash_0)
    Output: {split: {'md5', 'dhash_0', 'dhash_le_k', 'md5_other_package',
             'dhash_0_other_package', 'dhash_le_k_other_package'}}
    """
    pk_by_md5: Dict[str, set] = defaultdict(set)
    pk_by_hash: Dict[int, set] = defaultdict(set)
    for r in train:
        if r.md5 is not None:
            pk_by_md5[r.md5].add(_package_key(r))
        if r.dhash is not None:
            pk_by_hash[int(r.dhash)].add(_package_key(r))
    hashes = sorted(pk_by_hash)
    k = near_dup_max_dist if near_dup_max_dist is not None else 0
    index = HammingIndex(hashes, k, _hash_bits(train)) if hashes else None
    out: Dict[str, Dict[str, int]] = {}
    for split in sorted({r.split for r in evals}):
        c: Counter = Counter()
        for r in (e for e in evals if e.split == split):
            pk = _package_key(r)
            if r.md5 is not None and r.md5 in pk_by_md5:
                c["md5"] += 1
                c["md5_other_package"] += int(bool(pk_by_md5[r.md5] - {pk}))
            hits = index.query(r.dhash) if (index is not None and r.dhash is not None) else []
            exact = [i for i, d in hits if d == 0]
            if exact:
                c["dhash_0"] += 1
                c["dhash_0_other_package"] += int(any(pk_by_hash[hashes[i]] - {pk} for i in exact))
            if hits and near_dup_max_dist is not None:
                c["dhash_le_k"] += 1
                c["dhash_le_k_other_package"] += int(any(pk_by_hash[hashes[i]] - {pk} for i, _ in hits))
        out[split] = {key: int(c.get(key, 0)) for key in (
            "md5", "dhash_0", "dhash_le_k", "md5_other_package", "dhash_0_other_package",
            "dhash_le_k_other_package")}
    return out


def _stems_sha256(stems: Iterable[str]) -> str:
    """SHA-256 of a sorted stem list ("\n"-joined, UTF-8, with a final newline).

    Input:  stems — screen stems
    Output: hex digest (that of the empty string for no stems)
    """
    ordered = sorted(stems)
    text = "".join(f"{x}\n" for x in ordered)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_fingerprints(records: Sequence[ScreenRecord]) -> Dict[str, Dict[str, object]]:
    """Fingerprints of the split: per split the number of screens and a SHA-256 of the sorted stems.

    'originals_sha256' leaves out oversampled copies ('_os<k>' stems), so that
    the fingerprint of a training split does not depend on whether the dataset
    directory was oversampled.

    Input:  records — screen records with split set
    Output: {split: {'screens', 'originals', 'stems_sha256', 'originals_sha256'}}
    """
    out: Dict[str, Dict[str, object]] = {}
    for split in SPLITS:
        recs = [r for r in records if r.split == split]
        originals = [r.stem for r in recs if not r.is_copy]
        out[split] = {"screens": len(recs), "originals": len(originals),
                      "stems_sha256": _stems_sha256(r.stem for r in recs),
                      "originals_sha256": _stems_sha256(originals)}
    return out


def compare_fingerprints(got: Mapping[str, Mapping[str, object]],
                         expected: Mapping[str, Mapping[str, object]]) -> List[str]:
    """Compares split fingerprints with expected ones (e.g. a published summary.json).

    Only the original screens are compared ('originals', 'originals_sha256'),
    so oversampling does not matter.

    Input:  got — split_fingerprints() of this dataset; expected — the
            'split_fingerprints' entry of a reference summary.json
    Output: list of mismatch descriptions (empty if every split agrees)
    """
    problems = []
    for split in SPLITS:
        if split not in expected:
            continue
        for key in ("originals", "originals_sha256"):
            if got.get(split, {}).get(key) != expected[split].get(key):
                problems.append(f"{split}.{key}: got {got.get(split, {}).get(key)!r}, "
                                f"expected {expected[split].get(key)!r}")
    return problems


def build_unseen_report(records: Sequence[ScreenRecord],
                        names: Mapping[int, str],
                        near_dup_max_dist: Optional[int] = DEFAULT_NEAR_DUP_MAX_DIST,
                        builder_prefix: bool = False
                        ) -> Tuple[List[Dict[str, object]], Dict[str, List[ScreenRecord]],
                                   Dict[str, object]]:
    """Computes manifest rows, unseen val/test subsets and the summary.

    Input:  records — all screen records (split set); names — class names;
            near_dup_max_dist — dHash radius k; builder_prefix — flag
    Output: (manifest rows, {'val': [...], 'test': [...]} unseen records, summary dict)
    """
    train = [r for r in records if r.split == "train"]
    evals = [r for r in records if r.split in ("val", "test")]
    exclusions, gids = classify_exclusions(train, evals, near_dup_max_dist, builder_prefix)
    gid_of = {id(r): g for r, g in zip(train + evals, gids)}
    excl_of = {id(r): e for r, e in zip(evals, exclusions)}

    rows: List[Dict[str, object]] = []
    for rec in records:
        ex = excl_of.get(id(rec))
        seen = rec.split == "train" or ex is not None
        rows.append({
            "stem": rec.stem, "package": rec.package, "group_id": gid_of[id(rec)],
            "split": rec.split, "md5": rec.md5,
            "dhash": f"{rec.dhash:016x}" if rec.dhash is not None else "",
            "seen_in_train": int(seen),
            "exclusion_reason": ex.reason if ex else "",
            "matched_train_stem": ex.matched_stem if ex and ex.matched_stem else "",
            "match_distance": ex.distance if ex and ex.distance is not None else "",
            "is_oversampled_copy": int(rec.is_copy),
            "n_instances": sum(rec.class_counts.values()),
        })

    unseen: Dict[str, List[ScreenRecord]] = {}
    summary_unseen: Dict[str, object] = {}
    for split in ("val", "test"):
        full = [r for r in evals if r.split == split]
        sub = [r for r in full if excl_of[id(r)] is None]
        unseen[split] = sub
        reasons = Counter(excl_of[id(r)].reason for r in full if excl_of[id(r)] is not None)
        summary_unseen[split] = {
            **_split_stats(sub, [gid_of[id(r)] for r in sub]),
            "excluded": len(full) - len(sub),
            "excluded_by_reason": {k: reasons.get(k, 0) for k in EXCLUSION_REASONS},
        }

    train_md5 = {r.md5 for r in train if r.md5}
    group_sizes = Counter(gids)
    largest = []
    for g, n in group_sizes.most_common(10):
        recs = [r for r, gg in zip(train + evals, gids) if gg == g]
        largest.append({"group_id": g, "screens": n,
                        "packages": len({_package_key(r) for r in recs}),
                        "splits": dict(Counter(r.split for r in recs))})
    links = find_links(train + evals, near_dup_max_dist, builder_prefix)
    summary: Dict[str, object] = {
        "parameters": {"near_dup_max_dist": near_dup_max_dist,
                       "builder_prefix": builder_prefix,
                       "dhash_size": DEFAULT_DHASH_SIZE},
        "splits": {s: _split_stats([r for r in records if r.split == s],
                                   [gid_of[id(r)] for r in records if r.split == s])
                   for s in SPLITS},
        "unseen": summary_unseen,
        "byte_identical_to_train": {
            s: sum(1 for r in evals if r.split == s and r.md5 in train_md5) for s in ("val", "test")},
        "direct_matches_to_train": direct_matches_to_train(train, evals, near_dup_max_dist),
        "class_support": {
            "val_full": _class_support([r for r in evals if r.split == "val"], names),
            "val_unseen": _class_support(unseen["val"], names),
            "test_full": _class_support([r for r in evals if r.split == "test"], names),
            "test_unseen": _class_support(unseen["test"], names),
        },
        "links": {k: len(v) for k, v in links.items()},
        "largest_groups": largest,
        "split_fingerprints": split_fingerprints(records),
    }
    return rows, unseen, summary


def _write_list(path: Path, records: Sequence[ScreenRecord]) -> None:
    """Writes absolute image paths, one per line (an Ultralytics image list).

    Input:  path — output .txt; records — records with path set
    Output: none (file written)
    """
    path.write_text("".join(f"{Path(r.path).resolve()}\n" for r in records), encoding="utf-8")


def resplit_report(records: Sequence[ScreenRecord], names: Mapping[int, str],
                   ratios: Sequence[float], seed: int,
                   near_dup_max_dist: Optional[int], builder_prefix: bool
                   ) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Makes and audits a new group-level split of the original screens.

    Oversampled copies are left out (oversampling must be re-run on the new
    training subset).

    Input:  records — all records; names — class names; ratios, seed —
            split parameters; near_dup_max_dist, builder_prefix — grouping
    Output: (manifest rows with stem/package/group_id/split/old_split, summary dict)
    """
    originals = [r for r in records if not r.is_copy]
    gids = group_screens(originals, near_dup_max_dist, builder_prefix)
    new = split_by_group(originals, ratios, seed, True, gids)
    audit = audit_split(originals, new, near_dup_max_dist, builder_prefix, gids)
    rows = [{"stem": r.stem, "package": r.package, "group_id": g, "split": s,
             "old_split": r.split, "md5": r.md5,
             "dhash": f"{r.dhash:016x}" if r.dhash is not None else ""}
            for r, g, s in zip(originals, gids, new)]
    by_split = {s: [r for r, ss in zip(originals, new) if ss == s] for s in SPLITS}
    summary = {
        "parameters": {"ratios": list(ratios), "seed": seed,
                       "near_dup_max_dist": near_dup_max_dist,
                       "builder_prefix": builder_prefix, "stratify": True},
        "splits": {s: _split_stats(by_split[s], [g for g, ss in zip(gids, new) if ss == s])
                   for s in SPLITS},
        "class_support": {s: _class_support(by_split[s], names) for s in SPLITS},
        "audit": audit,
    }
    return rows, summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Defines and reads the command-line arguments.

    Input:  argv — argument list (default sys.argv[1:])
    Output: argparse namespace
    """
    ap = argparse.ArgumentParser(
        description="App-level leak audit, unseen-app subset and app-level split "
                    "of a YOLO-format ReDraw dataset")
    ap.add_argument("--dataset-dir", required=True,
                    help="YOLO dataset root (images/{train,val,test}, labels/{...})")
    ap.add_argument("--out", default="results/splits", help="output directory")
    ap.add_argument("--dataset-yaml", default=None,
                    help="dataset.yaml with class names (default: <dataset-dir>/dataset.yaml)")
    ap.add_argument("--near-dup-dist", type=int, default=DEFAULT_NEAR_DUP_MAX_DIST,
                    help="max dHash Hamming distance for near-duplicates (provisional default 4; "
                         "negative disables near-duplicate grouping)")
    ap.add_argument("--builder-prefix", action="store_true",
                    help="also group apps of the same builder family (e.g. com.andromo.devNNNN)")
    ap.add_argument("--resplit", action="store_true",
                    help="also write a new app-level split manifest (for retraining), audited")
    ap.add_argument("--ratios", type=float, nargs=3, default=list(DEFAULT_RATIOS),
                    help="train/val/test ratios for --resplit")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed for --resplit")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="processes for dHash computation")
    ap.add_argument("--expect-fingerprints", default=None, metavar="SUMMARY_JSON",
                    help="reference summary.json: exit with status 3 if the original screens "
                         "of any split differ from its split_fingerprints")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Runs the leak audit and writes the manifest, unseen lists and summary.

    Input:  argv — argument list (default sys.argv[1:])
    Output: exit code 0 (1 without images, 3 if --expect-fingerprints does not
            match); files written to --out: manifest.csv, unseen_val.txt,
            unseen_test.txt, summary.json and, with --resplit,
            app_split_manifest.csv and app_split_summary.json
    """
    args = parse_args(argv)
    dataset_dir = Path(args.dataset_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    k = args.near_dup_dist if args.near_dup_dist >= 0 else None
    yaml_path = Path(args.dataset_yaml) if args.dataset_yaml else dataset_dir / "dataset.yaml"
    names = load_class_names(yaml_path)

    records = scan_yolo_dataset(dataset_dir, workers=args.workers)
    if not records:
        print(f"[ERR] no images found under {dataset_dir}/images/{{train,val,test}}")
        return 1
    rows, unseen, summary = build_unseen_report(records, names, k, args.builder_prefix)
    summary["dataset_dir"] = str(dataset_dir)
    write_manifest(out / "manifest.csv", rows, MANIFEST_COLUMNS)
    _write_list(out / "unseen_val.txt", unseen["val"])
    _write_list(out / "unseen_test.txt", unseen["test"])
    summary["note"] = ("unseen_*.txt contain absolute paths (machine-specific); "
                       "manifest.csv is the portable record of the subset")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for split in ("val", "test"):
        u = summary["unseen"][split]
        print(f"{split}: {summary['splits'][split]['screens']} screens -> "
              f"{u['screens']} unseen ({u['apps']} apps); excluded {u['excluded']} "
              f"{u['excluded_by_reason']}")

    if args.resplit:
        rs_rows, rs_summary = resplit_report(records, names, args.ratios, args.seed,
                                             k, args.builder_prefix)
        write_manifest(out / "app_split_manifest.csv", rs_rows,
                       ("stem", "package", "group_id", "split", "old_split", "md5", "dhash"))
        (out / "app_split_summary.json").write_text(json.dumps(rs_summary, indent=2),
                                                    encoding="utf-8")
        print(f"app-level split: { {s: v['screens'] for s, v in rs_summary['splits'].items()} } "
              f"audit ok")
    for split, fp in summary["split_fingerprints"].items():
        print(f"{split}: {fp['originals']} original screens, sha256 {fp['originals_sha256'][:16]}...")
    print(f"Outputs: {out.resolve()}")
    if args.expect_fingerprints:
        ref = json.loads(Path(args.expect_fingerprints).read_text(encoding="utf-8"))
        problems = compare_fingerprints(summary["split_fingerprints"],
                                        ref.get("split_fingerprints", {}))
        if problems or "split_fingerprints" not in ref:
            print("[FAIL] the split differs from " + args.expect_fingerprints + ": "
                  + ("; ".join(problems) or "no split_fingerprints in the reference"))
            return 3
        print(f"[OK] split fingerprints equal {args.expect_fingerprints}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
