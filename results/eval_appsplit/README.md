<!--
SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>

SPDX-License-Identifier: MIT
-->

# results/eval_appsplit — evaluation on the app-level split (v1.2.0)

The evaluation of `results/eval/` repeated on a **leak-free split**: ReDraw re-split by app
(reviewer 1: "re-split by app") and YOLOv8s retrained on it. Methods, metrics, grids, rules
and report formats are those of `results/eval/README.md`; only the split, the weights and
the numbers change. All choices were made on the new validation split; the new test split
was evaluated once (layout: 2026-10-21 04:54 UTC; detector: one `evaluate_detector` run per
checkpoint).

## Files

| file | content |
|---|---|
| `package_verification.json` | independent checks of the data package (counts, fingerprints, leakage, MD5 / dHash of every val + test image, XML hashes; `scripts/appsplit_checks.py package`) |
| `detector/{baseline,oversample_5k}/detector_{val,test}_full.json` | `ui2compose-evaluate-detector`: AP with image and app-group percentile CIs, operating point, YOLO.val cross-check |
| `detector/original_5k_contaminated/detector_test_full.json` | the original released weights on the new test split — **contaminated, not a result** (see below) |
| **`detector_table2.{json,csv,md}`** | **Table 2**: baseline vs oversample_5k on val and test, BCa intervals over app groups, paired differences, per-class AP50 with support, P / R / F1 at conf 0.3, what the checkpoints store about their training (`detector_ablation`, which writes all three) |
| `detector_colab_check.json` | the local CPU evaluation against the Colab evaluation of the same checkpoints (`scripts/appsplit_checks.py colab`) |
| `val_sweep.{json,md}`, `selected_thresholds.json`, `val_sweep_crossval.json` | threshold sweep (same 120-configuration grid and rule) on the new val split, selection, half-split check |
| `val_selected_vs_previous_default.json` | new val: the selected configuration against the previous default (`cut_tol_frac` 0.0025), paired (`scripts/appsplit_checks.py previous`) |
| **`layout_test.{json,md}`** | **Table 1**: layout evaluation C1–C5, C1′ on the new test split, paired differences, correction-effort proxy, strata |
| **`failure_taxonomy.{json,md}`** | cause of every edit on 50 random C4 test screens (seed 42), three traced cases |

Per-screen records, prediction caches, the val layout run, sweep records and logs are in
`$OUT` (outside the repository; here `data/eval_outputs_appsplit/`, about 0.5 GB). Two
assets of the [v1.2.0 release](https://github.com/gbilgeturk/ui2compose/releases/tag/v1.2.0)
hold them and the data: `ui2compose_v1.2.0_eval_outputs_appsplit.zip` (predictions, caches,
per-screen layout, sweep and taxonomy records, without logs; SHA-256 72fde1fc…66b8cc) and
`ui2compose_redraw_appsplit_v1.2.0_without_images.zip` (the data package without the 3,828
val and test images, which are ReDraw screens identified by MD5 in its manifest, and without
the weights, which are in `runs/`; SHA-256 e059f0f2…f89ef).

## Data and weights

* **Package** `ui2compose_redraw_appsplit_v1.2.0` (Colab, 2026-10-21; zip SHA-256
  d33c6b75…fd28; corrected archive `ui2compose_redraw_appsplit_v1.2.0_r1.zip`, SHA-256
  3f8e5b3f…d4c8, see "Discrepancies and caveats"): the 18,694 labelled screens of the v1.1.0 preprocessing re-split with
  `app_split --resplit` (groups = connected components of "same package OR identical MD5 OR
  dHash ≤ 4", greedy multi-label stratification, seed 42). Train / val / test **14,866 /
  1,928 / 1,900** screens; 4,354 / 642 / 632 packages; 2,199 / 609 / 586 groups (largest
  test group 60 screens, val 29; the 9,071-screen near-duplicate chain is entirely in
  train). Label lines 172,657 / 22,321 / 22,697.
* **Checks repeated here** (`package_verification.json`, all passed): file lists = manifest
  for every split; stem fingerprints = `split_fingerprints.json`; no group, package
  (case-insensitive), MD5 or identical dHash in two splits; brute-force dHash distance ≤ 4
  between val/test and train and between val and test: 0 pairs; MD5 and dHash recomputed for
  all 3,828 val + test images: 0 mismatches; the grouping recomputed from the manifest equals
  its `group_id`; `raw_xml/mapping.csv` covers exactly val + test, with the manifest's
  group ids, every XML present and its SHA-256 as recorded.
* **Weights.** `baseline_app` (no oversampling; MD5 a4cb282d…) and `oversample_5k_app`
  (rare classes of the train split oversampled to ≈ 5,000 instances; MD5 4d5763e8…, SHA-256
  24986d97…), both 60 epochs from COCO `yolov8s.pt` on an A100. Their `train_args` equal the
  `train:` sections of `configs/train_baseline.yaml` / `configs/train_oversample_5k.yaml`
  (56 keys) except `cache` (`ram` vs `True`: both cache images in RAM); outside the configs
  they differ from each other only in `data` and `name` (`detector_table2.json`,
  `provenance`). Training read copies of the screenshots pre-resized to 640 px with
  Ultralytics' own resize (identical network input, checked on Colab); every evaluation
  here uses the original screenshots.
* `runs/oversample_5k_appsplit/weights/best.pt` is the app-split 5k checkpoint (with its
  `args.yaml` and `results.csv`) and the pipeline's default weights;
  `runs/baseline_appsplit/weights/best.pt` is the app-split baseline checkpoint (SHA-256
  66efdfcf…, equal to `weights/baseline_app/best.pt` of the package and its
  `train_summary.json`; with its `args.yaml` and `results.csv`), so both rows of Table 2 can
  be recomputed from the repository; `runs/oversample_5k/` (the original release) is kept for
  reproducibility.
* **Evaluation view.** The evaluation tools read `splits/full/manifest.csv` and
  `splits/package_md5/manifest.csv`; `scripts/run_evaluation.sh app_setup` builds a view of
  the package in `$OUT/pkg_view` with the app split's manifest as `full` and a `package_md5`
  grouping computed from it (631 test / 640 val groups, largest 12 / 9 screens).
* **No unseen subsets.** Every val and test screen belongs to an app (group) that has no
  train screen, so the "unseen" subsets of `results/eval` are the whole splits here.
* **Bundled examples.** On the new split 8 of the 10 `examples/` screens are in **train**,
  battlesimulator in val, drivesafe in test. They were held-out screens for the original
  weights (all ten were old test screens), but most are not for the app-split (default)
  weights; the worked example of the figures is therefore drivesafe.

## Confidence intervals

As in `results/eval`: 1,000 bootstrap replicates, seed 0, each drawing as many app groups as
the set has (with replacement) and every screen of every drawn group. Primary grouping:
the split's own groups (`full` / `app_groups`); `package_md5` in addition. The full grouping
is no longer degenerate (largest test group 60 of 1,900 screens), and the two groupings give
nearly the same intervals. Detector: BCa intervals (AP is not a mean over screens; its
percentile bootstrap is biased, +0.003 for the test mAP50 of oversample_5k); layout:
percentile intervals of means. Paired differences use identical draws. One training run per
configuration: the intervals do not cover run-to-run variation.

## Table 2 — detector (`detector_table2.md`)

| split | checkpoint | mAP50 [BCa] | mAP50-95 [BCa] | P / R / F1 at conf 0.3 (F1 [BCa]) |
|---|---|---|---|---|
| test (1,900; 586 groups) | baseline | 0.404 [0.370, 0.432] | 0.307 [0.279, 0.326] | 0.624 / 0.457 / 0.527 [0.507, 0.549] |
| | oversample_5k | **0.442** [0.395, 0.480] | 0.347 [0.307, 0.381] | 0.585 / 0.498 / 0.538 [0.517, 0.558] |
| | 5k − baseline | +0.038 [+0.014, +0.067], P(≤0) 0.003 | +0.040 [+0.016, +0.069] | P −0.039 [−0.051, −0.025], R +0.041 [+0.030, +0.053], F1 +0.010 [+0.001, +0.021] |
| val (1,928; 609 groups) | baseline | 0.381 [0.347, 0.408] | 0.291 [0.267, 0.312] | 0.600 / 0.466 / 0.525 [0.500, 0.544] |
| | oversample_5k | 0.421 [0.385, 0.456] | 0.330 [0.301, 0.358] | 0.564 / 0.509 / 0.535 [0.511, 0.556] |
| | 5k − baseline | +0.040 [+0.018, +0.067], P(≤0) 0.001 | +0.039 [+0.021, +0.064] | F1 +0.010 [−0.001, +0.027] |

* Percentile intervals over app groups (the Colab table's): test 0.404 [0.376, 0.439] and
  0.442 [0.402, 0.485]; val 0.381 [0.354, 0.418] and 0.421 [0.390, 0.460].
* Every class has ≥ 20 instances in both splits, so the fixed-class mean AP50 that led
  Table 2 in `results/eval` equals mAP50 here.
* Per class (test, Δ = 5k − baseline [BCa]): gains for Spinner +0.222 [+0.132, +0.342],
  SeekBar +0.148 [+0.030, +0.287], CheckBox +0.108 [+0.015, +0.234] and RadioButton +0.100
  [−0.004, +0.235]; losses for Button −0.027 [−0.057, −0.010] and ListView −0.034 [−0.060,
  −0.013]; WebView −0.052 [−0.136, +0.018]. Supports and both splits in the `.md`.
* **Reading.** On apps never seen in training, oversampling to 5k raises mAP50 by about 4
  points on test and on val, through the rare classes, and trades precision for recall at
  the pipeline's operating point (F1 +0.010). This replaces the post-hoc ablation of
  `results/eval`, which found no evidence of a gain on the small unseen-app subsets of the
  screenshot split.
* **Against the screenshot split** (`results/eval`, released weights): test mAP50 0.487 there
  vs 0.442 here, the operating-point F1 0.580 vs 0.538. The two test sets differ, so the
  difference is not an estimate of the leak alone.
* **Colab agreement** (`detector_colab_check.json`): the local CPU FP32 numbers equal the
  Colab ones within 0.00021 mAP50, 0.00009 mAP50-95, 0.0034 per-class AP50 and 0.00023 on
  an interval bound, for both checkpoints and both splits (tolerances fixed a priori: 0.005 /
  0.02 / 0.01). ap.py agrees with Ultralytics' YOLO.val on the same predictions within 4e-6 (mAP) and 5e-5
  (per-class AP50).

**Not a result: the original released weights on the new test split.** They were trained on
the screenshot split; 1,512 of the 1,900 new test screens were in their training split and
190 in their validation split. On the new test split they reach mAP50 0.735 (percentile
[0.702, 0.763]), mAP50-95 0.613, F1 0.656 — memorisation, reported only to show that the
released weights cannot be evaluated on this split
(`detector/original_5k_contaminated/`).

## Threshold selection on the new validation split

Same grid (120 C4 configurations), objective (mean TEDS-o of C4), tie rule and detector
check as in `results/eval`, run on the 1,928 new val screens with the cached detections of
the app-split 5k weights (`val_sweep.md`).

* **Selected: de-duplication looser (0.4 / 0.85 / 0.5), `containment_ratio` 0.9,
  `cut_tol_frac` 0, `y_sorted`**: mean TEDS-o 0.4846 [0.4681, 0.5004]; a priori 0.4780;
  paired +0.0066 [+0.0039, +0.0097]. No tie (the next configuration, `containment_ratio`
  0.95, is 0.0004 lower; the old default is 0.0046 lower).
* **Decision: the shipped defaults change in one parameter**, `cut_tol_frac` 0.0025 → 0 (the
  other three are re-selected unchanged). Against the previous default on new val:
  TEDS-o +0.0046 [+0.0023, +0.0075], TEDS-g −0.0001 [−0.0023, +0.0025], mean grouping F1
  −0.006 [−0.015, +0.002], micro grouping F1 −0.025 [−0.049, −0.001]; 186 screens better, 144
  worse (`val_selected_vs_previous_default.json`). The rule optimises TEDS-o only; the lower
  micro grouping F1 is the price recorded here.
* **Stability** (`val_sweep_crossval.json`, half-splits of the 640 `package_md5` groups,
  200 × 2): the rule selects `cut_tol_frac` 0 in 398 of 400 halves and the full selected
  configuration in 66 % (screenshot split: 10.5 %); out-of-sample gain over the a priori
  values +0.0058 (2.5–97.5 %: +0.0022 to +0.0094, positive in 99.75 %), optimism +0.0006.
* Detector check: every de-duplication setting raises the operating-point F1 (off 0.5350,
  looser 0.5375). Grid edges: the selected de-duplication and `containment_ratio` are the
  loosest values tried and `cut_tol_frac` 0 the smallest.
* With ground-truth boxes (C1) the rule selects `containment_ratio` 0.95 and `cut_tol_frac`
  0.0025 (0.802), reported only.
* Applied in v1.2.0: `DEFAULT_CUT_TOL_FRAC = 0.0` (`infer/generate_compose_code.py`),
  `configs/pipeline_config.yaml`, README; examples, golden files, code-generator fixtures and
  harness renders regenerated (16 / 16 compile and render). With the original weights the
  Kotlin of foxfi, Shazam and drivesafe changes; on the examples the fallback nodes go 2 → 11.
  The examples were then regenerated once more with the app-split weights as the default
  (16 / 16 compile and render; 141 raw detections, 8 removed, 5 fallback nodes).

## Table 1 — layout evaluation, new test split (`layout_test.md`)

1,900 screens, all scored (no exclusion; every label line aligned to its XML node; zss
agrees with APTED on 600 / 600 checked screens). Mean [95 % CI, app groups]; grouping F1
over the screens where it is defined (n). Edit operations: TEDS-o mapping, median (IQR).

| | TEDS-g | TEDS-o | grouping F1 (n) | edit ops | nodes gen / ref |
|---|---|---|---|---|---|
| C1 GT + XY-cut | 0.809 [0.796, 0.822] | 0.786 [0.772, 0.801] | 0.734 [0.719, 0.750] (1,793) | 2 (0–6) | 10 / 11 |
| C2 GT + flat Column | 0.710 [0.700, 0.721] | 0.648 [0.636, 0.660] | 0.460 [0.445, 0.475] (1,812) | 3 (2–9) | 8 / 11 |
| C3 v1.1.0 as submitted | 0.522 [0.505, 0.537] | 0.481 [0.465, 0.497] | 0.614 [0.591, 0.636] (1,468) | 6 (3–14) | 8 / 11 |
| C4 v1.2.0 pipeline | 0.530 [0.513, 0.546] | 0.491 [0.475, 0.508] | 0.690 [0.668, 0.711] (1,464) | 6 (3–13) | 8 / 11 |
| C5 pred + flat Column | 0.476 [0.462, 0.489] | 0.410 [0.397, 0.422] | 0.464 [0.444, 0.483] (1,513) | 7 (3–14) | 7 / 11 |
| C1′ GT + v1.1.0 layout | 0.807 [0.796, 0.818] | 0.786 [0.774, 0.798] | 0.728 [0.712, 0.744] (1,793) | 2 (0–5) | 9 / 11 |

Paired differences (app groups; `package_md5` intervals in the `.md`, nearly identical):

| | TEDS-g | TEDS-o | grouping F1 (n) |
|---|---|---|---|
| C1 − C2 | +0.099 [+0.086, +0.112] | +0.138 [+0.123, +0.156] | +0.270 [+0.251, +0.289] (1,793) |
| C4 − C5 | +0.054 [+0.046, +0.061] | +0.081 [+0.071, +0.091] | +0.210 [+0.189, +0.230] (1,464) |
| C4 − C3 | +0.008 [+0.004, +0.013] | +0.010 [+0.005, +0.015] | +0.080 [+0.064, +0.094] (1,449) |
| C1 − C4 | +0.279 [+0.264, +0.295] | +0.295 [+0.280, +0.310] | +0.048 [+0.030, +0.067] (1,464) |
| C1 − C1′ | +0.002 [−0.003, +0.007] | +0.001 [−0.005, +0.006] | +0.007 [+0.000, +0.013] (1,793) |

* On the 1,449 screens where every condition defines grouping F1: C1 0.741, C2 0.445, C3
  0.618, C4 0.697, C5 0.485, C1′ 0.733.
* Detection: 52.3 % of the ground-truth boxes are matched after de-duplication, 67.7 % of the
  boxes given to the layout are matched, 1,057 matched pairs have the wrong class;
  de-duplication removes 1,766 of 19,327 raw detections. v1.1.0 (C3) emits 843 duplicated
  boxes, loses 1,994 and writes 475 placeholders on 287 screens; C4 none. Fallback nodes: C4
  1,643 on 863 screens, C3 2,528 on 982, C1 3,107 on 1,164.
* Micro grouping F1 [app groups]: C1 0.515 [0.407, 0.633], C4 0.624 [0.582, 0.665], C3 0.601
  [0.555, 0.646], C2 0.227 [0.192, 0.267] (dominated by a few large screens; C1 − C4 −0.109
  [−0.222, +0.018]).
* Strata (post hoc, as in `results/eval`): on the 308 flat dumps (16.2 %) the flat Column wins
  (C1 − C2 TEDS-g −0.084 [−0.105, −0.064]); on the 1,592 others XY-cut wins (+0.134 [+0.121,
  +0.148]). 88 screens have a single GT box (C1 = C2 there).
* **Against the screenshot split** (`results/eval/layout_test.md`, other test screens, other
  weights, `cut_tol_frac` 0.0025): C1 0.825 / 0.803, C4 0.549 / 0.506 (TEDS-g / TEDS-o);
  C1 − C2 +0.115, C4 − C3 +0.010 TEDS-g. The conclusions keep their direction; the
  containment-Box sensitivity (`--box-repr container`) was not re-run.

**Correction-effort proxy** (structure only; edits = TED, `layout_test.md`): C1 median 2
(0–6), 35.0 % of the screens need no edit, 74.4 % [72.0, 76.8] ≤ 5 edits; C4 median 6
(3–13), 9.6 % none, 46.9 % [44.5, 49.3] ≤ 5; edits per reference node 0.15 vs 0.57 (TEDS-o
mapping; with TEDS-g: 76.4 % and 49.6 % ≤ 5).

## Failure taxonomy (`failure_taxonomy.md`)

Same rules, sample size and seed (50 test screens, `default_rng(42)`); the C4 trees and TEDs
of all 50 screens equal those of the test run. 547 edits (median 7 per screen, IQR 2–15):

| family | edits | share | dominant on |
|---|---:|---:|---:|
| detector (fn 183, fp 84, wrong class 12) | 279 | 51.0 % | 26 screens |
| layout rules (xycut_fallback 58, containment 35, grouping 27, dedup_removed 11, tolerance_cut 0) | 131 | 23.9 % | 10 |
| reference (artifact 92, flat2d 34, overlap 11) | 137 | 25.0 % | 11 |

3 screens are perfect. `tolerance_cut` is 0 because the tolerance is 0. Traced cases (raw
detections identical to the cache in all three): `com.hrfy.plus-screens_screenshot_1`
(detector_fp: a detected ListView, containing 18 detected boxes laid out as 9 rows, that the
dump does not label),
`com.app.nl3-screens_screenshot_3` (xycut_fallback: three overlapping boxes ordered by
y-centre), `name.supreeth.android.spoticon-screens_screenshot_1` (reference_artifact: a
detected button outside the dumped window). Screenshot split for comparison: detector 57.5 %,
layout rules 24.9 %, reference 17.5 % of 457 edits.

## How to reproduce

```
PKG=/path/to/ui2compose_redraw_appsplit_v1.2.0 OUT=/path/outside/the/repo/eval_outputs_appsplit \
PY=python WORKERS=4 scripts/run_evaluation.sh app_setup app_pred_cache app_detector \
    app_sweep app_layout_test app_taxonomy
python scripts/appsplit_checks.py package "$PKG"
python scripts/appsplit_checks.py colab "$PKG"
python scripts/appsplit_checks.py previous "$OUT/pkg_view" "$OUT/layout_sweep_val"
```

`app_setup` takes the v1.1.0 files for C3 / C1′ from `git show v1.1.0:…` or from
`V110_CODE=<folder>` (SHA-256 checked by `layout_eval`). What was run: the evaluate_detector,
pred_cache, sweep and detector_ablation commands of the stages were run with the stages'
arguments from the folder holding `data/` and `ui2compose/` (paths inside those JSON files
are relative to it; `EVAL_BASE=..` when re-running `detector_ablation` from the repository
root); `app_layout_test` and `app_taxonomy` were run through the script. Environment: Apple
M2 (8 cores, 16 GB, CPU only, shared with an Android emulator), Python 3.13.3, torch 2.10.0,
ultralytics 8.4.24, numpy 2.4.3. Wall time under load: pred_cache ≈ 9.5 min per split,
evaluate_detector 30–47 min per checkpoint and split (four in parallel), sweep 24 min (4
processes), layout_test 1 min, taxonomy 18 s, detector_ablation 15 min.

## Discrepancies and caveats

* `splits/split_fingerprints.json` records a `manifest_sha256` (e7ee1a11…) that is not the
  SHA-256 of `app_split_manifest.csv` as shipped (c38d75e0…). The cause is the definition, not
  a rewritten file: the Colab notebook (`notebooks/retrain_app_split_colab.ipynb`) hashes the
  text of one `<stem>,<split>` line per screen, stems sorted, not the CSV bytes. Recomputed
  that way from the shipped manifest it is exactly e7ee1a11…, and the three stem fingerprints
  still equal `split_<split>_stems.txt`, so no split assignment changed. The corrected
  archive `ui2compose_redraw_appsplit_v1.2.0_r1.zip` (SHA-256
  3f8e5b3f764c329521d5809a170ec77a4e59553a94bc7589f5faa34d2ab1d4c8) keeps the value and adds
  `manifest_sha256_definition`, `manifest_file_sha256` (c38d75e0…) and a `note` to
  `split_fingerprints.json`, and a revision note to `README_PACKAGE.txt`; its other 26,450
  files are byte-identical to the first archive (CRC-32 and size checked).
* The package's `table1_app_split` is the detector table, i.e. Table 2 here.
* The thresholds were selected with the app-split weights, which are now also the pipeline's
  default weights (`runs/oversample_5k_appsplit/weights/best.pt`); the original release
  (`runs/oversample_5k/`) is kept for reproducing `results/eval/`.
* The shipped `cut_tol_frac` changed after the screenshot-split evaluation; `results/eval`
  documents the earlier selection (0.0025) and is left unchanged.
