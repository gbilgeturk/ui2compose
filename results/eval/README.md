<!--
SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>

SPDX-License-Identifier: MIT
-->

# results/eval — evaluation of the SoftwareX revision

> **Supplementary.** This is the evaluation of the v1.1.0 weights on the screenshot-level
> split of v1.1.0, with the thresholds selected on that split (`cut_tol_frac` 0.0025). The
> numbers of the revised paper and the shipped defaults (`cut_tol_frac` 0) come from the
> app-level split in `results/eval_appsplit/`; this folder is kept for comparison and for
> reproducing the numbers reported for v1.1.0.

This folder holds the small summaries of the revision's evaluation:

* detector numbers without the app leak (R1.1, R2.5) with intervals (R2.6);
* a layout evaluation on whole splits against the ReDraw view hierarchies,
  with baselines, failure cases and a correction-effort proxy (R1.2, R2.1,
  R2.2, R2.9).

Per-screen outputs (JSONL), prediction caches, pipeline traces and logs are
kept outside the repository (to be archived on Zenodo). Every number here is
reproduced by `scripts/run_evaluation.sh` from the data package and the
released weights (see "How to reproduce").

Sections marked **post hoc** were added after an internal review of these
results. They add checks and sensitivity analyses; no threshold, rule or
primary number was changed by them.

## Files

| file | content | written by |
|---|---|---|
| `detector_{val,test}_{full,unseen_full,unseen_package_md5,unseen_package_only}.json` | AP / mAP of the released weights with bootstrap CIs, the pipeline operating point, the YOLO.val cross-check | `python -m ui2compose.evaluation.evaluate_detector` |
| `detector_summary.{json,csv}` | the eight evaluations side by side, CIs under every grouping, per-class support | `python -m ui2compose.evaluation.detector_breakdown summary` |
| `detector_seen_unseen_{test,val}.{json,csv}` | per-class AP on screens of apps seen / unseen in training, differences with CIs | `... detector_breakdown seen-unseen` |
| `detector_intervals.{json,csv}` | **post hoc**: bootstrap bias of AP, percentile vs basic vs BCa intervals from the same replicates | `... detector_breakdown intervals` |
| `ablation/{imgsz_640,oversample_2k}/detector_{val,test}_{full,unseen_full,unseen_package_only}.json` | **post hoc**: the no-oversampling baseline and the 2k checkpoint, evaluated exactly as the released weights | `... evaluate_detector` |
| **`detector_ablation.{json,csv,md}`** | **post hoc: oversampling ablation, baseline / 2k / 5k × all / unseen subsets of test and val, paired differences with BCa intervals, the eight oversampled classes, what the checkpoints store about their training** | `python -m ui2compose.evaluation.detector_ablation` |
| `pred_cache_summary.json`, `detector_eval_timings.json` | prediction cache checks, job timings | detector evaluation |
| `layout_val_reference.json` | reference-tree statistics of the val split | `ui2compose-eval-layout` |
| `val_sweep.{json,md}`, `selected_thresholds.json` | threshold grid on val and the selected (shipped) values | `python -m ui2compose.evaluation.layout_sweep report` (+ `crossval`) |
| `val_sweep_crossval.json` | **post hoc**: the selection rule applied on one half of the val app groups, scored on the other | `... layout_sweep crossval` |
| **`layout_test.{json,md}`** | **Table 1: the test split, every condition, all / unseen / package-only subsets, paired differences, correction-effort proxy; post-hoc strata and the containment-Box sensitivity** | `ui2compose-eval-layout` + `python -m ui2compose.evaluation.layout_report` |
| **`failure_taxonomy.{json,md}`** | **cause of every edit on 50 random C4 test screens, three traced cases** | `python -m ui2compose.evaluation.failure_taxonomy` |

`coco_init_check.json` was written by a separate job and is not part of this evaluation.

## Data provenance

* **Package.** `ui2compose_redraw_valtest_v1.1.0_UNVERIFIED`, regenerated on
  Colab with the v1.1.0 preprocessing (`README_PACKAGE.txt`): val and test
  images, the label files of all three splits, the raw ReDraw uiautomator
  dumps of val and test (`raw_xml/`, linked to the images by
  `raw_xml/mapping.csv`), the app grouping manifests (`splits/`) and the
  unseen-app lists (`dataset/unseen_*.txt`).
* **The screen split is the thesis split**: 14,955 / 1,869 / 1,870 screens,
  all 10 bundled examples in test, 214 package-unseen test screens. The published
  test and val mAP are reproduced within 0.05 points (`detector_*_full.json`,
  key `reproduction`).
* **The labels are not byte-identical to the thesis run.** Per-class
  instance counts differ from thesis Table 3.1 by about 0.1 % (val 21,658
  label lines vs 21,613; test 23,068 vs 23,046). The package's status line
  "split reproduced: False / FAIL" refers to these instance counts only; the
  screen lists are identical.
* **Label quirk (found here).** `parse_redraw.to_xywhn` clamps centre and
  size separately, so a node whose bounds extend beyond the screen gets a
  label box that is shifted and still extends beyond it (e.g. a RecyclerView
  with XML bounds y 178–3202 on a 1,920 px screen is labelled y 730–2650).
  This concerns 718 of the 23,068 test labels (3.1 %, on 239 screens) and
  625 of the 21,658 val labels, most often lists (ListView 78 / 445,
  RecyclerView 34 / 165 on test). The same preprocessing produced the
  training labels. The layout evaluation uses the labels as they are; the
  failure taxonomy counts edits caused by such boxes as reference artifacts.
* **Weights.** `runs/oversample_5k/weights/best.pt`, MD5
  eb6ad943fb29b4593b5d1fbe993b347f (the released checkpoint). There is one
  training run of it; every number outside the oversampling ablation uses
  this checkpoint only. The ablation (post hoc) adds the two other
  checkpoints of the thesis comparison, assets of the v1.0.2 release:
  `imgsz_640.pt` (MD5 2812ba6e418b2a056996c23fc2ea5be6, the no-oversampling
  baseline) and `oversample_2k.pt` (MD5 144754dd3f0c8e74bc651ace6e4803d1).
* **Predictions.** One cache per split (`pred_cache`), made with the
  pipeline's own `ComponentDetector.detect` at its operating point: conf 0.3,
  NMS IoU 0.7, imgsz 640 (from the checkpoint), class-aware NMS, max_det
  300, CPU. 19,799 test and 18,242 val detections. The 10 bundled examples
  are bit-identical to `results/v1.1.0_examples` and `tests/golden`, and the
  three traced failure cases re-run through the CLI gave detections
  identical to the cache.

## How to reproduce

```
pip install -e ".[eval]"        # apted 1.0.3, zss 1.2.0, scipy (used: 1.17.1)
PKG=/path/to/ui2compose_redraw_valtest_v1.1.0_UNVERIFIED \
OUT=/path/outside/the/repo/eval_outputs \
PY=python WORKERS=4 scripts/run_evaluation.sh            # all stages
scripts/run_evaluation.sh layout_test taxonomy             # or selected stages
DRY_RUN=1 scripts/run_evaluation.sh                        # print the commands
```

Stages, in order: `setup` (dataset yaml with an absolute path in `$OUT`;
`git show v1.1.0:src/ui2compose/infer/{build_ui_graph,generate_compose_code}.py`
into `$OUT/v110_code`, SHA-256 checked by `layout_eval`), `pred_cache`,
`detector`, `detector_intervals` (post hoc), `detector_ablation` (post hoc;
needs `ABLATION_WEIGHTS`, the folder with `imgsz_640.pt` and
`oversample_2k.pt`), `sweep` (val only),
`sweep_crossval` (post hoc, val only), `layout_val`, `layout_test` (with
`--allow-test`), `box_sensitivity` (post hoc; C1 and C4 on val, then once on
test; rewrites both layout reports with the sensitivity section),
`taxonomy`. Each stage prints its commands. `EVAL_BASE` names the directory
the `detector` stage was run from (paths inside its JSON files are relative
to it; default: the repository root).

* Environment of the runs: Python 3.13.3, numpy 2.4.3, Pillow 12.1.1,
  ultralytics 8.4.24, torch 2.10.0 (CPU), apted 1.0.3, zss 1.2.0, scipy
  1.17.1, pytest 9.1.1, on an Apple M2 (8 cores, 16 GB) shared with other jobs.
* What was executed through the script: `setup` (into a scratch folder; the
  yaml and both v1.1.0 files came out identical to the ones used),
  `layout_val` (all 1,869 C4 records identical to the earlier run),
  `layout_test`, `taxonomy`, and the post-hoc stages `detector_intervals`
  (with `EVAL_BASE=..`), `sweep_crossval` and `box_sensitivity`.
  `pred_cache`, `detector` and `sweep` were run earlier with the same tools
  and settings (recorded in the JSON outputs: weights, conf, IoU, imgsz,
  groups, bootstrap, seed); the script repeats those commands, checked with
  a dry run only. Paths inside the JSON files are relative to the folder the
  earlier jobs were started from (the folder holding `data/` and
  `ui2compose/`). `detector_ablation`: its twelve `evaluate_detector`
  commands were run with the stage's arguments from that folder (the two
  checkpoints in parallel), then its `detector_ablation` command from the
  repository root with `EVAL_BASE=..`; the stage itself was checked with a
  dry run.
* Run times (wall, shared M2): pred_cache 13–14 min per split; detector
  evaluations 3–40 min each (about 2.5 h in total); sweep 11 min on 6
  processes; layout_val 36 s and layout_test 99 s on 4 processes; taxonomy
  about 3 min (most of it the three pipeline runs); box_sensitivity 4.5 min
  and detector_intervals 62 min on 3 processes (machine
  heavily loaded); sweep_crossval 14 s; detector_ablation about 1.5 h of
  inference per checkpoint (test 41 min, val 35 min, the four unseen
  subsets 3–4 min each) and 15–25 min for the paired bootstrap on 3
  processes. Disk: about 340 MB in `$OUT` (166 MB of it the sweep's
  per-screen records), plus 192 MB of ablation predictions.
* **Test-split discipline.** The thresholds were selected on val
  (`selected_thresholds.json`); the layout metrics, the taxonomy rules and
  both report formats were fixed on val. The test split was then run once
  (`layout_eval --split test` refuses to run without `--allow-test`; so do
  `layout_sweep` and `failure_taxonomy`). Changes after that run:
  * the taxonomy case rule (which screens illustrate it, see there); no
    number depends on it;
  * **post hoc, one further test run**: C1 and C4 again with
    `--box-repr container` (`$OUT/layout_test_boxsens`, 2026-10-20 12:59
    UTC), a sensitivity analysis of the scoring of containment Boxes. The
    variant was implemented and checked on val first (it reproduces the
    review's val re-score to four decimals); it changes no threshold and no
    primary number;
  * post hoc, computed from the per-screen records of the single test run
    without re-running anything: the grouping-F1 bookkeeping, the
    common-screen means and the strata; and, from the saved predictions,
    the detector interval check (no new inference).

## Confidence intervals

* 95 % percentile bootstrap, 1,000 replicates, seed 0 (as `ap.bootstrap_ci`).
  Each replicate draws as many app groups as the evaluated set has, with
  replacement, and takes every screen of every drawn group.
* Every interval names its grouping:
  * `full` (primary, as specified): `group_id` of `splits/full/manifest.csv`
    (package OR MD5 OR dHash ≤ 4, transitively closed);
  * `package_md5`: `splits/package_md5/manifest.csv` (package OR MD5);
  * `screen` (layout JSON only) and `image` (detector): every screen its own unit.
* **The full grouping is degenerate on the full splits.** Near-duplicate
  dHashes chain 2,010 packages into one group of 9,071 screens: 880 of the
  1,870 test screens and 878 of the 1,869 val screens. A replicate then
  holds that half of the split 0, 1, 2, … times, which makes the `full`
  intervals on "all screens" wide and asymmetric. We recommend the
  `package_md5` intervals for the full splits (largest group: 4 screens).
  On the unseen subsets no screen is in that group and the two groupings
  agree.
* Paired differences (layout) use the same draws for both conditions and
  the per-screen differences; `P` is the fraction of replicates ≤ 0.
  Seen-vs-unseen detector differences resample the two subsets separately.
* **The percentile bootstrap of AP is biased (post hoc check,
  `detector_intervals.*`).** The layout metrics are means over screens:
  their replicates are centred on the point estimate to within 0.002 (up to
  0.005 below it under the degenerate `full` grouping on all screens, checked
  for C1 and C4). AP is not a mean over screens: its
  bootstrap replicates lie above the point estimate (on all test screens,
  +0.009 for mAP50 with `package_md5` groups, 63 % of the replicates
  above the point), and much more so on the small unseen subsets, where
  classes with one to ten instances drop in and out of the per-replicate
  class mean. A percentile interval then sits too high. The check recomputes
  the same replicates (same seed and units; the percentile intervals are
  reproduced exactly) and adds the bias, the basic (reverse percentile)
  interval and the BCa interval. On the unseen subsets the bias of mAP50 is
  +0.019 to +0.039 (by subset and grouping), that of the mean AP50 over the
  classes with at least 20 instances −0.004 to +0.016. We report BCa
  intervals for mAP and lead with the fixed-class mean.
* **One training run.** The intervals describe the sampling of screens and
  apps, not the variation between training runs (seeds), which this study
  cannot measure without retraining.

## Detector evaluation (R1.1, R2.5, R2.6)

The released weights are evaluated on the test (and val) screens whose app
never occurs in training, without retraining. Three unseen subsets: `full`
grouping (169 test screens: the screen's whole near-duplicate group is
unseen), `package_md5` (212) and `package_only` (214, the thesis
definition). AP uses the Ultralytics validation defaults (conf 0.001, NMS
IoU 0.7, multi-label) through an exact port of Ultralytics' matching
(`ap.py`, agrees with YOLO.val to 1e-9); the operating point uses the
pipeline's own detections (conf 0.3, IoU 0.5). Weights SHA-256 b7b2987f…
as on Colab; test mAP50 0.4874412 here vs 0.4874411 on Colab.

**Headline number: the mean AP50 over a fixed set of classes.** mAP is the
mean over the classes present in a set; on the unseen subsets most classes
have 0–10 instances, so mAP mixes in near-random per-class values and its
bootstrap is strongly biased (see "Confidence intervals"). The mean AP50
over the classes with at least 20 instances in the set (Button, EditText,
ImageView, TextView, WebView on the unseen test subsets; all 13 classes on
the full splits) is much less biased (on the unseen test subsets −0.004 to
+0.006) and is reported first.

**Table 2 (test; val below).** 95 % intervals over `package_md5` groups:
the percentile interval of the evaluation files and the bias-corrected BCa
interval from the same replicates (`detector_intervals.csv` has the
`image` and `full` groupings and the basic intervals too). Bias = mean of
the bootstrap replicates − point. P / R / F1: pipeline detections at conf
0.3, IoU 0.5, micro over boxes, percentile intervals.

| set | screens (groups) | mean AP50, classes ≥ 20 instances: point, BCa | mAP50: point, percentile / BCa, bias | mAP50-95: point, BCa | P / R / F1 (F1 interval) |
|---|---|---|---|---|---|
| test, all | 1,870 (1,586) | 0.487 [0.434, 0.527] (13 classes) | 0.487 [0.449, 0.543] / [0.434, 0.527], +0.009 | 0.391 [0.348, 0.423] | 0.628 / 0.539 / 0.580 [0.561, 0.599] |
| test, unseen (full grouping) | 169 (161) | 0.434 [0.339, 0.555] (5) | 0.289 [0.229, 0.442] / [0.186, 0.365], +0.037 | 0.205 [0.131, 0.257] | 0.412 / 0.429 / 0.420 [0.306, 0.544] |
| test, unseen (package_md5) | 212 (201) | 0.507 [0.405, 0.589] (5) | 0.370 [0.295, 0.508] / [0.265, 0.450], +0.026 | 0.281 [0.195, 0.361] | 0.500 / 0.507 / 0.504 [0.392, 0.602] |
| test, unseen (package only) | 214 (203) | 0.507 [0.409, 0.604] (5) | 0.370 [0.294, 0.506] / [0.264, 0.462], +0.024 | 0.281 [0.200, 0.371] | 0.500 / 0.507 / 0.504 [0.392, 0.613] |
| val, all | 1,869 (1,590) | 0.519 [0.474, 0.560] (13) | 0.519 [0.479, 0.567] / [0.474, 0.560], +0.004 | 0.403 [0.362, 0.438] | 0.614 / 0.517 / 0.562 [0.544, 0.580] |
| val, unseen (full grouping) | 203 (189) | 0.447 [0.350, 0.513] (5) | 0.402 [0.317, 0.542] / [0.289, 0.499], +0.020 | 0.266 [0.180, 0.312] | 0.451 / 0.410 / 0.429 [0.360, 0.512] |
| val, unseen (package_md5 = package only) | 241 (224) | 0.414 [0.332, 0.482] (6, with ListView) | 0.375 [0.317, 0.515] / [0.273, 0.428], +0.035 | 0.265 [0.197, 0.312] | 0.471 / 0.418 / 0.443 [0.376, 0.512] |

* Under the `full` grouping the test row is 0.487 with percentile
  [0.391, 0.545] and BCa [0.399, 0.552] (one unit holds 880 of the 1,870
  screens; acceleration 0.14); on the unseen subsets the groupings agree.
* The 43 screens that only the near-duplicate rule removes (212 vs 169) are
  noticeably easier: ImageView AP50 is 0.292 on the 169-screen set and 0.412
  on the 212-screen set; TextView 0.375 and 0.544.
* The operating-point P / R / F1 are ratios of box counts; in a separate
  check (same draws, not in the files) their replicates are centred within
  0.007 of the point, so their percentile intervals stand.

**Seen vs unseen apps (test; seen = the screen's full-grouping group has a
train screen).** 1,701 seen and 169 unseen screens; the two subsets are
resampled separately (`package_md5` groups). Mean AP50 over the five
classes with ≥ 20 instances in both: seen 0.573 [0.546, 0.607], unseen
0.434 [0.336, 0.546]; **difference −0.139, percentile [−0.245, −0.023],
basic [−0.255, −0.033]**, bias +0.004, P(diff ≤ 0) = 0.994. The all-class
mAP50 difference, −0.206 with percentile [−0.282, −0.042], is biased
(+0.032) like the unseen mAP; its basic interval is [−0.370, −0.130]. On
val the fixed-class difference is −0.112 [−0.192, −0.017] (basic [−0.206,
−0.031]). No BCa is computed for the two-sample differences.

| class | instances seen / unseen | AP50 seen | AP50 unseen | unseen − seen |
|---|---|---|---|---|
| Button | 2642 / 147 | .638 [.596,.679] | .419 [.290,.553] | −.219 [−.360,−.057] |
| EditText | 584 / 50 | .714 [.655,.775] | .547 [.303,.754] | −.167 [−.400,.046] |
| ImageView | 6105 / 359 | .572 [.533,.616] | .292 [.148,.478] | −.280 [−.426,−.092] |
| TextView | 11382 / 489 | .599 [.558,.636] | .375 [.171,.631] | −.224 [−.445,.032] |
| WebView | 72 / 36 | .344 [.249,.473] | .539 [.380,.705] | +.195 [−.006,.375] |
| mean of the five | | .573 [.546,.607] | .434 [.336,.546] | −.139 [−.245,−.023] |

Per-class intervals are percentile intervals (`package_md5` groups); they
were not bias-checked. Classes with too few unseen instances for a usable
AP (flagged in the files): CheckBox 1, ListView 2, ProgressBar 6,
RadioButton 6, RecyclerView 10, SeekBar 1, Spinner 0, Switch 3. On the full
splits every class has at least 20 instances, but some sit on few screens
(CheckBox 212 instances on 69 test images, Switch 61 on 23), so their
intervals are wide. Details: `detector_summary.csv`,
`detector_seen_unseen_test.csv`, `detector_intervals.csv`.

### Post hoc: oversampling ablation on all screens and on unseen apps (`detector_ablation.*`)

Does oversampling still help on apps never seen in training? No retraining:
the three checkpoints of the thesis comparison (no oversampling, rare
classes oversampled to 2k and to 5k instances) are evaluated like the
released weights on the same sets (test and val: all screens, unseen full
grouping, unseen package only), and every checkpoint is scored on the same
1,000 `package_md5` draws, so the differences are paired. 95 % BCa
intervals; `detector_ablation.md` has the Table-2-ready rows, the JSON and
CSV every interval under both groupings (they agree on the unseen subsets).

* **Which checkpoint is the baseline.** `imgsz_640.pt` records train data
  `/content/ablation_640/…` (no oversampling), initial weights
  `yolov8s.pt`, no resume; the validation mAP50 stored with it (0.43959)
  is the published baseline value. The `train:` sections of
  `configs/train_{baseline,oversample_2k,oversample_5k}.yaml` (56 keys)
  equal the checkpoints' `train_args`; the three checkpoints' `train_args`
  differ only in `data`, `model` / `resume` (the 5k run was resumed from
  its own `last.pt`), `name` and `project`.
* **Reproduction.** Test mAP50 43.93 / 45.69 / 48.74 % (published 43.89 /
  45.65 / 48.70); val 44.00 / 50.06 / 51.87 % (published 43.96 / 50.27 /
  51.82). The 2k checkpoint itself stores a val mAP50 of 0.50018, so the
  published 2k val value (50.27 %) is not from the epoch of the released
  2k weights; the order of the three is unchanged. `ap.py` agrees with
  YOLO.val to within 3e-6 on every set and checkpoint.
* **All screens.** 5k − baseline mAP50 +0.048 [+0.019, +0.071] on test
  (val +0.079 [+0.052, +0.110]); 5k − 2k +0.031 [+0.012, +0.059] (val
  +0.018 [−0.004, +0.038]); 2k − baseline +0.018 [−0.018, +0.040] (val
  +0.061 [+0.040, +0.090]). At the pipeline's operating point the gain is a
  trade of precision for recall: test P −0.031 [−0.043, −0.021], R +0.037
  [+0.027, +0.048], F1 +0.010 [+0.002, +0.019] (val F1 +0.002 [−0.008,
  +0.012]). Of the oversampled classes RadioButton (+0.138 [+0.006,
  +0.271]), SeekBar (+0.152 [+0.072, +0.254]) and Spinner (+0.115 [+0.054,
  +0.193]) gain on test; Switch +0.178 [−0.060, +0.367] (61 instances on 23
  screens).
* **Unseen apps.** The mean AP50 over the classes with ≥ 20 instances
  (Button, EditText, ImageView, TextView, WebView; on val package-only also
  ListView) is not higher with oversampling: 5k − baseline −0.023 [−0.064,
  +0.032] (test, 169 screens), −0.008 [−0.048, +0.040] (test package-only,
  214), −0.022 [−0.073, +0.012] and −0.022 [−0.060, +0.008] (val, 203 /
  241). mAP50 over the classes present: +0.037 [−0.031, +0.141] and +0.060
  [−0.014, +0.149] on test, +0.033 [−0.086, +0.089] and +0.008 [−0.058,
  +0.049] on val; these points are carried by classes with 1–17 instances.
  Operating point, test unseen: P −0.055 [−0.121, −0.005], R +0.041
  [+0.000, +0.111], F1 −0.003 [−0.046, +0.044]; val unseen F1 −0.037
  [−0.069, −0.010]. Only one oversampled class, WebView, has ≥ 20 unseen
  instances: −0.115 [−0.233, −0.012] (test, 36 instances), +0.067 [−0.043,
  +0.201] (val, 32); the other seven have 0–17 (flagged †; their intervals
  are not informative).
* **Reading.** Oversampling to 5k raises mAP on screens of apps seen in
  training, where the rare-class gains are measured; there is no evidence
  that it helps on apps never seen in training, where the frequent classes
  are flat to slightly worse and the rare classes are too rare to measure.
  One run per configuration: run-to-run variance is not covered.
* Checks (all passed): identical evaluation settings, images, labels and
  resampling units for the three checkpoints; percentile intervals equal to
  those of the evaluation files (18 comparisons); operating-point intervals
  equal to `operating_point_bootstrap`; point, percentile, basic and BCa
  intervals of the 5k model equal to `detector_intervals.json` (36
  comparisons).

## Layout evaluation (`ui2compose-eval-layout`)

Compares the layout tree behind the generated Kotlin with the view hierarchy
of the same ReDraw screen, on whole splits (R1.2, R2.1, R2.2, R2.9). The
module docstring of `src/ui2compose/evaluation/layout_eval.py` is the
normative description; this section restates it.

1. **Reference tree.** The dump is parsed and the labelling of
   `parse_redraw.py` is replayed node by node: the same traversal,
   container-name filter, bounds, area filter, `ALIASES` /
   `short_class_name`, 13-class whitelist and same-class IoU
   de-duplication, with the command-line defaults the labels were written
   with (`--min-area 0.00005 --dedup-iou 0.85`, containers dropped). Each
   replayed label is formatted like a label line, which links every line of
   the label file to its XML node (whole sequence, else line by line, else
   same class within 1 px). A screen with an unmatched line is excluded, so
   the reference leaf set always equals the screen's ground-truth boxes.
   Detection nodes carry the label box (denormalised, snapped to whole
   pixels within 0.01 px, undoing the 6-decimal rounding of the labels).
   All other XML nodes with a label below them are containers; a label
   node with labels below it (a ListView with items) stays an internal node
   labelled by its class.
2. **Generated tree.** `ComposeCodeGenerator.last_layout_tree` (v1.2.0) or
   the XY-cut tree of the v1.1.0 generator. Column / Row become containers,
   leaves detection nodes. A LazyColumn becomes a node labelled with the
   list's class whose children are its items; a containment Box becomes a
   node labelled with the container's class whose children are the Box's
   children without the container's own widget leaf (the mapping the
   evaluation was specified with). v1.1.0 lays out only the leaves of its
   containment forest: containers are lost and a box listed under two
   containers appears twice (`lost`, `duplicates`).
   **This Box mapping is a choice that matters (post hoc sensitivity
   below).** The Kotlin draws `Box { widget; content }`, widget and content
   as siblings, which is how a dump describes a background image under its
   content (a FrameLayout holding both), whereas the mapping scores the
   widget as the parent of its content. Only C1 and C4 have containment
   Boxes; C2, C3, C5 and C1′ are unaffected, so the mapping penalises v1.2.0
   against v1.1.0 and XY-cut against the flat Column.
3. **Canonical form, for both trees.** Containers without a detection below them are pruned; a container with one child is replaced by that
   child (single-child chains collapse; detection nodes are never removed);
   children are sorted in reading order (top, then left of their union
   box; siblings that are x-disjoint but not y-disjoint, a row, left
   first). The root is the top-most remaining container; a tree that
   collapses to one detection or nothing gets one root container.
4. **Labels.** TEDS-g: detections by class, containers 'Group'. TEDS-o:
   containers by the geometry of their children, one rule for both trees,
   tolerance 0 (touching does not overlap): 'Column' if the y-intervals are
   pairwise disjoint, else 'Row' if the x-intervals are, else 'Box'.
5. **Distance.** Unit-cost ordered tree edit distance (APTED, `apted`
   1.0.3); TEDS = 1 − TED / max(|T_ref|, |T_gen|). TEDS can be negative when
   TED exceeds the larger tree (a flat tree against a deep one): on test,
   14 screen-condition pairs for TEDS-o (minimum −0.200) and 4 for TEDS-g
   (minimum −0.162). The optimal mapping is split into inserts, deletes and
   renames (sum = TED, checked on every screen). Zhang-Shasha (`zss` 1.2.0)
   recomputes the first 100 screens of every condition: agreement 600 / 600
   on test and 600 / 600 on val (and 200 / 200 in each sensitivity run).
6. **Grouping F1.** Ground-truth boxes are matched one to one with the
   boxes given to the layout stage (identity for GT boxes; Hungarian on
   IoU ≥ 0.5 for predictions, class not required, class mismatches
   counted). A pair of matched boxes is "grouped" when the two detection
   nodes share their direct parent; F1 = 2·TP / (grouped-in-generated +
   grouped-in-reference), micro-averaged as well. **Its means are over
   different screens per condition.** F1 is undefined (counted, not
   averaged) when neither tree groups a pair: almost always because fewer
   than two boxes were matched (a single-box screen, or a detector that
   found less than two of the boxes), rarely because neither tree groups a
   pair of two or more matched boxes, an agreement that the mean leaves out
   (test, all screens: C1 116 undefined = 93 + 23; C4 429 = 367 + 62). With
   predicted boxes it is computed over the
   boxes the detector found. Every mean below therefore carries its number
   of screens, every paired difference is over the screens where both
   conditions define F1 (n given), and `layout_test.md` repeats the means
   on the screens where every condition defines F1.
7. **Conditions.**

   | key | boxes | layout |
   |-----|-------|--------|
   | C1  | ground truth | v1.2.0 containment + XY-cut (no de-duplication) |
   | C2  | ground truth | flat Column (baseline) |
   | C3  | raw predictions | v1.1.0 graph builder + code generator as submitted (v1.1.0 tag files, SHA-256 checked), no de-duplication |
   | C4  | predictions | v1.2.0 de-duplication, containment, XY-cut (the shipped pipeline) |
   | C5  | predictions | v1.2.0 de-duplication, then one flat Column (baseline) |
   | C1p (C1′) | ground truth | v1.1.0 layout |

8. **Exclusions** are counted per reason (`no_mapping`, `no_label_file`,
   `empty_labels`, `label_parse_error`, `image_unreadable`,
   `xml_parse_error`, `alignment_incomplete`; per condition
   `no_predictions`, `error:generation:<message>`, `error:tree_too_large`
   above 1,500 nodes). A screen with zero predicted boxes is scored. **On
   both splits nothing was excluded**: every val and test screen has a
   mapping, labels and a parseable XML, all 21,658 val and 23,068 test label
   lines were aligned to their XML node as a whole sequence, and every
   condition was evaluated on every screen.
9. **Aggregation** per condition on all screens, the unseen-app subset
   (`dataset/unseen_test.txt`, 169 screens) and the package-only unseen
   subset (214 screens): mean / median / quartiles of TEDS-g, TEDS-o and
   grouping F1, micro grouping F1, edit operations median (IQR), node
   counts, fallback nodes, placeholders (`Text("Unknown: …")`),
   duplicates, lost boxes, de-duplication removals; intervals as above.

Choices beyond the written specification, each applied to both trees and
every condition: rows are read left to right (a strict top-then-left order
scrambles a row of predicted boxes whose tops differ by 1 px); ground-truth
coordinates are snapped within 0.01 px (else touching boxes overlap by
0.001 px and a stack of them is labelled 'Box'); `package_md5` and `screen`
intervals in addition to `full`; trees above 1,500 nodes are not scored
(none occurred).

**Reference flatness and single-box screens.** Dumps whose canonical tree
has depth ≤ 2 (every box directly under the root) are "flat": 317 / 1,870
test screens (17.0 %), 348 / 1,869 val (18.6 %). The unseen subsets are much
flatter and smaller: 89 / 169 (52.7 %, median 3 boxes and 5 nodes) and
104 / 214 (48.6 %). Part of that is screens with a single ground-truth box,
on which every tree is one root over one box and C1 = C2 by construction:
47 of the 169 unseen screens (27.8 %; 47 of the 89 flat ones), 55 of the 214
package-only screens (25.7 %), 93 of all 1,870 test screens (5.0 %). On a
flat reference a flat Column is already the right structure, and XY-cut,
which nests, loses against it (see the strata below).

## Threshold selection on the validation split (`layout_sweep`)

Files: `val_sweep.md`, `val_sweep.json`, `selected_thresholds.json`,
`val_sweep_crossval.json`. The method (module docstring of
`layout_sweep.py`) was fixed before the sweep; only val was used.

* **Grid.** C4 for every combination of de-duplication (off / stricter
  0.6/0.95/0.7 / a priori 0.5/0.9/0.6 / looser 0.4/0.85/0.5),
  `containment_ratio` (1.0 / 0.95 / 0.9), `cut_tol_frac` (0 / 0.0025 /
  0.005 / 0.01 / 0.02) and `fallback` (`y_sorted` / `least_overlap`): 120
  configurations, each on all 1,869 val screens. C1 (30 layout settings)
  and C5 (4 de-duplication settings) for context.
* **Detector check.** A de-duplication setting is eligible only if the
  operating-point F1 (conf > 0.3, IoU 0.5) drops by at most 0.5 points.
  All three settings raise it (0.5618 without, 0.5638–0.5652 with).
* **Rule.** Maximum mean TEDS-o of C4; ties within 0.001 go to the higher
  mean grouping F1 (within 0.001), then to the setting closest to the a
  priori one.
* **Result (shipped).** De-duplication 0.4 / 0.85 / 0.5, containment 0.9,
  `cut_tol_frac` 0.0025, `y_sorted`: mean TEDS-o 0.5086 vs 0.5053 a priori,
  paired +0.0033 [+0.0018, +0.0072] (`full`) / [+0.0015, +0.0054]
  (`package_md5`). **This difference is in-sample**: it is measured on the
  screens on which the configuration won among 120, so it overstates the
  gain on new screens. On the val unseen subsets the selection changes
  TEDS-o by +0.003 but lowers grouping F1 by 0.016–0.019.
* **Out-of-sample check (post hoc, `val_sweep_crossval.json`).** The rule
  applied to the means on one random half of the 1,590 `package_md5` app
  groups and scored on the other half (200 splits, both directions, every
  de-duplication setting eligible): the selected configuration beats the a
  priori one on the unseen half by +0.0019 TEDS-o on average (2.5–97.5 % of
  the 400 selections: −0.0006 to +0.0043; positive in 92 %), against
  +0.0041 on the half it was selected on (optimism +0.0022). The selection
  is unstable: the shipped configuration wins on only 10.5 % of the halves;
  the three more frequent winners differ from it in one to three of the
  four parameters. The gain is small, about 0.002 TEDS-o out of sample.
* **Grid edges.** The selected de-duplication setting (looser) is the
  loosest one tried and `containment_ratio` 0.9 the smallest ratio tried;
  the grid does not show whether values beyond them would be better.
* **Grid summary.** C4 spans mean TEDS-o 0.456–0.509; de-duplication
  matters most (best without it 0.484), then strict containment (1.0:
  best 0.503) and large tolerances (0.02: best 0.500).

The val numbers of every condition with the selected thresholds are in the
run `$OUT/layout_val` (report `$OUT/layout_val/report/layout_val.md`, with the
same post-hoc sections as on test); they are in-sample for the thresholds
and are not used in the paper's tables.

## Test results (`layout_test.md`, `layout_test.json`)

Mean with 95 % CI. Intervals: `full` grouping / `package_md5` grouping for
all screens (see "Confidence intervals"); `full` only for the unseen subsets
(the two groupings agree there; both are in `layout_test.md`). Grouping F1:
mean over the screens where it is defined, their number in parentheses.
Edit operations: TEDS-o mapping, median (IQR); nodes: generated / reference,
median.

**All 1,870 test screens**

| | TEDS-g | TEDS-o | grouping F1 (n) | edit ops | nodes |
|---|---|---|---|---|---|
| C1 GT + XY-cut | 0.825 [0.810, 0.832] / [0.816, 0.834] | 0.803 [0.786, 0.811] / [0.793, 0.814] | 0.748 [0.722, 0.759] / [0.736, 0.761] (1,754) | 1 (0–5) | 10 / 11 |
| C2 GT + flat Column | 0.709 [0.699, 0.736] / [0.699, 0.718] | 0.649 [0.636, 0.681] / [0.638, 0.660] | 0.454 [0.444, 0.484] / [0.440, 0.469] (1,777) | 4 (2–9) | 8 / 11 |
| C3 v1.1.0 as submitted | 0.539 [0.506, 0.554] / [0.527, 0.550] | 0.501 [0.469, 0.515] / [0.488, 0.513] | 0.649 [0.598, 0.667] / [0.632, 0.665] (1,437) | 6 (2–13) | 8 / 11 |
| C4 v1.2.0 pipeline | 0.549 [0.515, 0.564] / [0.535, 0.562] | 0.506 [0.476, 0.518] / [0.492, 0.519] | 0.698 [0.664, 0.710] / [0.680, 0.715] (1,441) | 6 (2–13) | 8 / 11 |
| C5 pred + flat Column | 0.484 [0.464, 0.492] / [0.472, 0.495] | 0.416 [0.400, 0.424] / [0.404, 0.428] | 0.448 [0.436, 0.480] / [0.431, 0.466] (1,503) | 7 (3–14) | 7 / 11 |
| C1′ GT + v1.1.0 layout | 0.810 [0.800, 0.818] / [0.802, 0.818] | 0.788 [0.776, 0.797] / [0.779, 0.797] | 0.736 [0.696, 0.751] / [0.722, 0.748] (1,754) | 2 (0–5) | 9 / 11 |

**Unseen apps (full grouping, 169 screens)** and **unseen packages (214 screens)**, `full` CIs

| | TEDS-g (169) | TEDS-o (169) | grouping F1 (169: n) | TEDS-g (214) | TEDS-o (214) | grouping F1 (214: n) |
|---|---|---|---|---|---|---|
| C1 | 0.855 [0.818, 0.890] | 0.831 [0.786, 0.871] | 0.774 [0.720, 0.825] (122) | 0.839 [0.803, 0.871] | 0.812 [0.768, 0.850] | 0.763 [0.722, 0.806] (159) |
| C2 | 0.852 [0.821, 0.881] | 0.819 [0.784, 0.851] | 0.622 [0.566, 0.680] (122) | 0.829 [0.799, 0.860] | 0.798 [0.767, 0.831] | 0.593 [0.547, 0.644] (159) |
| C3 | 0.471 [0.428, 0.512] | 0.439 [0.398, 0.483] | 0.626 [0.542, 0.712] (80) | 0.485 [0.439, 0.529] | 0.456 [0.408, 0.503] | 0.604 [0.532, 0.671] (108) |
| C4 | 0.476 [0.434, 0.518] | 0.439 [0.398, 0.482] | 0.756 [0.687, 0.823] (78) | 0.494 [0.447, 0.540] | 0.456 [0.409, 0.502] | 0.719 [0.654, 0.777] (106) |
| C5 | 0.485 [0.441, 0.526] | 0.411 [0.366, 0.456] | 0.648 [0.579, 0.723] (81) | 0.497 [0.456, 0.535] | 0.427 [0.383, 0.467] | 0.586 [0.520, 0.660] (110) |
| C1′ | 0.865 [0.837, 0.893] | 0.840 [0.804, 0.872] | 0.726 [0.669, 0.778] (122) | 0.849 [0.821, 0.875] | 0.821 [0.787, 0.853] | 0.724 [0.670, 0.769] (159) |

The grouping-F1 column compares different screens: on the 169 unseen
screens C1 is averaged over 122 and C4 over 78 (the screens where the
detector matched at least two boxes and a tree groups a pair). On the 77
unseen screens where every condition defines F1: C1 0.811 [0.753, 0.867],
C2 0.579, C3 0.650, C4 0.766 [0.696, 0.829], C5 0.681, C1′ 0.716; on the
1,417 such screens of all test screens: C1 0.764, C2 0.437, C3 0.656, C4
0.709, C5 0.475, C1′ 0.749 (`layout_test.md`, "Grouping F1 on common
screens").

**Paired differences** (mean per-screen difference; all screens: `full` /
`package_md5` CI; unseen 169: `full` CI). TEDS differences are over every
screen of the subset; grouping-F1 differences over the screens where both
conditions define F1 (n).

| | TEDS-g, all | TEDS-o, all | grouping F1, all (n) | TEDS-g, unseen | TEDS-o, unseen | grouping F1, unseen (n) |
|---|---|---|---|---|---|---|
| C1 − C2 (XY-cut vs flat, GT boxes) | +0.115 [+0.083, +0.130] / [+0.105, +0.127] | +0.154 [+0.114, +0.171] / [+0.140, +0.168] | +0.288 [+0.243, +0.307] / [+0.273, +0.304] (1,754) | +0.003 [−0.024, +0.030] | +0.012 [−0.025, +0.050] | +0.152 [+0.086, +0.218] (122) |
| C4 − C5 (XY-cut vs flat, predictions) | +0.065 [+0.045, +0.073] / [+0.058, +0.071] | +0.090 [+0.070, +0.098] / [+0.082, +0.098] | +0.230 [+0.184, +0.247] / [+0.212, +0.246] (1,441) | −0.009 [−0.024, +0.005] | +0.028 [+0.009, +0.046] | +0.084 [+0.027, +0.140] (78) |
| C4 − C3 (v1.2.0 vs v1.1.0) | +0.010 [+0.004, +0.015] / [+0.004, +0.015] | +0.005 [+0.001, +0.013] / [−0.001, +0.011] | +0.053 [+0.042, +0.084] / [+0.039, +0.067] (1,417) | +0.005 [−0.013, +0.021] | +0.000 [−0.019, +0.019] | +0.116 [+0.059, +0.176] (77) |
| C1 − C4 (GT vs predicted boxes) | +0.276 [+0.264, +0.305] / [+0.264, +0.290] | +0.297 [+0.287, +0.322] / [+0.284, +0.312] | +0.064 [+0.053, +0.087] / [+0.050, +0.077] (1,441) | +0.379 [+0.330, +0.432] | +0.391 [+0.337, +0.447] | +0.053 [−0.006, +0.115] (78) |
| C1 − C1′ (v1.2.0 vs v1.1.0 layout, GT) | +0.015 [+0.004, +0.020] / [+0.009, +0.021] | +0.015 [+0.004, +0.021] / [+0.008, +0.021] | +0.013 [+0.003, +0.037] / [+0.001, +0.023] (1,754) | −0.010 [−0.027, +0.007] | −0.009 [−0.029, +0.010] | +0.048 [+0.008, +0.090] (122) |

The grouping-F1 differences involving predicted boxes (C4 − C3, C1 − C4,
C4 − C5) are conditioned on what the detector found: they compare the
grouping of the boxes both trees contain, not the share of the screen that
is laid out correctly.

Other counts on all test screens (`layout_test.md`, second table):

* Predicted boxes: 55.5 % of the ground-truth boxes are matched (IoU ≥ 0.5)
  after de-duplication, 71.3 % of the boxes given to the layout are matched;
  835 matched pairs have the wrong class. De-duplication removes 1,832 of
  19,799 raw detections.
* v1.1.0 (C3) emits 1,391 duplicated boxes, loses 1,961 boxes (containers)
  and 459 `Text("Unknown: …")` placeholders on 312 screens; v1.2.0 (C4) has
  none of these. Fallback nodes (no cut existed): C4 916 on 631 screens, C3
  2,826 on 969 screens, C1 707 on 429 screens.
* Micro grouping F1, all screens (`full` CI): C1 0.627 [0.566, 0.683], C4
  0.610 [0.530, 0.651], C3 0.548 [0.446, 0.589], C2 0.166 [0.117, 0.217].

### Post hoc: differences by stratum of the reference

Paired differences on the screens of a stratum, same draws restricted to it
(`full` CI; `layout_test.md` has every stratum and subset).

| subset | stratum | n | C1 − C2 TEDS-g | C1 − C2 TEDS-o | C4 − C5 TEDS-g | C4 − C5 TEDS-o |
|---|---|---:|---|---|---|---|
| all | flat dump | 317 | −0.060 [−0.083, −0.041] | −0.086 [−0.116, −0.058] | −0.033 [−0.043, −0.025] | +0.005 [−0.007, +0.021] |
| all | not flat | 1,553 | +0.151 [+0.122, +0.163] | +0.203 [+0.167, +0.217] | +0.085 [+0.066, +0.092] | +0.107 [+0.085, +0.116] |
| unseen | flat dump | 89 | −0.051 [−0.086, −0.020] | −0.073 [−0.124, −0.029] | −0.035 [−0.055, −0.020] | +0.019 [−0.008, +0.043] |
| unseen | not flat | 80 | +0.064 [+0.023, +0.106] | +0.106 [+0.052, +0.159] | +0.020 [−0.002, +0.043] | +0.038 [+0.011, +0.064] |
| unseen | ≥ 2 GT boxes | 122 | +0.005 [−0.032, +0.042] | +0.016 [−0.034, +0.068] | −0.006 [−0.027, +0.012] | +0.016 [−0.007, +0.037] |
| package-only | not flat | 110 | +0.070 [+0.025, +0.108] | +0.103 [+0.046, +0.152] | +0.028 [+0.003, +0.048] | +0.036 [+0.011, +0.058] |

On the unseen subset, the near-zero C1 − C2 is not only the single-box
screens (C1 = C2 there by construction): the 42 flat dumps with two or more
boxes favour the flat Column, and on the non-flat dumps XY-cut wins
(+0.064 TEDS-g), less than on all non-flat test screens (+0.151).

### Post hoc: scoring containment Boxes as containers (sensitivity)

C1 and C4 re-scored with `--box-repr container`: a containment Box becomes a
container whose children are the container's widget leaf and the content
(as the Kotlin draws it and as a dump describes a background under its
content); LazyColumns, the references and all other conditions are
unchanged. Test, `full` CIs (`package_md5` where it matters; both in
`layout_test.md`):

| | primary (composite) | sensitivity (container) | screens changed |
|---|---|---|---:|
| C1 TEDS-g / TEDS-o | 0.825 / 0.803 | 0.847 [0.836, 0.854] / 0.833 [0.820, 0.840] | 430 |
| C4 TEDS-g / TEDS-o | 0.549 / 0.506 | 0.559 [0.525, 0.573] / 0.510 [0.480, 0.523] | 501 |
| C4 − C3 TEDS-g, all | +0.010 [+0.004, +0.015] | +0.020 [+0.014, +0.026] / [+0.014, +0.025] | |
| C4 − C3 TEDS-o, all | +0.005 [+0.001, +0.013] / [−0.001, +0.011] | +0.009 [+0.005, +0.018] / [+0.003, +0.016] | |
| C1 − C1′ TEDS-g, all | +0.015 [+0.004, +0.020] | +0.038 [+0.029, +0.043] | |
| C1 − C2 TEDS-g, all | +0.115 [+0.083, +0.130] | +0.138 [+0.111, +0.151] | |
| C1 − C2 TEDS-g, unseen | +0.003 [−0.024, +0.030] | +0.029 [+0.011, +0.049] | |
| C4 − C5 TEDS-g, unseen | −0.009 [−0.024, +0.005] | +0.002 [−0.010, +0.014] | |
| C1 − C4 TEDS-g, all | +0.276 | +0.289 [+0.275, +0.320] | |

On val (in-sample for the thresholds) the same re-score gives C1 0.826 →
0.845 TEDS-g and 0.805 → 0.829 TEDS-o (406 screens changed), C4 0.554 →
0.563 and 0.509 → 0.514 (462), C4 − C3 +0.011 → +0.019 TEDS-g. The
container scoring is not uniformly closer to the dumps: of the 430 changed
C1 test screens 309 improve and 101 get worse (C4: 256 and 130 of 501). On
val, the Boxes of the screens that improve are mostly ImageView containers
(backgrounds under content); those of the screens that get worse are mostly
Spinner and Button containers, which the dump describes as parents, as the
primary mapping scores them. The primary metric stays as specified; the
conclusions that depend on the choice are marked below.

## Correction-effort proxy (structure only)

The number of edit operations (insert + delete + rename = TED) that turn the
generated layout tree into the reference tree. It ignores texts, sizes,
styles, modifiers and the Kotlin itself, so it is a lower bound of the work
on the structure, not an estimate of developer time. Test split
(`layout_test.md`, last table; `full` / `package_md5` CIs):

| subset | condition | edits TEDS-o, median (IQR) | share 0 edits | share ≤ 5 edits | edits per reference node, median |
|---|---|---|---|---|---|
| all (1,870) | C1 | 1 (0–5) | 0.366 | 0.764 [0.741, 0.782] / [0.744, 0.784] | 0.12 |
| all (1,870) | C4 | 6 (2–13) | 0.098 | 0.483 [0.459, 0.506] / [0.460, 0.505] | 0.55 |
| unseen (169) | C1 | 0 (0–3) | 0.586 | 0.858 [0.799, 0.912] | 0.00 |
| unseen (169) | C4 | 4 (1–8) | 0.130 | 0.604 [0.526, 0.675] | 0.67 |
| package-only (214) | C1 | 0 (0–3) | 0.542 | 0.827 [0.774, 0.878] | 0.00 |
| package-only (214) | C4 | 4 (1–8) | 0.136 | 0.594 [0.532, 0.659] | 0.67 |

With the TEDS-g mapping: C1 median 1 (0–5), 78.1 % ≤ 5; C4 median 5
(2–12), 51.4 % ≤ 5. For C4 most edits are inserts (median 3 of 6: boxes and
groups the generated tree lacks). The unseen screens are smaller (median 5
reference nodes; 28 % have a single box), so "≤ 5 edits" is easier there
while the edits per node are higher (0.67 vs 0.55). The proxy uses the
primary Box scoring.

## Failure taxonomy (`failure_taxonomy.md`, `failure_taxonomy.json`)

**Method** (module docstring of `failure_taxonomy.py`; rules fixed on 50
random val screens). 50 test screens drawn with
`numpy.random.default_rng(42).choice` from the sorted test stems, evaluated
under C4 exactly as above (TED and both trees cross-checked against the
test run: 50 / 50 identical). Every edit of the TEDS-o mapping gets one
category by the first matching rule; among equal-cost optimal mappings the
one pairing the most IoU-matched boxes is used. The rules use the box
matching (with and without de-duplication), the XML of the dump (window
bounds, later-drawn siblings, drawers) and the screenshot (grey-level
standard deviation inside a ground-truth box). Families: detector
(`detector_fp`, `detector_fn`, `wrong_class`), layout rules
(`dedup_removed`, `containment`, `xycut_fallback`, `tolerance_cut`,
`grouping` = residual: XY-cut groups differently from the hierarchy),
reference (`reference_overlap`: hidden or stacked nodes; `reference_flat2d`:
RelativeLayout-style 2-D containers that XY-cut, like Compose, nests;
`reference_artifact`: the dump does not describe the pixels).

**Counts** (457 edits on 50 screens; median 6 per screen, IQR 2–14):

| category | edits | share | dominant on | screens with ≥ 1 |
|---|---:|---:|---:|---:|
| detector_fn | 157 | 34.4 % | 20 | 32 |
| detector_fp | 97 | 21.2 % | 11 | 26 |
| wrong_class | 9 | 2.0 % | 0 | 7 |
| dedup_removed | 2 | 0.4 % | 0 | 2 |
| containment | 15 | 3.3 % | 0 | 9 |
| xycut_fallback | 38 | 8.3 % | 2 | 16 |
| tolerance_cut | 25 | 5.5 % | 2 | 11 |
| grouping | 34 | 7.4 % | 3 | 14 |
| reference_overlap | 26 | 5.7 % | 1 | 8 |
| reference_flat2d | 15 | 3.3 % | 1 | 5 |
| reference_artifact | 39 | 8.5 % | 7 | 12 |
| none (perfect screen) | – | – | 3 | – |

Families: detector 263 edits (57.5 %, dominant on 31 screens), layout rules
114 (24.9 %, 7 screens), reference 80 (17.5 %, 9 screens). The taxonomy
uses the primary Box scoring; the `containment` category depends on it and
was not recomputed under the sensitivity variant.

**Post-hoc check (not part of the fixed rules).** 28 of the 119 unmatched
detections on these screens (24 %) coincide (IoU ≥ 0.5) with an XML node
that carries no label, mostly `FrameLayout` (9), `android.view.View` (8)
and `ToggleButton` (6): visible widgets whose class is outside the 13
classes or not aliased. Part of `detector_fp` is therefore a gap of the
reference labels rather than a detector error.

**Illustrative cases** (traced with `ui2compose <screenshot> --explain <id>`;
raw detections identical to the cache in all three):

1. `si.modula.android.instantheartrate-screens_screenshot_3` (detector:
   `detector_fp`, 3 of 3 edits, TEDS-o 0.625). The two gender icons are
   detected as ImageViews, but the dump draws them as `android.view.View`,
   which is not labelled, so the reference has no box there. Trace:
   `[detection] ImageView conf=0.77 bbox=[226, 785, 427, 992]`, kept, top-level,
   placed by `Row (rule=gap, axis=x, 1 cut)`. Correct pixels, penalised by
   the label set.
2. `com.andromo.dev535138.app530249-screens_screenshot_2` (layout rules:
   `grouping`, 2 of 3 edits, TEDS-o 0.700). The toolbar's refresh and
   overflow buttons sit in their own LinearLayout in the reference; XY-cut
   cuts the toolbar into four groups at one level (`Row (rule=gap, axis=x,
   3 cuts, tolerance 3 px)`), and a false-positive ImageView (an image
   inside the WebView-rendered article) adds a root Column; the rules
   attribute that root Column to `grouping` too, although the false
   positive causes it.
3. `com.andromo.dev442382.app431230-screens_screenshot_2` (reference:
   `reference_artifact`, 1 of 1 edits, TEDS-o 0.750). The dump holds only
   the open overflow menu (window [776, 146, 1200, 274]); the detector also
   finds the action-bar title underneath (`[detection] TextView conf=0.38
   bbox=[834, 83, 947, 131]`), 0 % inside the dumped window.

The case rule (one screen per family, the most frequent dominant category,
the highest share of edits) was fixed before the test run; after it, the
condition "the dominant category has an edit with a detection to trace" was
added, because the first reference-family case
(`com.wageworks.ezreceipts-screens_screenshot_3`, an undetected ImageView
over uniform pixels) had nothing to `--explain`. No count changed.

**Limits of the taxonomy.** The attribution is heuristic and local: one
cause per edit, decided from geometry, the dump and pixel statistics. It
was checked by eye on val screens and on the three test cases, not by a
blind manual annotation. `reference_overlap` is an upper bound of
occlusion; `detector_fp` includes unlabelled visible widgets (above);
`grouping` is a residual that also absorbs consequences of other errors
(case 2); 50 screens give counts, not precise shares.

## Findings in brief

* With ground-truth boxes, XY-cut recovers much of the developers' grouping
  (C1 TEDS-g 0.825, grouping F1 0.748 on 1,754 screens) and clearly beats a
  flat Column on all test screens (+0.115 TEDS-g, +0.288 grouping F1); on
  the non-flat dumps by +0.151 TEDS-g, while on the 17 % flat dumps the
  flat Column is better (−0.060).
* On the unseen-app subsets half the dumps are flat (52.7 % of 169, 48.6 %
  of 214), and 28 % (47 of 169) have a single box, where C1 = C2 by
  construction. There, with the primary scoring, TEDS does not separate
  XY-cut from a flat Column (C1 − C2 +0.003 [−0.024, +0.030] TEDS-g, +0.012
  [−0.025, +0.050] TEDS-o), while grouping F1 does (+0.152 [+0.086,
  +0.218], 122 screens); on the 80 non-flat unseen dumps XY-cut wins
  (+0.064 [+0.023, +0.106] TEDS-g). With containment Boxes scored as
  containers (post hoc) C1 − C2 on unseen is +0.029 [+0.011, +0.049].
* Detection dominates the error: replacing ground-truth by predicted boxes
  costs 0.276 TEDS-g on all screens and 0.379 on unseen apps; only 55.5 % of
  the ground-truth boxes are found. In the taxonomy 57.5 % of the edits are
  detector errors (34 % missed boxes, 21 % unmatched detections, part of
  them unlabelled widgets).
* v1.2.0 removes the v1.1.0 output defects (1,391 duplicated and 1,961 lost
  boxes, 459 placeholders). Its TEDS gain over the submitted v1.1.0 pipeline
  is small: +0.010 TEDS-g, and +0.005 TEDS-o with a `package_md5` interval
  that includes 0; with Boxes scored as containers +0.020 TEDS-g and +0.009
  TEDS-o (both intervals above 0). On the boxes both trees contain it groups
  better (grouping F1 +0.053, 1,417 screens; +0.116 on 77 unseen screens).
* With predicted boxes on unseen apps, the full pipeline (C4) against a flat
  Column of the same detections (C5): TEDS-g −0.009 [−0.024, +0.005], TEDS-o
  +0.028 [+0.009, +0.046], grouping F1 +0.084 [+0.027, +0.140] (78
  screens). TEDS-g does not separate them, TEDS-o and grouping F1 favour
  C4.
* Structure-only correction: half of the C4 test screens need more than 5
  edits (median 6); with ground-truth boxes three quarters need at most 5.
* The threshold selection gains little: +0.0033 TEDS-o in-sample, about
  +0.002 out of sample, with two selected values on the edge of the grid.
* Detector without the app leak: on the 169 unseen-app test screens the mean
  AP50 over the five classes with ≥ 20 instances is 0.434 (BCa [0.339,
  0.555]) against 0.573 on seen-app screens (difference −0.139 [−0.245,
  −0.023]). Their mAP50 is 0.289 (BCa [0.186, 0.365]) against 0.487 on all
  test screens; that mAP averages classes with 0–10 instances, and its
  percentile interval [0.229, 0.442] is biased upward.
* Oversampling (post hoc ablation, three checkpoints, paired): +0.048
  [+0.019, +0.071] test mAP50 over no oversampling on all screens, but on
  the unseen-app subsets the fixed-class mean AP50 does not improve (−0.023
  [−0.064, +0.032] on 169 test screens, −0.008 [−0.048, +0.040] on 214) and
  the pipeline's F1 is flat (test) to lower (val).
