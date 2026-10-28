# Changelog

Reviewer points of the SoftwareX revision are given in brackets (R1.x = reviewer 1,
R2.x = reviewer 2). Every entry below is implemented and covered by the test suite.

## [1.2.0] – 2026-10-28

### Default detector weights: retrained on an app-level split [R1.1, R2.5, R2.6]
- ReDraw is re-split by app (`ui2compose-app-split --resplit`; groups = same package, identical
  file, or dHash distance ≤ 4; greedy multi-label stratification, seed 42): 14,866 / 1,928 /
  1,900 screens of 4,354 / 642 / 632 packages. An audit of the data package finds no group,
  package, MD5 or near-duplicate in two splits (`results/eval_appsplit/package_verification.json`).
- YOLOv8s is retrained on it with the training arguments of the v1.1.0 model
  (`configs/train_baseline.yaml`, `configs/train_oversample_5k.yaml`; 60 epochs from COCO
  `yolov8s.pt`), with and without oversampling. **The default weights of the pipeline, the
  config file, the demo and the evaluation commands are now
  `runs/oversample_5k_appsplit/weights/best.pt`** (MD5 4d5763e8a4887215f828a3a92cba2b3c; with
  its `args.yaml` and `results.csv`). Test mAP@50 0.442 (BCa 95 % CI [0.395, 0.480], app
  groups), 0.404 [0.370, 0.432] without oversampling; validation 0.421 / 0.381
  (`results/eval_appsplit/`).
- The app-split baseline without oversampling, the comparison model of Table 2, is shipped as
  `runs/baseline_appsplit/weights/best.pt` (SHA-256
  66efdfcf5a48f7665baffdba811dcb574b5020100513ede1decdff1166656e67, MD5
  a4cb282dd1d6e0233c3b451945ee96d8; with its `args.yaml` and `results.csv`), copied from
  `weights/baseline_app/` of the data package (hash checked against its `train_summary.json`),
  so both checkpoints of the app-split evaluation are in the repository.
- The app-split data package (without its 3,828 val and test images, which are ReDraw
  screens identified by MD5) and the per-screen outputs of its evaluation are assets of the
  v1.2.0 release. The `manifest_sha256` of its `splits/split_fingerprints.json` hashes the
  sorted `<stem>,<split>` lines, not the CSV bytes; no split assignment changed
  (`results/eval_appsplit/README.md`, `scripts/appsplit_checks.py`).
- `runs/oversample_5k/weights/best.pt`, the v1.1.0 weights, stays in the repository to
  reproduce the v1.1.0 numbers and the screenshot-split evaluation of `results/eval/`
  (pass `--model runs/oversample_5k/weights/best.pt`). It must not be evaluated on the
  app-level test split, 1,512 of whose 1,900 screens were in its training split.
- The bundled examples, golden files, harness renders and the figures are regenerated with the
  default weights. The worked example of the figures and of `docs/example_detections.png` is
  now the drivesafe screen, a test screen of the app-level split; of the ten examples, eight
  are training screens of the app-level split, battlesimulator is a validation screen.

### Detection post-processing [R1.3, R2.4]
- New de-duplication step (`infer/postprocess.py`, `deduplicate_detections`) with three
  explicit rules: same class with IoU ≥ 0.4; same non-nesting class (e.g. `TextView`,
  `Button`) with intersection / smaller area ≥ 0.85; different classes with IoU ≥ 0.5. The
  lower-confidence box is removed. Thresholds are named parameters, selected on the
  validation split (see "Threshold selection" below), and can be disabled one by one. Every removal is written to `1_dedup_removed.json` with
  its rule, overlap value and the id of the box that caused it, and drawn dashed grey in
  `1_detections_viz.png`. On the ten bundled examples it removed 10 of 150 boxes with the
  v1.1.0 weights (8 of 141 with the default app-split weights), among them
  a duplicate `ImageView`, a duplicate `TextView` and a spurious `TextView` of the Shazam
  example.
- Optional per-class confidence filter `class_conf` (off by default; values to be chosen on
  the validation split). Its class names are checked against the detector's classes before
  detection; a misspelt name stops the run instead of filtering nothing.
- The detector settings `iou_threshold`, `agnostic_nms` and `max_det` are now passed to the
  model, and `imgsz` when it is given; by default it is not passed, so Ultralytics uses the
  training size stored in the checkpoint, as in v1.1.0 (640 for the released weights). v1.1.0
  passed only the confidence; its config file listed an NMS IoU of 0.5 that was never used
  (0.7 ran). The config now says 0.7, and the raw detections are identical to v1.1.0 on all
  ten examples.

### UI graph [R1.3, R2.3, R2.4, R2.8]
- Containment is a forest: a box is contained in another when at least
  `containment_ratio` = 0.9 (selected on the validation split) of its area lies inside it and the other box
  ranks higher (area, confidence, id); each box gets exactly one parent, the smallest
  container. v1.1.0 listed a child of two partly overlapping parents twice (Shazam ids 10
  and 11) and lost both of two identical boxes.
- Spatial edges use a tolerance relative to the image width (`align_tol_frac` = 20/1200,
  i.e. v1.1.0's 20 px at 1200 px), so a screen at another resolution gives the same graph.
  They connect boxes with the same containment parent.
- The role of the graph parts is documented and tested: code generation reads only the
  containment forest; spatial edges and synthetic row/column groups are an inspection export
  (removing or shuffling them leaves the Kotlin byte-identical).
- `explain_containment` explains the parent of a node, including near misses.
- `build_graph` accepts plain box lists (`bbox` + `class_name`) and records the parameters
  and image size it used.
- API: `UIGraphBuilder` takes keyword arguments only; `vertical_threshold` /
  `horizontal_threshold` are deprecated (mapped to `align_tol_px`); the unused
  `iou_threshold` argument and `SIBLING` relation are removed.

### Layout and code generation [R1.3, R1.5, R2.4, R2.7]
- Every detection that reaches code generation is emitted exactly once. Containers are no
  longer dropped: a `ListView` / `RecyclerView` with children becomes a `LazyColumn` (one
  `item` per group), any other container a `Box` with its own widget and its children.
- XY-cut accepts cuts with an overlap of less than `cut_tol_frac` of the image extent (a named
  parameter; default 0, the v1.1.0 rule, re-selected on the app-split validation split; the
  screenshot-split selection had chosen 0.0025, see "Threshold selection"). Where no cut exists, the fallback
  is explicit and logged: `y_sorted` (v1.1.0 behaviour, the default) or the new
  `least_overlap`.
- Templates for the seven former placeholder classes: `RadioButton`, `SeekBar` (`Slider`),
  `ProgressBar` (linear or circular by box aspect), `Spinner` (`OutlinedButton` +
  `DropdownMenu`), `WebView` (`AndroidView`), and lists without detected items. Only class
  names outside the 13 detector classes still get an `Unknown` placeholder.
- Layout trace: `2_layout_tree.json` records, for every node, its rule and the geometric
  evidence (cut axis, gap intervals, tolerance, fallback, containing detection);
  `explain_path` and `ui2compose --explain ID` turn it into a readable trace through
  de-duplication, containment and layout (`2_explain_<id>.txt`).
- `layout='flat'` baseline (one `Column` ordered by y-centre, containment ignored) and
  layout of plain box lists, for a later quantitative layout evaluation [R1.2, R2.1, R2.2].
- Explicit imports instead of wildcards; full-width templates, and containment `Box`es,
  inside a `Row` use `weight(1f)`, so that no `Row` child takes the whole width and hides its
  siblings (a test checks this on every fixture and example); optional box-derived sizes
  (`size_from_boxes`); optional trace comments.
- In traces, a cut is printed as `[a, b]` (end of the earlier group, start of the next);
  overlapping cuts are printed with two decimals and an overlap that equals `a − b` of the
  printed numbers.
- `pipeline.layout_from_boxes(boxes, image_size)` lays out plain boxes (e.g. ground truth)
  with containment, exactly as the pipeline lays out its detections.
- New command `ui2compose-codegen` (Kotlin, layout tree and explanations from a saved graph).
- Removed: the unreachable v1.1.0 recursive emitter with its `Card` / `CardView` /
  `Toolbar` branches.

### Pipeline and command line [R1.5, R1.6]
- New outputs `0_run_config.json` (every setting, with defaults resolved to the values used,
  including the de-duplication thresholds, `cut_tol_frac` and `imgsz_effective`),
  `1_detections_raw.json`, `1_dedup_removed.json`, `2_layout_tree.json`,
  `2_explain_<id>.txt`.
- New flags `--config`, `--iou`, `--imgsz`, `--agnostic-nms`, `--max-det`, `--no-dedup`,
  `--layout`, `--size-from-boxes`, `--explain`; every threshold can be set in
  `configs/pipeline_config.yaml`, whose unknown keys are rejected.
- torch and ultralytics are imported only when a model is loaded.

### Demo and figures [R1.6]
- The Streamlit demo runs the command-line pipeline, so both give the same output.
- Its layout overlay draws the layout tree the Kotlin was generated from (`Column`, `Row`,
  `Box`, `LazyColumn`). The legend entry "Card" is removed: it referred to the legacy
  `CardView` class, which the 13-class detector cannot produce.
- New: number and table of de-duplicated detections, an Explain tab, previews for every
  code template, the containment forest and layout tree; the adjacency matrix maps ids to
  rows correctly when ids have gaps.
- Detection figures: one fixed colour per class, labels `#id Class confidence` (the id links
  the figure to the `--explain` traces) on a filled background sized to the image width and
  placed to avoid each other. `docs/example_detections.png` is regenerated.
- `scripts/make_figures.py` draws the five figures of the paper into `docs/figures/` from the
  example outputs, the compile harness renders and the evaluation results (Fig. 5: baseline
  vs oversampled to 5k on the app-level split, `results/eval_appsplit/detector_table2.json`;
  `docs/figures/README.md`).
- `scripts/make_mllm_side_by_side.py` redraws `results/mllm_claude/side_by_side*.png` (input,
  ui2compose render, Claude run-1 render) from the current renders, after checking each render
  against `results/compile_report.json` as Fig. 4 does; `results/mllm_claude/ANALYSIS.md`
  describes the current default output. The per-call Claude outputs these scripts read were
  lost before the release and are not included (`results/mllm_claude/README.md`); Fig. 4 is
  the committed image and `make_figures.py` skips it.

### Data split, evaluation and training [R1.1, R2.5, R2.6, R1.6]
- `ui2compose-app-split` groups screens into apps (same package, identical file, or
  near-identical image) and writes a manifest; it makes and audits the app-level split the
  default weights are trained on (`--resplit`, `ui2compose-build-dataset --split-by app`), and
  selects the unseen-app validation and test subsets of the screenshot-level split, on which
  the v1.1.0 weights were evaluated first (`results/eval/`). The default split of
  `ui2compose-build-dataset` is still the screenshot-level split of v1.1.0. `summary.json` records a SHA-256 of every
  split's sorted screen names; `--expect-fingerprints` compares a dataset with a published
  `summary.json`.
- `ui2compose-evaluate-detector`: AP with an exact port of Ultralytics' matching and AP code,
  on the predictions of `YOLO.val()` by default (comparable with the published results);
  percentile bootstrap confidence intervals over images and, with `--groups manifest.csv`,
  over app groups (cluster bootstrap); paired differences between models; the operating point
  of the pipeline's own detections (single-label `YOLO.predict()` at confidence 0.3);
  `--expect-metrics` / `--expect-images` check that the released weights reproduce the
  published val and test metrics on a regenerated split, the precondition for the
  unseen-app subset.
- `ui2compose-benchmark`: per-stage timing with host information and the load average at the
  start and end of the run; `results/benchmark_m2.json` is the run on the ten examples with the
  default weights (Apple M2 CPU: warm median 0.106 s per screenshot, cold start about 1.2 s).
- `ui2compose-train` with `configs/train_*.yaml`; `train_oversample_5k.yaml` equals the
  training arguments stored in the released checkpoint, and `train_baseline.yaml` and
  `train_oversample_2k.yaml` equal those of their own checkpoints (v1.0.2 release assets;
  `results/eval/detector_ablation.md`). The released checkpoint does not record its initial
  weights; `scripts/coco_init_check.py` compares them with the COCO-pretrained `yolov8s.pt`
  (`results/eval/coco_init_check.json`).
- `ui2compose-oversample` logs per-class counts before and after and documents the exact
  procedure. Behaviour change: it now visits files in sorted order (v1.1.0: file-system
  order), so the copied images can differ from those used for the released model.

### Testing and continuous integration [R1.6]
- Test suite (`pytest -q`) covering every stage, including golden tests that
  replay the saved detections of the ten examples through the pipeline and require the
  published layout trees and Kotlin exactly. It runs without torch and Ultralytics.
- GitHub Actions CI on Python 3.10, 3.11 and 3.12, with the small `eval` extra (apted, zss,
  scipy) so that the evaluation tests run there too. In CI, missing reference data (example
  outputs, compile report, weights) fails the tests instead of skipping them
  (`UI2COMPOSE_REQUIRE_REFERENCE_DATA=1`).
- Compile and render harness (`harness/`, not in CI: it needs JDK 17 and the Android SDK):
  the Kotlin of the ten examples and six code-generator fixtures compiles with Kotlin 2.0.21
  and the Compose BOM 2024.06.00 (material3 1.2.1) and renders with Paparazzi 1.3.4;
  `results/compile_report.json` records the result and a test checks it is current.

### Examples [R2.4]
- `scripts/run_examples.py` runs the pipeline on the ten examples
  (`results/v1.2.0_examples/`, keeping the harness renders) and writes
  `results/examples_comparison.md` / `.json` (v1.1.0 against v1.2.0, per screen and in
  total, including the fallback count at `cut_tol_frac` = 0 to separate the effect of the
  tolerance).

### Threshold selection on the validation split [R1.3, R2.4]
- The defaults of de-duplication, `containment_ratio`, `cut_tol_frac` and `fallback` were
  selected on the validation split (1,869 screens of the regenerated ReDraw package; the test
  split and the bundled examples were not used): `python -m ui2compose.evaluation.layout_sweep`
  runs a grid of 120 configurations (de-duplication off / stricter 0.6/0.95/0.7 / a priori
  0.5/0.9/0.6 / looser 0.4/0.85/0.5; `containment_ratio` 1.0 / 0.95 / 0.9; `cut_tol_frac`
  0 / 0.0025 / 0.005 / 0.01 / 0.02; `fallback` `y_sorted` / `least_overlap`) on the cached
  detections of the released weights and scores each with `ui2compose-eval-layout` (condition
  C4). Rule, fixed before the sweep: maximum mean TEDS-o; ties within 0.001 go to the higher
  mean grouping F1, then to the configuration closest to the a-priori values; a
  de-duplication setting may lower the detector's operating-point F1 (conf 0.3, IoU 0.5) by at
  most 0.5 points (all four settings passed; each raised F1 by 0.2–0.3 points).
- Selected: de-duplication 0.4 / 0.85 / 0.5, `containment_ratio` 0.9, `cut_tol_frac` 0.0025,
  `fallback` `y_sorted` (a priori: 0.5 / 0.9 / 0.6, 0.95, 0.005, `y_sorted`). Mean TEDS-o 0.5086
  vs 0.5053, paired difference +0.0033 (95 % CI [+0.0018, +0.0072], bootstrap over app groups);
  mean grouping F1 unchanged (−0.0000 [−0.0039, +0.0089]). This difference is in-sample (measured
  on the screens on which the configuration was picked as the best of 120); applied to one
  random half of the val app groups and scored on the other, the rule gains +0.0019 TEDS-o on
  average (2.5–97.5 % of the 400 half-split selections: −0.0006 to +0.0043), and it picks the
  shipped configuration in only 10 % of them (`results/eval/val_sweep_crossval.json`). The selected
  de-duplication setting and `containment_ratio` are the loosest values of the grid, so values
  beyond them were not tried. The ground-truth-box sweep (C1)
  selects the same tolerance and fallback but `containment_ratio` 0.95; C1 differs by 0.0005
  TEDS-o between the two ratios. Files: `results/eval/val_sweep.md` / `.json` (full grid with
  intervals) and `results/eval/selected_thresholds.json`.
- Effect on the ten bundled examples: only mineworld's code changes (a title `TextView` with
  92 % of its area on its card image is now contained in it); XY-cut fallback nodes 3 → 2,
  `Box` composites 13 → 14; the de-duplication removes the same 10 boxes. The Shazam tab bar
  is still split into an icon row and a label row (overlap 1.74 px < 0.0025 × 1920 px =
  4.8 px). Examples, golden files, code-generator fixtures (header line only) and the harness
  report are regenerated.

- **Re-selection on the app-level split.** With the detector retrained on an app-level split
  (no app, MD5 or near-duplicate in two splits) the sweep was repeated, unchanged, on that
  split's validation set (1,928 screens, detections of the app-split 5k weights;
  `results/eval_appsplit/val_sweep.md`). It keeps de-duplication 0.4 / 0.85 / 0.5,
  `containment_ratio` 0.9 and `y_sorted`, and selects `cut_tol_frac` 0 instead of 0.0025
  (mean TEDS-o 0.4846 vs 0.4800, paired +0.0046 [+0.0023, +0.0075]; stable on half-splits:
  `cut_tol_frac` 0 in 398 of 400, out-of-sample gain over the a-priori values +0.0058). The
  shipped default is now `cut_tol_frac` = 0. With the v1.1.0 weights the Kotlin of foxfi,
  Shazam and drivesafe changed (the Shazam tab bar is again one `Column` of icon and label per
  tab; fallback nodes 2 → 11). With the default app-split weights the ten examples have 5
  fallback nodes and 13 `Box` / 6 `LazyColumn` composites (`results/examples_comparison.md`).

### Evaluation of the revision [R1.1, R1.2, R2.1, R2.2, R2.5, R2.6, R2.9]
Results, method and reproduction: `results/eval/README.md` (`scripts/run_evaluation.sh` runs
every stage from the regenerated ReDraw val/test package and the released weights; per-screen
outputs are kept outside the repository).
- `ui2compose-eval-layout`: the layout tree behind the generated Kotlin against the ReDraw view
  hierarchy of the same screen, on whole splits: tree-edit-distance similarity (TEDS, APTED with
  a Zhang-Shasha cross-check) with group and orientation labels, grouping F1, the edit
  operations as a structure-only correction-effort proxy, for ground-truth and predicted boxes,
  v1.1.0 as submitted and v1.2.0, against a flat-Column baseline, with bootstrap intervals over
  app groups and paired differences. `layout_report` writes Table 1 (`results/eval/layout_test.*`).
- Detector numbers without the app leak: the released weights on the test and val screens whose
  app group never occurs in training, per class and seen vs. unseen, with cluster-bootstrap
  intervals under two groupings (`evaluation/detector_breakdown.py`, `pred_cache.py`).
- Failure taxonomy of 50 random test screens with traced cases (`evaluation/failure_taxonomy.py`).
- Oversampling ablation without retraining: the baseline, 2k and 5k checkpoints on all, unseen-app
  and package-unseen val and test screens with paired BCa intervals over app groups
  (`python -m ui2compose.evaluation.detector_ablation`, `results/eval/detector_ablation.*`).
- Added after the review of these results (all labelled post hoc; no threshold and no primary
  number changed): the bootstrap bias of AP with basic and BCa intervals next to the percentile
  intervals (`detector_breakdown intervals`, `ap.interval_methods`, `ap.jackknife`); the
  number of screens behind every grouping-F1 mean and difference, the reasons for an undefined
  F1 and the means on the screens every condition defines; paired differences by stratum of the
  reference (flat / not flat, one / several boxes); a sensitivity analysis of how a containment
  `Box` is scored (`--box-repr container`); the out-of-sample check of the threshold selection
  (`layout_sweep crossval`).

### Licensing
- README and `.zenodo.json` state the licenses of the bundled third-party material: AGPL-3.0
  for both weights files, CC BY 4.0 for the example screenshots, Apache-2.0 for the Gradle wrapper
  files of the harness. The copyright line of `LICENSE` names both authors, as the SPDX
  headers of the source files do.

### Not in this release
- Repeated training runs: there is one training run per configuration, so no interval covers
  run-to-run (training) variance.
- The ReDraw dataset and the per-screen evaluation outputs (to be archived separately); the
  repository holds the summaries in `results/eval/`.

## [1.1.0]

Archived release: [10.5281/zenodo.22096717](https://doi.org/10.5281/zenodo.22096717).
