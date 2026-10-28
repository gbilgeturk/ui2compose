<!--
SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>

SPDX-License-Identifier: MIT
-->

# docs/figures — figures of the SoftwareX article (v1.2.0)

All five figures are drawn by `scripts/make_figures.py` from files in this
repository. Nothing is drawn by hand. The exception is Fig. 4: the Claude renders it
needs are not in this release (`results/mllm_claude/README.md`), so the script skips it
and the committed image is the one drawn from those renders, identical to the one in the
manuscript.

```bash
# from the repository root
python scripts/make_figures.py              # all figures whose inputs exist
python scripts/make_figures.py --only 2 3   # selected figures
```

Format: PNG at 600 dpi at print size, 190 mm (7.48 in) wide for the
double-column figures and 90 mm (3.54 in) for Fig. 3. Every text is at least
8 pt at that size; `save()` refuses to write a figure with smaller text. Fonts are
the DejaVu fonts bundled with matplotlib. The output is byte-identical between runs
(no randomness in the drawing; the PNG metadata carry no software version).

Colours follow one palette with fixed meanings, checked for colour-vision
deficiency (all pairs that appear together: CVD ΔE ≥ 9.1, normal-vision
ΔE ≥ 16.3, OKLab × 100):

| Meaning | Colours |
|---|---|
| Detector class (Figs. 1, 3) | ImageView blue `#2a78d6`, TextView orange `#eb6834`, Button green `#1baf7a`, CheckBox violet `#4a3aa7` |
| Container rule (Fig. 2) | gap blue, containment green, list violet, fallback grey |
| Detection error, compile error | red `#d03b3b`, always with a number badge or a text label |
| Training configuration (Fig. 5) | the two ends of one blue ramp, light (baseline) and dark (5k), plus marker shape (square, circle) |

Colour never carries meaning alone: boxes carry text labels (`#id Class
confidence`), rules are written in the chips, and configurations also differ
in marker shape.

The worked example of Figs. 1-3 is the drivesafe screen
(`examples/westport.andrewirwin.com.drivesafe-screens_screenshot_3.png`), a test
screen of the app-level split, i.e. an app the default weights
(`runs/oversample_5k_appsplit`) never saw in training. All outputs shown are those of
the pipeline defaults (`results/v1.2.0_examples/<screen>/`).

## Fig. 1 — `fig1_pipeline.png` (pipeline)

The four stages as implemented in `src/ui2compose/infer/`, each with its
module, its output files and a small illustration:

1. Detection (`detect_components.py`): YOLOv8s, 13 classes, confidence 0.3,
   NMS IoU 0.7. Illustration: the 9 raw boxes of the worked example.
2. De-duplication (`postprocess.py`): the validation-selected thresholds
   (same class IoU 0.4, or 85 % of the smaller box for classes that cannot
   nest; other class IoU 0.5). The worked example has no removal, so the
   illustration is the `same_class_nested` removal of the battlesimulator screen
   (a validation screen of the app-level split): TextView #37 (0.54), 93 % nested
   in the more confident TextView #36 (0.55), is removed.
3. Containment (`build_ui_graph.py`): each box gets as parent the smallest box
   that holds at least 90 % of it (schematic: an ImageView holding a CheckBox and a
   TextView, as on the worked example). Only this containment forest is used for
   the code; the spatial edges (`above`, `left_of`, `right_of`) are written to
   `3_ui_graph.json` for inspection only.
4. Layout and code (`generate_compose_code.py`): containers become `Box` /
   `LazyColumn`, XY-cut gives `Column` / `Row`, every container records the
   rule that made it (`2_layout_tree.json`), and templates emit
   `4_GeneratedScreen.kt`.

Sources: `0_run_config.json` (every threshold shown) and `1_detections_raw.json` of the
worked example; `1_detections.json` and `1_dedup_removed.json` of
`results/v1.2.0_examples/com.luigiedeveloper.battlesimulator-screens_screenshot_3/`.

## Fig. 2 — `fig2_layout_example.png` (worked example: layout tree)

- (a) the screenshot with the cut of the root `Column` (dashed), the cuts inside
  nested `Row` / `Column` nodes (dotted) and the composites in the colour of their
  rule; the detection error is framed in red;
- (b) the layout tree: every container with its rule (`gap`, `containment`; `list`
  and `fallback` do not occur on this screen) and cut count or container id, every
  leaf with id, class and confidence; the subtree affected by the error is shaded;
- (c) the error, with the numbers from the trace: the menu drawer is detected as
  ImageView #2 (0.67); the six boxes of its three rows lie entirely inside it
  (coverage 100 % ≥ 90 %), so the rows become a `Column` in a `Box` over an image
  instead of the items of a `LazyColumn`, which a drawer detected as `ListView` or
  `RecyclerView` would have become. Nothing behind the drawer's dimmed overlay is
  detected. The settings shown are those of `0_run_config.json` (cut tolerance 0,
  containment 0.9, fallback `y_sorted`; 9 detections, none removed).

Sources: `2_layout_tree.json`, `3_ui_graph.json` (coverage of the children of #2),
`0_run_config.json` of the example folder; the traces
`ui2compose examples/westport.andrewirwin.com.drivesafe-screens_screenshot_3.png --explain 3`
(also 2 and 4, saved in the example folder) print the same numbers.

## Fig. 3 — `fig3_detections.png` (worked example: detections)

Single-column version of `docs/example_detections.png`: every raw detection of
the worked example with its id, class and confidence (boxes removed by
de-duplication would be dashed grey; there are none on this screen). The ids are
those of Fig. 2 and of the `--explain` traces. Sources: `1_detections.json`,
`1_dedup_removed.json` of the example folder.

## Fig. 4 — `fig4_claude_comparison.png` (comparison with Claude)

Four of the ten example screens, each titled with its split on the app-level split:
drivesafe #3 (test) and battlesimulator #3 (validation), the two examples the default
weights never saw, and com.foxfi #1 and com.mineworld.swordsaddon #2 (both in the
app-level train split), chosen in `results/mllm_claude/ANALYSIS.md` (section 2) as
the typical case and the case where Claude's run 1 does not compile. For each: the
input screenshot, the Paparazzi render of the ui2compose v1.2.0 Kotlin
(`results/v1.2.0_examples/<screen>/render.png`, default weights) and the render of
Claude's run-1 Kotlin (`results/mllm_claude/<screen>/run1/render.png`, unchanged).
Claude's run 1 of mineworld #2 does not compile (`16:43 Cannot access 'val
RowColumnParentData?.weight: Float': it is internal in file.`, from
`results/mllm_claude/compile_report.json`); its panel is hatched with a red frame.

Before drawing, the script checks that every ui2compose render belongs to the
current Kotlin file: the SHA-256 of `4_GeneratedScreen.kt` must equal the one
recorded by the harness in `results/compile_report.json` together with that
render. `results/mllm_claude/side_by_side*.png` (all ten screens, and four as columns) are
drawn from the same current renders by `scripts/make_mllm_side_by_side.py`, with the same
check. Neither can be redrawn from this release: the Claude files named above were lost
before the release (see the note in `results/mllm_claude/README.md`).

## Fig. 5 — `fig5_detector.png` (detector on the app-level split)

Both checkpoints are YOLOv8s retrained on the app-level split with the training
arguments of the v1.1.0 model: the baseline (no oversampling;
`runs/baseline_appsplit/weights/best.pt`, MD5 a4cb282d…, also in the app-split data package) and the default weights (rare classes oversampled to about
5,000 instances; `runs/oversample_5k_appsplit/weights/best.pt`, MD5 4d5763e8…).

- (a) AP@50 per class on the 1,900 test screens (632 apps not seen in training), with
  95 % BCa intervals. The y labels give the number of test instances of each class.
  The eight classes oversampled in training (`RARE_CLASSES` of
  `src/ui2compose/preprocess/oversample_dataset.py`) are grouped at the top. The right
  column gives the paired difference (5k − baseline, percentage points, 95 % BCa
  interval); bold = the interval excludes 0. The class order within each group
  follows that difference.
- (b) mAP@50 (13 classes, all with at least 20 instances) on the test and validation
  splits, with 95 % BCa intervals, and the paired difference 5k − baseline.

All intervals resample the split's own app groups (586 test / 609 validation groups;
1,000 replicates, seed 0), the same draws for both checkpoints, so the differences are
paired. Values: `results/eval_appsplit/detector_table2.json`; the subset drawn is
written to `fig5_detector_data.json` (in this folder) with the SHA-256 of its source.

| Set | baseline | 5k (default) | 5k − baseline, BCa 95 % |
|---|---|---|---|
| test (1,900 screens) | 40.4 [37.0, 43.2] | 44.2 [39.5, 48.0] | +3.8 [+1.4, +6.7] |
| validation (1,928 screens) | 38.1 [34.7, 40.8] | 42.1 [38.5, 45.6] | +4.0 [+1.8, +6.7] |

Per class (test): Spinner +22.2 [+13.2, +34.2], SeekBar +14.8 [+3.0, +28.7] and
CheckBox +10.8 [+1.5, +23.4] points; Button −2.7 [−5.7, −1.0] and ListView −3.4
[−6.0, −1.3]; the other intervals include 0. The submitted figure showed the v1.1.0
weights on the screenshot-level split and its unseen-app subsets
(`results/eval/`); that version of the data is no longer drawn.
