# Detector ablation: oversampling on all screens and on unseen apps

Written by `python -m ui2compose.evaluation.detector_ablation` (`scripts/run_evaluation.sh detector_ablation`). No retraining: the three checkpoints of the thesis ablation, evaluated with the settings of the released model (AP: YOLO.val predictions, conf 0.001, NMS IoU 0.7, max_det 300, imgsz 640; operating point: the pipeline's detections at conf > 0.3, IoU 0.5, micro). Intervals: 95 % BCa over `app_groups` app groups, 1000 paired replicates (seed 0); every checkpoint is evaluated on the same draws, so differences are paired. `*` = BCa undefined, percentile interval shown.

## Checkpoints

| name | weights (MD5) | Ultralytics (eval) | published test mAP50 → here | published val mAP50 → here |
|---|---|---|---|---|
| baseline (no oversampling) | `runs/baseline_appsplit/weights/best.pt` (a4cb282d…) | 8.4.24 | – | – |
| oversample_5k (oversampled to 5k (released)) | `runs/oversample_5k_appsplit/weights/best.pt` (4d5763e8…) | 8.4.24 | – | – |

What the checkpoints store about their training (`train_args`, the validation metrics saved with the weights):

| name | written with | train data | initial weights (`model`) | resume | stored val mAP50 | training config equal |
|---|---|---|---|---|---|---|
| baseline | Ultralytics 8.4.24, 2026-10-20 | `/content/work/data/yolo_app_640/dataset.yaml` | `/content/work/weights/yolov8s.pt` | `False` | 0.38105 | `configs/train_baseline.yaml`: NO (56 keys) |
| oversample_5k | Ultralytics 8.4.24, 2026-10-21 | `/content/work/data/yolo_app_640_os5k/dataset.yaml` | `/content/work/weights/yolov8s.pt` | `False` | 0.42090 | `configs/train_oversample_5k.yaml`: NO (56 keys) |

train_args that differ between the checkpoints: `data`, `name` (all others equal; baseline: 108, oversample_5k: 108 keys).

## Table 2 rows (checkpoint × set)

Mean AP50 over the classes with ≥ 20 instances in the set (the headline metric of Table 2), mAP50 and mAP50-95 with BCa intervals; P / R / F1 at the pipeline's operating point, F1 with its percentile interval (as in Table 2; the BCa intervals of P / R / F1 are in the JSON and CSV).

| set | screens (groups) | checkpoint | mean AP50, classes ≥ 20 inst. | mAP50 | mAP50-95 | P / R / F1 (F1 interval) |
|---|---|---|---|---|---|---|
| test, all screens | 1,900 (586) | baseline | 0.404 [0.370, 0.432] (13) | 0.404 [0.370, 0.432] | 0.307 [0.279, 0.326] | 0.624 / 0.457 / 0.527 [0.505, 0.547] |
|  | | oversample_5k | 0.442 [0.395, 0.480] (13) | 0.442 [0.395, 0.480] | 0.347 [0.307, 0.381] | 0.585 / 0.498 / 0.538 [0.515, 0.557] |
| val, all screens | 1,928 (609) | baseline | 0.381 [0.347, 0.408] (13) | 0.381 [0.347, 0.408] | 0.291 [0.267, 0.312] | 0.600 / 0.466 / 0.525 [0.501, 0.545] |
|  | | oversample_5k | 0.421 [0.385, 0.456] (13) | 0.421 [0.385, 0.456] | 0.330 [0.301, 0.358] | 0.564 / 0.509 / 0.535 [0.512, 0.556] |

## Paired differences

Point [95 % BCa], and P = share of the replicates with a difference ≤ 0.

| set | difference | Δ mean AP50 (classes ≥ 20) | P | Δ mAP50 | P | Δ mAP50-95 | Δ precision | Δ recall | Δ F1 | P |
|---|---|---|---|---|---|---|---|---|---|---|
| test, all screens (1,900) | oversample_5k - baseline | +0.038 [+0.014, +0.067] | 0.003 | +0.038 [+0.014, +0.067] | 0.003 | +0.040 [+0.016, +0.069] | −0.039 [−0.051, −0.025] | +0.041 [+0.030, +0.053] | +0.010 [+0.001, +0.021] | 0.022 |
| val, all screens (1,928) | oversample_5k - baseline | +0.040 [+0.018, +0.067] | 0.001 | +0.040 [+0.018, +0.067] | 0.001 | +0.039 [+0.021, +0.064] | −0.036 [−0.051, −0.018] | +0.043 [+0.030, +0.061] | +0.010 [−0.001, +0.027] | 0.060 |

## The eight oversampled classes

AP50 per checkpoint and the paired differences [BCa]. Support = ground-truth instances (screens). **†** = fewer than 20 instances: AP50 rests on a handful of boxes and its interval is not informative.

| set | class | support (screens) | baseline | oversample_5k | Δ oversample_5k - baseline |
|---|---|---|---|---|---|
| test, all screens | Button | 2719 (1215) | 0.644 | 0.617 | −0.027 [−0.057, −0.010] |
|  | CheckBox | 179 (87) | 0.515 | 0.622 | +0.108 [+0.015, +0.234] |
|  | EditText | 627 (316) | 0.667 | 0.662 | −0.004 [−0.036, +0.020] |
|  | ImageView | 6227 (1157) | 0.485 | 0.495 | +0.010 [−0.005, +0.026] |
|  | ListView | 459 (402) | 0.658 | 0.624 | −0.034 [−0.060, −0.013] |
|  | ProgressBar | 56 (55) | 0.150 | 0.180 | +0.030 [−0.030, +0.079] |
|  | RadioButton | 135 (44) | 0.203 | 0.304 | +0.100 [−0.004, +0.235] |
|  | RecyclerView | 157 (135) | 0.380 | 0.352 | −0.028 [−0.094, +0.048] |
|  | SeekBar | 45 (23) | 0.230 | 0.378 | +0.148 [+0.030, +0.287] |
|  | Spinner | 77 (62) | 0.174 | 0.396 | +0.222 [+0.132, +0.342] |
|  | Switch | 40 (27) | 0.305 | 0.342 | +0.038 [−0.142, +0.288] |
|  | TextView | 11860 (1705) | 0.539 | 0.529 | −0.010 [−0.023, +0.006] |
|  | WebView | 116 (116) | 0.303 | 0.251 | −0.052 [−0.136, +0.018] |
| val, all screens | Button | 2719 (1185) | 0.615 | 0.603 | −0.011 [−0.029, +0.004] |
|  | CheckBox | 180 (79) | 0.123 | 0.217 | +0.094 [+0.026, +0.179] |
|  | EditText | 607 (327) | 0.702 | 0.657 | −0.045 [−0.072, −0.022] |
|  | ImageView | 6120 (1169) | 0.478 | 0.488 | +0.010 [−0.007, +0.032] |
|  | ListView | 459 (412) | 0.670 | 0.645 | −0.025 [−0.041, −0.008] |
|  | ProgressBar | 56 (55) | 0.145 | 0.196 | +0.052 [−0.018, +0.148] |
|  | RadioButton | 135 (42) | 0.061 | 0.160 | +0.099 [+0.019, +0.235] |
|  | RecyclerView | 157 (141) | 0.345 | 0.357 | +0.012 [−0.060, +0.066] |
|  | SeekBar | 46 (29) | 0.355 | 0.407 | +0.051 [−0.104, +0.187] |
|  | Spinner | 77 (63) | 0.253 | 0.385 | +0.131 [+0.065, +0.220] |
|  | Switch | 40 (27) | 0.373 | 0.536 | +0.162 [+0.027, +0.382] |
|  | TextView | 11609 (1702) | 0.525 | 0.519 | −0.006 [−0.023, +0.017] |
|  | WebView | 116 (116) | 0.304 | 0.305 | +0.001 [−0.061, +0.071] |

## Checks

* Evaluation settings identical for every checkpoint and set: True ({"conf": 0.001, "conf_op": 0.3, "imgsz": null, "imgsz_effective": 640, "iou": 0.7, "max_det": 300, "multi_label": true, "pred_source": "val", "seed_n": [0, 1000]}).
* Same images, labels and resampling units for every checkpoint: True.
* Percentile intervals of mAP50 / mAP50-95 equal those of the evaluation files (their own grouping): True (4 comparisons).
* Operating-point intervals equal `operating_point_bootstrap` (same draws): True.
