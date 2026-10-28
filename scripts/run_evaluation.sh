#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT
#
# Reproduces every number of results/eval/ from the regenerated ReDraw val/test
# package and the released weights (runs/oversample_5k/weights/best.pt).
#
# Usage (from the repository root):
#   PKG=/path/to/ui2compose_redraw_valtest_v1.1.0_UNVERIFIED \
#   OUT=/path/outside/the/repo/eval_outputs \
#   scripts/run_evaluation.sh [stage ...]
#
# Stages (default: all, in this order):
#   setup        dataset yaml with an absolute path (in $OUT), v1.1.0 code (git show)
#   pred_cache   cached pipeline detections of val and test (conf 0.3, NMS IoU 0.7, imgsz 640)
#   detector     detector AP / operating point with CIs on val + test (full split and the three
#                unseen subsets), the summary tables and seen vs unseen (results/eval/detector_*)
#   detector_intervals
#                bias of the percentile bootstrap of AP: basic and BCa intervals from the same
#                replicates (results/eval/detector_intervals.*; added after the review)
#   detector_ablation
#                oversampling ablation (added after the review): the no-oversampling baseline
#                (imgsz_640.pt) and oversample_2k.pt of the v1.0.2 release, evaluated exactly as the
#                released weights on test (all / unseen full grouping / unseen package-only) and val
#                (the same), then baseline / 2k / 5k with paired differences and BCa intervals
#                (results/eval/ablation/<tag>/detector_*.json, results/eval/detector_ablation.*);
#                needs ABLATION_WEIGHTS (folder with imgsz_640.pt and oversample_2k.pt)
#   sweep        threshold sweep and selection on val (results/eval/val_sweep.*,
#                selected_thresholds.json); the selected values are the shipped defaults
#   layout_val   layout evaluation of all conditions on val + its report ($OUT/layout_val)
#   layout_test  the same on test, ONCE, after every choice is fixed on val
#                (results/eval/layout_test.json / .md, per-screen JSONL in $OUT/layout_test)
#   box_sensitivity
#                post-hoc sensitivity (after the review) of the scoring of containment Boxes:
#                C1 and C4 again with --box-repr container on val, then once on test; the
#                reports of layout_val / layout_test are rewritten with a sensitivity section
#   sweep_crossval
#                post-hoc check of the threshold selection: the selection rule applied on one
#                half of the val app groups and scored on the other (results/eval/val_sweep_crossval.json)
#   taxonomy     failure taxonomy of 50 random C4 test screens (seed 42) and the three
#                illustrative cases with the pipeline's --explain
#                (results/eval/failure_taxonomy.json / .md)
#
#
# App-level split (v1.2.0, results/eval_appsplit/; never part of the default stages). PKG is then
# the package ui2compose_redraw_appsplit_v1.2.0 and OUT a separate output folder:
#   app_setup        dataset yaml, the evaluation view $OUT/pkg_view (package + splits/full =
#                    app_split_manifest.csv + splits/package_md5 computed from it), v1.1.0 code
#                    (git show, or V110_CODE=<folder with the two files>)
#   app_pred_cache   pipeline detections of val and test with the app-split 5k weights
#   app_detector     evaluate_detector: baseline_app and 5k_app on val + test, the original released
#                    5k weights on test (CONTAMINATED: trained on screens of the new test split,
#                    reported for reference only), then Table 2 (detector_ablation: BCa, paired)
#   app_sweep        threshold sweep + selection on the new val, then the cross-validation check
#   app_layout_test  layout evaluation C1..C5, C1' on the new test split, ONCE
#   app_taxonomy     failure taxonomy of 50 C4 test screens + the traced cases
# Weights: APP_WEIGHTS (default runs/oversample_5k_appsplit/weights/best.pt), APP_BASELINE
# (default runs/baseline_appsplit/weights/best.pt, the same file as $PKG/weights/baseline_app/best.pt),
# WEIGHTS (the original 5k, contamination row).
#
# Environment: PY (python with `pip install -e ".[eval]"`, default: python), WORKERS (default 4),
# EVAL_BASE (directory the detector stage was run from, against which the paths inside its JSON
# files are resolved; default: the repository root), ABLATION_WEIGHTS (detector_ablation only),
# DRY_RUN=1 prints the commands without running them.
# Run time on an Apple M2 (CPU only): pred_cache ~15 min per split, detector ~2.5 h,
# sweep ~11 min (6 processes), layout_val / layout_test < 1 min each (4 processes),
# taxonomy ~3 min, detector_ablation ~3 h (inference of two checkpoints on six sets, then the
# paired bootstrap). Disk: about 330 MB in $OUT (+ about 190 MB of ablation predictions).

set -euo pipefail

PY="${PY:-python}"
WORKERS="${WORKERS:-4}"
: "${PKG:?set PKG to the val/test package (ui2compose_redraw_valtest_v1.1.0_UNVERIFIED)}"
: "${OUT:?set OUT to an output directory outside the repository}"
WEIGHTS="${WEIGHTS:-runs/oversample_5k/weights/best.pt}"
APP_WEIGHTS="${APP_WEIGHTS:-runs/oversample_5k_appsplit/weights/best.pt}"
APP_BASELINE="${APP_BASELINE:-runs/baseline_appsplit/weights/best.pt}"
RES=results/eval
RES_APP=results/eval_appsplit
STAGES=("$@")
EVAL_BASE="${EVAL_BASE:-.}"
[ ${#STAGES[@]} -eq 0 ] && STAGES=(setup pred_cache detector detector_intervals detector_ablation
                                   sweep sweep_crossval layout_val layout_test box_sensitivity
                                   taxonomy)

run() {
    # Prints a command, then runs it unless DRY_RUN=1.
    echo "+ $*"
    if [ "${DRY_RUN:-0}" != "1" ]; then "$@"; fi
}

[ -f pyproject.toml ] && [ -d src/ui2compose ] || { echo "run from the repository root"; exit 2; }
[ -f "$PKG/README_PACKAGE.txt" ] || { echo "PKG is not the val/test package: $PKG"; exit 2; }
mkdir -p "$OUT/logs" "$RES"
PKG_ABS="$(cd "$PKG" && pwd)"

stage_setup() {
    # Dataset yaml with an absolute path (Ultralytics resolves split folders from it).
    if [ "${DRY_RUN:-0}" != "1" ]; then
        { echo "path: \"$PKG_ABS/dataset\""; grep -v '^path:' "$PKG/dataset/dataset.yaml"; } \
            > "$OUT/dataset_abs.yaml"
    fi
    echo "+ wrote $OUT/dataset_abs.yaml"
    # v1.1.0 graph builder and code generator (condition C3 / C1'); layout_eval checks their SHA-256.
    mkdir -p "$OUT/v110_code"
    for f in build_ui_graph.py generate_compose_code.py; do
        if [ "${DRY_RUN:-0}" != "1" ]; then
            git show "v1.1.0:src/ui2compose/infer/$f" > "$OUT/v110_code/$f"
        fi
        echo "+ git show v1.1.0:src/ui2compose/infer/$f > $OUT/v110_code/$f"
    done
}

stage_pred_cache() {
    for split in val test; do
        run "$PY" -m ui2compose.evaluation.pred_cache --model "$WEIGHTS" \
            --images "$PKG/dataset/images/$split" --out-dir "$OUT/pred_cache/$split" \
            --verify results/v1.1.0_examples 1_detections.json
        # (tests/golden holds the outputs of the app-split default weights, not of $WEIGHTS)
    done
}

stage_detector() {
    local split sub images expect
    for split in val test; do
        for sub in full unseen_full unseen_package_md5 unseen_package_only; do
            case "$sub" in
                full) images="$PKG/dataset/images/$split" ;;
                unseen_full) images="$PKG/dataset/unseen_$split.txt" ;;
                *) images="$PKG/dataset/unseen_${split}_${sub#unseen_}.txt" ;;
            esac
            expect=()
            if [ "$sub" = full ] && [ "$split" = test ]; then
                expect=(--expect-metrics results/perclass_ap_results.json --expect-key oversample_5k)
            elif [ "$sub" = full ]; then
                expect=(--expect-metrics results/oversample_5k_metrics.json)
            fi
            run "$PY" -m ui2compose.evaluation.evaluate_detector --model "$WEIGHTS" \
                --images "$images" --labels "$PKG/dataset/labels/$split" \
                --dataset-yaml "$OUT/dataset_abs.yaml" --groups "$PKG/splits/full/manifest.csv" \
                --bootstrap 1000 --seed 0 --cross-check-val --preds-dir "$OUT/preds" \
                --out "$RES/detector_${split}_${sub}.json" ${expect[@]+"${expect[@]}"}
        done
    done
    run "$PY" -m ui2compose.evaluation.detector_breakdown summary \
        --groups "full_grouping=$PKG/splits/full/manifest.csv" \
        --groups "package_md5=$PKG/splits/package_md5/manifest.csv" \
        --out-json "$RES/detector_summary.json" --out-csv "$RES/detector_summary.csv" \
        "$RES"/detector_val_{full,unseen_full,unseen_package_md5,unseen_package_only}.json \
        "$RES"/detector_test_{full,unseen_full,unseen_package_md5,unseen_package_only}.json
    for split in test val; do
        run "$PY" -m ui2compose.evaluation.detector_breakdown seen-unseen \
            "$RES/detector_${split}_full.json" --manifest "$PKG/splits/full/manifest.csv" \
            --groups "package_md5=$PKG/splits/package_md5/manifest.csv" \
            --groups "full_grouping=$PKG/splits/full/manifest.csv" --diff-groups package_md5 \
            --out-json "$RES/detector_seen_unseen_$split.json" \
            --out-csv "$RES/detector_seen_unseen_$split.csv"
    done
}

stage_detector_intervals() {
    run "$PY" -m ui2compose.evaluation.detector_breakdown intervals --base-dir "$EVAL_BASE" \
        --groups "full_grouping=$PKG/splits/full/manifest.csv" \
        --groups "package_md5=$PKG/splits/package_md5/manifest.csv" \
        --bootstrap 1000 --seed 0 --workers "$WORKERS" \
        --seen-unseen "$RES/detector_test_full.json" --seen-unseen "$RES/detector_val_full.json" \
        --manifest "$PKG/splits/full/manifest.csv" --diff-groups package_md5 \
        --out-json "$RES/detector_intervals.json" --out-csv "$RES/detector_intervals.csv" \
        "$RES"/detector_val_{full,unseen_full,unseen_package_md5,unseen_package_only}.json \
        "$RES"/detector_test_{full,unseen_full,unseen_package_md5,unseen_package_only}.json
}

stage_detector_ablation() {
    # Post hoc (after the review, R2.5 / R2.6): does oversampling still help on unseen apps? No
    # retraining: the ablation checkpoints of the thesis (v1.0.2 release assets; imgsz_640.pt is the
    # no-oversampling baseline, train data 'ablation_640'), evaluated with the detector stage's
    # settings on the same sets; the 5k files are those of the detector stage. The published test /
    # val mAP of each checkpoint is checked (--expect-metrics).
    : "${ABLATION_WEIGHTS:?set ABLATION_WEIGHTS to the folder with imgsz_640.pt and oversample_2k.pt}"
    local tag job split sub n images expect
    for tag in imgsz_640 oversample_2k; do
        for job in test:full:1870 test:unseen_full:169 test:unseen_package_only:214 \
                   val:full:1869 val:unseen_full:203 val:unseen_package_only:241; do
            IFS=: read -r split sub n <<< "$job"
            case "$sub" in
                full) images="$PKG/dataset/images/$split" ;;
                unseen_full) images="$PKG/dataset/unseen_$split.txt" ;;
                *) images="$PKG/dataset/unseen_${split}_${sub#unseen_}.txt" ;;
            esac
            expect=()
            if [ "$sub" = full ] && [ "$split" = test ]; then
                expect=(--expect-metrics results/perclass_ap_results.json --expect-key "$tag")
            elif [ "$sub" = full ]; then
                expect=(--expect-metrics "results/${tag}_metrics.json")
            fi
            run "$PY" -m ui2compose.evaluation.evaluate_detector --model "$ABLATION_WEIGHTS/$tag.pt" \
                --images "$images" --labels "$PKG/dataset/labels/$split" \
                --dataset-yaml "$OUT/dataset_abs.yaml" --groups "$PKG/splits/full/manifest.csv" \
                --bootstrap 1000 --seed 0 --cross-check-val --preds-dir "$OUT/preds" \
                --expect-images "$n" --out "$RES/ablation/$tag/detector_${split}_${sub}.json" \
                ${expect[@]+"${expect[@]}"}
        done
    done
    run "$PY" -m ui2compose.evaluation.detector_ablation --base-dir "$EVAL_BASE" \
        --model "baseline=$RES/ablation/imgsz_640" --model "oversample_2k=$RES/ablation/oversample_2k" \
        --model "oversample_5k=$RES" \
        --groups "package_md5=$PKG/splits/package_md5/manifest.csv" \
        --groups "full_grouping=$PKG/splits/full/manifest.csv" \
        --intervals-reference "$RES/detector_intervals.json" --bootstrap 1000 --seed 0 \
        --checkpoint "baseline=$ABLATION_WEIGHTS/imgsz_640.pt" \
        --checkpoint "oversample_2k=$ABLATION_WEIGHTS/oversample_2k.pt" \
        --checkpoint "oversample_5k=$WEIGHTS" --train-config baseline=configs/train_baseline.yaml \
        --train-config oversample_2k=configs/train_oversample_2k.yaml \
        --train-config oversample_5k=configs/train_oversample_5k.yaml \
        --workers "$WORKERS" --cache-dir "$OUT/ablation_cache" --out-json "$RES/detector_ablation.json" \
        --out-csv "$RES/detector_ablation.csv" --out-md "$RES/detector_ablation.md"
}

stage_sweep() {
    run "$PY" -m ui2compose.evaluation.layout_sweep run --data-dir "$PKG" --split val \
        --pred-cache "$OUT/pred_cache/val" --out "$OUT/layout_sweep_val" --workers "$WORKERS"
    run "$PY" -m ui2compose.evaluation.layout_sweep report --data-dir "$PKG" --split val \
        --pred-cache "$OUT/pred_cache/val" --sweep "$OUT/layout_sweep_val" --results-dir "$RES" \
        --date 2026-10-20
}

stage_sweep_crossval() {
    # Post hoc (after the review): the selection rule on random halves of the val app groups;
    # rewrites val_sweep.md and selected_thresholds.json from val_sweep.json (nothing re-selected).
    run "$PY" -m ui2compose.evaluation.layout_sweep crossval --data-dir "$PKG" --split val \
        --sweep "$OUT/layout_sweep_val" --results-dir "$RES" --repeats 200 --seed 0
}

split_params() {
    # Thresholds of this split: its own val selection ($RES/selected_thresholds.json, written by
    # the sweep stage), not the pipeline defaults, which were re-selected on the app-level split
    # (results/eval_appsplit/selected_thresholds.json; the two differ in cut_tol_frac).
    "$PY" -c 'import json, sys
v = json.load(open(sys.argv[1]))["values"]; d = v["dedup"]
print("--same-class-iou", d["same_class_iou"], "--same-class-iomin", d["same_class_iomin"],
      "--cross-class-iou", d["cross_class_iou"], "--containment-ratio", v["containment_ratio"],
      "--cut-tol-frac", v["cut_tol_frac"], "--fallback", v["fallback"])' "$RES/selected_thresholds.json"
}

split_pipeline_config() {
    # The same thresholds as a pipeline config (for the --explain traces of the taxonomy cases).
    run "$PY" -c 'import json, sys, yaml
v = json.load(open(sys.argv[1]))["values"]; c = yaml.safe_load(open("configs/pipeline_config.yaml"))
c["dedup"].update({k: v["dedup"][k] for k in ("same_class_iou", "same_class_iomin", "cross_class_iou")})
c["graph"]["containment_ratio"] = v["containment_ratio"]
c["layout"].update(cut_tol_frac=v["cut_tol_frac"], fallback=v["fallback"])
yaml.safe_dump(c, open(sys.argv[2], "w"), sort_keys=False)' "$RES/selected_thresholds.json" "$OUT/pipeline_config_split.yaml"
}

layout_split() {
    # Layout evaluation of every condition on one split, then its Table-1 report.
    local split="$1" report_json="$2" report_md="$3"
    local guard=()
    [ "$split" = test ] && guard=(--allow-test)
    run "$PY" -m ui2compose.evaluation.layout_eval --data-dir "$PKG" --split "$split" \
        ${guard[@]+"${guard[@]}"} --conditions C1,C2,C3,C4,C5,C1p $(split_params) \
        --pred-cache "$OUT/pred_cache/$split" --v110-code "$OUT/v110_code" \
        --out "$OUT/layout_$split" --workers "$WORKERS" --bootstrap 1000 --seed 0 --zss-check 100
    run "$PY" -m ui2compose.evaluation.layout_report --run "$OUT/layout_$split" --data-dir "$PKG" \
        --out-json "$report_json" --out-md "$report_md"
}

stage_layout_val() {
    layout_split val "$OUT/layout_val/report/layout_val.json" "$OUT/layout_val/report/layout_val.md"
}

stage_layout_test() {
    layout_split test "$RES/layout_test.json" "$RES/layout_test.md"
}

stage_box_sensitivity() {
    # Post hoc (after the review): containment Boxes scored as containers, C1 and C4 only; the
    # other conditions have no containment Boxes. Then both reports again, with the sensitivity.
    local split guard
    for split in val test; do
        guard=()
        [ "$split" = test ] && guard=(--allow-test)
        run "$PY" -m ui2compose.evaluation.layout_eval --data-dir "$PKG" --split "$split" \
            ${guard[@]+"${guard[@]}"} --conditions C1,C4 --box-repr container $(split_params) \
            --pred-cache "$OUT/pred_cache/$split" --out "$OUT/layout_${split}_boxsens" \
            --workers "$WORKERS" --bootstrap 1000 --seed 0 --zss-check 100
    done
    run "$PY" -m ui2compose.evaluation.layout_report --run "$OUT/layout_val" --data-dir "$PKG" \
        --box-sensitivity-run "$OUT/layout_val_boxsens" \
        --out-json "$OUT/layout_val/report/layout_val.json" \
        --out-md "$OUT/layout_val/report/layout_val.md"
    run "$PY" -m ui2compose.evaluation.layout_report --run "$OUT/layout_test" --data-dir "$PKG" \
        --box-sensitivity-run "$OUT/layout_test_boxsens" \
        --out-json "$RES/layout_test.json" --out-md "$RES/layout_test.md"
}

stage_taxonomy() {
    # The taxonomy takes its thresholds from the layout_test run it is checked against.
    split_pipeline_config
    run "$PY" -m ui2compose.evaluation.failure_taxonomy run --data-dir "$PKG" --split test \
        --allow-test --pred-cache "$OUT/pred_cache/test" --run "$OUT/layout_test" \
        --out-dir "$OUT/failure_taxonomy_test" --n 50 --seed 42
    run "$PY" -m ui2compose.evaluation.failure_taxonomy cases --data-dir "$PKG" --split test \
        --allow-test --pred-cache "$OUT/pred_cache/test" --model "$WEIGHTS" \
        --pipeline-config "$OUT/pipeline_config_split.yaml" \
        --taxonomy "$OUT/failure_taxonomy_test/failure_taxonomy.json" \
        --out-dir "$OUT/failure_taxonomy_test/cases" --results-dir "$RES" --name failure_taxonomy
}

# --- app-level split (results/eval_appsplit) -------------------------------------------------
VIEW="$OUT/pkg_view"

stage_app_setup() {
    # Evaluation view of the app-split package: the tools read splits/full/manifest.csv and
    # splits/package_md5/manifest.csv; the app split's own grouping (package OR MD5 OR dHash <= 4)
    # is the 'full' one, package_md5 is recomputed from the manifest (app_split.group_screens).
    run mkdir -p "$VIEW/splits/full" "$VIEW/splits/package_md5" "$OUT/v110_code"
    run ln -sfn "$PKG_ABS/dataset" "$VIEW/dataset"
    run ln -sfn "$PKG_ABS/raw_xml" "$VIEW/raw_xml"
    run ln -sf "$PKG_ABS/README_PACKAGE.txt" "$VIEW/README_PACKAGE.txt"
    run ln -sf "$PKG_ABS/splits/app_split_manifest.csv" "$VIEW/splits/full/manifest.csv"
    run "$PY" -c 'import csv, sys
from ui2compose.preprocess.app_split import ScreenRecord, group_screens
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
g = group_screens([ScreenRecord(stem=r["stem"], split=r["split"], package=r["package"],
                                md5=r["md5"]) for r in rows], None)
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f); w.writerow(["stem", "package", "group_id", "split"])
    w.writerows([r["stem"], r["package"], k, r["split"]] for r, k in zip(rows, g))' \
        "$PKG/splits/app_split_manifest.csv" "$VIEW/splits/package_md5/manifest.csv"
    if [ "${DRY_RUN:-0}" != "1" ]; then
        { echo "path: \"$PKG_ABS/dataset\""; grep -v '^path:' "$PKG/dataset/dataset.yaml"; } \
            > "$OUT/dataset_abs.yaml"
    fi
    for f in build_ui_graph.py generate_compose_code.py; do
        if [ -n "${V110_CODE:-}" ]; then
            run cp "$V110_CODE/$f" "$OUT/v110_code/$f"
        else
            echo "+ git show v1.1.0:src/ui2compose/infer/$f > $OUT/v110_code/$f"
            [ "${DRY_RUN:-0}" = "1" ] || git show "v1.1.0:src/ui2compose/infer/$f" > "$OUT/v110_code/$f"
        fi
    done
}

stage_app_pred_cache() {
    for split in val test; do
        run "$PY" -m ui2compose.evaluation.pred_cache --model "$APP_WEIGHTS" \
            --images "$PKG/dataset/images/$split" --out-dir "$OUT/pred_cache/$split"
    done
}

app_det() {
    # One evaluate_detector run: tag, weights, split, expected number of images.
    run "$PY" -m ui2compose.evaluation.evaluate_detector --model "$2" \
        --images "$PKG/dataset/images/$3" --labels "$PKG/dataset/labels/$3" \
        --dataset-yaml "$OUT/dataset_abs.yaml" --groups "$PKG/splits/app_split_manifest.csv" \
        --bootstrap 1000 --seed 0 --cross-check-val --preds-dir "$OUT/preds/$1" \
        --expect-images "$4" --out "$RES_APP/detector/$1/detector_$3_full.json"
}

stage_app_detector() {
    app_det baseline "$APP_BASELINE" val 1928
    app_det baseline "$APP_BASELINE" test 1900
    app_det oversample_5k "$APP_WEIGHTS" val 1928
    app_det oversample_5k "$APP_WEIGHTS" test 1900
    # Reference only: the released weights were trained on 1,512 screens of the new test split.
    app_det original_5k_contaminated "$WEIGHTS" test 1900
    run "$PY" -m ui2compose.evaluation.detector_ablation --base-dir "$EVAL_BASE" \
        --model "baseline=$RES_APP/detector/baseline" \
        --model "oversample_5k=$RES_APP/detector/oversample_5k" \
        --sets test_full,val_full --pairs oversample_5k-baseline \
        --groups "app_groups=$VIEW/splits/full/manifest.csv" \
        --groups "package_md5=$VIEW/splits/package_md5/manifest.csv" \
        --bootstrap 1000 --seed 0 --workers "$WORKERS" --cache-dir "$OUT/ablation_cache" \
        --classes Button,CheckBox,EditText,ImageView,ListView,ProgressBar,RadioButton,RecyclerView,SeekBar,Spinner,Switch,TextView,WebView \
        --checkpoint "baseline=$APP_BASELINE" --checkpoint "oversample_5k=$APP_WEIGHTS" \
        --train-config baseline=configs/train_baseline.yaml \
        --train-config oversample_5k=configs/train_oversample_5k.yaml \
        --out-json "$RES_APP/detector_table2.json" --out-csv "$RES_APP/detector_table2.csv" \
        --out-md "$RES_APP/detector_table2.md"
}

stage_app_sweep() {
    run "$PY" -m ui2compose.evaluation.layout_sweep run --data-dir "$VIEW" --split val \
        --pred-cache "$OUT/pred_cache/val" --out "$OUT/layout_sweep_val" --workers "$WORKERS"
    run "$PY" -m ui2compose.evaluation.layout_sweep report --data-dir "$VIEW" --split val \
        --pred-cache "$OUT/pred_cache/val" --sweep "$OUT/layout_sweep_val" --results-dir "$RES_APP" \
        --date 2026-10-21
    run "$PY" -m ui2compose.evaluation.layout_sweep crossval --data-dir "$VIEW" --split val \
        --sweep "$OUT/layout_sweep_val" --results-dir "$RES_APP" --repeats 200 --seed 0
}

stage_app_layout_test() {
    run "$PY" -m ui2compose.evaluation.layout_eval --data-dir "$VIEW" --split test --allow-test \
        --conditions C1,C2,C3,C4,C5,C1p --pred-cache "$OUT/pred_cache/test" \
        --v110-code "$OUT/v110_code" --out "$OUT/layout_test" --workers "$WORKERS" \
        --bootstrap 1000 --seed 0 --zss-check 100
    run "$PY" -m ui2compose.evaluation.layout_report --run "$OUT/layout_test" --data-dir "$VIEW" \
        --out-json "$RES_APP/layout_test.json" --out-md "$RES_APP/layout_test.md"
}

stage_app_taxonomy() {
    run "$PY" -m ui2compose.evaluation.failure_taxonomy run --data-dir "$VIEW" --split test \
        --allow-test --pred-cache "$OUT/pred_cache/test" --run "$OUT/layout_test" \
        --out-dir "$OUT/failure_taxonomy_test" --n 50 --seed 42
    run "$PY" -m ui2compose.evaluation.failure_taxonomy cases --data-dir "$VIEW" --split test \
        --allow-test --pred-cache "$OUT/pred_cache/test" --model "$APP_WEIGHTS" \
        --taxonomy "$OUT/failure_taxonomy_test/failure_taxonomy.json" \
        --out-dir "$OUT/failure_taxonomy_test/cases" --results-dir "$RES_APP" --name failure_taxonomy
}

for stage in "${STAGES[@]}"; do
    echo "=== $stage ($(date '+%H:%M:%S'))"
    start=$(date +%s)
    case "$stage" in
        setup|pred_cache|detector|detector_intervals|detector_ablation|sweep|sweep_crossval\
        |layout_val|layout_test|box_sensitivity|taxonomy) "stage_$stage" ;;
        app_setup|app_pred_cache|app_detector|app_sweep|app_layout_test|app_taxonomy)
            mkdir -p "$RES_APP"; "stage_$stage" ;;
        *) echo "unknown stage: $stage"; exit 2 ;;
    esac
    echo "=== $stage done in $(( $(date +%s) - start )) s"
done
