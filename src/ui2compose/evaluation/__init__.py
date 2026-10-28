# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Detector evaluation and timing.

Modules:
    ap                          exact port of Ultralytics' matching and AP computation,
                                operating-point metrics, percentile bootstrap intervals, and
                                the bootstrap bias with basic / BCa intervals (jackknife)
    evaluate_detector           ui2compose-evaluate-detector: AP of one or more weights on an
                                image list (e.g. the unseen-app test subset) with bootstrap CIs
    detector_breakdown          python -m ...: summary tables of saved detector evaluations
                                (CIs under several groupings, operating point, seen vs unseen,
                                percentile vs basic vs BCa intervals)
    pred_cache                  python -m ...: cache of the pipeline's raw detections of many
                                screens (input of layout evaluations without the model)
    layout_eval                 ui2compose-eval-layout: generated layout trees vs the ReDraw
                                view hierarchies (TEDS-g / TEDS-o, edit operations, grouping
                                F1) per condition, with app-group bootstrap CIs; optional
                                post-hoc scoring of containment Boxes as containers
    layout_sweep                python -m ...: grid sweep of the de-duplication, containment,
                                cut-tolerance and fallback thresholds on the validation split
                                and the selection rule of the shipped defaults; post-hoc
                                out-of-sample check of the rule on halves of the val apps
    layout_report               python -m ...: Table-1 report of a layout evaluation run
                                (means with CIs, paired differences), the structure-only
                                correction-effort proxy and the post-hoc strata and
                                containment-Box sensitivity
    failure_taxonomy            python -m ...: every edit of a random sample of C4 screens
                                attributed to a cause (detector, layout rules, reference), with
                                illustrative cases traced by the pipeline's --explain
    benchmark                  ui2compose-benchmark: per-stage wall-clock timing of the pipeline
    calculate_metrics           ui2compose-evaluate: the v1.1.0 metric script (own matching at
                                conf 0.15; not Ultralytics-equivalent)
    visualize_successful_cases  ui2compose-visualize: qualitative case figures of the thesis
"""
