# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""ui2compose — Jetpack Compose UI code generation from Android screenshots.

Subpackages and modules:
    preprocess  ReDraw hierarchies + screenshots -> YOLO dataset
                (parse_redraw, build_dataset, oversample_dataset) and the
                app-level leak audit / unseen-app subset / app split (app_split)
    infer       component detection (detect_components), de-duplication
                (postprocess), UI-graph construction (build_ui_graph), layout
                and code generation (generate_compose_code), end-to-end
                pipeline and CLI (pipeline)
    evaluation  Ultralytics-equivalent AP with bootstrap intervals (ap,
                evaluate_detector), per-stage timing (benchmark), the v1.1.0
                metric script (calculate_metrics) and qualitative figures
                (visualize_successful_cases)
    train       detector training from configs/train_*.yaml
    webapp      Streamlit user interface
"""

__version__ = "1.2.0"
