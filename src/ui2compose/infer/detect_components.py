# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""UI component detection with the trained YOLO model, and detection drawing.

ComponentDetector runs the YOLO weights (default: the app-split checkpoint) on a screenshot and returns
the detections sorted top to bottom. ultralytics (and therefore torch) is
imported only when a ComponentDetector is created, so the drawing helpers in
this module (draw_detections, layout_labels, label_style, class_color) can be
used on any detection list (e.g. ground-truth boxes or saved JSON files)
without loading the model.

Drawing is meant for figures: every class has a fixed colour, labels
("#5 Class 0.77", i.e. detection id, class and confidence; the id can be
switched off) sit on a filled background with black or white text chosen for
contrast, the font scale grows with the image width, and labels are placed so
that they do not overlap each other where possible. Detections removed by the
de-duplication step (ui2compose.infer.postprocess) can be drawn as dashed grey
boxes tagged 'removed'.
"""

import json
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ui2compose.infer.postprocess import intersection_area

BGR = Tuple[int, int, int]

# Fixed, mutually distinct colour per detector class (BGR, as used by OpenCV).
# Based on the Tableau-10 palette plus three extra hues for the 13 classes.
CLASS_COLORS_BGR: Dict[str, BGR] = {
    'Button': (44, 160, 44),         # green
    'CheckBox': (34, 189, 188),      # olive
    'EditText': (207, 190, 23),      # cyan
    'ImageView': (40, 39, 214),      # red
    'ListView': (189, 103, 148),     # purple
    'ProgressBar': (75, 86, 140),    # brown
    'RadioButton': (194, 119, 227),  # pink
    'RecyclerView': (14, 127, 255),  # orange
    'SeekBar': (128, 128, 0),        # teal
    'Spinner': (0, 0, 128),          # maroon
    'Switch': (128, 0, 0),           # navy
    'TextView': (180, 119, 31),      # blue
    'WebView': (0, 215, 255),        # gold
}
# Colours for class names outside the 13 detector classes (e.g. the legacy
# 'Card' class); chosen by a stable hash of the name, so always the same.
_FALLBACK_COLORS_BGR: Tuple[BGR, ...] = (
    (255, 0, 255), (0, 255, 0), (255, 255, 0), (0, 128, 255), (128, 0, 255),
)
# Colour of detections removed by de-duplication (drawn dashed).
REMOVED_COLOR_BGR: BGR = (150, 150, 150)

# Label geometry. Font scale is proportional to the image width:
# LABEL_FONT_SCALE_AT_REF_WIDTH at LABEL_REF_WIDTH pixels (0.9 at 1200 px,
# the width of the ReDraw screenshots), never below MIN_FONT_SCALE.
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_REF_WIDTH = 1200
LABEL_FONT_SCALE_AT_REF_WIDTH = 0.9
MIN_FONT_SCALE = 0.4
# Minimum stroke width (pixels) for both text and boxes.
MIN_LINE_THICKNESS = 2


def label_style(img_w: int) -> Dict[str, float]:
    """Computes font and line sizes for drawing on an image of the given width.

    Input:  img_w — image width in pixels
    Output: dict with 'font_scale' (0.9 at 1200 px, proportional to width,
            >= MIN_FONT_SCALE), 'text_thickness' and 'box_thickness'
            (>= MIN_LINE_THICKNESS), 'pad' (label padding in pixels),
            'dash' and 'gap' (dash pattern of removed boxes, in pixels)
    """
    font_scale = max(MIN_FONT_SCALE,
                     LABEL_FONT_SCALE_AT_REF_WIDTH * float(img_w) / LABEL_REF_WIDTH)
    return {
        'font_scale': font_scale,
        'text_thickness': max(MIN_LINE_THICKNESS, int(round(2.2 * font_scale))),
        'box_thickness': max(MIN_LINE_THICKNESS, int(round(3.0 * font_scale))),
        'pad': max(2, int(round(6.0 * font_scale))),
        'dash': max(6, int(round(18.0 * font_scale))),
        'gap': max(4, int(round(10.0 * font_scale))),
    }


def class_color(class_name: str) -> BGR:
    """Returns the fixed drawing colour of a class.

    Input:  class_name — detector class name
    Output: BGR colour; one of CLASS_COLORS_BGR for the 13 detector classes,
            otherwise a fallback colour chosen by a stable (CRC32) hash of the name
    """
    if class_name in CLASS_COLORS_BGR:
        return CLASS_COLORS_BGR[class_name]
    index = zlib.crc32(class_name.encode('utf-8')) % len(_FALLBACK_COLORS_BGR)
    return _FALLBACK_COLORS_BGR[index]


def text_color_for(background_bgr: Sequence[int]) -> BGR:
    """Chooses black or white text for legibility on a given background colour.

    Input:  background_bgr — label background colour (B, G, R)
    Output: (0, 0, 0) if the background's luma (ITU-R BT.601) exceeds 140, else (255, 255, 255)
    """
    b, g, r = background_bgr
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luma > 140 else (255, 255, 255)


def format_label(detection: Dict, removed: bool = False, show_id: bool = False) -> str:
    """Builds the label text of a detection.

    Input:  detection — dict with 'class_name' and 'confidence' (and 'id' when
                show_id is True);
            removed — True for a detection removed by de-duplication;
            show_id — True to prefix the detection id, so that a figure can be
                read together with the --explain traces, which name detections by id
    Output: "Class 0.77" ("#5 Class 0.77" with show_id), with " removed"
            appended when removed is True
    """
    text = f"{detection['class_name']} {float(detection['confidence']):.2f}"
    if show_id:
        text = f"#{detection['id']} {text}"
    return f"{text} removed" if removed else text


def _label_size(text: str, style: Dict[str, float]) -> Tuple[int, int]:
    """Measures the size of a label rectangle including padding.

    Input:  text — label text; style — output of label_style()
    Output: (width, height) in pixels
    """
    (tw, th), baseline = cv2.getTextSize(text, LABEL_FONT, style['font_scale'],
                                         int(style['text_thickness']))
    pad = int(style['pad'])
    return tw + 2 * pad, th + baseline + 2 * pad


def layout_labels(detections: List[Dict], img_w: int, img_h: int,
                  labels: Optional[List[str]] = None) -> List[List[int]]:
    """Places one label rectangle per detection so that labels avoid each other.

    Labels are placed in list order; earlier detections get the preferred
    positions. For each box the candidates are tried in order: above the box,
    inside its top edge, below the box, inside its bottom edge (all left-
    aligned with the box and clamped to the image). If all four collide with
    already placed labels, free slots are searched downwards and then upwards
    from the first candidate in half-label-height steps; if none is free the
    first candidate is used (labels then overlap).

    Input:  detections — list of dicts with 'bbox' [x1, y1, x2, y2] in pixels
                (and 'class_name', 'confidence' when labels is None)
            img_w, img_h — image size in pixels
            labels — label texts (default: format_label of each detection)
    Output: list of label rectangles [x1, y1, x2, y2] (ints), one per detection, in input order
    """
    style = label_style(img_w)
    if labels is None:
        labels = [format_label(d) for d in detections]
    placed: List[List[int]] = []
    for det, text in zip(detections, labels):
        lw, lh = _label_size(text, style)
        lw, lh = min(lw, img_w), min(lh, img_h)
        bx1, by1, bx2, by2 = (int(round(v)) for v in det['bbox'])

        def clamp(x: int, y: int) -> List[int]:
            """Clamps a label's top-left corner so the label lies inside the image.

            Input:  x, y — proposed top-left corner
            Output: label rectangle [x1, y1, x2, y2]
            """
            x = min(max(0, x), img_w - lw)
            y = min(max(0, y), img_h - lh)
            return [x, y, x + lw, y + lh]

        def free(rect: List[int]) -> bool:
            """Checks that a rectangle does not overlap any placed label.

            Input:  rect — [x1, y1, x2, y2]
            Output: True if the intersection with every placed label is empty
            """
            return all(intersection_area(rect, other) <= 0.0 for other in placed)

        candidates = [clamp(bx1, by1 - lh), clamp(bx1, by1), clamp(bx1, by2),
                      clamp(bx1, by2 - lh)]
        chosen = next((c for c in candidates if free(c)), None)
        if chosen is None:
            step = max(1, lh // 2)
            start = candidates[0][1]
            search = list(range(start + step, img_h - lh + 1, step)) + \
                list(range(start - step, -1, -step))
            chosen = next((clamp(bx1, y) for y in search if free(clamp(bx1, y))),
                          candidates[0])
        placed.append(chosen)
    return placed


def _draw_dashed_rectangle(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                           color: BGR, thickness: int, dash: int, gap: int) -> None:
    """Draws a dashed axis-aligned rectangle in place.

    Input:  img — BGR image (modified in place); pt1, pt2 — opposite corners;
            color — BGR colour; thickness — line width; dash, gap — dash pattern in pixels
    Output: none (side effect of drawing on img)
    """
    x1, y1 = pt1
    x2, y2 = pt2
    edges = [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
             ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]
    for (sx, sy), (ex, ey) in edges:
        length = max(abs(ex - sx), abs(ey - sy))
        if length == 0:
            continue
        pos = 0
        while pos < length:
            end = min(pos + dash, length)
            a = (int(sx + (ex - sx) * pos / length), int(sy + (ey - sy) * pos / length))
            b = (int(sx + (ex - sx) * end / length), int(sy + (ey - sy) * end / length))
            cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
            pos += dash + gap


def draw_detections(image: np.ndarray, detections: List[Dict],
                    removed: Optional[List[Dict]] = None, show_ids: bool = False) -> np.ndarray:
    """Draws detections (and optionally removed detections) on a copy of an image.

    Kept detections are drawn as solid boxes in their class colour with a thin
    black outline (visible on any background). Removed detections are drawn
    first, underneath, as dashed grey boxes. Every box gets a label "Class 0.77"
    ("Class 0.31 removed" for removed ones; with show_ids "#5 Class 0.77") on a
    filled background in the box colour, with black or white text chosen for
    contrast. Labels of kept
    detections are placed first (see layout_labels), then those of removed ones.

    Input:  image — BGR image (numpy array, H x W x 3); not modified
            detections — list of dicts with 'class_name', 'confidence', 'bbox'
            removed — optional list of removed detections (same keys)
            show_ids — True to prefix every label with the detection id
                (the detections then need an 'id')
    Output: annotated copy of the image (numpy array)
    """
    img = image.copy()
    img_h, img_w = img.shape[:2]
    style = label_style(img_w)
    box_t = int(style['box_thickness'])
    removed = removed or []

    def corners(d: Dict) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Returns integer box corners of a detection.

        Input:  d — detection dict with 'bbox'
        Output: ((x1, y1), (x2, y2))
        """
        x1, y1, x2, y2 = (int(round(v)) for v in d['bbox'])
        return (x1, y1), (x2, y2)

    # Boxes: removed (dashed, underneath), then kept (outline + class colour).
    for d in removed:
        p1, p2 = corners(d)
        _draw_dashed_rectangle(img, p1, p2, REMOVED_COLOR_BGR, box_t,
                               int(style['dash']), int(style['gap']))
    for d in detections:
        p1, p2 = corners(d)
        cv2.rectangle(img, p1, p2, (0, 0, 0), box_t + 2)
        cv2.rectangle(img, p1, p2, class_color(d['class_name']), box_t)

    # Labels: kept first so they get the preferred positions.
    all_dets = list(detections) + list(removed)
    texts = [format_label(d, show_id=show_ids) for d in detections] + \
        [format_label(d, removed=True, show_id=show_ids) for d in removed]
    colors = [class_color(d['class_name']) for d in detections] + \
        [REMOVED_COLOR_BGR] * len(removed)
    rects = layout_labels(all_dets, img_w, img_h, labels=texts)
    pad = int(style['pad'])
    for text, color, (lx1, ly1, lx2, ly2) in zip(texts, colors, rects):
        cv2.rectangle(img, (lx1, ly1), (lx2, ly2), color, cv2.FILLED)
        (_, th), _ = cv2.getTextSize(text, LABEL_FONT, style['font_scale'],
                                     int(style['text_thickness']))
        cv2.putText(img, text, (lx1 + pad, ly1 + pad + th), LABEL_FONT,
                    style['font_scale'], text_color_for(color),
                    int(style['text_thickness']), cv2.LINE_AA)
    return img


class ComponentDetector:
    """YOLO-based UI component detector"""

    def __init__(self, model_path: str, dataset_yaml: str):
        """Loads the trained YOLO model and takes the class names from the model.

        Input:  model_path — path to trained YOLO weights file (e.g. 'runs/train/exp/weights/best.pt')
                dataset_yaml — unused; kept for backward compatibility of the
                    constructor (class names are read from the model itself)
        Output: none (side effect of assigning self.model and self.class_names)
        """
        # Imported here so that this module (and its drawing helpers) can be
        # imported without ultralytics/torch.
        from ultralytics import YOLO

        self.model = YOLO(model_path)

        # Use model's own class names (safest - avoids index mismatch)
        self.class_names = list(self.model.names.values())

        print(f"✓ Model loaded: {model_path}")
        print(f"✓ Classes ({len(self.class_names)}): {self.class_names}")

    def effective_imgsz(self, imgsz: Optional[int] = None) -> Optional[int]:
        """Returns the inference resolution that detect() uses for a given imgsz argument.

        Input:  imgsz — the value passed to detect() (None = not passed)
        Output: imgsz if it is not None; otherwise the imgsz stored in the
                checkpoint's training arguments (Ultralytics keeps it in
                model.overrides), or None if the model records none (Ultralytics
                then uses its own default)
        """
        if imgsz is not None:
            return imgsz
        overrides = getattr(self.model, 'overrides', None) or {}
        return overrides.get('imgsz')

    def detect(self, image_path: str, conf_threshold: float = 0.3,
               iou_threshold: float = 0.7, imgsz: Optional[int] = None,
               agnostic_nms: bool = False, max_det: int = 300) -> List[Dict]:
        """Detects UI components in the image and returns them sorted top to bottom.

        The defaults reproduce v1.1.0, which passed only conf=0.3 and otherwise
        used the ultralytics predict defaults (NMS IoU 0.7, class-aware NMS,
        at most 300 detections) and the checkpoint's training resolution: with
        imgsz None the argument is not passed to predict(), so Ultralytics uses
        the imgsz stored in the checkpoint (640 for both bundled weights), for
        any checkpoint. None of these values was tuned on the validation split;
        they are provisional.

        Input:  image_path — screenshot file path
                conf_threshold — minimum detection confidence (default 0.3)
                iou_threshold — IoU threshold of the detector's NMS (default 0.7)
                imgsz — inference resolution in pixels (default None = the
                    checkpoint's training size, see effective_imgsz)
                agnostic_nms — if True, NMS suppresses across classes (default False)
                max_det — maximum number of detections per image (default 300)
        Output: list of detection dictionaries: {'id', 'class_name', 'class_id',
                'confidence', 'bbox' [x1,y1,x2,y2 pixels], 'bbox_norm' [xc,yc,w,h 0-1],
                'center' [xc,yc pixels], 'width', 'height', 'img_w', 'img_h'
                [image size in pixels]}, sorted by y1 and numbered 0..n-1 in that
                order. Raises FileNotFoundError if the image cannot be read.
        """
        # Load image to get dimensions (and fail early on a bad path)
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        img_h, img_w = img.shape[:2]

        # Run inference (imgsz only when given: otherwise the checkpoint's own)
        predict_kwargs = dict(source=image_path, conf=conf_threshold, iou=iou_threshold,
                              agnostic_nms=agnostic_nms, max_det=max_det, verbose=False)
        if imgsz is not None:
            predict_kwargs['imgsz'] = imgsz
        results = self.model.predict(**predict_kwargs)[0]

        detections = []

        # Parse results
        boxes = results.boxes
        for idx, box in enumerate(boxes):
            # Get box coordinates (xyxy format)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Calculate center and normalized coordinates
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            # Normalized [0-1]
            xc_norm = xc / img_w
            yc_norm = yc / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            detection = {
                'id': idx,
                'class_name': self.class_names[class_id],
                'class_id': class_id,
                'confidence': confidence,
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'bbox_norm': [float(xc_norm), float(yc_norm), float(w_norm), float(h_norm)],
                'center': [float(xc), float(yc)],
                'width': float(w),
                'height': float(h),
                'img_w': int(img_w),
                'img_h': int(img_h)
            }
            detections.append(detection)

        # Sort by y-coordinate (top to bottom)
        detections.sort(key=lambda x: x['bbox'][1])

        # Reassign IDs after sorting
        for idx, det in enumerate(detections):
            det['id'] = idx

        return detections

    def visualize(self, image_path: str, detections: List[Dict], output_path: str = None,
                  removed: Optional[List[Dict]] = None, show_ids: bool = True) -> np.ndarray:
        """Draws detection boxes and labels onto the image (see draw_detections).

        Input:  image_path — screenshot file path
                detections — detection list output by detect()
                output_path — file path to save to (optional)
                removed — detections removed by de-duplication, drawn as dashed
                    grey boxes tagged 'removed' (optional)
                show_ids — prefix every label with the detection id (default
                    True, so that the figure can be read with --explain traces)
        Output: image with boxes drawn (numpy array); also saves to file if output_path is given.
                Raises FileNotFoundError if the image cannot be read.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        img = draw_detections(img, detections, removed=removed, show_ids=show_ids)

        if output_path:
            cv2.imwrite(output_path, img)
            print(f"✓ Saved visualization: {output_path}")

        return img


def main():
    """Runs the detector on a sample test image.

    Input:  none
    Output: none (side effect of writing output/detections.png and output/detections.json)
    """
    # Paths (ADJUST THESE!)
    model_path = "runs/oversample_5k_appsplit/weights/best.pt"
    dataset_yaml = "configs/dataset.yaml"
    test_image = "examples/com.shazam.android-screens_screenshot_3.png"

    # Initialize detector
    detector = ComponentDetector(model_path, dataset_yaml)

    # Detect
    detections = detector.detect(test_image, conf_threshold=0.3)

    print(f"\n📦 Detected {len(detections)} components:")
    for det in detections[:5]:  # Show first 5
        print(f"  {det['id']}: {det['class_name']} (conf={det['confidence']:.2f})")

    # Visualize
    detector.visualize(test_image, detections, "output/detections.png")

    # Save JSON
    output_json = "output/detections.json"
    with open(output_json, 'w') as f:
        json.dump(detections, f, indent=2)
    print(f"✓ Saved JSON: {output_json}")


if __name__ == "__main__":
    main()
