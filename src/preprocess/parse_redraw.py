from pathlib import Path
import argparse
import re
import yaml
import xml.etree.ElementTree as ET
from PIL import Image

# ------------ Arguments ------------
def parse_args():
    """Defines and reads the command-line arguments.

    Input:  none (takes values from the command line)
    Output: args object — settings such as args.raw_root, args.min_area
    """
    ap = argparse.ArgumentParser(description="ReDraw XML + IMG -> YOLO labels")
    ap.add_argument("--raw-root", default="data/raw", type=str, help="Where the raw dataset (XML+PNG) is read from")
    ap.add_argument("--out-root", default="data/interim/labels", type=str, help="Where the generated output .txt files are written")
    ap.add_argument("--dataset-yaml", default="configs/dataset.yaml", type=str, help="Config file the 13-class list is read from, id and class mapping")
    ap.add_argument("--drop-container", default=True, type=lambda x: str(x).lower() in {"1","true","yes"}, help ="Skips structural nodes whose name contains 'container', Table 3.2 Container Removal")
    ap.add_argument("--min-area", default=0.00005, type=float, help="Ratio to screen area (w*h), the area filter from Table 3.2")
    ap.add_argument("--dedup-iou", default=0.85, type=float, help="If two boxes of the same class overlap by %%85, treats them as duplicates and removes one")
    ap.add_argument("--limit", default=0, type=int, help="To try only the first N screen pairs and speed up development, 0 = all")
    ap.add_argument("--verbose", default=False, type=lambda x: str(x).lower() in {"1","true","yes"}, help = "to print detailed logs during processing")
    return ap.parse_args()

# ------------ Helpers ------------
IMG_EXTS = (".png", ".jpg", ".jpeg") #file extensions
BOUNDS_RE = re.compile(r"\[(\d+),\s*(\d+)\]\[(\d+),\s*(\d+)\]") #regex that extracts bounds info from the XML

ALIASES = {
    # Text family
    "AppCompatTextView": "TextView",
    "MaterialTextView": "TextView",
    "CheckedTextView": "TextView",
    "TextInputEditText": "EditText",
    "AutoCompleteTextView": "EditText",
    # Image & button
    "AppCompatImageView": "ImageView",
    "AppCompatButton": "Button",
    "MaterialButton": "Button",
    "ImageButton": "Button",
    # Switch
    "SwitchCompat": "Switch",
}

def load_names(yaml_path: Path):
    """Reads the 13-class list from dataset.yaml; the source of the whitelist and id mapping.

    Input:  yaml_path — path to the YAML file containing the 'names' list
    Output: (names, name2id) — ordered class list and name->number dictionary
            e.g. {"Button": 0, "CheckBox": 1, ...}
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {yaml_path}")
    data = yaml.safe_load(open(yaml_path, "r"))
    names = data.get("names", [])
    if not names:
        raise ValueError(f"'names' missing or empty in dataset.yaml: {yaml_path}")
    return names, {n: i for i, n in enumerate(names)}

def short_class_name(full: str) -> str:
    """Shortens the full class name and normalizes it via ALIASES (Table 3.3).

    Input:  full — full name from the XML, e.g. "android.widget.AppCompatTextView"
    Output: normalized short name, e.g. "TextView" (if no match, the short name is returned as-is)
    """
    s = (full or "").split(".")[-1]
    return ALIASES.get(s, s)

def parse_bounds(s: str):
    """Converts the bounds string into four pixel coordinates.

    Input:  s — text in the form "[x1,y1][x2,y2]", e.g. "[0,63][1080,231]"
    Output: (x1, y1, x2, y2) integer quadruple; None if the format is malformed
            or the box is inverted/zero-sized
    """
    if not s:
        return None
    m = BOUNDS_RE.match(s)
    if not m:
        return None
    x1, y1, x2, y2 = map(int, m.groups())
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def to_xywhn(box, W, H):
    """Converts corner coordinates to YOLO format: center + size, normalized to 0-1.

    Input:  box — (x1, y1, x2, y2) pixel coordinates; W, H — image size
    Output: (xc, yc, w, h) — center and size normalized to the screen dimensions
    """
    x1, y1, x2, y2 = box
    xc = (x1 + x2) / 2.0 / W
    yc = (y1 + y2) / 2.0 / H
    w  = (x2 - x1) / W
    h  = (y2 - y1) / H
    # clamp
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    w  = min(max(w, 1e-6), 1.0)
    h  = min(max(h, 1e-6), 1.0)
    return (xc, yc, w, h)

def iou_xywh(a, b):
    """Computes the IoU (intersection-over-union ratio) of two boxes; used for dedup.

    Input:  a, b — two normalized boxes in (xc, yc, w, h) form
    Output: ratio between 0.0-1.0 (0 = no overlap, 1 = identical)
    """
    ax1, ay1 = a[0] - a[2]/2, a[1] - a[3]/2
    ax2, ay2 = a[0] + a[2]/2, a[1] + a[3]/2
    bx1, by1 = b[0] - b[2]/2, b[1] - b[3]/2
    bx2, by2 = b[0] + b[2]/2, b[1] + b[3]/2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter/union if union > 0 else 0.0

def match_image_for(xml_path: Path):
    """Finds the screenshot corresponding to the XML file.

    Input:  xml_path — path to the hierarchy_N.xml file
    Output: path to screenshot_N.png/jpg/jpeg in the same folder; None if absent
    """
    m = re.search(r"hierarchy_(\d+)\.xml$", xml_path.name)
    if not m:
        return None
    idx = m.group(1)
    stem = f"screenshot_{idx}"
    for ext in IMG_EXTS:
        p = xml_path.with_name(stem + ext)
        if p.exists():
            return p
    return None

def walk_nodes(elem):
    """Traverses all nodes in the XML tree (itself + all descendants) in order.

    Input:  elem — starting XML node (usually the root)
    Output: generator yielding each node one by one
    """
    yield elem
    for ch in list(elem):
        yield from walk_nodes(ch)

# ------------ Main functions ------------
def extract_labels(xml_path: Path, W: int, H: int, name2id: dict,
                   drop_container=True,
                   min_area=0.00010, dedup_iou=0.95, verbose=False):
    """Processes a single XML and produces all YOLO labels for that screen (the heart of the script).

    Traverses the tree; for each node applies, in order: container removal ->
    box extraction -> area filter -> name normalization -> whitelist -> IoU dedup.

    Input:  xml_path — the hierarchy XML; W, H — size of the paired image;
            name2id — class whitelist/id dictionary; others — filter settings
    Output: [(cid, (xc, yc, w, h)), ...] — list of labels that passed the filters
    """
    labels = []  # [(cid, (xc,yc,w,h))]
    root = ET.parse(xml_path).getroot()

    for n in walk_nodes(root):
        cls_full = n.attrib.get("class") or n.attrib.get("className") or ""
        # Drop containers outright
        if drop_container and "container" in cls_full.lower():
            continue

        # extract the box
        b = None
        if "bounds" in n.attrib:
            b = parse_bounds(n.attrib.get("bounds"))
        if not b:
            try:
                x = int(n.attrib.get("x", "0"))
                y = int(n.attrib.get("y", "0"))
                w = int(n.attrib.get("width", "0"))
                h = int(n.attrib.get("height", "0"))
                b = (x, y, x+w, y+h) if (w > 0 and h > 0) else None
            except Exception:
                b = None
        if not b:
            continue

        box = to_xywhn(b, W, H)
        # min area filter
        if box[2] * box[3] < min_area:
            continue

        cname = short_class_name(cls_full)

        # class id
        if cname not in name2id:
            # skip if this class is not in dataset.yaml
            continue
        cid = name2id[cname]

        # intra-class dedup
        keep = True
        for cid2, box2 in labels:
            if cid2 == cid and iou_xywh(box, box2) >= dedup_iou:
                keep = False
                break
        if not keep:
            continue

        labels.append((cid, box))

    if verbose:
        by_class = {}
        for cid, _ in labels:
            by_class[cid] = by_class.get(cid, 0) + 1
        if by_class:
            print(f"    kept {sum(by_class.values())} labels -> {by_class}")

    return labels

def main():
    """Orchestrates the flow: walks all app folders, matches XML-image pairs,
    extracts labels with extract_labels, and writes them to .txt files.

    Input:  none (takes settings from parse_args)
    Output: data/interim/labels/<app>/screenshot_N.txt files + summary output
    """
    args = parse_args()
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    names, name2id = load_names(Path(args.dataset_yaml))

    app_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir() and p.name.endswith("-screens")])

    total_pairs = 0
    total_written = 0

    for app in app_dirs:
        xmls = sorted(app.glob("hierarchy_*.xml"),
                      key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)) if re.search(r"(\d+)", p.stem) else 0)
        if not xmls:
            continue

        app_out = out_root / app.name
        app_out.mkdir(parents=True, exist_ok=True)

        for xml_path in xmls:
            if args.limit and total_pairs >= args.limit:
                break

            img_path = match_image_for(xml_path)
            if img_path is None or not img_path.exists():
                if args.verbose:
                    print(f"[skip] image not found for: {xml_path}")
                continue

            try:
                with Image.open(img_path) as im:
                    W, H = im.size
            except Exception as e:
                if args.verbose:
                    print(f"[skip] cannot open image: {img_path} ({e})")
                continue

            total_pairs += 1
            if args.verbose:
                print(f"[pair] {app.name} :: {xml_path.name} <-> {img_path.name}  ({W}x{H})")

            try:
                labels = extract_labels(
                    xml_path, W, H, name2id,
                    drop_container=args.drop_container,
                    min_area=args.min_area,
                    dedup_iou=args.dedup_iou,
                    verbose=args.verbose
                )
            except Exception as e:
                if args.verbose:
                    print(f"[skip] parse error at {xml_path}: {e}")
                continue

            if not labels:
                continue

            m = re.search(r"hierarchy_(\d+)\.xml$", xml_path.name)
            idx = m.group(1) if m else "0"
            out_txt = app_out / f"screenshot_{idx}.txt"
            with open(out_txt, "w", encoding="utf-8") as f:
                for cid, (xc, yc, w, h) in labels:
                    f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            total_written += 1
            if args.verbose:
                print(f"    wrote: {out_txt}")

    print(f"✅ Done. Total pairs: {total_pairs}, label files written: {total_written}")
    print(f"Outputs: {out_root.resolve()}")

if __name__ == "__main__":
    main()