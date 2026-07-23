from pathlib import Path
import argparse, shutil, random, json
import sys
import yaml
from typing import Optional


SEED = 42
SPLIT = (0.80, 0.10, 0.10)  # train, val, test
IMG_EXTS = (".png", ".jpg", ".jpeg")

def parse_args():
    """Defines and reads the command-line arguments.

    Input:  none (takes values from the command line)
    Output: args object — settings such as args.labels_root, args.yolo_root
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", default="data/interim/labels", type=str,
                    help="Root of the parse_redraw outputs")
    ap.add_argument("--images-root", default="data/raw", type=str,
                    help="Root of the ReDraw images")
    ap.add_argument("--yolo-root", default="data/yolo", type=str,
                    help="YOLO dataset destination")
    ap.add_argument("--dataset-yaml", default="configs/dataset.yaml", type=str,
                    help="(opt) existing YAML; can be written if absent")
    ap.add_argument("--write-yaml", action="store_true",
                    help="Rewrite the existing YAML (reads the same file for names)")
    ap.add_argument("--clean", action="store_true",
                    help="Clean the data/yolo folder first")
    ap.add_argument("--symlink", action="store_true",
                    help="Create symlinks instead of copying images")
    ap.add_argument("--names-json", default=None, type=str,
                    help="(opt) Optionally provide the class list as JSON: \"['Button', ...]\"")
    return ap.parse_args()

# ---------- helpers ----------

def link_or_copy(src: Path, dst: Path, symlink=False):
    """Links the image file to the destination as a symlink or copies it.

    Input:  src — source file path; dst — destination file path;
            symlink — if True, tries a symlink instead of a copy
    Output: none (side effect of creating a file/symlink at dst;
            falls back to copying if the symlink fails)
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if symlink:
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)
        except OSError:
            # Fall back to copying if some filesystems lack permission
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)

def non_empty_label(path: Path) -> bool:
    """Checks whether the label file has at least one valid YOLO line (5+ fields).

    Input:  path — path to the .txt label file to check
    Output: bool — True if there is a valid line, False if empty/unreadable
    """
    try:
        for ln in path.read_text(encoding="utf-8").splitlines():
            if len(ln.split()) >= 5:
                return True
    except Exception:
        pass
    return False

def unique_name(app: str, stem: str) -> str:
    """Combines the app name and the file stem into a collision-free file name.

    Input:  app — name of the app folder; stem — screenshot stem, e.g. "screenshot_1"
    Output: unique name in the form "app_stem" (long but collision-free)
    """
    return f"{app}_{stem}"

def find_image_for(app_dir: Path, stem: str):
    """Searches for the image file matching the given stem by trying extensions.

    Input:  app_dir — the app's raw image folder; stem — file stem, e.g. "screenshot_1"
    Output: Path of the found image (.png/.jpg/.jpeg); None if absent
    """
    for ext in IMG_EXTS:
        p = app_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def load_names(yaml_path: Path, override_json: Optional[str]):
    """Reads the class name list first from the JSON override, otherwise from dataset.yaml.

    Input:  yaml_path — path to the YAML file containing the 'names' list;
            override_json — (opt) JSON text or path to a JSON file
    Output: names — ordered list of class names; empty list if no source exists
    """
    # 1) Override via JSON
    if override_json:
        try:
            if Path(override_json).exists():
                return json.loads(Path(override_json).read_text(encoding="utf-8"))
            return json.loads(override_json)
        except Exception:
            print("[WARN] names-json could not be parsed; dataset.yaml will be used.")
    # 2) Read from dataset.yaml
    if yaml_path.exists():
        data = yaml.safe_load(open(yaml_path, "r", encoding="utf-8"))
        names = data.get("names", [])
        if names:
            return names
    return []

def write_dataset_yaml(out_yaml: Path, yolo_root: Path, names: list[str]):
    """Generates a dataset.yaml with path/train/val/test/nc/names fields for YOLO training.

    Input:  out_yaml — path to the YAML file to write; yolo_root — dataset root directory;
            names — list of class names (if empty, nc=0 is written)
    Output: none (side effect of creating/overwriting the out_yaml file)
    """
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    nc = len(names) if names else 0
    text = (
        f"# autogenerated\n"
        f"path: {yolo_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {nc}\n"
        f"names:\n"
    )
    for n in names:
        text += f"  - {n}\n"
    out_yaml.write_text(text, encoding="utf-8")

# ---------- main flow ----------

def main():
    """Matches label-image pairs and builds the YOLO dataset with an 80/10/10 split.

    Input:  none (takes settings from the command line via parse_args)
    Output: none (side effects of copying images/labels under data/yolo,
            optionally writing dataset.yaml, and printing a summary)
    """
    args = parse_args()
    labels_root = Path(args.labels_root)
    images_root = Path(args.images_root)
    yolo_root   = Path(args.yolo_root)
    yaml_path   = Path(args.dataset_yaml)

    if not labels_root.exists():
        print(f"[ERR] labels-root does not exist: {labels_root}"); sys.exit(1)
    if not images_root.exists():
        print(f"[ERR] images-root does not exist: {images_root}"); sys.exit(1)

    # Start clean
    if args.clean and yolo_root.exists():
        shutil.rmtree(yolo_root)

    # Prepare target directories
    for sub in ["images/train", "images/val", "images/test",
                "labels/train", "labels/val", "labels/test"]:
        (yolo_root / sub).mkdir(parents=True, exist_ok=True)

    # Get class names (from yaml or json)
    names = load_names(yaml_path, args.names_json)

    # Walk the label folders: data/interim/labels/<app>/*.txt
    pairs = []
    missing_images = []

    for app_dir in sorted(labels_root.iterdir()):
        if not app_dir.is_dir():
            continue
        app = app_dir.name

        # Skip empty folders
        label_files = sorted([p for p in app_dir.glob("*.txt") if non_empty_label(p)])
        if not label_files:
            continue

        # Match
        raw_app_dir = images_root / app
        for lf in label_files:
            stem = lf.stem  # "screenshot_i"
            img = find_image_for(raw_app_dir, stem)
            if img is None:
                missing_images.append(str(lf))
                continue
            pairs.append((lf, img, app, stem))

    n = len(pairs)
    if n == 0:
        print("✅ YOLO dataset prepared.")
        print("Total pairs: 0  | train:0 val:0 test:0")
        print(f"Unmatched image count (label exists but image missing): {len(missing_images)}")
        print(f"YAML: {yaml_path.resolve()}")
        print(f"Folder: {yolo_root.resolve()}")
        return

    # Deterministic split
    random.seed(SEED)
    random.shuffle(pairs)
    t = int(n * SPLIT[0]); v = int(n * SPLIT[1])
    train, val, test = pairs[:t], pairs[t:t+v], pairs[t+v:]

    # Copy/link
    for subset, items in (("train", train), ("val", val), ("test", test)):
        for lf, img, app, stem in items:
            base = unique_name(app, stem)
            out_img = yolo_root / "images" / subset / (base + img.suffix.lower())
            out_lab = yolo_root / "labels" / subset / (base + ".txt")
            link_or_copy(img, out_img, symlink=args.symlink)
            shutil.copy2(lf, out_lab)

    # Write YAML (if requested)
    if args.write-yaml if False else False:
        pass  # (guard for IDE syntax highlighters)

    if args.write_yaml:
        # produce the file even if no name list was found (nc=0)
        write_dataset_yaml(yaml_path, yolo_root, names)

    # Summary
    print("✅ YOLO dataset prepared.")
    print(f"Total pairs: {n}  | train:{len(train)} val:{len(val)} test:{len(test)}")
    print(f"Unmatched image count (label exists but image missing): {len(missing_images)}")
    print(f"YAML: {yaml_path.resolve()}")
    print(f"Folder: {yolo_root.resolve()}")

if __name__ == "__main__":
    main()