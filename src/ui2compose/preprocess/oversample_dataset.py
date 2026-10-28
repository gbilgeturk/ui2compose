# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Image-level oversampling of rare classes in the YOLO training subset.

Usage:
    python oversample_dataset.py --yolo-root data/yolo --target-min 2000
    (the released checkpoint runs/oversample_5k used --target-min 5000)

What the procedure does (this is the exact behaviour of oversample()):

  1. Instance counts n_c and the list of images containing each rare class c
     (RARE_CLASSES) are computed ONCE from the label files present in
     labels/train before any copying. They are not updated while copying.
  2. Rare classes are processed in ascending class-id order. A class is
     skipped if n_c == 0 or n_c >= target_min.
  3. For a processed class, the number of image copies is estimated once as
     needed = int((target_min - n_c) / (n_c / m_c)) + 1, where m_c is the
     number of images containing c (i.e. assuming every copy adds the mean
     number of c-instances per image).
  4. The images containing c are copied whole (image + label file) in a
     round-robin over their sorted file names, one copy per image per pass,
     named "<stem>_os<i>" where the suffix index i starts at 0 and is
     incremented after every pass, until `needed` copies have been written.
     If a copy name already exists (written for an earlier rare class), that
     image is skipped in this pass (not overwritten, not counted) and i is
     incremented for the rest of the pass. After every pass copying stops
     once i exceeds 100, i.e. at most 101 passes (copies per image per class).

Consequences, which the before/after report makes visible:
  * the final count of a rare class is approximate: it overshoots when the
    copies made for other rare classes also contain it (co-occurrence is not
    counted), and undershoots when the 101-pass cap is reached (classes with
    few source images);
  * every other class in a copied image is duplicated too, so frequent
    classes (TextView, ImageView, Button, ...) also grow;
  * there is no random selection: `seed` is passed to random.seed() but no
    random draw follows, so the result is fully determined by the sorted file
    names.

Determinism note: v1.1.0 iterated label files in os.scandir() order (not
sorted), which is filesystem-dependent; v1.2.0 sorts them. When `needed` is
not a multiple of the number of source images, the set of copied images can
therefore differ from the one used to train the released checkpoint.
"""

import argparse
import shutil
from pathlib import Path
from collections import Counter, defaultdict
import random
from typing import Dict

# ReDraw class names (13 class — remapped, CardView & Toolbar removed)
CLASS_NAMES = [
    'Button', 'CheckBox', 'EditText', 'ImageView', 'ListView',
    'ProgressBar', 'RadioButton', 'RecyclerView', 'SeekBar',
    'Spinner', 'Switch', 'TextView', 'WebView'
]

# Rare classes (oversampling targets — 13-class remapped indices)
# CheckBox(1), ProgressBar(5), RadioButton(6), RecyclerView(7),
# SeekBar(8), Spinner(9), Switch(10), WebView(12)
RARE_CLASSES = {1, 5, 6, 7, 8, 9, 10, 12}


def count_class_instances(labels_dir: Path) -> Counter:
    """Scans the label files and computes the total instance count for each class.

    Input:  labels_dir — directory containing the YOLO .txt label files
    Output: Counter — class id -> total instance count mapping
    """
    class_counts = Counter()
    for label_file in sorted(labels_dir.glob("*.txt")):
        content = label_file.read_text().strip()
        if not content:
            continue
        for line in content.split("\n"):
            if line.strip():
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
    return class_counts


def count_class_images(labels_dir: Path) -> Counter:
    """Counts, for each class, the label files (images) containing at least one instance.

    Input:  labels_dir — directory containing the YOLO .txt label files
    Output: Counter — class id -> number of images containing the class
    """
    image_counts = Counter()
    for label_file in sorted(labels_dir.glob("*.txt")):
        classes_in_image = set()
        for line in label_file.read_text().strip().split("\n"):
            parts = line.split()
            if len(parts) >= 5:
                classes_in_image.add(int(parts[0]))
        image_counts.update(classes_in_image)
    return image_counts


def get_images_by_rare_class(labels_dir: Path) -> dict:
    """Finds, for each rare class (RARE_CLASSES), the images that contain that class.

    Input:  labels_dir — directory containing the YOLO .txt label files
    Output: dict — rare class id -> list of image stems containing that class,
            in sorted file-name order
    """
    class_to_images = defaultdict(list)

    for label_file in sorted(labels_dir.glob("*.txt")):
        content = label_file.read_text().strip()
        if not content:
            continue

        classes_in_image = set()
        for line in content.split("\n"):
            if line.strip():
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id in RARE_CLASSES:
                        classes_in_image.add(class_id)

        for cls_id in classes_in_image:
            class_to_images[cls_id].append(label_file.stem)

    return class_to_images


def find_image_file(images_dir: Path, stem: str) -> Path:
    """Searches for the image file matching the given stem by trying extensions.

    Input:  images_dir — directory containing the images; stem — file stem (name without extension)
    Output: Path of the found image (.png/.jpg/.jpeg); None if absent
    """
    for ext in [".png", ".jpg", ".jpeg"]:
        img_path = images_dir / f"{stem}{ext}"
        if img_path.exists():
            return img_path
    return None


def _print_distribution(title: str, inst: Counter, imgs: Counter,
                        old_inst: Counter = None, old_imgs: Counter = None) -> None:
    """Prints per-class instance and image counts (with differences to a baseline).

    Input:  title — heading; inst, imgs — class id -> instance / image counts;
            old_inst, old_imgs — optional baseline counts for "(+diff)" columns
    Output: none (console output)
    """
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(f"{'':2s} {'class':15s}: {'instances':>16s} {'images':>16s}")
    for i, name in enumerate(CLASS_NAMES):
        n, m = inst.get(i, 0), imgs.get(i, 0)
        d_n = f"(+{n - old_inst.get(i, 0)})" if old_inst is not None and n > old_inst.get(i, 0) else ""
        d_m = f"(+{m - old_imgs.get(i, 0)})" if old_imgs is not None and m > old_imgs.get(i, 0) else ""
        marker = " <-- RARE" if i in RARE_CLASSES else ""
        print(f"{i:2d} {name:15s}: {n:7d} {d_n:>8s} {m:7d} {d_m:>8s}{marker}")


def oversample(yolo_root: Path, target_min: int, seed: int = 42) -> Dict[str, Dict[int, int]]:
    """Copies whole training images that contain rare classes (see module docstring).

    Input:  yolo_root — YOLO dataset root directory;
            target_min — target instance count per rare class (approximate,
            see the module docstring);
            seed — passed to random.seed(); has no effect because no random
            draw is made (kept for CLI compatibility with v1.1.0)
    Output: dict with per-class counts {'instances_before', 'images_before',
            'instances_after', 'images_after', 'copies_per_class'} (empty if
            labels/train is missing); side effects: "_os<k>"-suffixed copies
            under labels/train and images/train and a before/after report
    """
    random.seed(seed)  # no random draws follow; kept from v1.1.0

    train_labels = yolo_root / "labels" / "train"
    train_images = yolo_root / "images" / "train"

    if not train_labels.exists():
        print(f"[ERROR] Labels directory not found: {train_labels}")
        return {}

    # Compute the current distribution (once; not updated while copying)
    class_counts = count_class_instances(train_labels)
    image_counts = count_class_images(train_labels)
    _print_distribution("CURRENT CLASS DISTRIBUTION (before oversampling)",
                        class_counts, image_counts)

    # Find the images for each rare class
    class_to_images = get_images_by_rare_class(train_labels)

    print("\n" + "=" * 60)
    print("OVERSAMPLING PROCESS")
    print("=" * 60)

    total_copies = 0
    copies_per_class: Dict[int, int] = {}

    for cls_id in sorted(RARE_CLASSES):
        current_count = class_counts.get(cls_id, 0)

        if current_count == 0:
            print(f"\n[SKIP] {CLASS_NAMES[cls_id]}: No samples at all, skipping.")
            continue

        if current_count >= target_min:
            print(f"\n[SKIP] {CLASS_NAMES[cls_id]}: Already sufficient ({current_count} >= {target_min})")
            continue

        images_with_class = class_to_images.get(cls_id, [])
        if not images_with_class:
            print(f"\n[SKIP] {CLASS_NAMES[cls_id]}: No images found.")
            continue

        # How many instances do we need to add?
        needed_instances = target_min - current_count

        # How many instances per image on average?
        avg_instances_per_image = current_count / len(images_with_class)

        # How many copies are needed? (estimated once, from the pre-copy counts)
        needed_copies = int(needed_instances / avg_instances_per_image) + 1

        print(f"\n[PROCESS] {CLASS_NAMES[cls_id]}:")
        print(f"  - Current: {current_count} instances, {len(images_with_class)} images")
        print(f"  - Target: {target_min} instances")
        print(f"  - Copies needed: ~{needed_copies} images")

        # Copy the images containing the class round-robin, in sorted file-name
        # order (no random selection), one "_os<copy_idx>" copy per image per pass
        copy_count = 0
        copy_idx = 0

        while copy_count < needed_copies:
            for img_stem in images_with_class:
                if copy_count >= needed_copies:
                    break

                new_stem = f"{img_stem}_os{copy_idx}"

                # Copy the label
                src_label = train_labels / f"{img_stem}.txt"
                dst_label = train_labels / f"{new_stem}.txt"

                if dst_label.exists():
                    copy_idx += 1
                    continue

                # Copy the image
                src_img = find_image_file(train_images, img_stem)
                if src_img is None:
                    continue

                dst_img = train_images / f"{new_stem}{src_img.suffix}"

                shutil.copy2(src_label, dst_label)
                shutil.copy2(src_img, dst_img)

                copy_count += 1
                total_copies += 1

            copy_idx += 1

            # Prevent an infinite loop (caps the number of passes)
            if copy_idx > 100:
                break

        copies_per_class[cls_id] = copy_count
        print(f"  - Copied: {copy_count} images")

    # Compute the new distribution
    new_counts = count_class_instances(train_labels)
    new_images = count_class_images(train_labels)
    print()
    _print_distribution("NEW CLASS DISTRIBUTION (after oversampling)",
                        new_counts, new_images, class_counts, image_counts)

    print("\n" + "=" * 72)
    print(f"TOTAL: {total_copies} images copied")
    print("=" * 72)
    return {
        "instances_before": dict(class_counts), "images_before": dict(image_counts),
        "instances_after": dict(new_counts), "images_after": dict(new_images),
        "copies_per_class": copies_per_class,
    }


def main():
    """Reads the command-line arguments and runs the oversample function.

    Input:  none (takes --yolo-root, --target-min, --seed from the command line)
    Output: none (dataset-copying side effect of the oversample call)
    """
    parser = argparse.ArgumentParser(description="Oversampling for class imbalance")
    parser.add_argument("--yolo-root", type=str, default="data/yolo",
                        help="YOLO dataset root directory")
    parser.add_argument("--target-min", type=int, default=2000,
                        help="Target instance count per rare class (approximate; "
                             "the released checkpoint used 5000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Passed to random.seed(); no effect (no random draws)")

    args = parser.parse_args()

    oversample(Path(args.yolo_root), args.target_min, args.seed)


if __name__ == "__main__":
    main()