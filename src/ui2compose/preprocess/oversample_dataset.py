# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""
Usage:
    python oversample_dataset.py --yolo-root data/yolo --target-min 2000
"""

import argparse
import shutil
from pathlib import Path
from collections import Counter, defaultdict
import random

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
    for label_file in labels_dir.glob("*.txt"):
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


def get_images_by_rare_class(labels_dir: Path) -> dict:
    """Finds, for each rare class (RARE_CLASSES), the images that contain that class.

    Input:  labels_dir — directory containing the YOLO .txt label files
    Output: dict — rare class id -> list of image stems containing that class
    """
    class_to_images = defaultdict(list)

    for label_file in labels_dir.glob("*.txt"):
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


def oversample(yolo_root: Path, target_min: int, seed: int = 42):
    """Oversamples up to the target count by copying train images that contain rare classes.

    Input:  yolo_root — YOLO dataset root directory;
            target_min — minimum target instance count per rare class;
            seed — random seed (for reproducibility)
    Output: none (side effects of creating "_osN"-suffixed copies under
            labels/train and images/train and printing a before/after distribution report)
    """
    random.seed(seed)

    train_labels = yolo_root / "labels" / "train"
    train_images = yolo_root / "images" / "train"

    if not train_labels.exists():
        print(f"[ERROR] Labels directory not found: {train_labels}")
        return

    # Compute the current distribution
    print("=" * 60)
    print("CURRENT CLASS DISTRIBUTION (before oversampling)")
    print("=" * 60)

    class_counts = count_class_instances(train_labels)
    for i, name in enumerate(CLASS_NAMES):
        count = class_counts.get(i, 0)
        marker = " <-- RARE" if i in RARE_CLASSES else ""
        print(f"{i:2d} {name:15s}: {count:6d}{marker}")

    # Find the images for each rare class
    class_to_images = get_images_by_rare_class(train_labels)

    print("\n" + "=" * 60)
    print("OVERSAMPLING PROCESS")
    print("=" * 60)

    total_copies = 0

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

        # How many copies are needed?
        needed_copies = int(needed_instances / avg_instances_per_image) + 1

        print(f"\n[PROCESS] {CLASS_NAMES[cls_id]}:")
        print(f"  - Current: {current_count} instances, {len(images_with_class)} images")
        print(f"  - Target: {target_min} instances")
        print(f"  - Copies needed: ~{needed_copies} images")

        # Pick images randomly and copy them
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

            # Prevent an infinite loop
            if copy_idx > 100:
                break

        print(f"  - Copied: {copy_count} images")

    # Compute the new distribution
    print("\n" + "=" * 60)
    print("NEW CLASS DISTRIBUTION (after oversampling)")
    print("=" * 60)

    new_counts = count_class_instances(train_labels)
    for i, name in enumerate(CLASS_NAMES):
        old_count = class_counts.get(i, 0)
        new_count = new_counts.get(i, 0)
        diff = new_count - old_count
        diff_str = f" (+{diff})" if diff > 0 else ""
        marker = " <-- RARE" if i in RARE_CLASSES else ""
        print(f"{i:2d} {name:15s}: {new_count:6d}{diff_str}{marker}")

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_copies} images copied")
    print("=" * 60)


def main():
    """Reads the command-line arguments and runs the oversample function.

    Input:  none (takes --yolo-root, --target-min, --seed from the command line)
    Output: none (dataset-copying side effect of the oversample call)
    """
    parser = argparse.ArgumentParser(description="Oversampling for class imbalance")
    parser.add_argument("--yolo-root", type=str, default="data/yolo",
                        help="YOLO dataset root directory")
    parser.add_argument("--target-min", type=int, default=2000,
                        help="Minimum target instance count per rare class")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    oversample(Path(args.yolo_root), args.target_min, args.seed)


if __name__ == "__main__":
    main()