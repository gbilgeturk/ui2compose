# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Generates the synthetic YOLO dataset tests/fixtures/tooling/yolo_tiny/.

The images are random 9x16 gray-level block grids upscaled to 90x160 (so
their dHashes are far apart), named with the ReDraw dataset convention. Each
screen is designed to exercise one case of the leak audit:

  train  com.alpha.app 1, 2 (+ oversampled copy 2_os0), com.beta.app 1,
         com.andromo.dev111111.app222222 1
  val    com.alpha.app 3            -> excluded: package
         com.gamma.app 1            -> unseen
         com.clone.of.beta 1        -> excluded: md5 (byte copy of beta 1)
  test   com.delta.app 1            -> excluded: near_dup (beta 1 + noise)
         com.epsilon.app 1          -> unseen
         com.eta.app 1              -> excluded: transitive (same package as eta 2)
         com.eta.app 2              -> excluded: md5 (byte copy of alpha 1)
         com.andromo.dev111111.app333333 1
                                    -> unseen; builder_prefix with the flag

Labels are random but fixed (seeded). Run from the repository root:
    python tests/fixtures/tooling/make_yolo_tiny.py
"""

import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent / "yolo_tiny"
NAMES = ["Button", "CheckBox", "EditText", "ImageView", "ListView", "ProgressBar",
         "RadioButton", "RecyclerView", "SeekBar", "Spinner", "Switch", "TextView",
         "WebView"]


def grid_image(seed: int, noise: int = 0) -> Image.Image:
    """Makes a 90x160 grayscale block image from a seeded 9x16 random grid.

    Input:  seed — RNG seed of the grid; noise — amplitude of added pixel noise
    Output: PIL image (mode 'L')
    """
    rng = np.random.default_rng(seed)
    grid = rng.integers(0, 256, size=(16, 9)).astype(np.int16)
    img = np.kron(grid, np.ones((10, 10), dtype=np.int16))
    if noise:
        img = img + np.random.default_rng(seed + 1000).integers(-noise, noise + 1, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="L")


def write_label(path: Path, seed: int) -> None:
    """Writes a random YOLO label file with 1-4 boxes.

    Input:  path — .txt path; seed — RNG seed
    Output: none (file written)
    """
    rng = random.Random(seed)
    lines = []
    for _ in range(rng.randint(1, 4)):
        c = rng.randrange(len(NAMES))
        w, h = rng.uniform(0.1, 0.5), rng.uniform(0.05, 0.3)
        xc, yc = rng.uniform(w / 2, 1 - w / 2), rng.uniform(h / 2, 1 - h / 2)
        lines.append(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    path.write_text("".join(lines), encoding="utf-8")


# (split, stem, image seed, noise, label seed); "copy:<split>/<stem>" copies bytes
SCREENS = [
    ("train", "com.alpha.app-screens_screenshot_1", 1, 0, 1),
    ("train", "com.alpha.app-screens_screenshot_2", 2, 0, 2),
    ("train", "com.alpha.app-screens_screenshot_2_os0", "copy:train/com.alpha.app-screens_screenshot_2", 0, 2),
    ("train", "com.beta.app-screens_screenshot_1", 3, 0, 3),
    ("train", "com.andromo.dev111111.app222222-screens_screenshot_1", 4, 0, 4),
    ("val", "com.alpha.app-screens_screenshot_3", 5, 0, 5),
    ("val", "com.gamma.app-screens_screenshot_1", 6, 0, 6),
    ("val", "com.clone.of.beta-screens_screenshot_1", "copy:train/com.beta.app-screens_screenshot_1", 0, 7),
    ("test", "com.delta.app-screens_screenshot_1", 3, 6, 8),
    ("test", "com.epsilon.app-screens_screenshot_1", 7, 0, 9),
    ("test", "com.eta.app-screens_screenshot_1", 8, 0, 10),
    ("test", "com.eta.app-screens_screenshot_2", "copy:train/com.alpha.app-screens_screenshot_1", 0, 11),
    ("test", "com.andromo.dev111111.app333333-screens_screenshot_1", 9, 0, 12),
]


def main() -> None:
    """Regenerates the fixture directory from scratch.

    Input:  none
    Output: none (files written under yolo_tiny/)
    """
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
    for split, stem, img_seed, noise, lab_seed in SCREENS:
        dst = OUT / "images" / split / f"{stem}.png"
        if isinstance(img_seed, str):
            shutil.copyfile(OUT / "images" / f"{img_seed[5:]}.png", dst)
        else:
            grid_image(img_seed, noise).save(dst, optimize=False)
        if stem.endswith("_os0"):
            src = OUT / "labels" / split / f"{stem[:-4]}.txt"
            shutil.copyfile(src, OUT / "labels" / split / f"{stem}.txt")
        else:
            write_label(OUT / "labels" / split / f"{stem}.txt", lab_seed)
    yaml_text = ("path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
                 f"nc: {len(NAMES)}\nnames:\n" + "".join(f"  - {n}\n" for n in NAMES))
    (OUT / "dataset.yaml").write_text(yaml_text, encoding="utf-8")


if __name__ == "__main__":
    main()
