"""
Kullanım:
    python oversample_dataset.py --yolo-root data/yolo --target-min 2000
"""

import argparse
import shutil
from pathlib import Path
from collections import Counter, defaultdict
import random

# ReDraw sınıf isimleri (13 class — remapped, CardView & Toolbar removed)
CLASS_NAMES = [
    'Button', 'CheckBox', 'EditText', 'ImageView', 'ListView',
    'ProgressBar', 'RadioButton', 'RecyclerView', 'SeekBar',
    'Spinner', 'Switch', 'TextView', 'WebView'
]

# Nadir sınıflar (oversampling hedefleri — 13-class remapped indices)
# CheckBox(1), ProgressBar(5), RadioButton(6), RecyclerView(7),
# SeekBar(8), Spinner(9), Switch(10), WebView(12)
RARE_CLASSES = {1, 5, 6, 7, 8, 9, 10, 12}


def count_class_instances(labels_dir: Path) -> Counter:
    """Label dosyalarını tarayıp her sınıfın toplam instance sayısını hesaplar.

    Girdi:  labels_dir — YOLO .txt label dosyalarını içeren dizin
    Çıktı:  Counter — sınıf id -> toplam instance sayısı eşlemesi
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
    """Her nadir sınıf (RARE_CLASSES) için o sınıfı içeren görüntüleri bulur.

    Girdi:  labels_dir — YOLO .txt label dosyalarını içeren dizin
    Çıktı:  dict — nadir sınıf id -> o sınıfı içeren görüntü stem'leri listesi
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
    """Verilen köke karşılık gelen görüntü dosyasını uzantıları deneyerek arar.

    Girdi:  images_dir — görüntülerin bulunduğu dizin; stem — dosya kökü (uzantısız ad)
    Çıktı:  bulunan görüntünün Path'i (.png/.jpg/.jpeg); yoksa None
    """
    for ext in [".png", ".jpg", ".jpeg"]:
        img_path = images_dir / f"{stem}{ext}"
        if img_path.exists():
            return img_path
    return None


def oversample(yolo_root: Path, target_min: int, seed: int = 42):
    """Nadir sınıfları içeren train görüntülerini kopyalayarak hedef sayıya kadar oversample eder.

    Girdi:  yolo_root — YOLO dataset kök dizini;
            target_min — her nadir sınıf için minimum hedef instance sayısı;
            seed — random seed (reproducibility için)
    Çıktı:  yok (labels/train ve images/train altına "_osN" ekli kopyalar
            oluşturma ve öncesi/sonrası dağılım raporu basma yan etkisi)
    """
    random.seed(seed)

    train_labels = yolo_root / "labels" / "train"
    train_images = yolo_root / "images" / "train"

    if not train_labels.exists():
        print(f"[ERROR] Labels dizini bulunamadı: {train_labels}")
        return

    # Mevcut dağılımı hesapla
    print("=" * 60)
    print("MEVCUT SINIF DAĞILIMI (Oversampling öncesi)")
    print("=" * 60)

    class_counts = count_class_instances(train_labels)
    for i, name in enumerate(CLASS_NAMES):
        count = class_counts.get(i, 0)
        marker = " <-- RARE" if i in RARE_CLASSES else ""
        print(f"{i:2d} {name:15s}: {count:6d}{marker}")

    # Her nadir sınıf için görüntüleri bul
    class_to_images = get_images_by_rare_class(train_labels)

    print("\n" + "=" * 60)
    print("OVERSAMPLING İŞLEMİ")
    print("=" * 60)

    total_copies = 0

    for cls_id in sorted(RARE_CLASSES):
        current_count = class_counts.get(cls_id, 0)

        if current_count == 0:
            print(f"\n[SKIP] {CLASS_NAMES[cls_id]}: Hiç örnek yok, atlanıyor.")
            continue

        if current_count >= target_min:
            print(f"\n[SKIP] {CLASS_NAMES[cls_id]}: Zaten yeterli ({current_count} >= {target_min})")
            continue

        images_with_class = class_to_images.get(cls_id, [])
        if not images_with_class:
            print(f"\n[SKIP] {CLASS_NAMES[cls_id]}: Görüntü bulunamadı.")
            continue

        # Kaç instance eklememiz gerekiyor?
        needed_instances = target_min - current_count

        # Her görüntüde ortalama kaç instance var?
        avg_instances_per_image = current_count / len(images_with_class)

        # Kaç kopya gerekiyor?
        needed_copies = int(needed_instances / avg_instances_per_image) + 1

        print(f"\n[PROCESS] {CLASS_NAMES[cls_id]}:")
        print(f"  - Mevcut: {current_count} instance, {len(images_with_class)} görüntü")
        print(f"  - Hedef: {target_min} instance")
        print(f"  - Gerekli kopya: ~{needed_copies} görüntü")

        # Görüntüleri rastgele seç ve kopyala
        copy_count = 0
        copy_idx = 0

        while copy_count < needed_copies:
            for img_stem in images_with_class:
                if copy_count >= needed_copies:
                    break

                new_stem = f"{img_stem}_os{copy_idx}"

                # Label kopyala
                src_label = train_labels / f"{img_stem}.txt"
                dst_label = train_labels / f"{new_stem}.txt"

                if dst_label.exists():
                    copy_idx += 1
                    continue

                # Image kopyala
                src_img = find_image_file(train_images, img_stem)
                if src_img is None:
                    continue

                dst_img = train_images / f"{new_stem}{src_img.suffix}"

                shutil.copy2(src_label, dst_label)
                shutil.copy2(src_img, dst_img)

                copy_count += 1
                total_copies += 1

            copy_idx += 1

            # Sonsuz döngüyü önle
            if copy_idx > 100:
                break

        print(f"  - Kopyalanan: {copy_count} görüntü")

    # Yeni dağılımı hesapla
    print("\n" + "=" * 60)
    print("YENİ SINIF DAĞILIMI (Oversampling sonrası)")
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
    print(f"TOPLAM: {total_copies} görüntü kopyalandı")
    print("=" * 60)


def main():
    """Komut satırı argümanlarını okuyup oversample fonksiyonunu çalıştırır.

    Girdi:  yok (--yolo-root, --target-min, --seed değerlerini komut satırından alır)
    Çıktı:  yok (oversample çağrısının dataset kopyalama yan etkisi)
    """
    parser = argparse.ArgumentParser(description="Oversampling for class imbalance")
    parser.add_argument("--yolo-root", type=str, default="data/yolo",
                        help="YOLO dataset kök dizini")
    parser.add_argument("--target-min", type=int, default=2000,
                        help="Her nadir sınıf için minimum hedef instance sayısı")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    oversample(Path(args.yolo_root), args.target_min, args.seed)


if __name__ == "__main__":
    main()