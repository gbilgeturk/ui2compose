from pathlib import Path
import argparse
import re
import yaml
import xml.etree.ElementTree as ET
from PIL import Image

# ------------ Argümanlar ------------
def parse_args():
    """Komut satırı argümanlarını tanımlar ve okur.

    Girdi:  yok (değerleri komut satırından alır)
    Çıktı:  args nesnesi — args.raw_root, args.min_area gibi ayarlar
    """
    ap = argparse.ArgumentParser(description="ReDraw XML + IMG -> YOLO labels")
    ap.add_argument("--raw-root", default="data/raw", type=str, help="Ham veri setinin (XML+PNG) nereden okunacağı")
    ap.add_argument("--out-root", default="data/interim/labels", type=str, help="Çıktı olarak oluşturulan .txt'lerin nereye yazılacağı")
    ap.add_argument("--dataset-yaml", default="configs/dataset.yaml", type=str, help="13 sınıflık listenin okunacağı config dosyası, id ve sınıf eşleşmesi")
    ap.add_argument("--drop-container", default=True, type=lambda x: str(x).lower() in {"1","true","yes"}, help ="Adında container geçen yapısal düğümleri atlatan kısım, Tablo 3.2 Container Removal")
    ap.add_argument("--min-area", default=0.00005, type=float, help="Ekran alanına oran (w*h), Tablo 3.2'de bulunan alan filtresi")
    ap.add_argument("--dedup-iou", default=0.85, type=float, help="Aynı sınıftan iki kutu %%85 oranında çakışıyorsa duplicate kabul ederek silmeye yarar")
    ap.add_argument("--limit", default=0, type=int, help="Sadece ilk N ekran çiftini denemek için, geliştirmeyi hızlandırmak için hepsi = 0")
    ap.add_argument("--verbose", default=False, type=lambda x: str(x).lower() in {"1","true","yes"}, help = "işlemler sırasında ayrıntılı log basmak için")
    return ap.parse_args()

# ------------ Yardımcılar ------------
IMG_EXTS = (".png", ".jpg", ".jpeg") #dosya uzantıları
BOUNDS_RE = re.compile(r"\[(\d+),\s*(\d+)\]\[(\d+),\s*(\d+)\]") #XML içinden bounds bilgilerini çeken regex

ALIASES = {
    # Text ailesi
    "AppCompatTextView": "TextView",
    "MaterialTextView": "TextView",
    "CheckedTextView": "TextView",
    "TextInputEditText": "EditText",
    "AutoCompleteTextView": "EditText",
    # Görsel & buton
    "AppCompatImageView": "ImageView",
    "AppCompatButton": "Button",
    "MaterialButton": "Button",
    "ImageButton": "Button",
    # Switch
    "SwitchCompat": "Switch",
}

def load_names(yaml_path: Path):
    """dataset.yaml'dan 13 sınıflık listeyi okur; whitelist ve id eşlemesinin kaynağı.

    Girdi:  yaml_path — 'names' listesini içeren YAML dosyasının yolu
    Çıktı:  (names, name2id) — sıralı sınıf listesi ve isim->numara sözlüğü
            örn. {"Button": 0, "CheckBox": 1, ...}
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"dataset.yaml bulunamadı: {yaml_path}")
    data = yaml.safe_load(open(yaml_path, "r"))
    names = data.get("names", [])
    if not names:
        raise ValueError(f"dataset.yaml içinde 'names' yok veya boş: {yaml_path}")
    return names, {n: i for i, n in enumerate(names)}

def short_class_name(full: str) -> str:
    """Tam sınıf adını kısaltıp ALIASES ile normalize eder (Tablo 3.3).

    Girdi:  full — XML'deki tam ad, örn. "android.widget.AppCompatTextView"
    Çıktı:  normalize kısa ad, örn. "TextView" (eşleşme yoksa kısa ad aynen döner)
    """
    s = (full or "").split(".")[-1]
    return ALIASES.get(s, s)

def parse_bounds(s: str):
    """bounds metnini dört piksel koordinatına çevirir.

    Girdi:  s — "[x1,y1][x2,y2]" biçiminde metin, örn. "[0,63][1080,231]"
    Çıktı:  (x1, y1, x2, y2) tamsayı dörtlüsü; biçim bozuksa veya
            kutu ters/sıfır boyutluysa None
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
    """Köşe koordinatlarını YOLO biçimine çevirir: merkez + boyut, 0-1 normalize.

    Girdi:  box — (x1, y1, x2, y2) piksel koordinatları; W, H — görüntü boyutu
    Çıktı:  (xc, yc, w, h) — ekran oranına normalize edilmiş merkez ve boyut
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
    """İki kutunun IoU'sunu (kesişim/birleşim oranını) hesaplar; dedup için kullanılır.

    Girdi:  a, b — (xc, yc, w, h) biçiminde iki normalize kutu
    Çıktı:  0.0-1.0 arası oran (0 = hiç çakışmıyor, 1 = birebir aynı)
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
    """XML dosyasına karşılık gelen ekran görüntüsünü bulur.

    Girdi:  xml_path — hierarchy_N.xml dosya yolu
    Çıktı:  aynı klasördeki screenshot_N.png/jpg/jpeg yolu; yoksa None
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
    """XML ağacındaki tüm düğümleri (kendisi + tüm torunları) sırayla gezer.

    Girdi:  elem — başlangıç XML düğümü (genelde kök)
    Çıktı:  her düğümü tek tek veren generator
    """
    yield elem
    for ch in list(elem):
        yield from walk_nodes(ch)

# ------------ Ana işlevler ------------
def extract_labels(xml_path: Path, W: int, H: int, name2id: dict,
                   drop_container=True,
                   min_area=0.00010, dedup_iou=0.95, verbose=False):
    """Tek bir XML'i işleyip o ekranın tüm YOLO etiketlerini üretir (script'in kalbi).

    Ağacı gezer; her düğüm için sırayla: container atma -> kutu çıkarma ->
    alan filtresi -> isim normalize -> whitelist -> IoU dedup uygular.

    Girdi:  xml_path — hierarchy XML'i; W, H — eş görüntünün boyutu;
            name2id — sınıf whitelist/id sözlüğü; diğerleri — filtre ayarları
    Çıktı:  [(cid, (xc, yc, w, h)), ...] — filtrelerden geçen etiket listesi
    """
    labels = []  # [(cid, (xc,yc,w,h))]
    root = ET.parse(xml_path).getroot()

    for n in walk_nodes(root):
        cls_full = n.attrib.get("class") or n.attrib.get("className") or ""
        # Container'ı direkt ele
        if drop_container and "container" in cls_full.lower():
            continue

        # kutu çıkar
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
        # min alan filtresi
        if box[2] * box[3] < min_area:
            continue

        cname = short_class_name(cls_full)

        # sınıf id
        if cname not in name2id:
            # bu sınıf dataset.yaml'da yoksa geç
            continue
        cid = name2id[cname]

        # sınıf içi dedup
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
    """Akışı yönetir: tüm uygulama klasörlerini gezer, XML-görüntü çiftlerini
    eşler, extract_labels ile etiketleri çıkarır ve .txt dosyalarına yazar.

    Girdi:  yok (ayarları parse_args'tan alır)
    Çıktı:  data/interim/labels/<app>/screenshot_N.txt dosyaları + özet çıktı
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

    print(f"✅ Bitti. Toplam çift: {total_pairs}, yazılan label dosyası: {total_written}")
    print(f"Çıktılar: {out_root.resolve()}")

if __name__ == "__main__":
    main()