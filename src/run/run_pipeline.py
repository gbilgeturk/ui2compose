from pathlib import Path
import sys
import os
from datetime import datetime
import yaml

# ============================================================================
# IMPORT PATH AYARI
# ============================================================================

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # 3 seviye yukarı

# Proje kökünü sys.path'in EN BAŞINA ekle
if str(project_root) in sys.path:
    sys.path.remove(str(project_root))
sys.path.insert(0, str(project_root))

# Çalışma dizinini proje köküne ayarla
os.chdir(project_root)

print(f"📂 Proje kök dizini: {project_root}")
print(f"📂 Çalışma dizini: {Path.cwd()}")

# ============================================================================
# CONFIG DOSYASINI OKU
# ============================================================================

CONFIG_PATH = project_root / "configs" / "pipeline_config.yaml"

print(f"\n📋 Config dosyası yükleniyor: {CONFIG_PATH.relative_to(project_root)}")

# Default değerler (config dosyası yoksa)
DEFAULT_CONFIG = {
    'input': {
        'image': 'examples/sign_in.png',
        'model': 'runs/oversample_5k/weights/best.pt',
        'dataset_yaml': 'configs/dataset.yaml'
    },
    'output': {
        'directory': 'output'
    },
    'detection': {
        'confidence_threshold': 0.3
    },
    'features': {
        'visualization': True
    }
}

# Config dosyasını oku
try:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ Config yüklendi")
    else:
        print(f"⚠️  Config dosyası bulunamadı, default değerler kullanılıyor")
        print(f"💡 Şu konuma oluşturun: {CONFIG_PATH}")
        config = DEFAULT_CONFIG
except Exception as e:
    print(f"⚠️  Config okuma hatası: {e}")
    print(f"⚠️  Default değerler kullanılıyor")
    config = DEFAULT_CONFIG

# Config'den değerleri al
IMAGE_PATH = config['input']['image']
MODEL_PATH = config['input']['model']
DATASET_YAML = config['input']['dataset_yaml']
CONFIDENCE_THRESHOLD = config['detection']['confidence_threshold']
ENABLE_VISUALIZATION = config['features']['visualization']

# Otomatik tarih-saat klasörü oluştur
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = f"{config['output']['directory']}/{TIMESTAMP}"

# ============================================================================
# PIPELINE MODÜLÜNÜ IMPORT ET
# ============================================================================

print("\n📦 Pipeline modülü yükleniyor...")
try:
    from src.infer.pipeline_end_to_end import run_pipeline

    print(f"✅ pipeline_end_to_end modülü yüklendi")
except ImportError as e:
    print(f"\n❌ Import hatası: {e}")
    print(f"\n🔍 Debug:")
    print(f"   sys.path[0]: {sys.path[0]}")
    print(f"   Pipeline dosyası: {project_root / 'src/infer/pipeline_end_to_end.py'}")
    print(f"   Dosya var mı?: {(project_root / 'src/infer/pipeline_end_to_end.py').exists()}")

    # src/infer/ içeriğini göster
    infer_dir = project_root / 'src/infer'
    if infer_dir.exists():
        print(f"\n📂 {infer_dir} içeriği:")
        for item in sorted(infer_dir.iterdir()):
            if item.suffix == '.py':
                print(f"   ✅ {item.name}")

    import traceback

    traceback.print_exc()
    sys.exit(1)


# ============================================================================
# SCRIPT BAŞLANGICI
# ============================================================================

def main():
    """Config'deki ayarlarla uçtan uca pipeline'ı doğrulayıp onay alarak çalıştırır.

    Girdi:  yok (görsel/model/eşik ayarlarını pipeline_config.yaml'dan
            okunan modül sabitlerinden alır)
    Çıktı:  yok (run_pipeline'ı çağırma, OUTPUT_DIR altına sonuç dosyaları
            üretme ve özet basma yan etkisi; hata durumunda sys.exit)
    """

    # Dosya kontrolleri
    if not Path(IMAGE_PATH).exists():
        print(f"\n❌ Hata: Görsel bulunamadı: {IMAGE_PATH}")
        print(f"💡 İpucu: configs/pipeline_config.yaml dosyasında 'input.image' yolunu kontrol edin")
        sys.exit(1)

    if not Path(MODEL_PATH).exists():
        print(f"\n❌ Hata: Model bulunamadı: {MODEL_PATH}")
        print(f"💡 İpucu: configs/pipeline_config.yaml dosyasında 'input.model' yolunu kontrol edin")

        # Mevcut modelleri ara
        runs_dir = Path("runs")
        if runs_dir.exists():
            found_models = list(runs_dir.rglob("best.pt"))
            if found_models:
                print(f"\n📁 Bulunan modeller:")
                for i, model in enumerate(found_models[:5], 1):
                    print(f"   {i}. {model}")
                print(f"\n💡 Config dosyasında 'input.model' değerini yukarıdaki modellerden biriyle değiştirin")
        sys.exit(1)

    if not Path(DATASET_YAML).exists():
        print(f"\n❌ Hata: Dataset YAML bulunamadı: {DATASET_YAML}")
        sys.exit(1)

    # Ayarları yazdır
    print("\n" + "=" * 70)
    print("🚀 PIPELINE AYARLARI (configs/pipeline_config.yaml)")
    print("=" * 70)
    print(f"📷 Görsel:           {IMAGE_PATH}")
    print(f"🤖 Model:            {Path(MODEL_PATH).name}")
    print(f"📊 Dataset YAML:     {DATASET_YAML}")
    print(f"📁 Çıktı dizini:     {OUTPUT_DIR}")
    print(f"🕐 Timestamp:        {TIMESTAMP}")
    print(f"🎯 Confidence:       {CONFIDENCE_THRESHOLD}")
    print(f"🖼️  Visualization:    {'✅ Açık' if ENABLE_VISUALIZATION else '❌ Kapalı'}")
    print("=" * 70 + "\n")

    # Onay iste (etkileşimsiz ortamda — ör. CI — onay atlanır)
    try:
        response = input("Devam etmek için ENTER'a basın (iptal için 'q'): ")
        if response.lower() == 'q':
            print("İptal edildi.")
            sys.exit(0)
    except EOFError:
        pass

    # Pipeline'ı çalıştır
    try:
        print("\n🚀 Pipeline başlatılıyor...\n")

        results = run_pipeline(
            image_path=IMAGE_PATH,
            model_path=MODEL_PATH,
            dataset_yaml=DATASET_YAML,
            output_dir=OUTPUT_DIR,
            conf_threshold=CONFIDENCE_THRESHOLD,
            visualize=ENABLE_VISUALIZATION
        )

        print("\n" + "=" * 70)
        print("✅ BAŞARILI!")
        print("=" * 70)
        print(f"📁 Sonuçlar: {results['output_dir']}")
        print(f"🔢 Detection sayısı: {len(results['detections'])}")
        print(f"📄 Kod satırı: {len(results['code'].split(chr(10)))}")
        print("=" * 70)

        # Dosya listesi
        output_path = Path(results['output_dir'])
        if output_path.exists():
            files = sorted(output_path.iterdir())
            if files:
                print(f"\n📄 Oluşturulan dosyalar:")
                for f in files:
                    size = f.stat().st_size / 1024
                    print(f"   • {f.name:<35s} ({size:>6.1f} KB)")

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Kullanıcı tarafından durduruldu")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()