from pathlib import Path
from ultralytics import YOLO
import cv2
import yaml
import json
from typing import List, Dict, Tuple


class ComponentDetector:
    """YOLO-based UI component detector"""

    def __init__(self, model_path: str, dataset_yaml: str):
        """Eğitilmiş YOLO modelini yükler ve sınıf isimlerini modelden alır.

        Girdi:  model_path — eğitilmiş YOLO ağırlık dosyası yolu (örn. 'runs/train/exp/weights/best.pt')
                dataset_yaml — sınıf isimlerini içeren veri kümesi yapılandırma dosyası
        Çıktı:  yok (self.model ve self.class_names atama yan etkisi)
        """
        self.model = YOLO(model_path)

        # Use model's own class names (safest - avoids index mismatch)
        self.class_names = list(self.model.names.values())

        print(f"✓ Model loaded: {model_path}")
        print(f"✓ Classes ({len(self.class_names)}): {self.class_names}")

    def detect(self, image_path: str, conf_threshold: float = 0.3) -> List[Dict]:
        """Görüntüdeki UI bileşenlerini tespit eder ve yukarıdan aşağıya sıralı döndürür.

        Girdi:  image_path — ekran görüntüsü dosya yolu
                conf_threshold — tespit güven eşiği (varsayılan 0.3)
        Çıktı:  tespit sözlüklerinin listesi: {'id', 'class_name', 'class_id',
                'confidence', 'bbox' [x1,y1,x2,y2 piksel], 'bbox_norm' [xc,yc,w,h 0-1],
                'center' [xc,yc piksel], 'width', 'height'}
        """
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            verbose=False
        )[0]

        # Load image to get dimensions
        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]

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
                'height': float(h)
            }
            detections.append(detection)

        # Sort by y-coordinate (top to bottom)
        detections.sort(key=lambda x: x['bbox'][1])

        # Reassign IDs after sorting
        for idx, det in enumerate(detections):
            det['id'] = idx

        return detections

    def visualize(self, image_path: str, detections: List[Dict], output_path: str = None):
        """Tespit kutularını ve etiketlerini görüntü üzerine çizer.

        Girdi:  image_path — ekran görüntüsü dosya yolu
                detections — detect() çıktısı tespit listesi
                output_path — kaydedilecek dosya yolu (opsiyonel)
        Çıktı:  kutular çizilmiş görüntü (numpy dizisi); output_path verildiyse dosyaya da kaydeder
        """
        img = cv2.imread(image_path)

        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_name = det['class_name']
            conf = det['confidence']

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{class_name} {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if output_path:
            cv2.imwrite(output_path, img)
            print(f"✓ Saved visualization: {output_path}")

        return img


def main():
    """Dedektörü örnek bir test görüntüsü üzerinde çalıştırır.

    Girdi:  yok
    Çıktı:  yok (output/detections.png ve output/detections.json dosya yan etkisi)
    """
    # Paths (ADJUST THESE!)
    model_path = "runs/oversample_5k/weights/best.pt"
    dataset_yaml = "configs/dataset.yaml"
    test_image = "examples/sign_in.png"

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