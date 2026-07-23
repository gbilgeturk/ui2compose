from ultralytics import YOLO
from pathlib import Path
import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import sys

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent.absolute()

# If inside src/evaluation/, go two levels up (to the project root)
if SCRIPT_DIR.name == 'evaluation':
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

# Add the project root to sys.path
sys.path.insert(0, str(PROJECT_ROOT))

# Set the working directory to the project root
os.chdir(PROJECT_ROOT)


class MetricsCalculator:
    def __init__(self, model_path="runs/oversample_5k/weights/best.pt"):
        """Loads the YOLO model to be evaluated.

        Input:  model_path — path to the trained YOLO weights file
        Output: none (side effect of assigning self.model)
        """
        self.model = YOLO(model_path)
        print(f"✅ Model loaded: {model_path}")

    def parse_yolo_label(self, label_path, img_width, img_height):
        """Parses a YOLO label file and converts boxes to absolute pixel coordinates.

        Input:  label_path — path to the YOLO-format .txt label file
                img_width — image width (pixels)
                img_height — image height (pixels)
        Output: list of {'class_id', 'bbox' [x1,y1,x2,y2], 'matched': False} dicts; empty list if the file does not exist
        """
        boxes = []

        if not label_path.exists():
            return boxes

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])

                # Normalized -> absolute
                x1 = (xc - w / 2) * img_width
                y1 = (yc - h / 2) * img_height
                x2 = (xc + w / 2) * img_width
                y2 = (yc + h / 2) * img_height

                boxes.append({
                    'class_id': class_id,
                    'bbox': [x1, y1, x2, y2],
                    'matched': False
                })

        return boxes

    def calculate_iou(self, box1, box2):
        """Calculates the IoU (intersection over union) ratio between two boxes.

        Input:  box1 — [x1, y1, x2, y2] box coordinates
                box2 — [x1, y1, x2, y2] box coordinates
        Output: IoU value in the 0-1 range; 0 if the union area is 0
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = box1_area + box2_area - inter

        return inter / union if union > 0 else 0

    def match_predictions_to_ground_truth(self, predictions, ground_truth, iou_threshold=0.5):
        """Matches predictions to same-class ground truth boxes based on IoU.

        Input:  predictions — list of prediction boxes
                ground_truth — list of ground truth (label) boxes
                iou_threshold — minimum IoU required for a match (default 0.5)
        Output: list of matches consisting of {'pred', 'gt', 'iou'} dicts;
                the 'matched' field of matched ground truth boxes is set to True
        """
        matches = []

        for pred in predictions:
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truth):
                if gt['matched']:
                    continue

                if pred['class_id'] != gt['class_id']:
                    continue

                iou = self.calculate_iou(pred['bbox'], gt['bbox'])

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                ground_truth[best_gt_idx]['matched'] = True
                matches.append({
                    'pred': pred,
                    'gt': ground_truth[best_gt_idx],
                    'iou': best_iou
                })

        return matches

    def calculate_metrics_for_dataset(self, data_dir, conf_threshold=0.15, iou_threshold=0.5):
        """Calculates per-class TP/FP/FN statistics for all images in the test set.

        Input:  data_dir — YOLO dataset root directory (contains images/test and labels/test)
                conf_threshold — prediction confidence threshold (default 0.15)
                iou_threshold — matching IoU threshold (default 0.5)
        Output: (class_stats, number of images) — per class id a
                {'tp','fp','fn','total_gt','total_pred','ious'} dict
        """

        images_dir = Path(data_dir) / "images/test"
        labels_dir = Path(data_dir) / "labels/test"

        image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))

        print(f"\n📊 Analyzing {len(image_files)} test images...")
        print(f"   Confidence threshold: {conf_threshold}")
        print(f"   IoU threshold: {iou_threshold}\n")

        # Per-class metrics
        class_stats = defaultdict(lambda: {
            'tp': 0,  # True positives
            'fp': 0,  # False positives
            'fn': 0,  # False negatives
            'total_gt': 0,
            'total_pred': 0,
            'ious': []
        })

        all_matches = []

        for img_path in image_files:
            # Get image dimensions
            import cv2
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]

            # Ground truth
            label_path = labels_dir / (img_path.stem + ".txt")
            gt_boxes = self.parse_yolo_label(label_path, w, h)

            # Predictions
            results = self.model.predict(
                source=str(img_path),
                conf=conf_threshold,
                iou=0.45,
                max_det=100,
                save=False,
                verbose=False
            )

            result = results[0]
            pred_boxes = []

            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                pred_boxes.append({
                    'class_id': int(box.cls[0]),
                    'bbox': xyxy.tolist(),
                    'confidence': float(box.conf[0])
                })

            # Match predictions to ground truth
            matches = self.match_predictions_to_ground_truth(pred_boxes, gt_boxes, iou_threshold)
            all_matches.extend(matches)

            # Update stats
            for gt in gt_boxes:
                cls_id = gt['class_id']
                class_stats[cls_id]['total_gt'] += 1

                if gt['matched']:
                    class_stats[cls_id]['tp'] += 1
                else:
                    class_stats[cls_id]['fn'] += 1

            for pred in pred_boxes:
                cls_id = pred['class_id']
                class_stats[cls_id]['total_pred'] += 1

                # Check if this pred matched any GT
                matched = any(m['pred'] == pred for m in matches)
                if not matched:
                    class_stats[cls_id]['fp'] += 1

            # Store IoUs
            for match in matches:
                cls_id = match['pred']['class_id']
                class_stats[cls_id]['ious'].append(match['iou'])

        return class_stats, len(image_files)

    def calculate_precision_recall_f1(self, stats):
        """Calculates precision, recall, and F1 values from class statistics.

        Input:  stats — class statistics output by calculate_metrics_for_dataset
        Output: per class id a {'precision','recall','f1','tp','fp','fn',
                'total_gt','total_pred','avg_iou'} dict
        """
        metrics = {}

        for cls_id, stat in stats.items():
            tp = stat['tp']
            fp = stat['fp']
            fn = stat['fn']

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            avg_iou = np.mean(stat['ious']) if stat['ious'] else 0

            metrics[cls_id] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'total_gt': stat['total_gt'],
                'total_pred': stat['total_pred'],
                'avg_iou': avg_iou
            }

        return metrics

    def visualize_metrics(self, metrics, class_names, output_dir):
        """Plots the metrics as a four-panel chart and saves it to a PNG file.

        Input:  metrics — metrics dict output by calculate_precision_recall_f1
                class_names — class id → class name mapping
                output_dir — folder where the chart will be saved
        Output: none (performance_metrics.png file side effect)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Prepare data
        classes = []
        precisions = []
        recalls = []
        f1_scores = []

        for cls_id, metric in sorted(metrics.items()):
            classes.append(class_names.get(cls_id, f"Class_{cls_id}"))
            precisions.append(metric['precision'])
            recalls.append(metric['recall'])
            f1_scores.append(metric['f1'])

        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold')

        x = np.arange(len(classes))
        width = 0.25

        # 1. Precision, Recall, F1
        axes[0, 0].bar(x - width, precisions, width, label='Precision', alpha=0.8)
        axes[0, 0].bar(x, recalls, width, label='Recall', alpha=0.8)
        axes[0, 0].bar(x + width, f1_scores, width, label='F1', alpha=0.8)
        axes[0, 0].set_ylabel('Score', fontsize=12)
        axes[0, 0].set_title('Precision, Recall, F1 by Class', fontsize=13)
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(classes, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].grid(axis='y', alpha=0.3)

        # 2. TP, FP, FN
        tp_vals = [metrics[cls_id]['tp'] for cls_id in sorted(metrics.keys())]
        fp_vals = [metrics[cls_id]['fp'] for cls_id in sorted(metrics.keys())]
        fn_vals = [metrics[cls_id]['fn'] for cls_id in sorted(metrics.keys())]

        axes[0, 1].bar(x - width, tp_vals, width, label='TP', color='green', alpha=0.7)
        axes[0, 1].bar(x, fp_vals, width, label='FP', color='red', alpha=0.7)
        axes[0, 1].bar(x + width, fn_vals, width, label='FN', color='orange', alpha=0.7)
        axes[0, 1].set_ylabel('Count', fontsize=12)
        axes[0, 1].set_title('True Positives, False Positives, False Negatives', fontsize=13)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(classes, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)

        # 3. Average IoU
        avg_ious = [metrics[cls_id]['avg_iou'] for cls_id in sorted(metrics.keys())]
        axes[1, 0].bar(classes, avg_ious, color='steelblue', alpha=0.7)
        axes[1, 0].set_ylabel('Average IoU', fontsize=12)
        axes[1, 0].set_title('Average IoU by Class', fontsize=13)
        axes[1, 0].set_xticklabels(classes, rotation=45, ha='right')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(axis='y', alpha=0.3)

        # 4. Summary table
        axes[1, 1].axis('off')

        # Overall metrics
        overall_precision = np.mean(precisions)
        overall_recall = np.mean(recalls)
        overall_f1 = np.mean(f1_scores)

        summary_text = f"""
        OVERALL PERFORMANCE
        {'=' * 35}

        Mean Precision:  {overall_precision:.3f}
        Mean Recall:     {overall_recall:.3f}
        Mean F1 Score:   {overall_f1:.3f}

        {'=' * 35}

        Total Classes:   {len(classes)}

        Best Precision:  {classes[np.argmax(precisions)]} ({max(precisions):.3f})
        Best Recall:     {classes[np.argmax(recalls)]} ({max(recalls):.3f})
        Best F1:         {classes[np.argmax(f1_scores)]} ({max(f1_scores):.3f})
        """

        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11,
                        family='monospace', verticalalignment='center')

        plt.tight_layout()
        output_file = output_path / "performance_metrics.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Performance metrics saved: {output_file}")

    def generate_metrics_table(self, metrics, class_names, output_dir):
        """Generates and saves the metrics table in LaTeX and Markdown formats.

        Input:  metrics — per-class metrics dict
                class_names — class id → class name mapping
                output_dir — folder where the tables will be saved
        Output: none (table_performance_metrics.tex and .md file side effects)
        """
        output_path = Path(output_dir)

        # LaTeX
        latex = r"""\begin{table}[h]
\centering
\caption{Detection Performance Metrics by Class}
\label{tab:performance_metrics}
\begin{tabular}{lrrrr}
\hline
\textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Avg IoU} \\
\hline
"""

        for cls_id, metric in sorted(metrics.items()):
            cls_name = class_names.get(cls_id, f"Class_{cls_id}")
            latex += f"{cls_name} & {metric['precision']:.3f} & {metric['recall']:.3f} & {metric['f1']:.3f} & {metric['avg_iou']:.3f} \\\\\n"

        # Overall
        precisions = [m['precision'] for m in metrics.values()]
        recalls = [m['recall'] for m in metrics.values()]
        f1s = [m['f1'] for m in metrics.values()]
        ious = [m['avg_iou'] for m in metrics.values()]

        latex += r"""\hline
\textbf{Mean} & """ + f"{np.mean(precisions):.3f} & {np.mean(recalls):.3f} & {np.mean(f1s):.3f} & {np.mean(ious):.3f} \\\\\n"
        latex += r"""\hline
\end{tabular}
\end{table}"""

        latex_file = output_path / "table_performance_metrics.tex"
        with open(latex_file, 'w') as f:
            f.write(latex)

        print(f"✅ LaTeX table saved: {latex_file}")

        # Markdown
        md = "| Class | Precision | Recall | F1 | Avg IoU |\n"
        md += "|-------|-----------|--------|----|---------|\n"

        for cls_id, metric in sorted(metrics.items()):
            cls_name = class_names.get(cls_id, f"Class_{cls_id}")
            md += f"| {cls_name} | {metric['precision']:.3f} | {metric['recall']:.3f} | {metric['f1']:.3f} | {metric['avg_iou']:.3f} |\n"

        md += f"| **Mean** | **{np.mean(precisions):.3f}** | **{np.mean(recalls):.3f}** | **{np.mean(f1s):.3f}** | **{np.mean(ious):.3f}** |\n"

        md_file = output_path / "table_performance_metrics.md"
        with open(md_file, 'w') as f:
            f.write(md)

        print(f"✅ Markdown table saved: {md_file}")

    def run_full_evaluation(self, data_dir="data/yolo", output_dir="thesis_figures"):
        """Runs the full evaluation: metric calculation, visualization, table and JSON generation.

        Input:  data_dir — YOLO dataset root directory (default "data/yolo")
                output_dir — output folder (default "thesis_figures")
        Output: none (chart, table, and quantitative_metrics.json file side effects)
        """
        print("\n" + "=" * 70)
        print("QUANTITATIVE METRICS CALCULATION")
        print("=" * 70 + "\n")

        # Calculate metrics
        class_stats, n_images = self.calculate_metrics_for_dataset(data_dir)

        print(f"✅ Analyzed {n_images} test images\n")

        # Calculate precision, recall, F1
        metrics = self.calculate_precision_recall_f1(class_stats)

        # Get class names
        class_names = self.model.names

        # Print results
        print("📊 RESULTS BY CLASS:")
        print("-" * 70)
        print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
        print("-" * 70)

        for cls_id, metric in sorted(metrics.items()):
            cls_name = class_names.get(cls_id, f"Class_{cls_id}")
            print(f"{cls_name:<15} {metric['precision']:>10.3f} {metric['recall']:>10.3f} "
                  f"{metric['f1']:>10.3f} {metric['tp']:>6} {metric['fp']:>6} {metric['fn']:>6}")

        # Overall
        precisions = [m['precision'] for m in metrics.values()]
        recalls = [m['recall'] for m in metrics.values()]
        f1s = [m['f1'] for m in metrics.values()]

        print("-" * 70)
        print(f"{'MEAN':<15} {np.mean(precisions):>10.3f} {np.mean(recalls):>10.3f} "
              f"{np.mean(f1s):>10.3f}")
        print("=" * 70)

        # Visualize
        print("\n📊 Creating visualizations...")
        self.visualize_metrics(metrics, class_names, output_dir)

        # Generate tables
        print("\n📄 Generating tables...")
        self.generate_metrics_table(metrics, class_names, output_dir)

        # Save JSON
        output_path = Path(output_dir)
        json_file = output_path / "quantitative_metrics.json"

        # Convert metrics to serializable format
        metrics_json = {}
        for cls_id, metric in metrics.items():
            metrics_json[str(cls_id)] = {
                'class_name': class_names.get(cls_id, f"Class_{cls_id}"),
                **{k: float(v) if isinstance(v, np.floating) else v
                   for k, v in metric.items()}
            }

        with open(json_file, 'w') as f:
            json.dump(metrics_json, f, indent=2)

        print(f"✅ JSON saved: {json_file}")

        print("\n" + "=" * 70)
        print("✅ QUANTITATIVE EVALUATION COMPLETE")
        print("=" * 70)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Calculate quantitative metrics')
    parser.add_argument('--model', type=str,
                        default='runs/oversample_5k/weights/best.pt',
                        help='Path to model weights')
    parser.add_argument('--data', type=str, default='data/yolo',
                        help='Path to YOLO dataset')
    parser.add_argument('--output', type=str, default='thesis_figures',
                        help='Output directory')

    args = parser.parse_args()

    calculator = MetricsCalculator(model_path=args.model)
    calculator.run_full_evaluation(data_dir=args.data, output_dir=args.output)