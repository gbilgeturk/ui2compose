from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import json
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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


class SuccessfulCaseVisualizer:
    def __init__(self, model_path="runs/oversample_5k/weights/best.pt"):
        """Loads the YOLO model and defines the successful case list and color map.

        Input:  model_path — path to the trained YOLO weights file
        Output: none (side effect of assigning self.model, self.successful_cases, and self.colors)
        """
        self.model = YOLO(model_path)
        print(f"✅ Model loaded: {model_path}")

        # Successful cases (manual selection)
        self.successful_cases = [
            "com.microsoft.office.outlook-screens_screenshot_3",
            "com.securus.videoclient-screens_screenshot_1",
            "com.alesig.wmb-screens_screenshot_3",
            "com.Livewallpaper.LivingRoom-screens_screenshot_2",
        ]

        # Color map (thesis-friendly)
        self.colors = {
            'Button': (0, 122, 204),  # Blue
            'TextView': (255, 140, 0),  # Orange
            'EditText': (34, 139, 34),  # Green
            'ImageView': (220, 20, 60),  # Crimson
            'ListView': (148, 0, 211),  # Purple
            'CheckBox': (255, 215, 0),  # Gold
        }

    def find_test_image(self, stem):
        """Finds the image matching the given name in the test folder.

        Input:  stem — image file name without extension
        Output: .png/.jpg/.jpeg path under data/yolo/images/test; None if not found
        """
        test_dir = Path("data/yolo/images/test")

        for ext in ['.png', '.jpg', '.jpeg']:
            path = test_dir / f"{stem}{ext}"
            if path.exists():
                return path

        return None

    def detect_with_metrics(self, image_path):
        """Runs detection on the image and calculates summary metrics.

        Input:  image_path — screenshot file path
        Output: (result, metrics) — YOLO result object and a {'total_detections',
                'avg_confidence', 'class_distribution', 'confidence_distribution'} dict
        """
        results = self.model.predict(
            source=str(image_path),
            conf=0.15,
            iou=0.45,
            max_det=100,
            save=False,
            verbose=False
        )

        result = results[0]
        boxes = result.boxes

        # Metrics
        metrics = {
            'total_detections': len(boxes),
            'avg_confidence': float(boxes.conf.mean()) if len(boxes) > 0 else 0,
            'class_distribution': {},
            'confidence_distribution': []
        }

        if len(boxes) > 0:
            classes = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()

            # Class distribution
            class_counts = Counter([int(c) for c in classes])
            for cls_id, count in class_counts.items():
                cls_name = result.names[cls_id]
                metrics['class_distribution'][cls_name] = count

            # Confidence per class
            for cls_name in metrics['class_distribution'].keys():
                cls_confs = [
                    float(boxes[i].conf[0])
                    for i in range(len(boxes))
                    if result.names[int(boxes[i].cls[0])] == cls_name
                ]
                metrics['confidence_distribution'].append({
                    'class': cls_name,
                    'avg_conf': sum(cls_confs) / len(cls_confs),
                    'min_conf': min(cls_confs),
                    'max_conf': max(cls_confs)
                })

        return result, metrics

    def visualize_single_case(self, image_path, output_dir):
        """Plots the original and detected images side by side for a single case and saves in thesis format.

        Input:  image_path — screenshot file path
                output_dir — output folder
        Output: metrics dict for the case; a <name>_visualized.png file is also saved
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Detection
        result, metrics = self.detect_with_metrics(image_path)
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create figure - side by side
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))
        fig.suptitle(f'{image_path.stem}', fontsize=16, fontweight='bold')

        # Original
        axes[0].imshow(img_rgb)
        axes[0].set_title('Original Screenshot', fontsize=14)
        axes[0].axis('off')

        # Detected
        axes[1].imshow(img_rgb)
        axes[1].set_title(f'Detections (n={metrics["total_detections"]})', fontsize=14)
        axes[1].axis('off')

        # Draw boxes
        boxes = result.boxes
        for i in range(len(boxes)):
            box = boxes[i]
            xyxy = box.xyxy[0].cpu().numpy()
            cls_name = result.names[int(box.cls[0])]
            conf = float(box.conf[0])

            # Color
            color = self.colors.get(cls_name, (128, 128, 128))
            color_norm = tuple([c / 255 for c in color])

            # Rectangle
            x1, y1, x2, y2 = xyxy
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2,
                edgecolor=color_norm,
                facecolor='none'
            )
            axes[1].add_patch(rect)

            # Label
            label = f'{cls_name} {conf:.2f}'
            axes[1].text(
                x1, y1 - 5,
                label,
                color='white',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color_norm, alpha=0.7)
            )

        # Metrics text
        metrics_text = f"Avg Confidence: {metrics['avg_confidence']:.3f}\n\n"
        metrics_text += "Class Distribution:\n"
        for cls, count in sorted(metrics['class_distribution'].items()):
            metrics_text += f"  • {cls}: {count}\n"

        fig.text(0.02, 0.02, metrics_text, fontsize=10,
                 family='monospace', verticalalignment='bottom')

        # Save
        output_file = output_path / f"{image_path.stem}_visualized.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Saved: {output_file}")

        return metrics

    def create_comparison_grid(self, output_dir):
        """Plots all successful cases in a grid layout with original/detected rows.

        Input:  output_dir — output folder
        Output: list of metrics for all cases; comparison_grid.png is also saved
        """
        output_path = Path(output_dir)

        n_cases = len(self.successful_cases)
        fig, axes = plt.subplots(2, n_cases, figsize=(n_cases * 4, 8))
        fig.suptitle('Successful Detection Cases', fontsize=18, fontweight='bold')

        all_metrics = []

        for idx, case_stem in enumerate(self.successful_cases):
            img_path = self.find_test_image(case_stem)
            if not img_path:
                print(f"⚠️  Not found: {case_stem}")
                continue

            # Detection
            result, metrics = self.detect_with_metrics(img_path)
            all_metrics.append(metrics)

            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Original (top row)
            axes[0, idx].imshow(img_rgb)
            axes[0, idx].set_title(f'{case_stem.split("-")[0][:20]}', fontsize=10)
            axes[0, idx].axis('off')

            # Detected (bottom row)
            img_detected = img_rgb.copy()
            boxes = result.boxes

            for i in range(len(boxes)):
                box = boxes[i]
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cls_name = result.names[int(box.cls[0])]
                conf = float(box.conf[0])

                color = self.colors.get(cls_name, (128, 128, 128))

                # Draw
                cv2.rectangle(img_detected, (xyxy[0], xyxy[1]),
                              (xyxy[2], xyxy[3]), color, 2)

                label = f'{cls_name}'
                cv2.putText(img_detected, label, (xyxy[0], xyxy[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            axes[1, idx].imshow(img_detected)
            axes[1, idx].set_title(f'n={metrics["total_detections"]}, '
                                   f'conf={metrics["avg_confidence"]:.2f}',
                                   fontsize=9)
            axes[1, idx].axis('off')

        plt.tight_layout()
        output_file = output_path / "comparison_grid.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Comparison grid saved: {output_file}")

        return all_metrics

    def create_metrics_summary(self, all_metrics, output_dir):
        """Plots and saves summary charts of average confidence and total class distribution.

        Input:  all_metrics — list of case metrics
                output_dir — output folder
        Output: none (metrics_summary.png file side effect)
        """
        output_path = Path(output_dir)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Detection Performance Metrics', fontsize=16, fontweight='bold')

        # 1. Average confidence per case
        case_names = [case.split('-')[0][:15] for case in self.successful_cases]
        avg_confs = [m['avg_confidence'] for m in all_metrics]

        axes[0].bar(range(len(avg_confs)), avg_confs,
                    color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Test Case', fontsize=12)
        axes[0].set_ylabel('Average Confidence', fontsize=12)
        axes[0].set_title('Average Detection Confidence', fontsize=13)
        axes[0].set_xticks(range(len(case_names)))
        axes[0].set_xticklabels(case_names, rotation=45, ha='right')
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis='y', alpha=0.3)

        # 2. Class distribution (aggregated)
        all_classes = {}
        for m in all_metrics:
            for cls, count in m['class_distribution'].items():
                all_classes[cls] = all_classes.get(cls, 0) + count

        classes = list(all_classes.keys())
        counts = list(all_classes.values())
        colors = [self.colors.get(c, (128, 128, 128)) for c in classes]
        colors_norm = [[c / 255 for c in col] for col in colors]

        axes[1].barh(classes, counts, color=colors_norm, alpha=0.7)
        axes[1].set_xlabel('Detection Count', fontsize=12)
        axes[1].set_ylabel('Component Class', fontsize=12)
        axes[1].set_title('Component Class Distribution', fontsize=13)
        axes[1].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        output_file = output_path / "metrics_summary.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Metrics summary saved: {output_file}")

    def generate_latex_table(self, all_metrics, output_dir):
        """Generates and saves LaTeX and Markdown tables for the thesis from successful case metrics.

        Input:  all_metrics — list of case metrics
                output_dir — output folder
        Output: none (table_successful_cases.tex and .md file side effects)
        """
        output_path = Path(output_dir)

        # Calculate overall metrics
        total_dets = sum(m['total_detections'] for m in all_metrics)
        avg_conf = sum(m['avg_confidence'] for m in all_metrics) / len(all_metrics)

        # Class-wise aggregation
        class_totals = {}
        class_confs = {}

        for m in all_metrics:
            for cls, count in m['class_distribution'].items():
                class_totals[cls] = class_totals.get(cls, 0) + count

            for conf_info in m['confidence_distribution']:
                cls = conf_info['class']
                if cls not in class_confs:
                    class_confs[cls] = []
                class_confs[cls].append(conf_info['avg_conf'])

        # LaTeX table
        latex = r"""\begin{table}[h]
\centering
\caption{Detection Performance on Successful Test Cases}
\label{tab:successful_cases}
\begin{tabular}{lrr}
\hline
\textbf{Component Class} & \textbf{Count} & \textbf{Avg. Confidence} \\
\hline
"""

        for cls in sorted(class_totals.keys()):
            count = class_totals[cls]
            avg_c = sum(class_confs[cls]) / len(class_confs[cls])
            latex += f"{cls} & {count} & {avg_c:.3f} \\\\\n"

        latex += r"""\hline
\textbf{Total} & """ + f"{total_dets} & {avg_conf:.3f} \\\\\n"
        latex += r"""\hline
\end{tabular}
\end{table}"""

        # Save
        latex_file = output_path / "table_successful_cases.tex"
        with open(latex_file, 'w') as f:
            f.write(latex)

        print(f"✅ LaTeX table saved: {latex_file}")

        # Also save as markdown
        md = "| Component Class | Count | Avg. Confidence |\n"
        md += "|----------------|-------|----------------|\n"

        for cls in sorted(class_totals.keys()):
            count = class_totals[cls]
            avg_c = sum(class_confs[cls]) / len(class_confs[cls])
            md += f"| {cls} | {count} | {avg_c:.3f} |\n"

        md += f"| **Total** | **{total_dets}** | **{avg_conf:.3f}** |\n"

        md_file = output_path / "table_successful_cases.md"
        with open(md_file, 'w') as f:
            f.write(md)

        print(f"✅ Markdown table saved: {md_file}")

    def run_full_analysis(self, output_dir="thesis_figures"):
        """Runs the full analysis: produces individual visualizations, grid, summary chart, table, and JSON.

        Input:  output_dir — folder where all figures will be saved (default "thesis_figures")
        Output: none (figure, table, and metrics_summary.json file side effects)
        """
        print("\n" + "=" * 70)
        print("SUCCESSFUL CASES ANALYSIS")
        print("=" * 70 + "\n")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        all_metrics = []

        # 1. Detailed visualization for each case
        print("1️⃣  Individual case visualizations...")
        for case_stem in self.successful_cases:
            img_path = self.find_test_image(case_stem)
            if img_path:
                metrics = self.visualize_single_case(img_path, output_dir)
                all_metrics.append(metrics)

        # 2. Comparison grid
        print("\n2️⃣  Creating comparison grid...")
        all_metrics = self.create_comparison_grid(output_dir)

        # 3. Metrics summary
        print("\n3️⃣  Creating metrics summary...")
        self.create_metrics_summary(all_metrics, output_dir)

        # 4. LaTeX table
        print("\n4️⃣  Generating LaTeX table...")
        self.generate_latex_table(all_metrics, output_dir)

        # 5. JSON export
        print("\n5️⃣  Exporting metrics to JSON...")
        json_file = output_path / "metrics_summary.json"
        with open(json_file, 'w') as f:
            json.dump({
                'cases': self.successful_cases,
                'metrics': all_metrics
            }, f, indent=2)
        print(f"✅ JSON saved: {json_file}")

        print("\n" + "=" * 70)
        print(f"✅ ALL FIGURES READY: {output_path}")
        print("=" * 70)
        print("\nGenerated files:")
        print("  📊 comparison_grid.png - All cases side by side")
        print("  📊 metrics_summary.png - Performance charts")
        print("  📄 table_successful_cases.tex - LaTeX table")
        print("  📄 table_successful_cases.md - Markdown table")
        print("  💾 metrics_summary.json - Raw data")
        for case in self.successful_cases:
            print(f"  🖼️  {case}_visualized.png")
        print("\n✅ Ready to add to the thesis!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize successful detection cases')
    parser.add_argument('--model', type=str,
                        default='runs/oversample_5k/weights/best.pt',
                        help='Path to model weights')
    parser.add_argument('--output', type=str, default='thesis_figures',
                        help='Output directory for figures')

    args = parser.parse_args()

    visualizer = SuccessfulCaseVisualizer(model_path=args.model)
    visualizer.run_full_analysis(output_dir=args.output)