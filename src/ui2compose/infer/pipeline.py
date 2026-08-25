# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path
import json
import sys
from collections import Counter

from ui2compose.infer.detect_components import ComponentDetector
from ui2compose.infer.build_ui_graph import UIGraphBuilder
from ui2compose.infer.generate_compose_code import ComposeCodeGenerator


def print_detection_statistics(detections: list, title: str = "Detection Statistics"):
    """Prints detection statistics (class distribution, confidence range) to the console.

    Input:  detections — list of detection dictionaries
            title — heading text to show in the output
    Output: none (side effect of printing to the console)
    """
    print(f"\n{'=' * 60}")
    print(f"📊 {title}")
    print(f"{'=' * 60}")

    if not detections:
        print("  No detections found!")
        return

    # Count by class
    class_counts = Counter(d['class_name'] for d in detections)

    print(f"\n  Total components: {len(detections)}")
    print(f"\n  Class distribution:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        percentage = (count / len(detections)) * 100
        print(f"    {class_name:15s}: {count:3d} ({percentage:5.1f}%)")

    # Confidence stats
    confidences = [d['confidence'] for d in detections]
    print(f"\n  Confidence range:")
    print(f"    Min:  {min(confidences):.3f}")
    print(f"    Max:  {max(confidences):.3f}")
    print(f"    Avg:  {sum(confidences) / len(confidences):.3f}")


def run_pipeline(
        image_path: str,
        model_path: str,
        dataset_yaml: str,
        output_dir: str = "output",
        conf_threshold: float = 0.3,
        visualize: bool = True
):
    """Runs the full pipeline: detection -> graph building -> Compose code generation.

    Input:  image_path — input screenshot path
            model_path — YOLO model weights file path
            dataset_yaml — dataset configuration file path
            output_dir — output folder (default "output")
            conf_threshold — detection confidence threshold (default 0.3)
            visualize — if True, the detection visualization is also saved
    Output: result dictionary with keys {'detections', 'graph', 'code', 'output_dir'}
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("🚀 UI-to-Compose Pipeline (ENHANCED VERSION)")
    print("=" * 60)
    print(f"\n📷 Input: {image_path}")
    print(f"🤖 Model: {Path(model_path).name}")
    print(f"🎯 Confidence threshold: {conf_threshold}")

    # ========== STEP 1: DETECT COMPONENTS ==========
    print("\n" + "─" * 60)
    print("[1/4] 🔍 Detecting UI Components...")
    print("─" * 60)

    detector = ComponentDetector(model_path, dataset_yaml)
    detections = detector.detect(image_path, conf_threshold)

    print(f"      ✓ Detections: {len(detections)} components")

    # Save detections
    detections_json = output_path / "1_detections.json"
    with open(detections_json, 'w') as f:
        json.dump(detections, f, indent=2)
    print(f"      ✓ Saved: {detections_json}")

    # Print statistics
    print_detection_statistics(detections, "Detection Results")

    # Visualize detections
    if visualize:
        vis_path = output_path / "1_detections_viz.png"
        detector.visualize(image_path, detections, str(vis_path))
        print(f"      ✓ Visualization: {vis_path}")

    # ========== STEP 2: BUILD UI GRAPH ==========
    print("\n" + "─" * 60)
    print("[2/4] 🌳 Building UI Hierarchy Graph...")
    print("─" * 60)

    builder = UIGraphBuilder()
    graph = builder.build_graph(detections)

    print(f"      ✓ Nodes: {len(graph['nodes'])}")
    print(f"      ✓ Edges: {len(graph['edges'])}")
    print(f"      ✓ Root components: {len(graph['hierarchy']['roots'])}")

    # Save graph JSON
    graph_json = output_path / "3_ui_graph.json"
    with open(graph_json, 'w') as f:
        json.dump(graph, f, indent=2)
    print(f"      ✓ Saved: {graph_json}")

    # Print hierarchy preview
    print(f"\n      Hierarchy preview:")

    def print_tree(node, indent=0, max_depth=2):
        """Prints the hierarchy tree to the console in indented form.

        Input:  node — hierarchy node dictionary (contains 'class' and 'children')
                indent — current indentation level
                max_depth — maximum depth to print
        Output: none (side effect of printing to the console)
        """
        if indent > max_depth:
            return
        prefix = "      " + "  " * indent
        print(f"{prefix}└─ {node['class']}")
        for child in node.get('children', [])[:2]:
            print_tree(child, indent + 1, max_depth)
        if len(node.get('children', [])) > 2:
            print(f"{prefix}   └─ ... ({len(node.get('children', [])) - 2} more)")

    for i, root in enumerate(graph['hierarchy']['roots'][:2]):
        print(f"\n      Root {i + 1}:")
        print_tree(root)

    if len(graph['hierarchy']['roots']) > 2:
        print(f"\n      ... and {len(graph['hierarchy']['roots']) - 2} more roots")

    # ========== STEP 3: GENERATE COMPOSE CODE ==========
    print("\n" + "─" * 60)
    print("[3/4] 📝 Generating Jetpack Compose Code...")
    print("─" * 60)

    generator = ComposeCodeGenerator()
    code = generator.generate(graph)

    # Count lines
    num_lines = len(code.split('\n'))
    num_composables = code.count('@Composable')

    print(f"      ✓ Generated {num_lines} lines of code")
    print(f"      ✓ Composable functions: {num_composables}")

    # Save code
    code_path = output_path / "4_GeneratedScreen.kt"
    generator.save_to_file(code, str(code_path))

    # Print code preview
    print(f"\n      Code preview (first 15 lines):")
    code_lines = code.split('\n')
    for i, line in enumerate(code_lines[:15], 1):
        print(f"      {i:3d} | {line}")
    if len(code_lines) > 15:
        print(f"      ... | ({len(code_lines) - 15} more lines)")

    # ========== STEP 4: SUMMARY ==========
    print("\n" + "─" * 60)
    print("[4/4] 📊 Pipeline Summary")
    print("─" * 60)

    print(f"\n  📷 Input:  {Path(image_path).name}")
    print(f"  📁 Output: {output_dir}/")
    print(f"\n  📦 Results:")
    print(f"     • Components detected:  {len(detections)}")
    print(f"     • Relationships found: {len(graph['edges'])}")
    print(f"     • Root components:     {len(graph['hierarchy']['roots'])}")
    print(f"     • Generated code:      {num_lines} lines")

    print(f"\n  📄 Output files:")
    output_files = sorted(output_path.glob("*"))
    for f in output_files:
        size = f.stat().st_size / 1024
        print(f"     • {f.name:<30s} ({size:>6.1f} KB)")

    print("\n" + "=" * 60)
    print("✅ Pipeline Completed Successfully!")
    print("=" * 60)

    # Final class distribution
    print(f"\n📊 Final Component Distribution:")
    final_classes = Counter(d['class_name'] for d in detections)
    for cls, count in sorted(final_classes.items()):
        print(f"   {cls:15s}: {count:2d}")

    return {
        'detections': detections,
        'graph': graph,
        'code': code,
        'output_dir': str(output_path)
    }


def main():
    """Reads command-line arguments, validates the inputs and runs the pipeline.

    Input:  none (arguments are read from the command line)
    Output: none (pipeline outputs are written to the output folder; sys.exit on error)
    """
    parser = argparse.ArgumentParser(
        description="UI Screenshot to Jetpack Compose Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("image", type=str, help="Input screenshot path")
    parser.add_argument("--model", type=str,
                        default="runs/oversample_5k/weights/best.pt",
                        help="YOLO model path")
    parser.add_argument("--dataset-yaml", type=str,
                        default="configs/dataset.yaml",
                        help="Dataset config YAML")
    parser.add_argument("--output", type=str, default="output",
                        help="Output directory")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Detection confidence threshold")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization")

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.image).exists():
        print(f"❌ Error: Image not found: {args.image}")
        sys.exit(1)

    if not Path(args.model).exists():
        print(f"❌ Error: Model not found: {args.model}")
        print(f"\n🔍 Searching for available models...")
        runs_dir = Path("runs")
        if runs_dir.exists():
            found_models = list(runs_dir.rglob("best.pt"))
            if found_models:
                print(f"✓ Found {len(found_models)} model(s):")
                for model in found_models[:5]:
                    print(f"  - {model}")
                print(f"\n💡 Usage: python {sys.argv[0]} {args.image} --model {found_models[0]}")
            else:
                print("  No models found")
        sys.exit(1)

    # Run pipeline
    try:
        results = run_pipeline(
            image_path=args.image,
            model_path=args.model,
            dataset_yaml=args.dataset_yaml,
            output_dir=args.output,
            conf_threshold=args.conf,
            visualize=not args.no_viz
        )

        print(f"\n💾 All results saved to: {results['output_dir']}")

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()