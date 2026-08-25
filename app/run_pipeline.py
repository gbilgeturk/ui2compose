#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Config-driven entry point for the end-to-end pipeline.

Reads its settings from `configs/pipeline_config.yaml` and runs the pipeline
on the image named there. For an argument-driven run without a config file,
use the `ui2compose` command instead.

Run with:
    python app/run_pipeline.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

from ui2compose.infer.pipeline import run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline_config.yaml"

# Used when the config file is missing or unreadable
DEFAULT_CONFIG = {
    'input': {
        'image': 'examples/com.shazam.android-screens_screenshot_3.png',
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


def load_config() -> dict:
    """Reads the pipeline settings from configs/pipeline_config.yaml.

    Input:  none (uses the module-level CONFIG_PATH)
    Output: settings dictionary — DEFAULT_CONFIG if the file is missing or unreadable
    """
    print(f"\n📋 Loading config file: {CONFIG_PATH.relative_to(PROJECT_ROOT)}")
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ Config loaded")
            return config
        print(f"⚠️  Config file not found, using default values")
        print(f"💡 Create it at: {CONFIG_PATH}")
    except Exception as e:
        print(f"⚠️  Config read error: {e}")
        print(f"⚠️  Using default values")
    return DEFAULT_CONFIG


def main():
    """Validates and runs the end-to-end pipeline with the settings from the config, after confirmation.

    Input:  none (settings come from configs/pipeline_config.yaml)
    Output: none (side effects of calling run_pipeline, producing result files
            under the output directory, and printing a summary; sys.exit on error)
    """
    # Paths in the config are relative to the project root
    os.chdir(PROJECT_ROOT)
    print(f"📂 Project root directory: {PROJECT_ROOT}")
    print(f"📂 Working directory: {Path.cwd()}")

    config = load_config()

    image_path = config['input']['image']
    model_path = config['input']['model']
    dataset_yaml = config['input']['dataset_yaml']
    conf_threshold = config['detection']['confidence_threshold']
    enable_visualization = config['features']['visualization']

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"{config['output']['directory']}/{timestamp}"

    # File checks
    if not Path(image_path).exists():
        print(f"\n❌ Error: Image not found: {image_path}")
        print(f"💡 Hint: Check the 'input.image' path in configs/pipeline_config.yaml")
        sys.exit(1)

    if not Path(model_path).exists():
        print(f"\n❌ Error: Model not found: {model_path}")
        print(f"💡 Hint: Check the 'input.model' path in configs/pipeline_config.yaml")

        # Search for available models
        runs_dir = Path("runs")
        if runs_dir.exists():
            found_models = list(runs_dir.rglob("best.pt"))
            if found_models:
                print(f"\n📁 Models found:")
                for i, model in enumerate(found_models[:5], 1):
                    print(f"   {i}. {model}")
                print(f"\n💡 Change the 'input.model' value in the config file to one of the models above")
        sys.exit(1)

    if not Path(dataset_yaml).exists():
        print(f"\n❌ Error: Dataset YAML not found: {dataset_yaml}")
        sys.exit(1)

    # Print the settings
    print("\n" + "=" * 70)
    print("🚀 PIPELINE SETTINGS (configs/pipeline_config.yaml)")
    print("=" * 70)
    print(f"📷 Image:            {image_path}")
    print(f"🤖 Model:            {Path(model_path).name}")
    print(f"📊 Dataset YAML:     {dataset_yaml}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🕐 Timestamp:        {timestamp}")
    print(f"🎯 Confidence:       {conf_threshold}")
    print(f"🖼️  Visualization:    {'✅ On' if enable_visualization else '❌ Off'}")
    print("=" * 70 + "\n")

    # Ask for confirmation (skipped in non-interactive environments — e.g. CI)
    try:
        response = input("Press ENTER to continue (or 'q' to cancel): ")
        if response.lower() == 'q':
            print("Cancelled.")
            sys.exit(0)
    except EOFError:
        pass

    # Run the pipeline
    try:
        print("\n🚀 Starting pipeline...\n")

        results = run_pipeline(
            image_path=image_path,
            model_path=model_path,
            dataset_yaml=dataset_yaml,
            output_dir=output_dir,
            conf_threshold=conf_threshold,
            visualize=enable_visualization
        )

        print("\n" + "=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print(f"📁 Results: {results['output_dir']}")
        print(f"🔢 Detection count: {len(results['detections'])}")
        print(f"📄 Lines of code: {len(results['code'].split(chr(10)))}")
        print("=" * 70)

        # File list
        output_path = Path(results['output_dir'])
        if output_path.exists():
            files = sorted(output_path.iterdir())
            if files:
                print(f"\n📄 Generated files:")
                for f in files:
                    size = f.stat().st_size / 1024
                    print(f"   • {f.name:<35s} ({size:>6.1f} KB)")

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
