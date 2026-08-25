# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

from pathlib import Path
import sys
import os
from datetime import datetime
import yaml

# ============================================================================
# IMPORT PATH SETUP
# ============================================================================

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # 3 levels up

# Add the project root to the VERY BEGINNING of sys.path
if str(project_root) in sys.path:
    sys.path.remove(str(project_root))
sys.path.insert(0, str(project_root))

# Set the working directory to the project root
os.chdir(project_root)

print(f"📂 Project root directory: {project_root}")
print(f"📂 Working directory: {Path.cwd()}")

# ============================================================================
# READ THE CONFIG FILE
# ============================================================================

CONFIG_PATH = project_root / "configs" / "pipeline_config.yaml"

print(f"\n📋 Loading config file: {CONFIG_PATH.relative_to(project_root)}")

# Default values (if the config file does not exist)
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

# Read the config file
try:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ Config loaded")
    else:
        print(f"⚠️  Config file not found, using default values")
        print(f"💡 Create it at: {CONFIG_PATH}")
        config = DEFAULT_CONFIG
except Exception as e:
    print(f"⚠️  Config read error: {e}")
    print(f"⚠️  Using default values")
    config = DEFAULT_CONFIG

# Get values from the config
IMAGE_PATH = config['input']['image']
MODEL_PATH = config['input']['model']
DATASET_YAML = config['input']['dataset_yaml']
CONFIDENCE_THRESHOLD = config['detection']['confidence_threshold']
ENABLE_VISUALIZATION = config['features']['visualization']

# Create an automatic date-time folder
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = f"{config['output']['directory']}/{TIMESTAMP}"

# ============================================================================
# IMPORT THE PIPELINE MODULE
# ============================================================================

print("\n📦 Loading pipeline module...")
try:
    from src.infer.pipeline_end_to_end import run_pipeline

    print(f"✅ pipeline_end_to_end module loaded")
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print(f"\n🔍 Debug:")
    print(f"   sys.path[0]: {sys.path[0]}")
    print(f"   Pipeline file: {project_root / 'src/infer/pipeline_end_to_end.py'}")
    print(f"   Does the file exist?: {(project_root / 'src/infer/pipeline_end_to_end.py').exists()}")

    # Show the contents of src/infer/
    infer_dir = project_root / 'src/infer'
    if infer_dir.exists():
        print(f"\n📂 Contents of {infer_dir}:")
        for item in sorted(infer_dir.iterdir()):
            if item.suffix == '.py':
                print(f"   ✅ {item.name}")

    import traceback

    traceback.print_exc()
    sys.exit(1)


# ============================================================================
# SCRIPT START
# ============================================================================

def main():
    """Validates and runs the end-to-end pipeline with the settings from the config, after confirmation.

    Input:  none (takes image/model/threshold settings from module constants
            read from pipeline_config.yaml)
    Output: none (side effects of calling run_pipeline, producing result files
            under OUTPUT_DIR, and printing a summary; sys.exit on error)
    """

    # File checks
    if not Path(IMAGE_PATH).exists():
        print(f"\n❌ Error: Image not found: {IMAGE_PATH}")
        print(f"💡 Hint: Check the 'input.image' path in configs/pipeline_config.yaml")
        sys.exit(1)

    if not Path(MODEL_PATH).exists():
        print(f"\n❌ Error: Model not found: {MODEL_PATH}")
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

    if not Path(DATASET_YAML).exists():
        print(f"\n❌ Error: Dataset YAML not found: {DATASET_YAML}")
        sys.exit(1)

    # Print the settings
    print("\n" + "=" * 70)
    print("🚀 PIPELINE SETTINGS (configs/pipeline_config.yaml)")
    print("=" * 70)
    print(f"📷 Image:            {IMAGE_PATH}")
    print(f"🤖 Model:            {Path(MODEL_PATH).name}")
    print(f"📊 Dataset YAML:     {DATASET_YAML}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🕐 Timestamp:        {TIMESTAMP}")
    print(f"🎯 Confidence:       {CONFIDENCE_THRESHOLD}")
    print(f"🖼️  Visualization:    {'✅ On' if ENABLE_VISUALIZATION else '❌ Off'}")
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
            image_path=IMAGE_PATH,
            model_path=MODEL_PATH,
            dataset_yaml=DATASET_YAML,
            output_dir=OUTPUT_DIR,
            conf_threshold=CONFIDENCE_THRESHOLD,
            visualize=ENABLE_VISUALIZATION
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