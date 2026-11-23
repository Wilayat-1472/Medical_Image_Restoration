import os
import sys
import subprocess
import argparse

def run_script(script_name, args=None):
    """Utility function to run a python script as a subprocess."""
    if args is None:
        args = []

    from pathlib import Path
    script_name = str(Path(__file__).resolve().parent / script_name)
    cmd = [sys.executable, script_name] + args
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    try:
        # Use subprocess.run to execute the command and stream output to the console
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Script '{script_name}' failed with return code {e.returncode}")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Medical Image Restoration - Full Pipeline Execution")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset downloading")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip data preparation")
    parser.add_argument("--config", type=str, default=None,
                        help="Force a specific config (e.g., configs/gpu_config.yaml). Auto-detected if not set.")
    args = parser.parse_args()

    print("Starting Medical Image Restoration Pipeline...")

    # 1. Download Datasets
    if not args.skip_download:
        print("\n--- Step 1: Downloading Datasets ---")
        run_script("scripts/download_all_datasets.py")
    else:
        print("\n--- Step 1: Skipping Dataset Download ---")

    # Downloads already materialize references with official splits.
    # Prepare only manually supplied raw data when downloading was skipped.
    if args.skip_download and not args.skip_prepare:
        print("\n--- Step 2: Preparing Data ---")
        run_script("scripts/prepare_data.py")
    else:
        print("\n--- Step 2: Using prepared reference images ---")

    # 3. Training
    print("\n--- Step 3: Training Model ---")
    config_path = args.config

    if config_path is None:
        try:
            import torch
            if torch.cuda.is_available():
                print("CUDA detected! Using GPU configuration for maximum performance.")
                config_path = "configs/gpu_config.yaml"
            else:
                print("CUDA NOT detected. Using CPU configuration.")
                config_path = "configs/cpu_config.yaml"
        except ImportError:
            print("PyTorch not found. Defaulting to CPU configuration.")
            config_path = "configs/cpu_config.yaml"

    if not os.path.exists(config_path):
        print(f"\n[Error] Config file not found at {config_path}")
        sys.exit(1)

    run_script("scripts/train.py", ["--config", config_path])

    print("\n" + "="*60)
    print("Pipeline Execution Completed Successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
