from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
from inference.restore import RestorationPipeline
from models.factory import create_model


from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Batch restore medical images")
    parser.add_argument("input_dir", help="Directory with degraded images")
    parser.add_argument("--config", default="configs/cpu_config.yaml", help="Config file")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.pt)")
    parser.add_argument("--output_dir", default="outputs/restored", help="Output directory")
    parser.add_argument("--target_size", type=int, default=None, help="Target size")
    parser.add_argument("--extensions", nargs="+", default=[".png", ".jpg", ".jpeg", ".tif", ".bmp"], help="File extensions")
    parser.add_argument("--strength", type=float, default=0.5, help="Correction strength: 0 preserves the original, 1 applies the full model correction")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from utils.checkpoint import load_model
    model, config = load_model(args.checkpoint, config, device)

    pipeline = RestorationPipeline(model, device=device, target_size=args.target_size or config["data"]["target_size"], strength=args.strength)
    os.makedirs(args.output_dir, exist_ok=True)

    image_files = []
    for dirpath, _, filenames in os.walk(args.input_dir):
        for f in filenames:
            if any(f.lower().endswith(ext) for ext in args.extensions):
                image_files.append(os.path.join(dirpath, f))

    if not image_files:
        print(f"No images found in {args.input_dir}")
        return

    print(f"Restoring {len(image_files)} images...")
    failures = []
    for img_path in tqdm(image_files):
        try:
            result = pipeline.restore(img_path)
            restored = result["restored"]

            rel_path = os.path.relpath(img_path, args.input_dir)
            out_path = os.path.join(args.output_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            from data.io import save_image
            save_image(out_path, restored)
        except Exception as e:
            failures.append(f"{img_path}: {e}")
            print(f"Error processing {img_path}: {e}")

    if failures:
        raise RuntimeError(f"{len(failures)} images failed restoration")
    print(f"\nAll restored images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
