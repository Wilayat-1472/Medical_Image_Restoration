from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import argparse
import torch
import cv2
import numpy as np
from models.factory import create_model
from evaluation.edge_cases import EdgeCaseTester
from data.preprocessing import MedicalTransform


from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Test edge cases for restoration model")
    parser.add_argument("--config", default="configs/cpu_config.yaml", help="Config file")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.pt)")
    parser.add_argument("--data_dir", default="./datasets/processed", help="Test images directory")
    parser.add_argument("--output", default="outputs/edge_case_results.json", help="Output path")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from utils.checkpoint import load_model
    model, config = load_model(args.checkpoint, config, device)

    transform = MedicalTransform(target_size=(config["data"]["target_size"], config["data"]["target_size"]), augment=False)

    from data.dataset import MedicalRestorationDataset
    from data.io import load_image
    references = MedicalRestorationDataset(args.data_dir, split='test',
        seed=config['project'].get('seed', 42),
        train_split=config['data'].get('train_split', 0.8), val_split=config['data'].get('val_split', 0.1))
    clean_images = []
    degraded_images = []
    for path, _ in references.pairs[:20]:
        img = load_image(path)
        size = args.input_size if hasattr(args, 'input_size') else config['data']['target_size']
        img = cv2.resize(img, (size, size))
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        clean_images.append(img)
        degraded_images.append(cv2.GaussianBlur(img, (15, 15), 2.0))

    if not clean_images:
        print(f"No test images found in {args.data_dir}")
        return

    print(f"Testing edge cases with {len(clean_images)} clean images...")
    tester = EdgeCaseTester(model, device, target_size=size)
    results = tester.run_all(clean_images)

    print("\n=== Edge Case Test Results ===")
    for test_name, metrics in results.items():
        print(f"\n{test_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
