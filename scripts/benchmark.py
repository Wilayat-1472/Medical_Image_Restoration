from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import argparse
import torch
import cv2
import numpy as np
from models.factory import create_model
from evaluation.benchmark import benchmark_inference_time, compare_with_classical
from evaluation.edge_cases import EdgeCaseTester
from data.preprocessing import MedicalTransform


from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Benchmark model performance")
    parser.add_argument("--config", default="configs/cpu_config.yaml", help="Config file")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.pt)")
    parser.add_argument("--data_dir", default="./datasets/processed", help="Test data directory")
    parser.add_argument("--output", default="outputs/benchmark_results.json", help="Output path")
    parser.add_argument("--input_size", type=int, default=256, help="Input image size for benchmark")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from utils.checkpoint import load_model
    model, config = load_model(args.checkpoint, config, device)

    results = {}

    print("Benchmarking inference time...")
    speed = benchmark_inference_time(
        model,
        input_size=(1, config["model"]["in_channels"], args.input_size, args.input_size),
        device=device,
    )
    results["inference_speed"] = speed
    print(f"  Mean: {speed['mean_ms']:.2f}ms  |  FPS: {speed['fps']:.1f}")

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

    if clean_images:
        print("Comparing with classical methods...")
        classical = compare_with_classical(model, degraded_images, clean_images, device)
        results["classical_comparison"] = classical
        for method, scores in classical.items():
            print(f"  {method}: PSNR={scores['psnr']:.2f}, SSIM={scores['ssim']:.4f}")

        print("Testing edge cases...")
        tester = EdgeCaseTester(model, device, target_size=size)
        edge_results = tester.run_all(clean_images[:10])
        results["edge_cases"] = edge_results

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    import json

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            return super().default(obj)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\nBenchmark results saved to {args.output}")


if __name__ == "__main__":
    main()
