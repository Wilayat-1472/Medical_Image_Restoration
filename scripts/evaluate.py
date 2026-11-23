from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm

from data.dataloader import create_test_loader
from data.preprocessing import denormalize
from models.factory import create_model
from evaluation.metrics import compute_all_metrics


from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate raw-model synthetic restoration at training resolution; use audit_restoration.py for deployed detail preservation")
    parser.add_argument("--config", default="configs/cpu_config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--data_dir", default="./datasets/processed", help="Path to test data")
    parser.add_argument("--output", default="outputs/evaluation_results.json", help="Output path")
    args = parser.parse_args()

    config = load_config(args.config)
    config["data"]["root_dir"] = args.data_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from utils.checkpoint import load_model
    model, config = load_model(args.checkpoint, config, device)

    test_loader = create_test_loader(
        root_dir=args.data_dir,
        batch_size=1,
        num_workers=config["training"].get("num_workers", 0),
        target_size=config["data"]["target_size"],
        seed=config['project'].get('seed', 42),
        train_split=config['data'].get('train_split', 0.8),
        val_split=config['data'].get('val_split', 0.1),
        degradation_config=config['data'].get('degradation'),
    )

    all_metrics = []
    baseline_metrics = []
    print(f"Evaluating on {len(test_loader)} samples...")
    for batch in tqdm(test_loader):
        degraded = batch["degraded"].to(device)
        clean = batch["clean"].to(device)

        with torch.no_grad():
            restored = model(degraded)

        for i in range(degraded.shape[0]):
            metrics = compute_all_metrics(restored[i].cpu(), clean[i].cpu())
            all_metrics.append(metrics)
            baseline_metrics.append(compute_all_metrics(degraded[i].cpu(), clean[i].cpu()))

    avg_metrics = {}
    if not all_metrics:
        raise ValueError("No held-out test images were found")
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics if key in m]
        avg_metrics[key] = float(np.mean(values))
        avg_metrics[f"{key}_std"] = float(np.std(values))

    for key in ('psnr', 'ssim'):
        avg_metrics[f'input_{key}'] = float(np.mean([m[key] for m in baseline_metrics]))
        avg_metrics[f'{key}_gain'] = avg_metrics[key] - avg_metrics[f'input_{key}']
    avg_metrics['samples'] = len(all_metrics)
    print("\n=== Evaluation Results ===")
    for key, val in avg_metrics.items():
        print(f"  {key}: {val:.4f}")

    avg_metrics["evaluation_scope"] = "raw_model_on_resized_synthetic_inputs"
    avg_metrics["model_input_size"] = config["data"]["target_size"]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(args.output, "w") as f:
        json.dump(avg_metrics, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
