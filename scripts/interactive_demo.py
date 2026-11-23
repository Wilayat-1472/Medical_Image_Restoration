from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import sys
import argparse
import torch
import cv2
import numpy as np
from inference.restore import RestorationPipeline
from inference.confidence import ConfidenceScorer
from inference.failsafe import FailSafe
from models.factory import create_model
from data.degradation import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_motion_blur,
    add_gaussian_blur,
    adjust_contrast,
)


from utils.config import load_config


def draw_comparison(degraded: np.ndarray, restored: np.ndarray, title: str = "") -> np.ndarray:
    if degraded.ndim == 2:
        degraded = np.stack([degraded] * 3, axis=-1)
    if restored.ndim == 2:
        restored = np.stack([restored] * 3, axis=-1)
    h = max(degraded.shape[0], restored.shape[0])
    w1 = degraded.shape[1]
    w2 = restored.shape[1]
    canvas = np.ones((h + 65, w1 + w2 + 10, 3), dtype=np.uint8) * 255
    degraded_uint8 = (np.clip(degraded, 0, 1) * 255).astype(np.uint8)
    restored_uint8 = (np.clip(restored, 0, 1) * 255).astype(np.uint8)
    if degraded_uint8.shape[2] == 3:
        degraded_uint8 = cv2.cvtColor(degraded_uint8, cv2.COLOR_RGB2BGR)
        restored_uint8 = cv2.cvtColor(restored_uint8, cv2.COLOR_RGB2BGR)
    canvas[5:5 + degraded_uint8.shape[0], 5:5 + degraded_uint8.shape[1]] = degraded_uint8
    canvas[5:5 + restored_uint8.shape[0], 5 + w1 + 5:5 + w1 + 5 + restored_uint8.shape[1]] = restored_uint8
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Degraded", (10, h + 30), font, 0.6, (0, 0, 0), 1)
    cv2.putText(canvas, "Restored", (10 + w1 + 5, h + 30), font, 0.6, (0, 0, 0), 1)
    if title:
        cv2.putText(canvas, title, (10, h + 55), font, 0.5, (100, 100, 100), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Interactive demo for medical image restoration")
    parser.add_argument("--config", default="configs/cpu_config.yaml", help="Config file")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.pt)")
    parser.add_argument("--image", default=None, help="Path to test image (optional)")
    parser.add_argument("--output", default="outputs/demo_output.png", help="Output path")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from utils.checkpoint import load_model
    model, config = load_model(args.checkpoint, config, device)

    pipeline = RestorationPipeline(model, device=device, target_size=config["data"]["target_size"])

    if args.image:
        img = pipeline._load_image(args.image)
        img = cv2.resize(img, (pipeline.target_size, pipeline.target_size))
    else:
        img = np.random.default_rng(42).random((256, 256, 1), dtype=np.float32) * 0.5 + 0.25

    print("\n=== Medical Image Restoration Demo ===")
    print("Select degradation type:")
    print("  1. Gaussian Noise")
    print("  2. Salt & Pepper Noise")
    print("  3. Motion Blur")
    print("  4. Gaussian Blur")
    print("  5. Low Contrast")
    print("  6. Mixed Degradation")
    print("  7. Quit")

    while True:
        try:
            choice = input("\nEnter choice (1-7): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "7":
            break

        degraded = img.copy()
        label = ""
        try:
            if choice == "1":
                std = int(input("Noise std (5-70, default=25): ") or "25")
                degraded = add_gaussian_noise(img, std=std)
                label = f"Gaussian Noise (std={std})"
            elif choice == "2":
                amt = float(input("Noise amount (0.01-0.50, default=0.05): ") or "0.05")
                degraded = add_salt_pepper_noise(img, amount=amt)
                label = f"Salt & Pepper (amount={amt:.2f})"
            elif choice == "3":
                k = int(input("Kernel size (3-31 odd, default=15): ") or "15")
                if k % 2 == 0:
                    k += 1
                degraded = add_motion_blur(img, kernel_size=k)
                label = f"Motion Blur (k={k})"
            elif choice == "4":
                k = int(input("Kernel size (3-31 odd, default=15): ") or "15")
                if k % 2 == 0:
                    k += 1
                sigma = float(input("Sigma (0.5-5.0, default=2.0): ") or "2.0")
                degraded = add_gaussian_blur(img, kernel_size=k, sigma=sigma)
                label = f"Gaussian Blur (k={k}, sigma={sigma})"
            elif choice == "5":
                factor = float(input("Contrast factor (0.1-1.0, default=0.5): ") or "0.5")
                degraded = adjust_contrast(img, factor=factor)
                label = f"Low Contrast (factor={factor:.2f})"
            elif choice == "6":
                degraded = add_gaussian_noise(img, std=np.random.randint(10, 40))
                degraded = add_motion_blur(degraded, kernel_size=np.random.choice([5, 9, 15]))
                degraded = adjust_contrast(degraded, factor=np.random.uniform(0.3, 0.7))
                label = "Mixed (Noise + Blur + Low Contrast)"
            else:
                print("Invalid choice")
                continue

            print("Running restoration...")
            result = pipeline.restore(degraded)
            restored = result["restored"]

            scorer = ConfidenceScorer()
            confidence = scorer.score(degraded, restored)
            print(f"Confidence: {confidence['overall_confidence']:.3f} ({'heuristic pass' if confidence['is_reliable'] else 'needs review'})")

            failsafe = FailSafe()
            check = failsafe.check_restoration_quality(restored, confidence["overall_confidence"])
            if check["warnings"]:
                for w in check["warnings"]:
                    print(f"  Warning: {w}")

            comparison = draw_comparison(degraded, restored, label)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(args.output, comparison)
            print(f"Comparison saved to {args.output}")


        except Exception as e:
            print(f"Error: {e}")

    print("Demo ended.")


if __name__ == "__main__":
    main()
