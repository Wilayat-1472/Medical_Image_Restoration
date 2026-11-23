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
from inference.report import generate_report, save_report
from evaluation.explainability import visualize_explanations
from models.factory import create_model


from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Restore a single medical image")
    parser.add_argument("input", help="Path to input degraded image")
    parser.add_argument("--config", default="configs/cpu_config.yaml", help="Config file")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.pt)")
    parser.add_argument("--output", default=None, help="Output path for restored image")
    parser.add_argument("--report", default=None, help="Save quality report (json or html)")
    parser.add_argument("--explain", action="store_true", help="Generate explainability visualizations")
    parser.add_argument("--target_size", type=int, default=None, help="Target image size")
    parser.add_argument("--strength", type=float, default=0.5, help="Correction strength: 0 preserves the original, 1 applies the full model correction")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from utils.checkpoint import load_model
    model, config = load_model(args.checkpoint, config, device)

    pipeline = RestorationPipeline(model, device=device, target_size=args.target_size or config["data"]["target_size"], strength=args.strength)
    scorer = ConfidenceScorer(threshold=config["inference"].get("confidence_threshold", 0.6))
    failsafe = FailSafe(confidence_threshold=config["inference"].get("confidence_threshold", 0.6))

    import time
    start = time.perf_counter()
    result = pipeline.restore(args.input)
    elapsed = (time.perf_counter() - start) * 1000

    restored = result["restored"]

    degraded_orig = pipeline._load_image(args.input)
    degraded_display = degraded_orig

    confidence = scorer.score(degraded_display, restored)
    quality_check = failsafe.check_restoration_quality(restored, confidence["overall_confidence"])
    from evaluation.detail import compare_detail
    detail = compare_detail(degraded_orig, restored)
    if detail['weaker_edges']:
        quality_check['passes'] = False
        quality_check['warnings'].append('The output has substantially weaker edges. Reduce --strength and compare fine details.')

    output_path = args.output or f"outputs/restored_{os.path.basename(args.input)}"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    from data.io import save_image
    save_image(output_path, restored)

    print(f"\nRestored image saved to: {output_path}")
    print(f"Restoration time: {elapsed:.1f}ms")
    print(f"Confidence: {confidence['overall_confidence']:.3f} ({'heuristic pass' if confidence['is_reliable'] else 'needs review'})")
    if quality_check["warnings"]:
        print("Warnings:")
        for w in quality_check["warnings"]:
            print(f"  - {w}")

    if args.explain:
        input_tensor = pipeline.transform(degraded_orig).unsqueeze(0).to(device)
        exp_path = str(Path(output_path).with_name(Path(output_path).stem + "_explanation.png"))
        restored_tensor = None

        explanations = visualize_explanations(model, input_tensor, restored_tensor, save_path=exp_path)
        print(f"Explanation saved to: {exp_path}")

    if args.report:
        report = generate_report(
            degraded_path=args.input,
            metrics={},
            confidence=confidence,
            restoration_time_ms=elapsed,
            model_name=config["model"]["name"],
        )
        report["strength"] = args.strength
        report["detail_check"] = detail
        report["quality_check"] = quality_check
        if not quality_check["passes"]:
            report["status"] = "needs_review"
        save_report(report, args.report, format=Path(args.report).suffix.lstrip(".") or "json", degraded_image=degraded_display, restored_image=restored)
        print(f"Report saved to: {args.report}")


if __name__ == "__main__":
    main()
