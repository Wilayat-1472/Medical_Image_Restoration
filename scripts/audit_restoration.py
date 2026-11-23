"""Compare detail preservation, clean-input behavior and corruption-specific quality."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import math
import warnings
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from data.io import load_image, unit_float
from data.preprocessing import denormalize
from data.degradation import add_gaussian_noise, add_gaussian_blur, add_motion_blur
from evaluation.detail import edge_energy, grayscale
from evaluation.metrics import compute_all_metrics
from inference.restore import RestorationPipeline
from utils.checkpoint import load_model


@torch.inference_mode()
def legacy_restore(pipeline, image):
    tensor = pipeline.transform(image).unsqueeze(0).to(pipeline.device)
    output = pipeline.model(tensor)[0].float().cpu().numpy().transpose(1, 2, 0)
    output = denormalize(output)
    h, w = image.shape[:2]
    return cv2.resize(output, (w, h)) if output.shape[:2] != (h, w) else output


def measure(image, reference):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result = compute_all_metrics(grayscale(image), grayscale(reference), normalized=False)
    ref_energy = edge_energy(reference)
    result['reference_edge_energy_ratio'] = edge_energy(image) / ref_energy if ref_energy > 1e-8 else None
    return {k: float(v) if v is not None and math.isfinite(v) else None for k, v in result.items()}


def summarize(records):
    return {key: float(np.mean(values)) if values else None
            for key in records[0]
            for values in [[r[key] for r in records if r[key] is not None]]}


def run(checkpoint, data_dir, output_dir, samples=24):
    torch.set_num_threads(2)
    model, config = load_model(checkpoint)
    pipeline = RestorationPipeline(model, target_size=config['data']['target_size'])
    paths = sorted(Path(data_dir).rglob('*.png'))[:samples]
    if not paths:
        raise ValueError(f'No reference PNGs at {data_dir}')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = {
        'clean_input': lambda x: x.copy(),
        'mild_noise_std5': lambda x: add_gaussian_noise(x, 5, np_rng=np.random.default_rng(42)),
        'noise_std25': lambda x: add_gaussian_noise(x, 25, np_rng=np.random.default_rng(42)),
        'mild_blur_sigma0.6': lambda x: add_gaussian_blur(x, 3, 0.6),
        'strong_blur_sigma3': lambda x: add_gaussian_blur(x, 15, 3),
        'noise25_motion7': lambda x: add_motion_blur(add_gaussian_noise(x, 25, np_rng=np.random.default_rng(42)), 7),
    }
    report = {'checkpoint': str(checkpoint), 'reference_samples': len(paths),
              'reference_source': str(data_dir), 'reference_sizes': [],
              'model_proxy_size': pipeline.target_size, 'metrics': 'Grayscale reference metrics',
              'edge_ratio_note': 'Edge energy is a detail/noise proxy, not a standalone quality score.',
              'profiles': {}}
    sizes = set()
    for name, degrade in profiles.items():
        records = {strategy: [] for strategy in ['input', 'legacy_replacement', 'detail_25', 'detail_50', 'detail_100']}
        for path in paths:
            clean = load_image(path)
            sizes.add(tuple(clean.shape[:2]))
            image = unit_float(degrade(clean))
            results = {'input': image, 'legacy_replacement': legacy_restore(pipeline, image)}
            for percent in [25, 50, 100]:
                results[f'detail_{percent}'] = pipeline.restore(image, strength=percent / 100)['restored']
            for strategy, output in results.items():
                records[strategy].append(measure(output, clean))
        report['profiles'][name] = {key: summarize(value) for key, value in records.items()}
        print(name, json.dumps(report['profiles'][name]))
    report['reference_sizes'] = [list(size) for size in sorted(sizes)]
    # A controlled non-medical pattern demonstrates the resampling defect.
    y, x = np.mgrid[:384, :512]
    pattern = (0.45 + 0.2 * np.sin(2*np.pi*x/5) + 0.15 * np.sin(2*np.pi*y/7)).astype(np.float32)[..., None]
    legacy = legacy_restore(pipeline, pattern)
    corrected = pipeline.restore(pattern)['restored']
    report['high_resolution_pattern'] = {key: measure(value, pattern) for key, value in
                                         [('input', pattern), ('legacy_replacement', legacy), ('detail_50', corrected)]}
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    clean = load_image(paths[0])
    for ax, title, image in zip(axes[0], ['Clean reference', 'Previous replacement', 'Detail-preserving, 50%'],
                                 [clean, legacy_restore(pipeline, clean), pipeline.restore(clean)['restored']]):
        ax.imshow(grayscale(image), cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')
    for ax, title, image in zip(axes[1], ['Resolution test pattern (crop)', 'Previous replacement (crop)', 'Detail-preserving, 50% (crop)'],
                                 [pattern, legacy, corrected]):
        ax.imshow(grayscale(image)[80:176, 80:208], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle('Detail preservation audit — unchanged trained weights', fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / 'detail_comparison.png', dpi=160)
    plt.close(fig)
    (output_dir / 'quality_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Audit saved to', output_dir)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/colab_t4/light_unet_best.pt')
    parser.add_argument('--data_dir', default='datasets/processed/test/xray/clean')
    parser.add_argument('--output_dir', default='outputs/detail_audit')
    parser.add_argument('--samples', type=int, default=24)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == '__main__':
    main()
