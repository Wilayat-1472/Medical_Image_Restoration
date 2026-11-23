import time
import numpy as np
import torch
from typing import Dict, List, Optional, Callable
from evaluation.metrics import compute_psnr, compute_ssim


def benchmark_inference_time(
    model: torch.nn.Module,
    input_size: tuple = (1, 3, 256, 256),
    num_warmup: int = 10,
    num_runs: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    device = torch.device(device)
    if num_runs < 1 or num_warmup < 0:
        raise ValueError("Benchmark counts must be positive (warmup may be zero)")
    model = model.eval().to(device)
    dummy = torch.randn(input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - start)

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times) * 1000),
        "std_ms": float(np.std(times) * 1000),
        "min_ms": float(np.min(times) * 1000),
        "max_ms": float(np.max(times) * 1000),
        "fps": float(1.0 / np.mean(times)),
    }


def _median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    import cv2
    if img.ndim == 3:
        return cv2.medianBlur((img * 255).astype(np.uint8), ksize).astype(np.float32) / 255.0
    return cv2.medianBlur((img * 255).astype(np.uint8), ksize)[..., None].astype(np.float32) / 255.0


def _gaussian_filter(img: np.ndarray, ksize: int = 5, sigma: float = 1.5) -> np.ndarray:
    import cv2
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def _wiener_filter(img: np.ndarray, kernel_size: int = 5, noise_var: float = 0.01) -> np.ndarray:
    from scipy.signal import wiener
    if img.ndim == 3:
        return np.stack([wiener(img[..., c], kernel_size, noise_var) for c in range(img.shape[-1])], axis=-1)
    return wiener(img, kernel_size, noise_var)


def _bilateral_filter(img: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    import cv2
    if img.ndim == 3:
        return cv2.bilateralFilter((img * 255).astype(np.uint8), d, sigma_color, sigma_space).astype(np.float32) / 255.0
    return cv2.bilateralFilter((img * 255).astype(np.uint8), d, sigma_color, sigma_space)[..., None].astype(np.float32) / 255.0


def compare_with_classical(
    model: torch.nn.Module,
    degraded_images: list,
    clean_images: list,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Dict[str, float]]:
    if not clean_images or len(clean_images) != len(degraded_images):
        raise ValueError("Provide equally sized, nonempty reference/degraded image lists")
    def rgb(image):
        if image.ndim == 2:
            image = image[..., None]
        return np.repeat(image, 3, axis=-1) if image.shape[-1] == 1 else image
    degraded_images = [rgb(x) for x in degraded_images]
    clean_images = [rgb(x) for x in clean_images]
    methods = {
        "median_filter": lambda x: _median_filter(x, ksize=3),
        "gaussian_filter": lambda x: _gaussian_filter(x, ksize=5, sigma=1.5),
        "wiener_filter": lambda x: _wiener_filter(x, kernel_size=5, noise_var=0.01),
        "bilateral_filter": lambda x: _bilateral_filter(x, d=9, sigma_color=75, sigma_space=75),
    }

    results = {}

    for name, method in methods.items():
        psnrs, ssims = [], []
        for degraded, clean in zip(degraded_images, clean_images):
            restored = np.nan_to_num(np.clip(method(degraded), 0, 1), nan=0.0)
            psnrs.append(compute_psnr(restored, clean))
            ssims.append(compute_ssim(restored, clean))
        results[name] = {
            "psnr": float(np.mean(psnrs)),
            "ssim": float(np.mean(ssims)),
        }

    from inference.restore import RestorationPipeline
    pipeline = RestorationPipeline(model, device, target_size=clean_images[0].shape[0])
    psnrs, ssims = [], []
    for degraded, clean in zip(degraded_images, clean_images):
        restored = pipeline.restore(degraded)['restored']
        if clean.ndim == 3 and clean.shape[-1] == 1:
            clean = np.repeat(clean, 3, axis=-1)
        psnrs.append(compute_psnr(restored, clean))
        ssims.append(compute_ssim(restored, clean))
    results['deep_learning_model'] = {'psnr': float(np.mean(psnrs)), 'ssim': float(np.mean(ssims))}
    return results
