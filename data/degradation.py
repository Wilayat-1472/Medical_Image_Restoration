import random
import math
import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from .io import unit_float


def add_gaussian_noise(image: np.ndarray, std: float = 25, np_rng=None) -> np.ndarray:
    image = unit_float(image)
    np_rng = np_rng if np_rng is not None else np.random.default_rng()
    noise = np_rng.standard_normal(image.shape).astype(np.float32) * (std / 255.0)
    return np.clip(image.astype(np.float32) + noise, 0.0, 1.0)


def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.05, np_rng=None) -> np.ndarray:
    result = unit_float(image).copy()
    np_rng = np_rng if np_rng is not None else np.random.default_rng()
    mask = np_rng.random(result.shape[:2])
    result[mask < amount / 2] = 0.0
    result[mask > 1 - amount / 2] = 1.0
    return result


def add_speckle_noise(image: np.ndarray, std: float = 0.05, np_rng=None) -> np.ndarray:
    image = unit_float(image)
    np_rng = np_rng if np_rng is not None else np.random.default_rng()
    noise = np_rng.standard_normal(image.shape).astype(np.float32) * std
    return np.clip(image.astype(np.float32) + image.astype(np.float32) * noise, 0.0, 1.0)


def add_motion_blur(
    image: np.ndarray, kernel_size: int = 15, angle: float = 45
) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    rad = math.radians(angle)
    dx = int(round(center * math.cos(rad)))
    dy = int(round(center * math.sin(rad)))
    cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1.0, thickness=2)
    kernel /= kernel.sum() if kernel.sum() > 0 else 1

    if image.ndim == 2:
        blurred = cv2.filter2D(image, -1, kernel)
    else:
        blurred = np.stack([cv2.filter2D(image[..., c], -1, kernel) for c in range(image.shape[-1])], axis=-1)
    return np.clip(blurred, 0.0, 1.0)


def add_gaussian_blur(
    image: np.ndarray, kernel_size: int = 15, sigma: float = 2.0
) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    if image.ndim == 3 and blurred.ndim == 2:
        blurred = blurred[..., None]
    return np.clip(blurred, 0.0, 1.0)


def add_defocus_blur(image: np.ndarray, radius: int = 5) -> np.ndarray:
    kernel_size = 2 * radius + 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    cv2.circle(kernel, (radius, radius), radius, 1.0, thickness=-1)
    kernel /= kernel.sum() if kernel.sum() > 0 else 1

    if image.ndim == 2:
        blurred = cv2.filter2D(image, -1, kernel)
    else:
        blurred = np.stack([cv2.filter2D(image[..., c], -1, kernel) for c in range(image.shape[-1])], axis=-1)
    return np.clip(blurred, 0.0, 1.0)


def adjust_contrast(image: np.ndarray, factor: float = 0.5) -> np.ndarray:
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    result = mean + factor * (image.astype(np.float32) - mean)
    return np.clip(result, 0.0, 1.0)


def add_compression_artifacts(image: np.ndarray, quality: int = 30) -> np.ndarray:
    img_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    if img_uint8.ndim == 3 and img_uint8.shape[-1] == 3:
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc = cv2.imencode(".jpg", img_bgr, encode_param)
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        decoded = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    elif img_uint8.ndim == 3 and img_uint8.shape[-1] == 1:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc = cv2.imencode(".jpg", img_uint8, encode_param)
        decoded = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
        decoded = decoded[..., None]
    else:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc = cv2.imencode(".jpg", img_uint8, encode_param)
        decoded = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
        decoded = decoded[..., None]
    return decoded.astype(np.float32) / 255.0


def apply_mixed_degradation(image, max_degradations=3, rng=None, np_rng=None, config=None):
    rng = rng if rng is not None else random.Random()
    np_rng = np_rng if np_rng is not None else np.random.default_rng()
    config = config or {}
    image = unit_float(image)
    degraded = image.copy()
    specs = {
        'gaussian_noise': lambda c: add_gaussian_noise(degraded, rng.uniform(*c.get('std_range', [5, 70])), np_rng),
        'salt_pepper_noise': lambda c: add_salt_pepper_noise(degraded, rng.uniform(*c.get('amount_range', [0.01, 0.30])), np_rng),
        'speckle_noise': lambda c: add_speckle_noise(degraded, rng.uniform(*c.get('std_range', [0.02, 0.12])), np_rng),
        'motion_blur': lambda c: add_motion_blur(degraded, odd_kernel(c), rng.uniform(*c.get('angle_range', [0, 360]))),
        'gaussian_blur': lambda c: add_gaussian_blur(degraded, odd_kernel(c), rng.uniform(*c.get('sigma_range', [0.5, 4.0]))),
        'defocus_blur': lambda c: add_defocus_blur(degraded, rng.randint(*c.get('radius_range', [1, 10]))),
        'contrast_reduction': lambda c: adjust_contrast(degraded, rng.uniform(*c.get('factor_range', [0.3, 0.9]))),
        'compression': lambda c: add_compression_artifacts(degraded, rng.randint(*c.get('quality_range', [15, 80]))),
    }
    def odd_kernel(c):
        low, high = c.get('kernel_size_range', [3, 31])
        return rng.choice([k for k in range(low, high + 1) if k % 2])
    available = [name for name in specs if config.get(name, {}).get('enabled', True)]
    if not available:
        return {'degraded': degraded, 'degradations': []}
    mixed = config.get('mixed_degradation', {})
    maximum = mixed.get('max_degradations', max_degradations) if mixed.get('enabled', True) else 1
    if maximum < 1:
        raise ValueError('max_degradations must be positive')
    chosen = rng.sample(available, rng.randint(1, min(maximum, len(available))))
    for name in chosen:
        degraded = specs[name](config.get(name, {}))
    return {'degraded': unit_float(degraded), 'degradations': chosen}
