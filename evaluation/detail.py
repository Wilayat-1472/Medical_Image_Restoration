"""Relative edge measurements, not a diagnostic or noise-independent quality score."""
import cv2
import numpy as np


def grayscale(image):
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.shape[-1] == 1:
        return image[..., 0].astype(np.float32)
    return (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).astype(np.float32)


def edge_energy(image):
    gray = grayscale(image)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return float(np.mean(dx * dx + dy * dy))


def compare_detail(original, restored):
    before, after = edge_energy(original), edge_energy(restored)
    ratio = after / before if before > 1e-8 else None
    return {'edge_energy_ratio': ratio, 'weaker_edges': ratio is not None and ratio < 0.6,
            'interpretation': 'Lower edge energy can reflect noise removal or detail loss; inspect the original.'}
